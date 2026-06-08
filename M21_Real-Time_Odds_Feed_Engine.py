"""
The Match Oracle – Module 21: Real-Time Odds Feed Engine (REFINED)
==============================================================
Fetches and normalizes live odds from multiple sources.
Provides a unified interface for odds data across the pipeline.

REFINEMENTS IN THIS VERSION:
---------------------------
1. FIXED: Proper provider abstraction with BaseProvider class
2. FIXED: Retry logic and timeout handling with exponential backoff
3. FIXED: Odds normalization and consensus calculation
4. ADDED: Support for multiple real providers (The Odds API, API-Football)
5. ADDED: WebSocket support for real-time updates
6. ADDED: Provider health monitoring and automatic failover
7. ADDED: Odds movement tracking and alert generation
8. ADDED: Best odds aggregation across providers
9. ADDED: Arbitrage opportunity detection within feed
10. ADDED: Rate limiting with token bucket algorithm
11. ADDED: Persistent cache with TTL
12. ADDED: Comprehensive error recovery with fallback providers

Feeds into:
    Module 20 (market drift)
    Module 11 (final decision)
    Module 7 (AI intelligence)
    Module 22 (arbitrage detection)

Usage:
    from module21 import OddsFeedEngine, TheOddsApiProvider, ApiFootballProvider
    
    # Initialize engine
    engine = OddsFeedEngine(providers=[TheOddsApiProvider(api_key="xxx")])
    
    # Get latest odds
    odds = engine.get_latest("arsenal_chelsea")
    print(f"Home: {odds.home_odds:.2f}, Away: {odds.away_odds:.2f}")
    
    # Subscribe to real-time updates
    engine.subscribe(lambda update: print(f"Odds changed: {update}"))
    
    # Refresh all odds
    engine.refresh_all()
"""
from __future__ import annotations

import json
import logging
import time
import threading
import queue
import urllib.request
import urllib.parse
import urllib.error
import asyncio
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any, Tuple, Callable, Union
from collections import OrderedDict
from enum import Enum
from contextlib import contextmanager

# Set up logging
logger = logging.getLogger("oracle_beast.module21")

# WebSocket support (optional)
try:
    import websocket
    _WEBSOCKET_AVAILABLE = True
except ImportError:
    _WEBSOCKET_AVAILABLE = False
    logger.warning("websocket-client not installed - WebSocket support disabled")


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — ENUMS
# ═══════════════════════════════════════════════════════════════

class ProviderStatus(Enum):
    """Status of an odds provider."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    DISABLED = "DISABLED"


class OddsUpdateType(Enum):
    """Type of odds update."""
    SIGNIFICANT = "SIGNIFICANT"  # >2% change
    MINOR = "MINOR"              # 1-2% change
    TRIVIAL = "TRIVIAL"          # <1% change
    ARBITRAGE = "ARBITRAGE"      # Arbitrage opportunity detected


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class MarketOdds:
    """Normalized odds from a single source."""
    match_id: str
    home_odds: float
    draw_odds: float
    away_odds: float
    source: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    last_updated: Optional[str] = None
    bookmaker: str = ""
    
    @property
    def implied_home_prob(self) -> float:
        """Implied home win probability (with margin)."""
        return 1.0 / self.home_odds if self.home_odds > 0 else 0.0
    
    @property
    def implied_draw_prob(self) -> float:
        """Implied draw probability (with margin)."""
        return 1.0 / self.draw_odds if self.draw_odds > 0 else 0.0
    
    @property
    def implied_away_prob(self) -> float:
        """Implied away win probability (with margin)."""
        return 1.0 / self.away_odds if self.away_odds > 0 else 0.0
    
    @property
    def total_margin(self) -> float:
        """Bookmaker margin (sum of implied probabilities - 1)."""
        return self.implied_home_prob + self.implied_draw_prob + self.implied_away_prob - 1.0
    
    def normalized_probs(self) -> Tuple[float, float, float]:
        """Return normalized probabilities (sum to 1.0)."""
        total = self.implied_home_prob + self.implied_draw_prob + self.implied_away_prob
        if total > 0:
            return (self.implied_home_prob / total,
                    self.implied_draw_prob / total,
                    self.implied_away_prob / total)
        return (0.33, 0.34, 0.33)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "match_id": self.match_id,
            "home_odds": self.home_odds,
            "draw_odds": self.draw_odds,
            "away_odds": self.away_odds,
            "source": self.source,
            "bookmaker": self.bookmaker,
            "timestamp": self.timestamp,
            "last_updated": self.last_updated,
        }
    
    def is_fresh(self, max_age_seconds: int = 60) -> bool:
        """Check if odds are fresh enough."""
        return (time.time() - self.timestamp) <= max_age_seconds


@dataclass
class OddsUpdate:
    """Notification of odds update."""
    match_id: str
    old_odds: MarketOdds
    new_odds: MarketOdds
    home_move: float
    draw_move: float
    away_move: float
    home_move_pct: float
    draw_move_pct: float
    away_move_pct: float
    update_type: OddsUpdateType = OddsUpdateType.MINOR
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "match_id": self.match_id,
            "home_move": round(self.home_move, 3),
            "draw_move": round(self.draw_move, 3),
            "away_move": round(self.away_move, 3),
            "home_move_pct": round(self.home_move_pct, 3),
            "update_type": self.update_type.value,
            "timestamp": self.timestamp,
        }


@dataclass
class ProviderHealth:
    """Health status of a provider."""
    status: ProviderStatus = ProviderStatus.HEALTHY
    success_rate: float = 1.0
    avg_latency: float = 0.0
    consecutive_failures: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    last_success: Optional[float] = None
    last_error: Optional[str] = None
    
    def record_success(self, latency: float) -> None:
        """Record a successful request."""
        self.total_requests += 1
        self.consecutive_failures = 0
        self.last_success = time.time()
        self.avg_latency = (self.avg_latency * (self.total_requests - 1) + latency) / self.total_requests
        self._update_status()
    
    def record_failure(self, error: str) -> None:
        """Record a failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        self.consecutive_failures += 1
        self.last_error = error
        self._update_status()
    
    def _update_status(self) -> None:
        """Update status based on success rate and consecutive failures."""
        if self.total_requests > 0:
            self.success_rate = (self.total_requests - self.failed_requests) / self.total_requests
        
        if self.consecutive_failures >= 5:
            self.status = ProviderStatus.UNHEALTHY
        elif self.consecutive_failures >= 3:
            self.status = ProviderStatus.DEGRADED
        elif self.success_rate < 0.7 and self.total_requests > 10:
            self.status = ProviderStatus.DEGRADED
        else:
            self.status = ProviderStatus.HEALTHY


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — PROVIDER INTERFACE
# ═══════════════════════════════════════════════════════════════

class OddsProvider(ABC):
    """Abstract base class for odds providers."""
    
    def __init__(self, name: str, timeout: int = 15, retries: int = 3):
        self.name = name
        self.timeout = timeout
        self.retries = retries
        self.health = ProviderHealth()
        self._last_error: Optional[str] = None
    
    @abstractmethod
    def fetch_odds(self, match_id: str, **kwargs) -> Optional[MarketOdds]:
        """Fetch odds for a specific match."""
        pass
    
    @abstractmethod
    def fetch_all_odds(self, league_id: Optional[str] = None, **kwargs) -> List[MarketOdds]:
        """Fetch odds for all available matches."""
        pass
    
    def _request_with_retry(
        self, 
        url: str, 
        headers: Dict = None, 
        params: Dict = None,
        method: str = "GET",
        data: bytes = None,
    ) -> Optional[Dict]:
        """Make HTTP request with retry logic and exponential backoff."""
        for attempt in range(self.retries):
            start_time = time.time()
            try:
                full_url = url
                if params and method == "GET":
                    full_url = f"{url}?{urllib.parse.urlencode(params)}"
                
                req = urllib.request.Request(full_url, headers=headers or {}, method=method, data=data)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    response = json.loads(r.read().decode("utf-8"))
                    latency = time.time() - start_time
                    self.health.record_success(latency)
                    return response
                    
            except urllib.error.HTTPError as e:
                error_msg = f"HTTP {e.code} for {url}"
                self._last_error = error_msg
                
                # Rate limit - backoff and retry
                if e.code == 429:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"{self.name}: Rate limited, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    continue
                
                # Server error - retry
                if e.code >= 500 and attempt < self.retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                
                self.health.record_failure(error_msg)
                return None
                
            except Exception as e:
                error_msg = str(e)
                self._last_error = error_msg
                
                if attempt < self.retries - 1:
                    wait_time = 2 ** attempt
                    logger.debug(f"{self.name}: Request failed, retry {attempt+1}/{self.retries} in {wait_time}s")
                    time.sleep(wait_time)
                    continue
                
                self.health.record_failure(error_msg)
                return None
        
        return None
    
    def get_health(self) -> ProviderHealth:
        """Get provider health status."""
        return self.health
    
    def is_healthy(self) -> bool:
        """Check if provider is healthy enough to use."""
        return self.health.status in (ProviderStatus.HEALTHY, ProviderStatus.DEGRADED)


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — REAL PROVIDERS
# ═══════════════════════════════════════════════════════════════

class TheOddsApiProvider(OddsProvider):
    """
    Provider for The Odds API (the-odds-api.com).
    
    Requires API key. Provides odds from multiple bookmakers.
    """
    
    def __init__(self, api_key: str, timeout: int = 15):
        super().__init__("TheOddsAPI", timeout)
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"
    
    def fetch_odds(self, match_id: str, **kwargs) -> Optional[MarketOdds]:
        """Fetch odds for a specific match."""
        sport = kwargs.get("sport", "soccer_epl")
        url = f"{self.base_url}/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "uk,us,eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
        }
        
        data = self._request_with_retry(url, params=params)
        if not data or not isinstance(data, list):
            return None
        
        # Find the specific fixture
        for fixture in data:
            fixture_id = f"{fixture.get('home_team', '')}_{fixture.get('away_team', '')}".lower().replace(" ", "_")
            if fixture_id == match_id or match_id in fixture_id:
                return self._parse_fixture(fixture)
        
        return None
    
    def fetch_all_odds(self, league_id: Optional[str] = None, **kwargs) -> List[MarketOdds]:
        """Fetch odds for all available matches."""
        sport = kwargs.get("sport", "soccer_epl")
        url = f"{self.base_url}/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "uk,us,eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
        }
        
        if league_id:
            params["leagueId"] = league_id
        
        data = self._request_with_retry(url, params=params)
        if not data or not isinstance(data, list):
            return []
        
        results = []
        for fixture in data:
            odds = self._parse_fixture(fixture)
            if odds:
                results.append(odds)
        
        return results
    
    def _parse_fixture(self, fixture: Dict) -> Optional[MarketOdds]:
        """Parse fixture data into MarketOdds."""
        try:
            home_team = fixture.get("home_team", "")
            away_team = fixture.get("away_team", "")
            match_id = f"{home_team}_{away_team}".lower().replace(" ", "_")
            
            # Get consensus odds (average across bookmakers)
            bookmakers = fixture.get("bookmakers", [])
            home_odds = []
            away_odds = []
            draw_odds = []
            
            for bk in bookmakers:
                for market in bk.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name", "")
                        price = outcome.get("price", 0)
                        if name == home_team and price > 0:
                            home_odds.append(price)
                        elif name == away_team and price > 0:
                            away_odds.append(price)
                        elif name == "Draw" and price > 0:
                            draw_odds.append(price)
            
            if not home_odds or not away_odds:
                return None
            
            return MarketOdds(
                match_id=match_id,
                home_odds=round(sum(home_odds) / len(home_odds), 2),
                draw_odds=round(sum(draw_odds) / len(draw_odds), 2) if draw_odds else 3.50,
                away_odds=round(sum(away_odds) / len(away_odds), 2),
                source=self.name,
                bookmaker="consensus",
                last_updated=fixture.get("commence_time"),
            )
        except Exception as e:
            logger.error(f"{self.name}: Failed to parse fixture - {e}")
            return None


class ApiFootballProvider(OddsProvider):
    """
    Provider for API-Football (api-football.com).
    
    Requires API key. Provides odds and fixture data.
    """
    
    def __init__(self, api_key: str, timeout: int = 15):
        super().__init__("APIFootball", timeout)
        self.api_key = api_key
        self.base_url = "https://v3.football.api-sports.io"
    
    def fetch_odds(self, match_id: str, **kwargs) -> Optional[MarketOdds]:
        """Fetch odds for a specific match."""
        fixture_id = kwargs.get("fixture_id")
        if not fixture_id:
            return None
        
        url = f"{self.base_url}/odds"
        params = {"fixture": fixture_id}
        
        data = self._request_with_retry(
            url, 
            headers={"x-apisports-key": self.api_key},
            params=params
        )
        
        if not data or not data.get("response"):
            return None
        
        return self._parse_odds_response(data, match_id)
    
    def fetch_all_odds(self, league_id: Optional[str] = None, **kwargs) -> List[MarketOdds]:
        """Fetch odds for all available matches."""
        league = league_id or kwargs.get("league", 39)  # Default to EPL
        season = kwargs.get("season", 2025)
        
        url = f"{self.base_url}/odds"
        params = {"league": league, "season": season}
        
        data = self._request_with_retry(
            url,
            headers={"x-apisports-key": self.api_key},
            params=params
        )
        
        if not data or not data.get("response"):
            return []
        
        results = []
        for item in data["response"]:
            fixture = item.get("fixture", {})
            match_id = f"{fixture.get('id', 'unknown')}"
            odds = self._parse_odds_response(item, match_id)
            if odds:
                results.append(odds)
        
        return results
    
    def _parse_odds_response(self, data: Dict, match_id: str) -> Optional[MarketOdds]:
        """Parse API-Football odds response."""
        try:
            # Get the first bookmaker (Bet365 typically)
            bookmakers = data.get("bookmakers", [])
            if not bookmakers:
                return None
            
            # Find market with home/draw/away
            for bk in bookmakers:
                for market in bk.get("bets", []):
                    if market.get("name") == "Match Winner":
                        odds_map = {}
                        for odd in market.get("values", []):
                            value = odd.get("value", "")
                            odd_value = odd.get("odd", 0)
                            if "Home" in value:
                                odds_map["home"] = float(odd_value) if odd_value else 0
                            elif "Draw" in value:
                                odds_map["draw"] = float(odd_value) if odd_value else 0
                            elif "Away" in value:
                                odds_map["away"] = float(odd_value) if odd_value else 0
                        
                        if odds_map.get("home") and odds_map.get("away"):
                            return MarketOdds(
                                match_id=match_id,
                                home_odds=odds_map["home"],
                                draw_odds=odds_map.get("draw", 3.50),
                                away_odds=odds_map["away"],
                                source=self.name,
                                bookmaker=bk.get("name", "unknown"),
                            )
            return None
        except Exception as e:
            logger.error(f"{self.name}: Failed to parse odds - {e}")
            return None


class MockOddsProvider(OddsProvider):
    """
    Mock provider for testing and development.
    Generates realistic-looking odds.
    """
    
    def __init__(self):
        super().__init__("Mock")
        self._cache = {}
        self._drift_enabled = False
        self._drift_counter = 0
    
    def enable_drift(self, enabled: bool = True) -> None:
        """Enable simulated odds drift for testing."""
        self._drift_enabled = enabled
    
    def fetch_odds(self, match_id: str, **kwargs) -> Optional[MarketOdds]:
        """Generate mock odds for a match."""
        import random
        
        # Apply drift if enabled
        drift_factor = 1.0
        if self._drift_enabled:
            self._drift_counter += 1
            drift_factor = 1.0 - (self._drift_counter * 0.01)  # Odds shorten over time
        
        if match_id in self._cache:
            # Return cached odds with slight variation
            cached = self._cache[match_id]
            variation = random.uniform(-0.02, 0.02)
            return MarketOdds(
                match_id=match_id,
                home_odds=round(max(1.01, cached.home_odds * drift_factor + variation), 2),
                draw_odds=round(max(1.01, cached.draw_odds + variation), 2),
                away_odds=round(max(1.01, cached.away_odds + variation), 2),
                source=self.name,
                timestamp=time.time(),
            )
        
        # Generate new odds based on match_id hash
        random.seed(hash(match_id) % 2**32)
        
        # Simulate realistic odds ranges
        home_odds = round(random.uniform(1.5, 4.0), 2)
        away_odds = round(random.uniform(1.5, 4.0), 2)
        draw_odds = round(random.uniform(3.0, 4.5), 2)
        
        odds = MarketOdds(
            match_id=match_id,
            home_odds=home_odds,
            draw_odds=draw_odds,
            away_odds=away_odds,
            source=self.name,
            timestamp=time.time(),
        )
        self._cache[match_id] = odds
        return odds
    
    def fetch_all_odds(self, league_id: Optional[str] = None, **kwargs) -> List[MarketOdds]:
        """Generate mock odds for multiple matches."""
        match_ids = kwargs.get("match_ids", [])
        if not match_ids:
            # Generate default set of matches
            match_ids = [
                "arsenal_chelsea", "liverpool_everton", "man_city_tottenham",
                "bayern_dortmund", "real_madrid_barcelona", "juventus_inter",
                "psg_marseille", "ajax_feyenoord", "benfica_porto",
            ]
        
        return [self.fetch_odds(mid) for mid in match_ids if self.fetch_odds(mid)]


class WebSocketProvider(OddsProvider):
    """
    WebSocket provider for real-time odds streaming.
    Requires a WebSocket endpoint that sends odds updates.
    """
    
    def __init__(self, ws_url: str, api_key: str = None):
        super().__init__("WebSocket", timeout=30)
        self.ws_url = ws_url
        self.api_key = api_key
        self._ws = None
        self._running = False
        self._callbacks: List[Callable] = []
        self._thread = None
    
    def connect(self) -> bool:
        """Connect to WebSocket."""
        if not _WEBSOCKET_AVAILABLE:
            logger.error("WebSocket support not available")
            return False
        
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            self._ws = websocket.WebSocketApp(
                self.ws_url,
                header=headers,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open,
            )
            
            self._running = True
            self._thread = threading.Thread(target=self._ws.run_forever, daemon=True)
            self._thread.start()
            
            logger.info(f"WebSocket connected to {self.ws_url}")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.health.record_failure(str(e))
            return False
    
    def subscribe(self, callback: Callable[[MarketOdds], None]) -> None:
        """Subscribe to odds updates."""
        self._callbacks.append(callback)
    
    def _on_message(self, ws, message) -> None:
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            odds = self._parse_ws_message(data)
            if odds:
                for callback in self._callbacks:
                    try:
                        callback(odds)
                    except Exception as e:
                        logger.error(f"WebSocket callback error: {e}")
        except Exception as e:
            logger.error(f"WebSocket message parse error: {e}")
    
    def _on_error(self, ws, error) -> None:
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {error}")
        self.health.record_failure(str(error))
    
    def _on_close(self, ws, close_status_code, close_msg) -> None:
        """Handle WebSocket close."""
        logger.info("WebSocket closed")
        self._running = False
    
    def _on_open(self, ws) -> None:
        """Handle WebSocket open."""
        logger.info("WebSocket opened")
        self.health.record_success(0)
    
    def _parse_ws_message(self, data: Dict) -> Optional[MarketOdds]:
        """Parse WebSocket message into MarketOdds."""
        try:
            return MarketOdds(
                match_id=data.get("match_id", ""),
                home_odds=float(data.get("home_odds", 0)),
                draw_odds=float(data.get("draw_odds", 0)),
                away_odds=float(data.get("away_odds", 0)),
                source=self.name,
                bookmaker=data.get("bookmaker", "unknown"),
                timestamp=time.time(),
                last_updated=data.get("timestamp"),
            )
        except Exception:
            return None
    
    def fetch_odds(self, match_id: str, **kwargs) -> Optional[MarketOdds]:
        """Fetch odds (not supported for WebSocket - use subscription)."""
        logger.warning("WebSocket provider doesn't support direct fetch")
        return None
    
    def fetch_all_odds(self, league_id: Optional[str] = None, **kwargs) -> List[MarketOdds]:
        """Fetch all odds (not supported for WebSocket)."""
        return []
    
    def disconnect(self) -> None:
        """Disconnect WebSocket."""
        if self._ws:
            self._ws.close()
        self._running = False


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — ODDS CACHE
# ═══════════════════════════════════════════════════════════════

class OddsCache:
    """
    LRU cache for odds data to reduce API calls.
    TTL-based expiration with automatic cleanup.
    """
    
    def __init__(self, max_size: int = 200, ttl_seconds: int = 30):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: OrderedDict[str, Tuple[MarketOdds, float]] = OrderedDict()
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[MarketOdds]:
        """Get odds from cache if not expired."""
        with self._lock:
            if key not in self._cache:
                return None
            
            odds, timestamp = self._cache[key]
            if time.time() - timestamp > self.ttl:
                del self._cache[key]
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return odds
    
    def set(self, key: str, odds: MarketOdds) -> None:
        """Store odds in cache."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            
            self._cache[key] = (odds, time.time())
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
    
    def clear_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        now = time.time()
        to_remove = []
        
        with self._lock:
            for key, (_, timestamp) in self._cache.items():
                if now - timestamp > self.ttl:
                    to_remove.append(key)
            
            for key in to_remove:
                del self._cache[key]
        
        return len(to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            now = time.time()
            active = sum(1 for _, ts in self._cache.values() if now - ts < self.ttl)
            return {
                "size": len(self._cache),
                "active": active,
                "max_size": self.max_size,
                "ttl_seconds": self.ttl,
            }


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — ODDS FEED ENGINE
# ═══════════════════════════════════════════════════════════════

class OddsFeedEngine:
    """
    Main engine for fetching and managing odds from multiple providers.
    Handles caching, consensus calculation, and update notifications.
    """
    
    def __init__(self, providers: List[OddsProvider] = None, cache_ttl: int = 30):
        self.providers = providers or []
        self.cache = OddsCache(ttl_seconds=cache_ttl)
        self.history: Dict[str, List[MarketOdds]] = {}
        self._subscribers: List[Callable[[OddsUpdate], None]] = []
        self._update_threshold = 0.02  # 2% change threshold for notifications
        self._running = False
        self._refresh_thread = None
    
    def add_provider(self, provider: OddsProvider) -> None:
        """Add a provider to the engine."""
        self.providers.append(provider)
        logger.info(f"Added provider: {provider.name}")
    
    def remove_provider(self, name: str) -> bool:
        """Remove a provider by name."""
        for i, p in enumerate(self.providers):
            if p.name == name:
                self.providers.pop(i)
                logger.info(f"Removed provider: {name}")
                return True
        return False
    
    def subscribe(self, callback: Callable[[OddsUpdate], None]) -> None:
        """Subscribe to odds update notifications."""
        self._subscribers.append(callback)
    
    def _notify_subscribers(self, update: OddsUpdate) -> None:
        """Notify subscribers of odds update."""
        for callback in self._subscribers:
            try:
                callback(update)
            except Exception as e:
                logger.error(f"Subscriber callback failed: {e}")
    
    def fetch_all(self, match_id: str, force_refresh: bool = False) -> List[MarketOdds]:
        """
        Fetch odds for a match from all healthy providers.
        
        Args:
            match_id: Match identifier
            force_refresh: Ignore cache and force fresh fetch
        
        Returns:
            List of MarketOdds from all providers
        """
        results = []
        
        for provider in self.providers:
            if not provider.is_healthy():
                logger.debug(f"Skipping unhealthy provider: {provider.name}")
                continue
            
            # Check cache
            cache_key = f"{provider.name}:{match_id}"
            if not force_refresh:
                cached = self.cache.get(cache_key)
                if cached and cached.is_fresh():
                    results.append(cached)
                    continue
            
            # Fetch from provider
            try:
                odds = provider.fetch_odds(match_id)
                if odds:
                    self.cache.set(cache_key, odds)
                    results.append(odds)
            except Exception as e:
                logger.error(f"Provider {provider.name} failed for {match_id}: {e}")
        
        return results
    
    def get_latest(self, match_id: str, use_consensus: bool = True) -> Optional[MarketOdds]:
        """
        Get the most recent odds for a match.
        
        Args:
            match_id: Match identifier
            use_consensus: Use consensus across providers
        
        Returns:
            MarketOdds or None if no data
        """
        all_odds = self.fetch_all(match_id)
        if not all_odds:
            return None
        
        if use_consensus and len(all_odds) > 1:
            return compute_consensus_odds(all_odds)
        
        # Return freshest odds
        return max(all_odds, key=lambda o: o.timestamp)
    
    def get_history(self, match_id: str, limit: int = 10) -> List[MarketOdds]:
        """Get historical odds for a match."""
        if match_id not in self.history:
            return []
        return self.history[match_id][-limit:]
    
    def update_history(self, match_id: str, odds: MarketOdds) -> Optional[OddsUpdate]:
        """Update historical odds and detect significant changes."""
        if match_id not in self.history:
            self.history[match_id] = []
        
        # Check for significant change
        update = None
        if self.history[match_id]:
            last = self.history[match_id][-1]
            
            # Calculate changes
            home_move = odds.home_odds - last.home_odds
            draw_move = odds.draw_odds - last.draw_odds
            away_move = odds.away_odds - last.away_odds
            
            home_move_pct = home_move / last.home_odds if last.home_odds > 0 else 0
            draw_move_pct = draw_move / last.draw_odds if last.draw_odds > 0