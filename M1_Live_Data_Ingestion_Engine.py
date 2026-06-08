"""
The Match Oracle - Module 1: Live Data Ingestion Engine (ENHANCED v3)
================================================================================
Fetches today's fixtures from API-Football and The Odds API,
builds fully populated Leg objects ready for the pipeline.

ENHANCEMENTS FOR DIMENSION RTM (v3):
------------------------------
1. ADDED: Enhanced opponent tier classification with season progress weighting
2. ADDED: Venue context for each fixture (home/away with venue type)
3. ADDED: Tier metadata passed to TeamProfile for dimension RTM building
4. ADDED: H2H fixtures with venue tracking for H2H RTM
5. ADDED: League size detection for proper tier thresholds
6. ADDED: Season progress tracking for tier stability
7. ADDED: Enhanced metadata in LegData for dimension analysis
8. ADDED: Distortion factor collection for clean RTM building
9. ADDED: Context flags collection (dead rubber, six-pointer, derby)
10. ADDED: Days rest calculation for fatigue tracking
11. ADDED: Manager tenure tracking for new manager bounce detection

PREVIOUS ENHANCEMENTS:
---------------------
- Guardrail integration (M0)
- Module 24 cache integration
- Parallel league processing
- Comprehensive error handling
- Season detection
- Standings fetching
- H2H fetching with venue tracking

This module is the canonical ingestion layer.

Usage:
    from module1 import load_todays_legs, LegData

    legs = load_todays_legs(
        football_key = "your-apifootball-key",
        odds_key     = "your-theoddsapi-key",
        league_ids   = [39, 140, 78],
        verbose      = True,
    )
    
    for leg_data in legs:
        print(f"Venue: {leg_data.venue}")
        print(f"Opponent tier: {leg_data.opponent_tier}")
        print(f"Days rest: {leg_data.days_rest}")
"""
from __future__ import annotations

import os
import json
import urllib.request
import urllib.parse
import logging
import time
import random
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oracle_beast.module1")

# Guardrail integration
try:
    from module0 import (
        validate_fixture_date, safe_validate, validate_leg,
        GuardrailError, GuardrailReport
    )
    _GUARDRAIL_AVAILABLE = True
except ImportError:
    _GUARDRAIL_AVAILABLE = False
    logger.warning("Module 0 not available - guardrail disabled")

# Module 24 cache integration
try:
    from module24 import cached_football, cached_odds_api, get_budget_status
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False
    logger.warning("Module 24 not available - cache disabled")
    
    def cached_football(path: str, key: str, params: Dict = None) -> Dict:
        q = urllib.parse.urlencode(params or {})
        url = f"https://v3.football.api-sports.io{path}?{q}" if q else f"https://v3.football.api-sports.io{path}"
        req = urllib.request.Request(url, headers={"x-apisports-key": key})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    
    def cached_odds_api(path: str, key: str, params: Dict = None) -> List:
        p = {**(params or {}), "apiKey": key}
        url = f"https://api.the-odds-api.com/v4{path}?{urllib.parse.urlencode(p)}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
            return data if isinstance(data, list) else []
    
    def get_budget_status():
        return type('BudgetStatus', (), {'summary': lambda: 'No cache module'})()


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class LegData:
    """
    Clean, type-safe data structure returned by load_todays_legs().
    
    ENHANCED v3: Now includes dimension metadata and distortion tracking.
    """
    leg: Any  # Leg object from module2
    fav_is_home: bool
    home_odds: float
    away_odds: float
    draw_odds: float
    model_prob: float = 0.0
    edge: float = 0.0
    
    # Dimension metadata for RTM
    venue: str = ""               # "home" or "away"
    opponent_tier: str = ""       # "top6", "mid", "bottom6"
    home_team_position: int = 0
    away_team_position: int = 0
    league_size: int = 20
    season_progress: float = 0.0  # 0-1
    
    # NEW v3: Distortion tracking
    days_rest: int = 7            # Days since last match
    is_midweek: bool = False      # True if fixture is midweek
    is_derby: bool = False        # True if local derby
    manager_tenure_days_home: int = 365
    manager_tenure_days_away: int = 365
    key_players_missing_home: int = 0
    key_players_missing_away: int = 0
    
    # NEW v3: Context flags
    is_dead_rubber: bool = False
    is_six_pointer: bool = False
    is_early_season: bool = False
    is_late_season: bool = False
    
    # Timestamp
    kickoff: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.leg.match_id,
            "selection": self.leg.selection,
            "odds": self.leg.odds,
            "fav_is_home": self.fav_is_home,
            "home_odds": self.home_odds,
            "away_odds": self.away_odds,
            "draw_odds": self.draw_odds,
            "league": self.leg.league,
            "league_id": self.leg.league_id,
            "league_tier": self.leg.league_tier,
            "venue": self.venue,
            "opponent_tier": self.opponent_tier,
            "kickoff": self.kickoff,
            "days_rest": self.days_rest,
            "is_midweek": self.is_midweek,
            "is_derby": self.is_derby,
        }
    
    def summary(self) -> str:
        return (f"{self.leg.home_profile.team_name if self.leg.home_profile else '?'} vs "
                f"{self.leg.away_profile.team_name if self.leg.away_profile else '?'} | "
                f"Pick: {self.leg.selection} @ {self.leg.odds:.2f} | "
                f"Venue: {self.venue} | Opp Tier: {self.opponent_tier} | "
                f"Rest: {self.days_rest}d")


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — CONFIGURATION
# ═══════════════════════════════════════════════════════════════

MIN_EDGE_ODDS = 1.70
MIN_H2H_GAMES = 5
DEFAULT_MODEL_PROB = 0.50
DEFAULT_EDGE = 0.00

MAX_RETRIES = 3
RETRY_DELAY_BASE = 2
MAX_PARALLEL_LEAGUES = 5
REQUEST_TIMEOUT = 30

# Tier thresholds (for opponent classification)
TOP6_THRESHOLD = 6           # Positions 1-6 = top tier
BOTTOM6_OFFSET = 5           # Last 5 positions = bottom tier
EARLY_SEASON_GAMES = 10      # First 10 games = early season
LATE_SEASON_GAMES = 30       # Last 8 games = late season (for 38-game season)

# Midweek detection
MIDWEEK_DAYS = {1, 2, 3, 4}  # Monday-Tuesday-Wednesday-Thursday
WEEKEND_DAYS = {5, 6, 0}     # Friday-Saturday-Sunday

# Fatigue thresholds
FATIGUE_HIGH_REST_DAYS = 3   # Less than 3 days rest = high fatigue
FATIGUE_MEDIUM_REST_DAYS = 5  # 3-5 days = medium fatigue

PLAYOFF_KEYWORDS = [
    'playoff', 'play-off', 'promotion', 'relegation',
    'semi', 'final', 'quarter', 'knockout', 'elimination',
    'closing stage', 'opening stage', 'top 6', 'championship group'
]

DERBY_KEYWORDS = [
    'derby', 'derbi', 'clasico', 'rivalry', 'clássico',
    'superclasico', 'el clasico', 'derby della', 'north london',
    'merseyside', 'manchester derby', 'old firm', 'superclásico'
]


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — LEAGUE CONFIGURATION WITH METADATA
# ═══════════════════════════════════════════════════════════════

LEAGUE_MAP: Dict[int, Dict[str, Any]] = {
    # Tier 1 - Top European Leagues
    39: {
        "odds_key": "soccer_epl",
        "label": "Premier League",
        "country": "england",
        "tier": 1,
        "total_teams": 20,
        "season_start_month": 8,
    },
    140: {
        "odds_key": "soccer_spain_la_liga",
        "label": "La Liga",
        "country": "spain",
        "tier": 1,
        "total_teams": 20,
        "season_start_month": 8,
    },
    78: {
        "odds_key": "soccer_germany_bundesliga",
        "label": "Bundesliga",
        "country": "germany",
        "tier": 1,
        "total_teams": 18,
        "season_start_month": 8,
    },
    135: {
        "odds_key": "soccer_italy_serie_a",
        "label": "Serie A",
        "country": "italy",
        "tier": 1,
        "total_teams": 20,
        "season_start_month": 8,
    },
    61: {
        "odds_key": "soccer_france_ligue_one",
        "label": "Ligue 1",
        "country": "france",
        "tier": 1,
        "total_teams": 18,
        "season_start_month": 8,
    },
    
    # Tier 2 - Strong Secondary Leagues
    88: {
        "odds_key": "soccer_netherlands_eredivisie",
        "label": "Eredivisie",
        "country": "netherlands",
        "tier": 2,
        "total_teams": 18,
        "season_start_month": 8,
    },
    94: {
        "odds_key": "soccer_portugal_primeira_liga",
        "label": "Primeira Liga",
        "country": "portugal",
        "tier": 2,
        "total_teams": 18,
        "season_start_month": 8,
    },
    144: {
        "odds_key": "soccer_belgium_first_div",
        "label": "Pro League",
        "country": "belgium",
        "tier": 2,
        "total_teams": 16,
        "season_start_month": 7,
    },
    207: {
        "odds_key": "soccer_turkey_super_league",
        "label": "Super Lig",
        "country": "turkey",
        "tier": 2,
        "total_teams": 20,
        "season_start_month": 8,
    },
    
    # South America
    71: {
        "odds_key": "soccer_brazil_campeonato",
        "label": "Brasileiro",
        "country": "brazil",
        "tier": 1,
        "total_teams": 20,
        "season_start_month": 4,
    },
    128: {
        "odds_key": "soccer_argentina_primera",
        "label": "Primera Division",
        "country": "argentina",
        "tier": 1,
        "total_teams": 28,
        "season_start_month": 1,
    },
    
    # Asia
    292: {
        "odds_key": "soccer_japan_j_league",
        "label": "J1 League",
        "country": "japan",
        "tier": 1,
        "total_teams": 18,
        "season_start_month": 2,
    },
    307: {
        "odds_key": "soccer_saudi_pro_league",
        "label": "Saudi Pro League",
        "country": "saudi arabia",
        "tier": 1,
        "total_teams": 18,
        "season_start_month": 8,
    },
}


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — OPPONENT TIER CLASSIFICATION (ENHANCED)
# ═══════════════════════════════════════════════════════════════

def _classify_opponent_tier(
    position: int, 
    league_size: int = 20,
    season_progress: float = 0.5,
) -> str:
    """
    Classify opponent tier based on league position.
    
    Args:
        position: Team's position in league (1-indexed)
        league_size: Total number of teams
        season_progress: 0-1, how far into season (early season = less reliable)
    
    Returns:
        "top6", "mid", or "bottom6"
    """
    # Early season: more conservative classification
    if season_progress < 0.25:
        # Only extreme positions are classified
        if position <= 3:
            return "top6"
        elif position >= league_size - 2:
            return "bottom6"
        return "mid"
    
    # Mid season: standard classification
    if season_progress < 0.75:
        top_threshold = min(TOP6_THRESHOLD, max(4, int(league_size * 0.3)))
        bottom_threshold = league_size - max(3, int(league_size * 0.25))
    else:
        # Late season: classification becomes more reliable
        top_threshold = TOP6_THRESHOLD
        bottom_threshold = league_size - BOTTOM6_OFFSET
    
    if position <= top_threshold:
        return "top6"
    elif position >= bottom_threshold:
        return "bottom6"
    return "mid"


def _get_league_size(league_id: int) -> int:
    """Get total teams in league from config."""
    cfg = LEAGUE_MAP.get(league_id, {})
    return cfg.get("total_teams", 20)


def _calculate_season_progress(games_played: int, total_games: int) -> float:
    """Calculate season progress (0-1)."""
    if total_games <= 0:
        return 0.5
    return min(1.0, games_played / total_games)


def _is_early_season(games_played: int, total_games: int) -> bool:
    """Check if team is in early season phase."""
    progress = _calculate_season_progress(games_played, total_games)
    return progress < 0.25


def _is_late_season(games_played: int, total_games: int) -> bool:
    """Check if team is in late season phase."""
    progress = _calculate_season_progress(games_played, total_games)
    return progress > 0.85


def _is_midweek(kickoff_dt: datetime) -> bool:
    """Check if kickoff is on a midweek day."""
    return kickoff_dt.weekday() in MIDWEEK_DAYS


def _calculate_days_rest(last_match_date: Optional[datetime], current_date: datetime) -> int:
    """Calculate days of rest since last match."""
    if last_match_date is None:
        return 7  # Default to 7 days if no data
    days = (current_date - last_match_date).days
    return max(0, min(21, days))  # Cap between 0 and 21


def _is_derby_match(home_name: str, away_name: str) -> bool:
    """Check if fixture is a local derby based on keywords."""
    home_lower = home_name.lower()
    away_lower = away_name.lower()
    
    # Check for known derby keywords
    for keyword in DERBY_KEYWORDS:
        if keyword in home_lower or keyword in away_lower:
            return True
    
    # Check for same city patterns
    cities = [
        'manchester', 'liverpool', 'london', 'madrid', 'barcelona', 'milan',
        'rome', 'berlin', 'munich', 'paris', 'lisbon', 'porto', 'istanbul',
        'buenos aires', 'rio', 'sao paulo', 'bangkok', 'cairo', 'johannesburg'
    ]
    
    for city in cities:
        if city in home_lower and city in away_lower:
            return True
    
    return False


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — HTTP HELPERS
# ═══════════════════════════════════════════════════════════════

def _request_with_retry(
    url: str,
    headers: Dict = None,
    method: str = "GET",
    data: bytes = None,
    max_retries: int = MAX_RETRIES,
    timeout: int = REQUEST_TIMEOUT,
) -> Optional[Dict]:
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers or {}, method=method, data=data)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait_time = RETRY_DELAY_BASE * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Rate limited (429), waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                continue
            elif e.code >= 500 and attempt < max_retries:
                wait_time = RETRY_DELAY_BASE * (2 ** attempt)
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"HTTP {e.code} for {url}")
                return None
        except Exception as e:
            if attempt < max_retries:
                wait_time = RETRY_DELAY_BASE * (2 ** attempt)
                time.sleep(wait_time)
                continue
            logger.error(f"Request failed: {e}")
            return None
    return None


def _football(path: str, key: str, params: Dict = None) -> Dict:
    if _CACHE_AVAILABLE:
        return cached_football(path, key, params)
    
    q = urllib.parse.urlencode(params or {})
    url = f"https://v3.football.api-sports.io{path}?{q}" if q else f"https://v3.football.api-sports.io{path}"
    headers = {"x-apisports-key": key}
    result = _request_with_retry(url, headers=headers)
    return result if isinstance(result, dict) else {}


def _odds_api(path: str, key: str, params: Dict = None) -> List:
    if _CACHE_AVAILABLE:
        return cached_odds_api(path, key, params)
    
    p = {**(params or {}), "apiKey": key}
    url = f"https://api.the-odds-api.com/v4{path}?{urllib.parse.urlencode(p)}"
    result = _request_with_retry(url)
    return result if isinstance(result, list) else []


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — SEASON DETECTION
# ═══════════════════════════════════════════════════════════════

_SEASON_CACHE: Dict[int, int] = {}


def _detect_season_from_response(response: List[Dict]) -> int:
    for fx in response:
        season = fx.get("league", {}).get("season")
        if season and isinstance(season, int):
            return season
    now = datetime.now(timezone.utc)
    return now.year - 1 if now.month <= 5 else now.year


def get_league_season(league_id: int, football_key: str) -> int:
    if league_id in _SEASON_CACHE:
        return _SEASON_CACHE[league_id]
    
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    probe = _football("/fixtures", football_key, {"league": league_id, "date": today})
    probe_fx = probe.get("response", [])
    season = _detect_season_from_response(probe_fx)
    _SEASON_CACHE[league_id] = season
    return season


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — FIXTURE FETCHERS
# ═══════════════════════════════════════════════════════════════

def fetch_todays_fixtures(league_id: int, key: str) -> List[Dict]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    data = _football("/fixtures", key, {"league": league_id, "date": today})
    fixtures = data.get("response", [])
    return [f for f in fixtures if f.get("fixture", {}).get("status", {}).get("short") == "NS"]


def fetch_season_fixtures(league_id: int, key: str, season: int = None) -> List[Dict]:
    if season is None:
        season = get_league_season(league_id, key)
    data = _football("/fixtures", key, {"league": league_id, "season": season})
    return data.get("response", [])


def fetch_standings(
    league_id: int,
    key: str,
    season: Optional[int] = None,
) -> Dict[int, Dict]:
    if season is None:
        season = get_league_season(league_id, key)
    
    data = _football("/standings", key, {"league": league_id, "season": season})
    result = {}
    
    try:
        standings_data = data.get("response", [])
        if standings_data:
            league_data = standings_data[0].get("league", {})
            standings_arrays = league_data.get("standings", [])
            if standings_arrays and isinstance(standings_arrays, list):
                for entry in standings_arrays[0]:
                    tid = entry["team"]["id"]
                    result[tid] = {
                        "position": entry["rank"],
                        "points": entry["points"],
                        "played": entry["all"]["played"],
                        "wins": entry["all"]["win"],
                        "draws": entry["all"]["draw"],
                        "losses": entry["all"]["lose"],
                        "goals_for": entry["all"]["goals"]["for"],
                        "goals_against": entry["all"]["goals"]["against"],
                    }
    except (KeyError, IndexError, TypeError) as e:
        logger.warning(f"Failed to parse standings for league {league_id}: {e}")
    
    return result


def fetch_h2h(home_id: int, away_id: int, key: str) -> List[Dict]:
    try:
        data = _football("/fixtures/headtohead", key, {"h2h": f"{home_id}-{away_id}", "last": 15})
        return data.get("response", [])
    except Exception as e:
        logger.debug(f"Failed to fetch H2H for {home_id} vs {away_id}: {e}")
        return []


def fetch_todays_odds(odds_key: str, sports_key: str) -> List[Dict]:
    now = datetime.now(timezone.utc)
    tomorrow = now + timedelta(days=1)
    
    try:
        return _odds_api(
            f"/sports/{sports_key}/odds",
            odds_key,
            params={
                "regions": "uk,us,eu",
                "markets": "h2h",
                "oddsFormat": "decimal",
                "dateFormat": "iso",
                "commenceTimeFrom": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "commenceTimeTo": tomorrow.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
    except Exception as e:
        logger.error(f"Failed to fetch odds for {sports_key}: {e}")
        return []


def fetch_last_match_date(team_id: int, football_key: str) -> Optional[datetime]:
    """Fetch the date of the team's last match."""
    try:
        data = _football("/fixtures", football_key, {"team": team_id, "last": 1})
        fixtures = data.get("response", [])
        if fixtures:
            last_fixture = fixtures[0]
            status = last_fixture.get("fixture", {}).get("status", {}).get("short", "")
            if status in ("FT", "AET", "PEN"):
                date_str = last_fixture.get("fixture", {}).get("date", "")
                if date_str:
                    return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception as e:
        logger.debug(f"Failed to fetch last match date for team {team_id}: {e}")
    return None


def fetch_manager_tenure(team_id: int, football_key: str) -> int:
    """Fetch manager tenure in days."""
    try:
        data = _football("/coachs", football_key, {"team": team_id})
        coaches = data.get("response", [])
        if coaches:
            coach = coaches[0]
            start_date = coach.get("start_date")
            if start_date:
                start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                return (datetime.now(timezone.utc) - start).days
    except Exception as e:
        logger.debug(f"Failed to fetch manager tenure for team {team_id}: {e}")
    return 365


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — TRANSFORMERS
# ═══════════════════════════════════════════════════════════════

def _result_sequence(fixtures: List[Dict], team_id: int) -> List[str]:
    seq = []
    for fx in fixtures:
        status = fx.get("fixture", {}).get("status", {}).get("short", "")
        if status not in ("FT", "AET", "PEN"):
            continue
        
        hid = fx["teams"]["home"]["id"]
        aid = fx["teams"]["away"]["id"]
        hg = fx["goals"]["home"]
        ag = fx["goals"]["away"]
        
        if hg is None or ag is None:
            continue
        
        if team_id == hid:
            seq.append("W" if hg > ag else "D" if hg == ag else "L")
        elif team_id == aid:
            seq.append("W" if ag > hg else "D" if ag == hg else "L")
    return seq


def _goal_totals_sequence(fixtures: List[Dict], team_id: int) -> List[int]:
    totals = []
    for fx in fixtures:
        status = fx.get("fixture", {}).get("status", {}).get("short", "")
        if status not in ("FT", "AET", "PEN"):
            continue
        
        hid = fx["teams"]["home"]["id"]
        aid = fx["teams"]["away"]["id"]
        hg = fx["goals"]["home"]
        ag = fx["goals"]["away"]
        
        if hg is None or ag is None:
            continue
        
        if team_id in (hid, aid):
            totals.append(int(hg) + int(ag))
    return totals


def _avg_odds(bookmakers: List[Dict], team_name: str) -> Optional[float]:
    prices = [
        float(o["price"])
        for b in bookmakers
        for m in b.get("markets", []) if m.get("key") == "h2h"
        for o in m.get("outcomes", []) if o.get("name") == team_name
    ]
    return round(sum(prices) / len(prices), 3) if prices else None


def _days_since_last_match(fixtures: List[Dict], team_id: int) -> int:
    now = datetime.now(timezone.utc)
    latest = None
    
    for fx in fixtures:
        status = fx.get("fixture", {}).get("status", {}).get("short", "")
        if status not in ("FT", "AET", "PEN"):
            continue
        
        if fx["teams"]["home"]["id"] == team_id or fx["teams"]["away"]["id"] == team_id:
            try:
                dt = datetime.fromisoformat(fx["fixture"]["date"].replace("Z", "+00:00"))
                if latest is None or dt > latest:
                    latest = dt
            except ValueError:
                pass
    
    if latest is None:
        return 7
    days = (now - latest).days
    return min(max(days, 0), 21)


def _clean_team_name(name: str) -> str:
    n = name.lower().strip()
    suffixes = [
        " fc", " cf", " ac", " sc", " afc", " bc",
        " city", " united", " town", " wanderers",
        " hotspur", " athletic", " albion", " fc ", " cf "
    ]
    for s in suffixes:
        n = n.replace(s, " ")
    n = ''.join(c for c in n if c.isalnum() or c == ' ')
    return n.strip()


def _find_team_id(name: str, fixtures: List[Dict]) -> Optional[int]:
    seen: Dict[int, str] = {}
    for fx in fixtures:
        for side in ("home", "away"):
            seen[fx["teams"][side]["id"]] = fx["teams"][side]["name"]
    
    nl, nc = name.lower(), _clean_team_name(name)
    
    for tid, tname in seen.items():
        if nl == tname.lower():
            return tid
    
    for tid, tname in seen.items():
        if nc == _clean_team_name(tname):
            return tid
    
    for tid, tname in seen.items():
        tc = _clean_team_name(tname)
        if nc and tc and (nc in tc or tc in nc):
            return tid
    
    nw = set(nc.split())
    best_tid, best = None, 0
    for tid, tname in seen.items():
        tw = set(_clean_team_name(tname).split())
        ov = len(nw & tw)
        if ov > best:
            best, best_tid = ov, tid
    
    if best < 1 and len(nc) >= 3:
        prefix = nc[:3]
        for tid, tname in seen.items():
            tc = _clean_team_name(tname)
            if tc.startswith(prefix):
                return tid
    
    return best_tid if best >= 1 else None


def _infer_pattern(seq: List[str], goal_totals: List[int] = None) -> str:
    if len(seq) < 5:
        return "UNKNOWN"
    
    if goal_totals and len(goal_totals) >= 5:
        mean = sum(goal_totals) / len(goal_totals)
        variance = sum((g - mean) ** 2 for g in goal_totals) / len(goal_totals)
        std_dev = variance ** 0.5
        if std_dev > 2.2:
            return "HIGH_VARIANCE"
    
    w = seq.count("W") / len(seq)
    d = seq.count("D") / len(seq)
    l = seq.count("L") / len(seq)
    
    if w >= 0.60:
        return "SERIAL_WINNER"
    if l >= 0.55:
        return "LOSS_PRONE"
    if d >= 0.40:
        return "DRAW_SPECIALIST"
    
    alt = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1]) / max(len(seq)-1, 1)
    if alt >= 0.65:
        return "VOLATILE"
    if w >= 0.40:
        return "BOUNCER"
    return "INCONSISTENT"


def _build_h2h_record_with_venue(
    h2h_fixtures: List[Dict],
    home_id: int,
    away_id: int,
    min_games: int = MIN_H2H_GAMES,
) -> Optional[Tuple[Any, List[Dict]]]:
    """
    Build H2H record with venue tracking for dimension RTM.
    
    Returns:
        Tuple of (H2HRecord, list of fixture dicts with venue info) or None
    """
    from module2 import H2HRecord
    
    completed = [
        fx for fx in h2h_fixtures
        if fx.get("fixture", {}).get("status", {}).get("short") in ("FT", "AET", "PEN")
    ]
    
    if len(completed) < min_games:
        return None
    
    rec = H2HRecord()
    h2h_fixtures_detail = []
    
    for fx in completed:
        hid = fx["teams"]["home"]["id"]
        aid = fx["teams"]["away"]["id"]
        hg, ag = fx["goals"]["home"], fx["goals"]["away"]
        
        if hg is None or ag is None:
            continue
        
        rec.games += 1
        fav_home = (hid == home_id)
        
        # Determine result from home_id perspective
        if hg == ag:
            outcome = "D"
            rec.draws += 1
        elif (fav_home and hg > ag) or (not fav_home and ag > hg):
            outcome = "W"
            rec.fav_wins += 1
        else:
            outcome = "L"
            rec.und_wins += 1
        
        # Store fixture with venue for RTM
        date_str = fx.get("fixture", {}).get("date", "")
        try:
            match_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            match_date = datetime.now(timezone.utc)
        
        h2h_fixtures_detail.append({
            "result": outcome,
            "venue": "home" if hid == home_id else "away",
            "date": date_str,
            "home_id": hid,
            "away_id": aid,
            "home_goals": hg,
            "away_goals": ag,
            "timestamp": match_date.timestamp(),
        })
    
    return rec if rec.games >= min_games else None, h2h_fixtures_detail


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — PROFILE BUILDER (ENHANCED FOR DIMENSION RTM)
# ═══════════════════════════════════════════════════════════════

def build_team_profile(
    team_id: int,
    team_name: str,
    all_fixtures: List[Dict],
    standings: Dict[int, Dict],
    odds: Optional[float] = None,
    football_key: Optional[str] = None,
    league_id: int = 0,
    league_size: int = 20,
) -> Any:
    """
    Build a fully populated TeamProfile from API-Football season data.
    
    ENHANCED v3: Now includes comprehensive fixture metadata for dimension RTM
    and distortion tracking.
    """
    from module2 import TeamProfile, TransitionMatrix, MultiDimensionRTM, DimensionRTM
    from module10 import build_tally_matrix
    
    profile = TeamProfile(team_id=str(team_id), team_name=team_name)
    standing = standings.get(team_id, {})
    seq = _result_sequence(all_fixtures, team_id)
    goals_ts = _goal_totals_sequence(all_fixtures, team_id)
    
    games = max(standing.get("played", len(seq)), 1)
    wins = standing.get("wins", seq.count("W"))
    draws = standing.get("draws", seq.count("D"))
    losses = standing.get("losses", seq.count("L"))
    gf = standing.get("goals_for", 0)
    ga = standing.get("goals_against", 0)
    
    # Home/Away records
    home_fx = [
        f for f in all_fixtures
        if f["teams"]["home"]["id"] == team_id
        and f.get("fixture", {}).get("status", {}).get("short") in ("FT", "AET", "PEN")
    ]
    hw = sum(1 for f in home_fx if f["goals"]["home"] is not None and f["goals"]["home"] > f["goals"]["away"])
    
    away_fx = [
        f for f in all_fixtures
        if f["teams"]["away"]["id"] == team_id
        and f.get("fixture", {}).get("status", {}).get("short") in ("FT", "AET", "PEN")
    ]
    aw_wins = sum(1 for f in away_fx if f["goals"]["away"] is not None and f["goals"]["away"] > f["goals"]["home"])
    aw_rate = aw_wins / max(len(away_fx), 1)
    
    pos = standing.get("position", 10)
    pts = standing.get("points", 0)
    
    all_pts = [v.get("points", 0) for v in standings.values()]
    leader_pts = max(all_pts) if all_pts else pts
    pts_to_first = max(leader_pts - pts, 0)
    
    rel_pts = max(
        (standings.get(tid, {}).get("points", 0)
         for tid in standings if standings[tid].get("position", 99) >= 18),
        default=20
    )
    pts_from_rel = max(pts - rel_pts, 0)
    
    days_rest = _days_since_last_match(all_fixtures, team_id)
    draw_rate = draws / games if games > 0 else 0.0
    implied_prob = round(1.0 / odds, 4) if odds and odds > 1.0 else 0.0
    
    # New manager detection
    new_manager = 0.0
    if football_key:
        try:
            data = _football("/coachs", football_key, {"team": team_id})
            coaches = data.get("response", [])
            if coaches:
                coach = coaches[0]
                if coach.get("start_date"):
                    start = datetime.fromisoformat(coach["start_date"].replace("Z", "+00:00"))
                    days_as_manager = (datetime.now(timezone.utc) - start).days
                    if days_as_manager <= 60:
                        new_manager = 1.0
        except Exception:
            pass
    
    profile.update_metrics({
        "core.games": float(games),
        "core.wins": float(wins),
        "core.draws": float(draws),
        "core.losses": float(losses),
        "core.xg": float(gf),
        "core.xga": float(ga),
        "core.goals": float(gf),
        "core.goals_against": float(ga),
        "home_wins": float(hw),
        "home_games": float(max(len(home_fx), 1)),
        "away_losses": float(len(away_fx) - aw_wins),
        "away_win_rate": round(aw_rate, 3),
        "position": float(pos),
        "points": float(pts),
        "pts_to_first": float(pts_to_first),
        "draw_rate": round(draw_rate, 3),
        "core.implied_prob": implied_prob,
        "wins_vs_top": float(max(wins - hw, 0)),
        "wins_vs_bottom": float(hw),
        "motivation.fatigue_days": float(days_rest),
        "motivation.relegation_pressure": 1.0 if pts_from_rel <= 6 else 0.0,
        "motivation.desperation_phase": 1.0 if pos >= 15 and pts_from_rel <= 10 else 0.0,
        "new_manager": float(new_manager),
    })
    
    if seq:
        profile.form["recent_results"] = seq[-20:]
    
    # Build TransitionMatrix
    if len(seq) >= 5:
        tally = build_tally_matrix(str(team_id), seq, str(datetime.now(timezone.utc).year))
        pattern = _infer_pattern(seq, goal_totals=goals_ts)
        profile.transition = TransitionMatrix(
            pattern=pattern,
            probs={
                r: dict(tally.probs.get(r, {"W": 0.33, "D": 0.33, "L": 0.34}))
                for r in ["W", "D", "L"]
            },
            sample_size=tally.total_transitions,
        )
    
    # Build MultiDimensionRTM structure (will be populated by M10)
    profile.multi_rtm = MultiDimensionRTM(
        team_id=str(team_id),
        team_name=team_name,
    )
    
    return profile


# ═══════════════════════════════════════════════════════════════
# SECTION 10 — LEG BUILDER (ENHANCED v3)
# ═══════════════════════════════════════════════════════════════

def build_leg(
    odds_fixture: Dict,
    all_fixtures: List[Dict],
    standings: Dict[int, Dict],
    h2h_fixtures: List[Dict],
    league_label: str,
    league_id: int,
    league_tier: int,
    league_country: str,
    football_key: Optional[str] = None,
) -> Optional[LegData]:
    """
    Build a complete Leg from one Odds API fixture + API-Football season data.
    
    ENHANCED v3: Now includes opponent tier classification, venue metadata,
    and distortion tracking (rest days, midweek, derby, manager tenure).
    """
    from module2 import Leg, BetMarket, CompetitionFormat, classify_opponent_tier, VenueType
    
    home_name = odds_fixture.get("home_team", "")
    away_name = odds_fixture.get("away_team", "")
    kickoff = odds_fixture.get("commence_time", "")
    
    if not home_name or not away_name:
        return None
    
    # Parse kickoff datetime for midweek detection and days rest
    kickoff_dt = None
    try:
        if kickoff:
            kickoff_dt = datetime.fromisoformat(kickoff.replace('Z', '+00:00'))
    except (ValueError, TypeError):
        pass
    
    home_id = _find_team_id(home_name, all_fixtures)
    away_id = _find_team_id(away_name, all_fixtures)
    
    if home_id is None or away_id is None:
        logger.debug(f"Could not match team IDs for {home_name} vs {away_name}")
        return None
    
    # Fetch manager tenure and last match dates for distortion tracking
    manager_tenure_home = fetch_manager_tenure(home_id, football_key) if football_key else 365
    manager_tenure_away = fetch_manager_tenure(away_id, football_key) if football_key else 365
    
    last_match_home = fetch_last_match_date(home_id, football_key) if football_key else None
    last_match_away = fetch_last_match_date(away_id, football_key) if football_key else None
    
    days_rest_home = _calculate_days_rest(last_match_home, kickoff_dt) if kickoff_dt else 7
    days_rest_away = _calculate_days_rest(last_match_away, kickoff_dt) if kickoff_dt else 7
    
    bookmakers = odds_fixture.get("bookmakers", [])
    home_odds = _avg_odds(bookmakers, home_name)
    away_odds = _avg_odds(bookmakers, away_name)
    draw_odds = _avg_odds(bookmakers, "Draw")
    
    if not home_odds or home_odds <= 1.0:
        return None
    
    fav_is_home = home_odds <= (away_odds or 999)
    fav_odds = home_odds if fav_is_home else (away_odds or 0.0)
    
    if fav_odds < MIN_EDGE_ODDS:
        logger.debug(f"Skipping {home_name} vs {away_name}: fav odds {fav_odds:.2f} < {MIN_EDGE_ODDS}")
        return None
    
    # Get positions for tier classification
    home_standing = standings.get(home_id, {})
    away_standing = standings.get(away_id, {})
    home_pos = home_standing.get("position", 10)
    away_pos = away_standing.get("position", 10)
    
    # Calculate season progress for reliability
    league_size = _get_league_size(league_id)
    home_games_played = home_standing.get("played", 20)
    away_games_played = away_standing.get("played", 20)
    total_games = league_size * 2 - 1  # Approximate total games in season
    season_progress = _calculate_season_progress(max(home_games_played, away_games_played), total_games)
    
    # Check early/late season
    is_early = _is_early_season(max(home_games_played, away_games_played), total_games)
    is_late = _is_late_season(max(home_games_played, away_games_played), total_games)
    
    # Check if midweek fixture
    is_midweek = _is_midweek(kickoff_dt) if kickoff_dt else False
    
    # Check if derby
    is_derby = _is_derby_match(home_name, away_name)
    
    # Classify opponent tiers
    home_opponent_tier = _classify_opponent_tier(away_pos, league_size, season_progress)
    away_opponent_tier = _classify_opponent_tier(home_pos, league_size, season_progress)
    
    # Build profiles with enhanced data
    home_profile = build_team_profile(
        home_id, home_name, all_fixtures, standings,
        odds=home_odds, football_key=football_key,
        league_id=league_id, league_size=league_size,
    )
    away_profile = build_team_profile(
        away_id, away_name, all_fixtures, standings,
        odds=away_odds, football_key=football_key,
        league_id=league_id, league_size=league_size,
    )
    
    # Add rest days to profiles for fatigue tracking
    home_profile.update_metrics({"motivation.fatigue_days": float(days_rest_home)})
    away_profile.update_metrics({"motivation.fatigue_days": float(days_rest_away)})
    
    # Build match_id
    date_str = kickoff[:10].replace("-", "") if kickoff else "00000000"
    match_id = f"{league_label.replace(' ', '_')}_{date_str}_{home_name.replace(' ', '_')}_{away_name.replace(' ', '_')}"
    
    # Build H2H record with venue tracking (for dimension RTM)
    h2h_result = _build_h2h_record_with_venue(h2h_fixtures, home_id, away_id)
    h2h_record = h2h_result[0] if h2h_result else None
    h2h_detail_fixtures = h2h_result[1] if h2h_result else []
    
    # Store H2H detail in leg features for later RTM building
    h2h_detail = h2h_detail_fixtures if h2h_detail_fixtures else None
    
    # Extract stage and round
    stage = ""
    round_name = ""
    for fx in all_fixtures:
        if fx.get("teams", {}).get("home", {}).get("id") == home_id and \
           fx.get("teams", {}).get("away", {}).get("id") == away_id:
            league_obj = fx.get("league", {})
            stage = league_obj.get("stage", "")
            round_name = league_obj.get("round", "")
            break
    
    competition_type = "league"
    comp_format = CompetitionFormat.REGULAR_SEASON
    
    leg = Leg(
        match_id=match_id,
        selection=home_name if fav_is_home else away_name,
        odds=home_odds if fav_is_home else (away_odds or home_odds),
        market=BetMarket.STRAIGHT_WIN,
        league=league_label,
        league_id=league_id,
        league_tier=league_tier,
        league_country=league_country,
        competition_type=competition_type,
        stage=stage,
        round=round_name,
        kickoff=kickoff,
        home_profile=home_profile,
        away_profile=away_profile,
        h2h=h2h_record,
        competition_format=comp_format,
    )
    
    # Store H2H detail for dimension RTM
    if h2h_detail:
        leg.features["h2h_detail_fixtures"] = h2h_detail
        leg.features["h2h_home_id"] = home_id
        leg.features["h2h_away_id"] = away_id
    
    # Store distortion tracking data in leg features
    leg.features["days_rest_home"] = days_rest_home
    leg.features["days_rest_away"] = days_rest_away
    leg.features["manager_tenure_home"] = manager_tenure_home
    leg.features["manager_tenure_away"] = manager_tenure_away
    leg.features["is_midweek"] = is_midweek
    leg.features["is_derby"] = is_derby
    leg.features["is_early_season"] = is_early
    leg.features["is_late_season"] = is_late
    
    # Store odds on Leg
    leg.home_odds = home_odds
    leg.away_odds = away_odds
    leg.draw_odds = draw_odds
    
    # Determine venue from favourite's perspective
    venue = VenueType.HOME if fav_is_home else VenueType.AWAY
    opponent_tier = home_opponent_tier if fav_is_home else away_opponent_tier
    
    # Set initial model probability
    implied_prob = 1.0 / leg.odds if leg.odds > 1.0 else DEFAULT_MODEL_PROB
    leg.model_prob = implied_prob
    leg.adjusted_prob = implied_prob
    leg.edge = DEFAULT_EDGE
    leg.pre_verdict = "PENDING"
    leg.venue = venue
    
    # Create LegData with dimension metadata and distortion tracking
    return LegData(
        leg=leg,
        fav_is_home=fav_is_home,
        home_odds=home_odds or 0.0,
        away_odds=away_odds or 0.0,
        draw_odds=draw_odds or 0.0,
        model_prob=implied_prob,
        edge=DEFAULT_EDGE,
        venue=venue.value,
        opponent_tier=opponent_tier,
        home_team_position=home_pos,
        away_team_position=away_pos,
        league_size=league_size,
        season_progress=season_progress,
        days_rest=min(days_rest_home, days_rest_away),
        is_midweek=is_midweek,
        is_derby=is_derby,
        manager_tenure_days_home=manager_tenure_home,
        manager_tenure_days_away=manager_tenure_away,
        key_players_missing_home=0,  # Will be populated by M6
        key_players_missing_away=0,
        is_dead_rubber=False,  # Will be populated by M26
        is_six_pointer=False,   # Will be populated by M26
        is_early_season=is_early,
        is_late_season=is_late,
        kickoff=kickoff,
    )


# ═══════════════════════════════════════════════════════════════
# SECTION 11 — MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def load_todays_legs(
    football_key: str,
    odds_key: str,
    league_ids: List[int] = None,
    verbose: bool = True,
    validate_dates: bool = True,
    use_mock: bool = False,
    parallel: bool = True,
) -> List[LegData]:
    """
    Main entry point. Fetches today's fixtures and returns fully populated LegData objects.
    
    ENHANCED v3: Now includes dimension metadata, distortion tracking, and context flags.
    """
    if not use_mock and (not football_key or not odds_key):
        raise ValueError("Both APIFOOTBALL_KEY and ODDS_API_KEY are required (or use use_mock=True)")
    
    if league_ids is None:
        league_ids = list(LEAGUE_MAP.keys())
    
    if use_mock:
        logger.info("Running in MOCK mode - using generated data")
        return _load_todays_legs_mock(verbose)
    
    if verbose and _CACHE_AVAILABLE:
        budget = get_budget_status()
        if hasattr(budget, 'summary'):
            logger.info(budget.summary())
    
    league_configs = []
    for lid in league_ids:
        cfg = LEAGUE_MAP.get(lid)
        if not cfg:
            if verbose:
                logger.warning(f"Skipping unknown league_id {lid}")
            continue
        league_configs.append({
            "id": lid,
            "label": cfg["label"],
            "odds_key": cfg["odds_key"],
            "country": cfg.get("country", "unknown"),
            "tier": cfg.get("tier", 3),
            "total_teams": cfg.get("total_teams", 20),
        })
    
    if parallel and len(league_configs) > 1:
        return _load_todays_legs_parallel(league_configs, football_key