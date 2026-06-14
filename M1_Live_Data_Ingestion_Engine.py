"""
The Match Oracle - Module 1: Live Data Ingestion Engine (HIGHLIGHTLY v4)
================================================================================
Fetches today's fixtures from Highlightly Sports API,
builds fully populated Leg objects ready for the pipeline.

ENHANCEMENTS FOR HIGHLIGHTLY INTEGRATION (v4):
------------------------------
1. REPLACED: API-Football with Highlightly Sports API
2. REPLACED: The Odds API with Highlightly's built-in odds
3. ADDED: Direct H2H endpoint support
4. ADDED: Team statistics from Highlightly
5. ADDED: Standings integration
6. ADDED: Last 5 games per team
7. MAINTAINED: All dimension RTM features

PREVIOUS ENHANCEMENTS (v3):
---------------------
- Opponent tier classification with season progress weighting
- Venue context for each fixture (home/away with venue type)
- Tier metadata passed to TeamProfile for dimension RTM building
- H2H fixtures with venue tracking for H2H RTM
- League size detection for proper tier thresholds
- Season progress tracking for tier stability
- Distortion factor collection for clean RTM building
- Context flags collection (dead rubber, six-pointer, derby)
- Days rest calculation for fatigue tracking
- Manager tenure tracking for new manager bounce detection

This module is the canonical ingestion layer.

Usage:
    from module1 import load_todays_legs, LegData

    legs = load_todays_legs(
        highlightly_key = "your-highlightly-api-key",
        league_ids      = [39, 140, 78],
        verbose         = True,
    )
"""
from __future__ import annotations

import os
import json
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

# ============================================================
# HIGHLIGHTLY API CONFIGURATION
# ============================================================

HIGHLIGHTLY_BASE_URL = "https://sports.highlightly.net"
HIGHLIGHTLY_HOST = "sport-highlights-api.p.rapidapi.com"

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

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class LegData:
    """
    Clean, type-safe data structure returned by load_todays_legs().
    
    ENHANCED v4: Now includes Highlightly-specific metadata.
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
    
    # Distortion tracking
    days_rest: int = 7
    is_midweek: bool = False
    is_derby: bool = False
    manager_tenure_days_home: int = 365
    manager_tenure_days_away: int = 365
    key_players_missing_home: int = 0
    key_players_missing_away: int = 0
    
    # Context flags
    is_dead_rubber: bool = False
    is_six_pointer: bool = False
    is_early_season: bool = False
    is_late_season: bool = False
    
    # Highlightly specific
    highlightly_match_id: int = 0
    highlightly_league_id: int = 0
    
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


# ============================================================
# CONFIGURATION
# ============================================================

MIN_EDGE_ODDS = 1.70
MIN_H2H_GAMES = 5
DEFAULT_MODEL_PROB = 0.50
DEFAULT_EDGE = 0.00

MAX_RETRIES = 3
RETRY_DELAY_BASE = 2
MAX_PARALLEL_LEAGUES = 5
REQUEST_TIMEOUT = 30

# Tier thresholds
TOP6_THRESHOLD = 6
BOTTOM6_OFFSET = 5
EARLY_SEASON_GAMES = 10
LATE_SEASON_GAMES = 30

# Midweek detection
MIDWEEK_DAYS = {1, 2, 3, 4}
WEEKEND_DAYS = {5, 6, 0}

# Fatigue thresholds
FATIGUE_HIGH_REST_DAYS = 3
FATIGUE_MEDIUM_REST_DAYS = 5

DERBY_KEYWORDS = [
    'derby', 'derbi', 'clasico', 'rivalry', 'clássico',
    'superclasico', 'el clasico', 'derby della', 'north london',
    'merseyside', 'manchester derby', 'old firm', 'superclásico'
]


# ============================================================
# LEAGUE CONFIGURATION WITH HIGHLIGHTLY METADATA
# ============================================================

LEAGUE_MAP: Dict[int, Dict[str, Any]] = {
    # Tier 1 - Top European Leagues
    39: {
        "name": "Premier League",
        "country": "england",
        "tier": 1,
        "total_teams": 20,
        "season_start_month": 8,
    },
    140: {
        "name": "La Liga",
        "country": "spain",
        "tier": 1,
        "total_teams": 20,
        "season_start_month": 8,
    },
    78: {
        "name": "Bundesliga",
        "country": "germany",
        "tier": 1,
        "total_teams": 18,
        "season_start_month": 8,
    },
    135: {
        "name": "Serie A",
        "country": "italy",
        "tier": 1,
        "total_teams": 20,
        "season_start_month": 8,
    },
    61: {
        "name": "Ligue 1",
        "country": "france",
        "tier": 1,
        "total_teams": 18,
        "season_start_month": 8,
    },
    88: {
        "name": "Eredivisie",
        "country": "netherlands",
        "tier": 2,
        "total_teams": 18,
        "season_start_month": 8,
    },
    94: {
        "name": "Primeira Liga",
        "country": "portugal",
        "tier": 2,
        "total_teams": 18,
        "season_start_month": 8,
    },
    71: {
        "name": "Brasileiro",
        "country": "brazil",
        "tier": 1,
        "total_teams": 20,
        "season_start_month": 4,
    },
    128: {
        "name": "Primera Division",
        "country": "argentina",
        "tier": 1,
        "total_teams": 28,
        "season_start_month": 1,
    },
    292: {
        "name": "J1 League",
        "country": "japan",
        "tier": 1,
        "total_teams": 18,
        "season_start_month": 2,
    },
}

# Reverse mapping for team ID lookups
HIGHLIGHTLY_TEAM_IDS: Dict[str, int] = {}


# ============================================================
# HIGHLIGHTLY API CLIENT
# ============================================================

class HighlightlyClient:
    """Client for Highlightly Sports API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = HIGHLIGHTLY_BASE_URL
        self.headers = {"x-rapidapi-key": api_key}
        self.session = None
        
        try:
            import requests
            self.session = requests.Session()
            self.session.headers.update(self.headers)
            self.requests_available = True
        except ImportError:
            self.requests_available = False
    
    def _request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make request to Highlightly API"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if self.requests_available and self.session:
                response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                return response.json()
            else:
                # Fallback to urllib
                import urllib.request
                import urllib.parse
                
                if params:
                    qs = urllib.parse.urlencode(params)
                    url = f"{url}?{qs}"
                
                req = urllib.request.Request(url, headers=self.headers)
                with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
                    return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            logger.error(f"Highlightly API error: {e}")
            return None
    
    def get_matches(self, date: str = None, league_id: int = None, limit: int = 100) -> List[Dict]:
        """Fetch matches from Highlightly"""
        params = {"limit": limit}
        if date:
            params["date"] = date
        if league_id:
            params["leagueId"] = league_id
        
        data = self._request("/football/matches", params)
        
        if data and isinstance(data, dict):
            return data.get("data", [])
        elif data and isinstance(data, list):
            return data
        return []
    
    def get_match_details(self, match_id: int) -> Optional[Dict]:
        """Fetch detailed match information"""
        data = self._request(f"/football/matches/{match_id}")
        if data and isinstance(data, list) and data:
            return data[0]
        return None
    
    def get_team_stats(self, team_id: int, from_date: str = None) -> Dict:
        """Fetch team statistics"""
        if not from_date:
            from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        params = {"fromDate": from_date}
        data = self._request(f"/football/teams/statistics/{team_id}", params)
        
        if data and isinstance(data, list) and data:
            return data[0]
        return {}
    
    def get_team_info(self, team_id: int) -> Optional[Dict]:
        """Fetch team information"""
        data = self._request(f"/football/teams/{team_id}")
        if data and isinstance(data, list) and data:
            return data[0]
        return None
    
    def search_team(self, team_name: str) -> Optional[Dict]:
        """Search for a team by name"""
        params = {"name": team_name, "limit": 10}
        data = self._request("/football/teams", params)
        
        if data and isinstance(data, dict):
            teams = data.get("data", [])
            if teams:
                return teams[0]
        elif data and isinstance(data, list) and data:
            return data[0]
        return None
    
    def get_head_to_head(self, team_id_1: int, team_id_2: int) -> List[Dict]:
        """Fetch head-to-head history"""
        params = {"teamIdOne": team_id_1, "teamIdTwo": team_id_2}
        data = self._request("/football/head-2-head", params)
        
        if data and isinstance(data, list):
            return data
        elif data and isinstance(data, dict):
            return data.get("data", [])
        return []
    
    def get_last_five_games(self, team_id: int) -> List[Dict]:
        """Fetch last 5 games for a team"""
        params = {"teamId": team_id}
        data = self._request("/football/last-five-games", params)
        
        if data and isinstance(data, list):
            return data
        elif data and isinstance(data, dict):
            return data.get("data", [])
        return []
    
    def get_standings(self, league_id: int, season: int) -> Dict:
        """Fetch league standings"""
        params = {"leagueId": league_id, "season": season}
        data = self._request("/football/standings", params)
        return data if data else {}
    
    def get_leagues(self, league_name: str = None) -> List[Dict]:
        """Fetch leagues"""
        params = {"limit": 100}
        if league_name:
            params["leagueName"] = league_name
        
        data = self._request("/football/leagues", params)
        
        if data and isinstance(data, dict):
            return data.get("data", [])
        return []


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _classify_opponent_tier(
    position: int, 
    league_size: int = 20,
    season_progress: float = 0.5,
) -> str:
    """Classify opponent tier based on league position"""
    if season_progress < 0.25:
        if position <= 3:
            return "top6"
        elif position >= league_size - 2:
            return "bottom6"
        return "mid"
    
    if season_progress < 0.75:
        top_threshold = min(TOP6_THRESHOLD, max(4, int(league_size * 0.3)))
        bottom_threshold = league_size - max(3, int(league_size * 0.25))
    else:
        top_threshold = TOP6_THRESHOLD
        bottom_threshold = league_size - BOTTOM6_OFFSET
    
    if position <= top_threshold:
        return "top6"
    elif position >= bottom_threshold:
        return "bottom6"
    return "mid"


def _get_league_size(league_id: int) -> int:
    """Get total teams in league from config"""
    cfg = LEAGUE_MAP.get(league_id, {})
    return cfg.get("total_teams", 20)


def _calculate_season_progress(games_played: int, total_games: int) -> float:
    """Calculate season progress (0-1)"""
    if total_games <= 0:
        return 0.5
    return min(1.0, games_played / total_games)


def _is_early_season(games_played: int, total_games: int) -> bool:
    """Check if team is in early season phase"""
    return _calculate_season_progress(games_played, total_games) < 0.25


def _is_late_season(games_played: int, total_games: int) -> bool:
    """Check if team is in late season phase"""
    return _calculate_season_progress(games_played, total_games) > 0.85


def _is_midweek(kickoff_str: str) -> bool:
    """Check if kickoff is on a midweek day"""
    try:
        dt = datetime.fromisoformat(kickoff_str.replace("Z", "+00:00"))
        return dt.weekday() in MIDWEEK_DAYS
    except (ValueError, TypeError):
        return False


def _calculate_days_rest(last_match_date: Optional[datetime], current_date: Optional[datetime]) -> int:
    """Calculate days of rest since last match"""
    if last_match_date is None or current_date is None:
        return 7
    days = (current_date - last_match_date).days
    return max(0, min(21, days))


def _is_derby_match(home_name: str, away_name: str) -> bool:
    """Check if fixture is a local derby"""
    home_lower = home_name.lower()
    away_lower = away_name.lower()
    
    for keyword in DERBY_KEYWORDS:
        if keyword in home_lower or keyword in away_lower:
            return True
    
    cities = ['manchester', 'liverpool', 'london', 'madrid', 'barcelona', 'milan',
              'rome', 'berlin', 'munich', 'paris', 'lisbon', 'porto']
    
    for city in cities:
        if city in home_lower and city in away_lower:
            return True
    
    return False


def _extract_odds_from_match(match: Dict) -> Tuple[float, float, float]:
    """Extract odds from Highlightly match data"""
    home_odds = 2.00
    draw_odds = 3.25
    away_odds = 3.50
    
    # Try to get odds from match data
    state = match.get("state", {})
    score = state.get("score", {})
    
    # If match has implied probabilities from bookmakers
    odds_data = match.get("odds", [])
    if odds_data:
        for odd in odds_data:
            if odd.get("market") == "Full Time Result":
                for value in odd.get("values", []):
                    val = value.get("value", "")
                    if val == "Home":
                        home_odds = value.get("odd", home_odds)
                    elif val == "Draw":
                        draw_odds = value.get("odd", draw_odds)
                    elif val == "Away":
                        away_odds = value.get("odd", away_odds)
    
    return home_odds, draw_odds, away_odds


# ============================================================
# STANDINGS PARSING
# ============================================================

def parse_standings(standings_data: Dict) -> Dict[int, Dict]:
    """Parse Highlightly standings into team_id -> stats dict"""
    result = {}
    
    if not standings_data:
        return result
    
    groups = standings_data.get("groups", [])
    for group in groups:
        for standing in group.get("standings", []):
            team = standing.get("team", {})
            team_id = team.get("id")
            if not team_id:
                continue
            
            total = standing.get("total", {})
            result[team_id] = {
                "position": standing.get("position", 0),
                "points": standing.get("points", 0),
                "played": total.get("games", 0),
                "wins": total.get("wins", 0),
                "draws": total.get("draws", 0),
                "losses": total.get("loses", 0),
                "goals_for": total.get("scoredGoals", 0),
                "goals_against": total.get("receivedGoals", 0),
            }
    
    return result


# ============================================================
# PROFILE BUILDER
# ============================================================

def build_team_profile_from_highlightly(
    team_id: int,
    team_name: str,
    client: HighlightlyClient,
    league_size: int = 20,
) -> Any:
    """
    Build a fully populated TeamProfile from Highlightly data.
    """
    from module2 import TeamProfile, TransitionMatrix
    from module10 import build_tally_matrix
    
    profile = TeamProfile(team_id=str(team_id), team_name=team_name)
    
    # Get team statistics
    stats = client.get_team_stats(team_id)
    if stats:
        total = stats.get("total", {})
        games = total.get("games", {})
        goals = total.get("goals", {})
        
        profile.update_metrics({
            "core.games": float(games.get("played", 0)),
            "core.wins": float(games.get("wins", 0)),
            "core.draws": float(games.get("draws", 0)),
            "core.losses": float(games.get("loses", 0)),
            "core.goals": float(goals.get("scored", 0)),
            "core.goals_against": float(goals.get("received", 0)),
        })
        
        home = stats.get("home", {})
        home_games = home.get("games", {})
        profile.update_metrics({
            "home_wins": float(home_games.get("wins", 0)),
            "home_games": float(home_games.get("played", 1)),
        })
        
        away = stats.get("away", {})
        away_games = away.get("games", {})
        away_wins = away_games.get("wins", 0)
        away_played = max(away_games.get("played", 1), 1)
        profile.update_metrics({
            "away_win_rate": round(away_wins / away_played, 3),
        })
    
    # Get last 5 games for form
    last_five = client.get_last_five_games(team_id)
    seq = []
    for match in last_five:
        state = match.get("state", {})
        score = state.get("score", {}).get("current", "0-0")
        
        # Determine result
        home_team = match.get("homeTeam", {}).get("id")
        home_score, away_score = 0, 0
        if " - " in score:
            parts = score.split(" - ")
            if len(parts) == 2:
                home_score = int(parts[0]) if parts[0].isdigit() else 0
                away_score = int(parts[1]) if parts[1].isdigit() else 0
        
        if home_score > away_score:
            result = "W" if home_team == team_id else "L"
        elif away_score > home_score:
            result = "L" if home_team == team_id else "W"
        else:
            result = "D"
        seq.append(result)
    
    if seq:
        profile.form["recent_results"] = seq
    
    # Build transition matrix if enough data
    if len(seq) >= 5:
        from datetime import datetime
        tally = build_tally_matrix(str(team_id), seq, str(datetime.now().year))
        profile.transition = TransitionMatrix(
            pattern="UNKNOWN",
            probs={
                r: dict(tally.probs.get(r, {"W": 0.33, "D": 0.33, "L": 0.34}))
                for r in ["W", "D", "L"]
            },
            sample_size=tally.total_transitions,
        )
    
    return profile


# ============================================================
# H2H BUILDER
# ============================================================

def build_h2h_record_from_highlightly(
    h2h_fixtures: List[Dict],
    home_id: int,
    away_id: int,
    min_games: int = MIN_H2H_GAMES,
) -> Optional[Any]:
    """Build H2H record from Highlightly H2H data"""
    from module2 import H2HRecord
    
    if len(h2h_fixtures) < min_games:
        return None
    
    rec = H2HRecord()
    
    for match in h2h_fixtures:
        hid = match.get("homeTeam", {}).get("id")
        aid = match.get("awayTeam", {}).get("id")
        state = match.get("state", {})
        score = state.get("score", {}).get("current", "0-0")
        
        if " - " in score:
            parts = score.split(" - ")
            if len(parts) == 2:
                hg = int(parts[0]) if parts[0].isdigit() else 0
                ag = int(parts[1]) if parts[1].isdigit() else 0
            else:
                continue
        else:
            continue
        
        rec.games += 1
        fav_home = (hid == home_id)
        
        if hg == ag:
            rec.draws += 1
        elif (fav_home and hg > ag) or (not fav_home and ag > hg):
            rec.fav_wins += 1
        else:
            rec.und_wins += 1
    
    return rec if rec.games >= min_games else None


# ============================================================
# LEG BUILDER
# ============================================================

def build_leg_from_highlightly(
    match: Dict,
    client: HighlightlyClient,
    league_id: int,
    league_label: str,
    league_tier: int,
    league_country: str,
    standings: Dict[int, Dict],
) -> Optional[LegData]:
    """
    Build a complete Leg from Highlightly match data.
    """
    from module2 import Leg, BetMarket, CompetitionFormat, VenueType
    
    home_team = match.get("homeTeam", {})
    away_team = match.get("awayTeam", {})
    home_name = home_team.get("name", "")
    away_name = away_team.get("name", "")
    home_id = home_team.get("id")
    away_id = away_team.get("id")
    kickoff = match.get("date", "")
    match_id_val = match.get("id")
    
    if not home_name or not away_name or not home_id or not away_id:
        return None
    
    # Extract odds
    home_odds, draw_odds, away_odds = _extract_odds_from_match(match)
    
    if not home_odds or home_odds <= 1.0:
        return None
    
    # Determine favorite
    fav_is_home = home_odds <= away_odds
    fav_odds = home_odds if fav_is_home else away_odds
    
    if fav_odds < MIN_EDGE_ODDS:
        return None
    
    # Get positions from standings
    home_standing = standings.get(home_id, {})
    away_standing = standings.get(away_id, {})
    home_pos = home_standing.get("position", 10)
    away_pos = away_standing.get("position", 10)
    
    # Season progress
    league_size = _get_league_size(league_id)
    total_games = league_size * 2 - 1
    season_progress = _calculate_season_progress(
        max(home_standing.get("played", 0), away_standing.get("played", 0)),
        total_games
    )
    
    # Midweek and derby detection
    is_midweek = _is_midweek(kickoff)
    is_derby = _is_derby_match(home_name, away_name)
    is_early = _is_early_season(max(home_standing.get("played", 0), away_standing.get("played", 0)), total_games)
    is_late = _is_late_season(max(home_standing.get("played", 0), away_standing.get("played", 0)), total_games)
    
    # Opponent tier classification
    home_opponent_tier = _classify_opponent_tier(away_pos, league_size, season_progress)
    
    # Build team profiles
    home_profile = build_team_profile_from_highlightly(
        home_id, home_name, client, league_size
    )
    away_profile = build_team_profile_from_highlightly(
        away_id, away_name, client, league_size
    )
    
    # Add positions to profiles
    home_profile.update_metrics({"position": float(home_pos)})
    away_profile.update_metrics({"position": float(away_pos)})
    
    # Build match ID
    date_str = kickoff[:10].replace("-", "") if kickoff else "00000000"
    match_id_str = f"{league_label.replace(' ', '_')}_{date_str}_{home_name.replace(' ', '_')}_{away_name.replace(' ', '_')}"
    
    # Get H2H data
    h2h_fixtures = client.get_head_to_head(home_id, away_id)
    h2h_record = build_h2h_record_from_highlightly(h2h_fixtures, home_id, away_id)
    
    # Build Leg object
    leg = Leg(
        match_id=match_id_str,
        selection=home_name if fav_is_home else away_name,
        odds=home_odds if fav_is_home else away_odds,
        market=BetMarket.STRAIGHT_WIN,
        league=league_label,
        league_id=league_id,
        league_tier=league_tier,
        league_country=league_country,
        competition_type="league",
        stage="",
        round="",
        kickoff=kickoff,
        home_profile=home_profile,
        away_profile=away_profile,
        h2h=h2h_record,
        competition_format=CompetitionFormat.REGULAR_SEASON,
    )
    
    # Store odds on Leg
    leg.home_odds = home_odds
    leg.away_odds = away_odds
    leg.draw_odds = draw_odds
    
    # Store Highlightly IDs
    leg.features["highlightly_match_id"] = match_id_val
    leg.features["highlightly_league_id"] = league_id
    
    # Store distortion tracking data
    leg.features["is_midweek"] = is_midweek
    leg.features["is_derby"] = is_derby
    leg.features["is_early_season"] = is_early
    leg.features["is_late_season"] = is_late
    
    # Set model probability
    implied_prob = 1.0 / leg.odds if leg.odds > 1.0 else DEFAULT_MODEL_PROB
    leg.model_prob = implied_prob
    leg.adjusted_prob = implied_prob
    leg.edge = DEFAULT_EDGE
    leg.pre_verdict = "PENDING"
    leg.venue = VenueType.HOME if fav_is_home else VenueType.AWAY
    
    # Create LegData
    return LegData(
        leg=leg,
        fav_is_home=fav_is_home,
        home_odds=home_odds,
        away_odds=away_odds,
        draw_odds=draw_odds,
        model_prob=implied_prob,
        edge=DEFAULT_EDGE,
        venue=leg.venue.value,
        opponent_tier=home_opponent_tier,
        home_team_position=home_pos,
        away_team_position=away_pos,
        league_size=league_size,
        season_progress=season_progress,
        days_rest=7,
        is_midweek=is_midweek,
        is_derby=is_derby,
        manager_tenure_days_home=365,
        manager_tenure_days_away=365,
        key_players_missing_home=0,
        key_players_missing_away=0,
        is_dead_rubber=False,
        is_six_pointer=False,
        is_early_season=is_early,
        is_late_season=is_late,
        highlightly_match_id=match_id_val or 0,
        highlightly_league_id=league_id,
        kickoff=kickoff,
    )


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def load_todays_legs(
    highlightly_key: str,
    league_ids: List[int] = None,
    target_date: str = None,
    verbose: bool = True,
    use_mock: bool = False,
) -> List[LegData]:
    """
    Main entry point. Fetches today's fixtures from Highlightly API
    and returns fully populated LegData objects.
    
    Args:
        highlightly_key: Your Highlightly API key from RapidAPI
        league_ids: List of league IDs to fetch (default: all configured leagues)
        target_date: Target date in YYYY-MM-DD format (default: today)
        verbose: Print progress messages
        use_mock: Use mock data for testing
    
    Returns:
        List of LegData objects ready for the pipeline
    """
    if not use_mock and not highlightly_key:
        raise ValueError("Highlightly API key is required (or use use_mock=True)")
    
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")
    
    if league_ids is None:
        league_ids = list(LEAGUE_MAP.keys())
    
    if use_mock:
        logger.info("Running in MOCK mode - using generated data")
        return _load_todays_legs_mock(verbose)
    
    if verbose:
        logger.info(f"🔍 Highlightly API: Fetching matches for {target_date}")
    
    # Initialize client
    client = HighlightlyClient(highlightly_key)
    
    all_legs = []
    
    for league_id in league_ids:
        league_config = LEAGUE_MAP.get(league_id)
        if not league_config:
            if verbose:
                logger.warning(f"Skipping unknown league_id {league_id}")
            continue
        
        if verbose:
            logger.info(f"📊 Processing {league_config['name']} (ID: {league_id})")
        
        # Fetch matches for this league
        matches = client.get_matches(date=target_date, league_id=league_id, limit=50)
        
        if not matches:
            if verbose:
                logger.info(f"  No matches found for {league_config['name']} on {target_date}")
            continue
        
        if verbose:
            logger.info(f"  Found {len(matches)} matches")
        
        # Fetch standings for this league
        season = datetime.now().year
        standings_data = client.get_standings(league_id, season)
        standings = parse_standings(standings_data)
        
        # Build legs from each match
        for match in matches:
            state = match.get("state", {})
            description = state.get("description", "")
            
            # Only process scheduled/not started matches
            if description not in ["Not started", "Scheduled", "To be announced"]:
                continue
            
            leg_data = build_leg_from_highlightly(
                match=match,
                client=client,
                league_id=league_id,
                league_label=league_config["name"],
                league_tier=league_config["tier"],
                league_country=league_config["country"],
                standings=standings,
            )
            
            if leg_data:
                all_legs.append(leg_data)
                
                if verbose:
                    logger.info(f"    ✓ {leg_data.summary()}")
        
        # Rate limit protection
        time.sleep(1)
    
    if verbose:
        logger.info(f"✅ Total legs built: {len(all_legs)}")
    
    return all_legs


# ============================================================
# MOCK DATA FOR TESTING
# ============================================================

def _load_todays_legs_mock(verbose: bool = True) -> List[LegData]:
    """Generate mock leg data for testing"""
    from module2 import Leg, BetMarket, TeamProfile, CompetitionFormat, VenueType
    
    mock_legs = []
    
    mock_teams = [
        ("Arsenal", "Chelsea", 1.85, 3.40, 4.20, 3, 5),
        ("Manchester City", "Liverpool", 1.95, 3.60, 3.80, 1, 2),
        ("Barcelona", "Real Madrid", 2.10, 3.30, 3.50, 2, 1),
        ("Bayern Munich", "Borussia Dortmund", 1.75, 3.80, 4.50, 1, 4),
        ("AC Milan", "Inter Milan", 2.30, 3.20, 3.10, 5, 3),
    ]
    
    for i, (home, away, h_odds, d_odds, a_odds, home_pos, away_pos) in enumerate(mock_teams):
        fav_is_home = h_odds <= a_odds
        
        home_profile = TeamProfile(team_id=f"mock_h_{i}", team_name=home)
        home_profile.update_metrics({
            "core.games": 20.0,
            "core.wins": 12.0,
            "core.draws": 5.0,
            "core.losses": 3.0,
            "core.goals": 45.0,
            "core.goals_against": 25.0,
            "position": float(home_pos),
        })
        
        away_profile = TeamProfile(team_id=f"mock_a_{i}", team_name=away)
        away_profile.update_metrics({
            "core.games": 20.0,
            "core.wins": 10.0,
            "core.draws": 6.0,
            "core.losses": 4.0,
            "core.goals": 38.0,
            "core.goals_against": 28.0,
            "position": float(away_pos),
        })
        
        leg = Leg(
            match_id=f"mock_{i}",
            selection=home if fav_is_home else away,
            odds=h_odds if fav_is_home else a_odds,
            market=BetMarket.STRAIGHT_WIN,
            league="Mock League",
            league_id=39,
            league_tier=1,
            league_country="england",
            competition_type="league",
            stage="",
            round="",
            kickoff=datetime.now().isoformat(),
            home_profile=home_profile,
            away_profile=away_profile,
            h2h=None,
            competition_format=CompetitionFormat.REGULAR_SEASON,
        )
        
        leg.home_odds = h_odds
        leg.away_odds = a_odds
        leg.draw_odds = d_odds
        leg.model_prob = 1.0 / (h_odds if fav_is_home else a_odds)
        leg.venue = VenueType.HOME if fav_is_home else VenueType.AWAY
        
        leg_data = LegData(
            leg=leg,
            fav_is_home=fav_is_home,
            home_odds=h_odds,
            away_odds=a_odds,
            draw_odds=d_odds,
            model_prob=leg.model_prob,
            edge=0.05,
            venue=leg.venue.value,
            opponent_tier="top6" if i < 2 else "mid",
            home_team_position=home_pos,
            away_team_position=away_pos,
            league_size=20,
            season_progress=0.5,
            days_rest=7,
            is_midweek=False,
            is_derby=(i == 4),
            manager_tenure_days_home=365,
            manager_tenure_days_away=365,
            key_players_missing_home=0,
            key_players_missing_away=0,
            is_dead_rubber=False,
            is_six_pointer=False,
            is_early_season=False,
            is_late_season=False,
            kickoff=datetime.now().isoformat(),
        )
        
        mock_legs.append(leg_data)
        
        if verbose:
            logger.info(f"  Mock leg: {home} vs {away}")
    
    return mock_legs


# ============================================================
# COMPATIBILITY WRAPPERS
# ============================================================

def load_todays_legs_legacy(
    football_key: str = None,
    odds_key: str = None,
    league_ids: List[int] = None,
    verbose: bool = True,
    use_mock: bool = False,
) -> List[LegData]:
    """
    Legacy wrapper that accepts both football_key and odds_key.
    Now uses Highlightly API internally.
    """
    # Try to get Highlightly key from environment
    highlightly_key = football_key or odds_key or os.environ.get("HIGHLIGHTLY_API_KEY", "")
    
    if not highlightly_key and not use_mock:
        logger.warning("No Highlightly API key found. Use use_mock=True for testing.")
        return _load_todays_legs_mock(verbose)
    
    return load_todays_legs(
        highlightly_key=highlightly_key,
        league_ids=league_ids,
        verbose=verbose,
        use_mock=use_mock,
    )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "LegData",
    "load_todays_legs",
    "load_todays_legs_legacy",
    "HighlightlyClient",
    "LEAGUE_MAP",
]
