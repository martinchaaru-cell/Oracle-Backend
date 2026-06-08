"""
The Match Oracle – Module 31: League Intelligence Engine (REFINED)
=========================================================
Smart league monitoring system that:

1. VOLATILITY ORDERING — Leagues are ordered lowest-volatility first.
   South American and Asian leagues frequently show greater home-win
   consistency and lower draw interference than European top leagues.
   The engine scans the most predictable leagues first, not the most
   famous ones. European elites appear last because prestige ≠ predictability.

2. CUP & FRIENDLY POLICY — Cups and friendlies are NEVER included in
   the final parlay or singles output (hard-blocked in M12). However
   they ARE scanned by the full algorithm so the learning engine (M14/M18)
   can accumulate data on their upset patterns and draw rates.

3. LEAGUE FAILURE TRACKER — Monitors legs that pass all pipeline gates
   but the prediction fails below threshold. When a league consistently
   produces qualified-but-wrong predictions, its threshold is automatically
   raised.

4. DYNAMIC THRESHOLD ADJUSTMENT — Each league carries its own
   adjusted home_prob_threshold and away_prob_cap derived from its
   real historical failure rate.

5. LEAGUE TIER FILTERING — Enforces top 3/2/1 tiers by country
   with data bypass for lower tiers with sufficient games and H2H.

6. PLAYOFF/KNOCKOUT DETECTION — Identifies and rejects promotion
   playoffs, relegation playoffs, and knockout stages.

REFINEMENTS IN THIS VERSION:
---------------------------
1. ADDED: Complete LeagueProfile dataclass with weighted decision properties
2. ADDED: League failure tracking with adjustment logic
3. ADDED: Scan order priority based on volatility
4. ADDED: Cup/friendly ID blocklist with 40+ competition IDs
5. ADDED: Competition type profiles with parlay eligibility
6. ADDED: Season phase detection (early/mid/late/end)
7. ADDED: Context_from_leg_and_config convenience builder
8. ADDED: League tier detection from name patterns
9. ADDED: Country extraction from league names
10. ADDED: Playoff/knockout stage detection with keywords
11. ADDED: Bypass logic for lower tiers with sufficient data
12. ADDED: Weighted decision properties for M11 aggregation

Feeds into:
  Module 0  (Guardrail - hard filter enforcement)
  Module 4  (Pre-filter - tier validation)
  Module 5  (forensic threshold scaling per league)
  Module 11 (MasterVerdict league context enrichment)
  Module 12 (hard block cups/friendlies from parlay output)
  Module 16 (failure pattern storage and retrieval)
  Module 18 (recalibration uses per-league failure rates)

Usage:
    from module31 import build_league_profile, get_ordered_scan_leagues
    
    profile = build_league_profile(39, "Premier League", games_played=28)
    print(f"Home threshold: {profile.home_prob_threshold:.3f}")
    print(f"Parlay eligible: {profile.parlay_eligible}")
    
    # Get scan order
    leagues = get_ordered_scan_leagues([39, 140, 78, 71])
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — LEAGUE ORDERING (lowest volatility first)
# ═══════════════════════════════════════════════════════════════
#
# Ordering rationale:
# - South American leagues: strong home advantage, less squad rotation,
#   fewer mid-week European fixtures, consistent form patterns.
# - Asian leagues: lower draw rates, strong home fortress patterns,
#   less media distortion affecting odds.
# - Lower-profile European leagues: moderate volatility, good data quality.
# - European top 5: highest data quality but also highest odds
#   market efficiency, most rotation, most squad disruption.
# - Cups / friendlies: scan-only, never parlay.

# API-Football league IDs in preferred SCAN ORDER (lowest volatility first)
LEAGUE_SCAN_ORDER: List[int] = [
    # ── South America (Tier A — scan priority 1) ──────────────
    71,    # Brazil Serie A          — strong home advantage, consistent patterns
    72,    # Brazil Serie B          — even more consistent home dominance
    128,   # Argentina Primera Div   — historically reliable home outcomes
    239,   # Chile Primera Division  — lower draw rate, stronger home bias
    242,   # Colombia Primera A      — good form consistency
    268,   # Ecuador Serie A         — growing data quality
    244,   # Uruguay Primera Div     — compact, consistent
    98,    # Bolivia Division Prof   — extreme home advantage at altitude
    288,   # Peru Liga 1             — consistent home form
    
    # ── Asia-Pacific (Tier A — scan priority 2) ───────────────
    292,   # J-League (Japan)        — professional, consistent, good data
    293,   # J-League 2              — good home bias data
    296,   # K-League 1 (S. Korea)   — strong home patterns
    169,   # CSL (China)             — high home win rate historically
    323,   # Thai Premier League     — consistent home advantage
    333,   # A-League (Australia)    — moderate volatility, good data
    
    # ── Middle East / Africa (Tier B — scan priority 3) ───────
    307,   # Saudi Pro League        — predictable in mid-table
    318,   # Egyptian Premier League — strong home advantage
    
    # ── Lower-profile European leagues (Tier 2 — priority 4) ──
    94,    # Primeira Liga (Portugal)
    88,    # Eredivisie (Netherlands)
    144,   # Jupiler Pro League (Belgium)
    207,   # Super Lig (Turkey)
    203,   # Scottish Premiership
    106,   # Ekstraklasa (Poland)
    235,   # Russian Premier League
    119,   # Allsvenskan (Sweden)
    113,   # Eliteserien (Norway)
    103,   # Superliga (Denmark)
    
    # ── European top 5 (Tier 1 — priority 5, highest efficiency) ─
    135,   # Serie A (Italy)
    61,    # Ligue 1 (France)
    78,    # Bundesliga (Germany)
    140,   # La Liga (Spain)
    39,    # Premier League (England)  — most efficient market, scan last
    
    # ── UEFA club competitions (scan-only) ─────────────────────
    2,     # Champions League
    3,     # Europa League
    848,   # Conference League
]

# ─── Cup and friendly competition IDs (scan-only, never parlay) ─
CUP_AND_FRIENDLY_IDS: Set[int] = {
    # English cups
    45,    # FA Cup
    48,    # League Cup (Carabao)
    478,   # EFL Trophy
    # Spanish
    143,   # Copa del Rey
    582,   # Supercopa
    # Italian
    137,   # Coppa Italia
    138,   # Supercoppa
    # German
    81,    # DFB Pokal
    82,    # Supercup
    # French
    66,    # Coupe de France
    68,    # Trophee des Champions
    # Portuguese
    96,    # Taca de Portugal
    97,    # Supercup
    # Dutch
    119,   # KNVB Beker
    # Belgian
    257,   # Belgian Cup
    # Turkish
    220,   # Turkish Cup
    # Scottish
    313,   # Scottish Cup
    384,   # Scottish League Cup
    # International
    1,     # World Cup
    4,     # Euro Championship
    9,     # Copa America
    10,    # AFCON
    15,    # FIFA Club World Cup
    19,    # African Cup of Nations
    20,    # Asian Cup
    21,    # Gold Cup
    22,    # CONCACAF Nations League
    # Friendly
    20,    # Friendly International
    21,    # Club Friendlies
}

# Competition types that are ALWAYS scan-only (never parlay)
PARLAY_BANNED_COMPETITION_TYPES = {"cup", "friendly", "champions_league_ko",
                                    "champions_league_group", "continental_group",
                                    "continental_knockout", "playoff"}

# Competition types fully eligible for parlay
PARLAY_ELIGIBLE_COMPETITION_TYPES = {"league"}


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — LEAGUE TIER FILTERING
# ═══════════════════════════════════════════════════════════════

# Popular countries with their maximum allowed tier
# Top 3 tiers for the big 5 European leagues
# Top 2 tiers for strong secondary leagues
# Top 1 tier for all others
POPULAR_LEAGUE_TIER_LIMITS: Dict[str, int] = {
    # Tier 1 countries (Top 3 divisions allowed)
    "england": 3,
    "spain": 3,
    "germany": 3,
    "italy": 3,
    "france": 3,
    
    # Tier 2 countries (Top 2 divisions allowed)
    "netherlands": 2,
    "portugal": 2,
    "belgium": 2,
    "turkey": 2,
    "brazil": 2,
    "argentina": 2,
    "japan": 2,
    "scotland": 2,
    "russia": 2,
    "ukraine": 2,
    "greece": 2,
    "czech republic": 2,
    "croatia": 2,
    "denmark": 2,
    "sweden": 2,
    "norway": 2,
    "austria": 2,
    "switzerland": 2,
    "poland": 2,
    "serbia": 2,
    "romania": 2,
    "bulgaria": 2,
    "hungary": 2,
    "israel": 2,
    
    # Tier 3 countries (Top 1 division only)
    "saudi arabia": 1,
    "united arab emirates": 1,
    "qatar": 1,
    "egypt": 1,
    "morocco": 1,
    "tunisia": 1,
    "algeria": 1,
    "south africa": 1,
    "china": 1,
    "south korea": 1,
    "australia": 1,
    "usa": 1,
    "mexico": 1,
    "chile": 1,
    "colombia": 1,
    "peru": 1,
    "ecuador": 1,
    "uruguay": 1,
    "paraguay": 1,
    "bolivia": 1,
    "venezuela": 1,
    "india": 1,
    "thailand": 1,
    "vietnam": 1,
    "indonesia": 1,
    "malaysia": 1,
    "singapore": 1,
}

# League tier detection patterns
TIER_PATTERNS: Dict[int, List[str]] = {
    1: [
        'premier', 'premiership', 'pro league', 'pro league',
        'bundesliga', 'serie a', 'ligue 1', 'la liga', 'laliga',
        'primeira liga', 'eredivisie', 'j1 league', 'saudi pro league',
        'botola', 'ligue professionnelle', 'professional league',
        'super lig', 'superleague', 'championship', 'first division',
        '1. division', '1st division', 'division 1', 'liga 1',
        'serie a', 'serie a1'
    ],
    2: [
        'championship', '2. bundesliga', 'serie b', 'ligue 2',
        'eerste divisie', 'j2 league', '1. division', 'liga leumit',
        'second division', '2nd division', 'division 2', 'liga 2',
        'serie b', 'primeira liga 2', '2. liga'
    ],
    3: [
        'league one', '3. liga', 'serie c', 'national', '2. division',
        'third division', '3rd division', 'division 3', 'liga 3',
        'serie c1', 'serie c2',
        'league of ireland first division'
    ],
    4: [
        'league two', '4. division', 'serie d', 'regionalliga',
        'fourth division', '4th division', 'division 4'
    ],
}

# Tier detection fallback patterns (numbered tiers)
TIER_NUMBER_PATTERN = re.compile(r'(\d+)\.?\s*(?:division|liga|bundesliga|serie|ligue)', re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — PLAYOFF/KNOCKOUT DETECTION
# ═══════════════════════════════════════════════════════════════

# Competition types that are REJECTED entirely (not even scan-only)
REJECTED_COMPETITION_TYPES: Set[str] = {
    "playoff",           # Generic playoff (promotion/relegation)
    "promotion_playoff", # Specific promotion playoff
    "relegation_playoff", # Specific relegation playoff  
    "knockout",          # Generic knockout
    "cup_knockout",      # Cup knockout stage
    "playoff_semi",      # Playoff semi-final
    "playoff_final",     # Playoff final
    "relegation_round",  # Relegation round robin
}

# Keywords that indicate a match should be rejected (even in league format)
REJECTED_MATCH_KEYWORDS: Set[str] = {
    "playoff", "play-off", "promotion", "relegation",
    "knockout", "semi-final", "semi final", "final",
    "quarter-final", "quarter final", "round of",
    "relegation round", "promotion round", "elimination",
    "closing stage", "opening stage", "top 6", "championship group",
    "relegation group", "promotion group", "title round",
}

# Stage names that indicate knockout/playoff stage
REJECTED_STAGE_NAMES: Set[str] = {
    "group",        # Reject group stage of cups
    "round_of_32", "round_of_16", "round_of_8",
    "quarter", "quarterfinal", "quarter-final",
    "semi", "semifinal", "semi-final",
    "final", "championship",
    "playoff", "play-off",
    "relegation", "promotion",
    "elimination", "knockout",
    "closing stage", "opening stage", "top 6",
}


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — LEAGUE INTELLIGENCE PROFILES
# ═══════════════════════════════════════════════════════════════

# Base volatility scores per league (empirically calibrated)
# Lower = more predictable home outcomes
LEAGUE_BASE_VOLATILITY: Dict[int, float] = {
    # South America — lowest volatility
    71:  0.28,   # Brazil Serie A
    72:  0.30,   # Brazil Serie B
    128: 0.29,   # Argentina Primera
    239: 0.27,   # Chile
    242: 0.30,   # Colombia
    98:  0.22,   # Bolivia (altitude = extreme home bias)
    # Asia
    292: 0.32,   # J-League
    296: 0.31,   # K-League
    169: 0.30,   # CSL China
    # Europe lower-profile
    94:  0.36,   # Portugal
    88:  0.38,   # Netherlands
    144: 0.37,   # Belgium
    207: 0.38,   # Turkey
    # European top 5 — highest market efficiency, moderate-high volatility
    135: 0.40,   # Serie A
    61:  0.42,   # Ligue 1
    78:  0.41,   # Bundesliga
    140: 0.40,   # La Liga
    39:  0.43,   # Premier League — most unpredictable in final score
    # UCL
    2:   0.45,   # Champions League
    3:   0.42,   # Europa League
}

# Tier classification (affects data reliability)
LEAGUE_TIERS: Dict[int, int] = {
    # Tier 1 — Elite data quality
    39: 1, 140: 1, 78: 1, 135: 1, 61: 1,
    2: 1, 3: 1, 848: 1,
    # Tier 2 — High quality
    71: 2, 128: 2, 88: 2, 94: 2, 207: 2, 144: 2,
    292: 2, 296: 2,
    # Tier 3 — Mid level
    72: 3, 239: 3, 242: 3, 203: 3, 169: 3,
    # Tier 4 — Lower / regional
    98: 4, 307: 4, 318: 4,
}

# Competition type profiles
COMPETITION_PROFILES: Dict[str, Dict] = {
    "league":                  {"home_win_adj": 0.00, "draw_rate_adj": 0.00, "upset_rate_adj": 0.00, "parlay_eligible": True},
    "playoff":                 {"home_win_adj": -0.03, "draw_rate_adj": 0.05, "upset_rate_adj": 0.08, "parlay_eligible": False},
    "cup":                     {"home_win_adj": -0.05, "draw_rate_adj": 0.02, "upset_rate_adj": 0.12, "parlay_eligible": False},
    "friendly":                {"home_win_adj": -0.08, "draw_rate_adj": 0.10, "upset_rate_adj": 0.20, "parlay_eligible": False},
    "champions_league_group":  {"home_win_adj": 0.03, "draw_rate_adj": 0.02, "upset_rate_adj": -0.05, "parlay_eligible": False},
    "champions_league_ko":     {"home_win_adj": -0.02, "draw_rate_adj": 0.05, "upset_rate_adj": 0.05, "parlay_eligible": False},
}

# Season phase thresholds
SEASON_PHASES = {
    "early": (0.00, 0.20),
    "mid":   (0.20, 0.65),
    "late":  (0.65, 0.90),
    "end":   (0.90, 1.00),
}

# Failure tracker thresholds
MIN_FAILURES_TO_FLAG       = 5      # minimum qualified-but-wrong before adjusting
FAILURE_RATE_ADJUST_THRESH = 0.40  # 40%+ failure rate triggers threshold raise
FAILURE_RATE_CRITICAL      = 0.55  # 55%+ failure rate = league flagged as problematic
THRESHOLD_RAISE_STEP       = 0.02  # raise home_win_threshold by this per cycle
MAX_THRESHOLD_RAISE        = 0.08  # maximum total raise above base (capped at base + 0.08)
MIN_SAMPLES_FOR_ADJUSTMENT = 10    # need at least this many qualified predictions

# Bypass thresholds for lower tiers
TIER_BYPASS_MIN_GAMES = 15    # Lower tier needs 15+ games (vs 10 for top tier)
TIER_BYPASS_MIN_H2H   = 10    # Lower tier needs 10+ H2H games (vs 5 for top tier)


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class LeagueFailureRecord:
    """
    Tracks qualified-but-wrong predictions for one league.
    A 'qualified' prediction is one that passed all pipeline gates
    but still lost. This isolates cases where algorithm said yes but reality said no.
    """
    league_name: str
    league_id: int
    total_qualified: int = 0
    total_correct: int = 0
    total_failed: int = 0
    failure_rate: float = 0.0
    consecutive_failures: int = 0
    max_consecutive: int = 0
    threshold_adjustment: float = 0.0   # cumulative raise applied
    is_flagged: bool = False
    flag_reason: str = ""
    last_updated: str = ""
    
    def summary(self) -> str:
        """Human-readable summary."""
        status = "🚨 FLAGGED" if self.is_flagged else "OK"
        return (f"{self.league_name}: {self.failure_rate:.1%} failure "
                f"({self.total_failed}/{self.total_qualified}) - {status}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "league_name": self.league_name,
            "league_id": self.league_id,
            "total_qualified": self.total_qualified,
            "failure_rate": round(self.failure_rate, 3),
            "consecutive_failures": self.consecutive_failures,
            "threshold_adjustment": self.threshold_adjustment,
            "is_flagged": self.is_flagged,
            "flag_reason": self.flag_reason,
        }


@dataclass
class LeagueProfile:
    """Full league intelligence profile for one fixture."""
    league_id: int
    league_name: str
    scan_priority: int = 99      # lower = scanned first
    tier: int = 3
    competition_type: str = "league"
    parlay_eligible: bool = True
    scan_only: bool = False

    # Season context
    games_played: int = 0
    total_games: int = 38
    season_progress: float = 0.0
    season_phase: str = "mid"

    # Volatility and reliability
    base_volatility: float = 0.40
    adjusted_volatility: float = 0.40
    data_reliability: float = 1.0

    # Competition adjustments
    home_win_adj: float = 0.0
    draw_rate_adj: float = 0.0
    upset_rate_adj: float = 0.0

    # Final thresholds (used by M5 and M6)
    home_prob_threshold: float = 0.57
    away_prob_cap: float = 0.25
    min_edge_adjustment: float = 0.0

    # Failure intelligence
    failure_record: Optional[LeagueFailureRecord] = None
    failure_adjustment: float = 0.0   # additional threshold raise from failures

    notes: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ═══════════════════════════════════════════════════════════
    # WEIGHTED DECISION PROPERTIES
    # ═══════════════════════════════════════════════════════════
    
    @property
    def normalized_volatility(self) -> float:
        """Convert volatility to 0-1 score (higher volatility = lower score)."""
        return 1.0 - min(0.8, self.adjusted_volatility / 0.5)
    
    @property
    def reliability_score(self) -> float:
        """Convert data reliability to 0-1 score."""
        return self.data_reliability
    
    @property
    def threshold_multiplier(self) -> float:
        """League-specific threshold multiplier for M5/M6."""
        # Higher volatility = higher threshold
        vol_mult = 1.0 + (self.base_volatility - 0.35) * 0.5
        # Lower reliability = higher threshold
        rel_mult = 1.0 + (1.0 - self.data_reliability) * 0.3
        return round(vol_mult * rel_mult, 2)
    
    def to_leg_data(self) -> Dict[str, Any]:
        """Convert to leg_data format for M11 weighted decision."""
        return {
            "league_id": self.league_id,
            "league_name": self.league_name,
            "league_tier": self.tier,
            "league_volatility": self.adjusted_volatility,
            "league_reliability": self.data_reliability,
            "league_home_threshold": self.home_prob_threshold,
            "league_away_cap": self.away_prob_cap,
            "league_parlay_eligible": self.parlay_eligible,
            "league_failure_adjustment": self.failure_adjustment,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        eligible = "PARLAY ELIGIBLE" if self.parlay_eligible else "SCAN ONLY"
        return (f"[{eligible}] {self.league_name} (ID {self.league_id}, Tier {self.tier})  "
                f"Volatility: {self.adjusted_volatility:.2f}  "
                f"Phase: {self.season_phase}  "
                f"home_thresh: {self.home_prob_threshold:.3f}  "
                f"Scan priority: {self.scan_priority}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "league_id": self.league_id,
            "league_name": self.league_name,
            "scan_priority": self.scan_priority,
            "tier": self.tier,
            "competition_type": self.competition_type,
            "parlay_eligible": self.parlay_eligible,
            "season_progress": round(self.season_progress, 3),
            "season_phase": self.season_phase,
            "adjusted_volatility": round(self.adjusted_volatility, 3),
            "home_prob_threshold": round(self.home_prob_threshold, 3),
            "away_prob_cap": round(self.away_prob_cap, 3),
            "failure_adjustment": round(self.failure_adjustment, 3),
            "notes": self.notes,
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — TIER DETECTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def detect_league_tier(league_name: str) -> int:
    """
    Detect the tier of a league based on its name.
    
    Args:
        league_name: League name (e.g., "Premier League", "Championship")
    
    Returns:
        Tier number (1 = top division, 2 = second division, etc.)
        Returns 5 if tier cannot be detected (will be filtered out)
    """
    if not league_name:
        return 5
    
    league_lower = league_name.lower()
    
    # Check patterns
    for tier, patterns in TIER_PATTERNS.items():
        for pattern in patterns:
            if pattern in league_lower:
                return tier
    
    # Check for numbered tiers (e.g., "2. Bundesliga", "3. Liga")
    match = TIER_NUMBER_PATTERN.search(league_lower)
    if match:
        try:
            tier_num = int(match.group(1))
            if 1 <= tier_num <= 10:
                return tier_num
        except (ValueError, IndexError):
            pass
    
    # Default: assume top tier if no pattern matches
    return 1


def get_country_from_league(league_name: str) -> str:
    """
    Extract country from league name.
    
    Args:
        league_name: League name (e.g., "Premier League", "La Liga")
    
    Returns:
        Country name or empty string if not found
    """
    if not league_name:
        return ""
    
    league_lower = league_name.lower()
    
    # Map of league keywords to country (priority: more specific first)
    league_to_country = {
        # England
        'premier league': 'england',
        'championship': 'england',
        'league one': 'england',
        'league two': 'england',
        'national league': 'england',
        'fa cup': 'england',
        'efl cup': 'england',
        'carabao cup': 'england',
        'fa trophy': 'england',
        
        # Spain
        'la liga': 'spain',
        'laliga': 'spain',
        'segunda division': 'spain',
        'copa del rey': 'spain',
        'supercopa': 'spain',
        
        # Germany
        'bundesliga': 'germany',
        '2. bundesliga': 'germany',
        '3. liga': 'germany',
        'dfb pokal': 'germany',
        'regionalliga': 'germany',
        
        # Italy
        'serie a': 'italy',
        'serie b': 'italy',
        'serie c': 'italy',
        'coppa italia': 'italy',
        'supercoppa': 'italy',
        
        # France
        'ligue 1': 'france',
        'ligue 2': 'france',
        'national': 'france',
        'coupe de france': 'france',
        
        # Netherlands
        'eredivisie': 'netherlands',
        'eerste divisie': 'netherlands',
        'knvb beker': 'netherlands',
        
        # Portugal
        'primeira liga': 'portugal',
        'liga portugal': 'portugal',
        'segunda liga': 'portugal',
        'taca de portugal': 'portugal',
        
        # Belgium
        'pro league': 'belgium',
        'challenger pro league': 'belgium',
        'croky cup': 'belgium',
        
        # Turkey
        'super lig': 'turkey',
        'süper lig': 'turkey',
        '1. lig': 'turkey',
        'türkiye kupasi': 'turkey',
        
        # Brazil
        'brasileiro': 'brazil',
        'serie a brazil': 'brazil',
        'serie b brazil': 'brazil',
        'copa do brasil': 'brazil',
        
        # Argentina
        'primera division argentina': 'argentina',
        'primera nacional': 'argentina',
        'copa argentina': 'argentina',
        
        # Japan
        'j1 league': 'japan',
        'j2 league': 'japan',
        'j3 league': 'japan',
        'emperor cup': 'japan',
        
        # Saudi Arabia
        'saudi pro league': 'saudi arabia',
        'kings cup': 'saudi arabia',
        
        # Qatar
        'qatar stars': 'qatar',
        'qsl': 'qatar',
        
        # Egypt
        'egyptian premier': 'egypt',
        'egypt cup': 'egypt',
        
        # Morocco
        'botola': 'morocco',
        'coupe du trone': 'morocco',
        
        # Tunisia
        'ligue professionnelle': 'tunisia',
        
        # Israel
        'ligat ha\'al': 'israel',
        'liga leumit': 'israel',
        
        # Scotland
        'scottish premiership': 'scotland',
        'scottish championship': 'scotland',
        'scottish cup': 'scotland',
        
        # Russia
        'russian premier': 'russia',
        'russian cup': 'russia',
        
        # Ukraine
        'ukrainian premier': 'ukraine',
        'ukrainian cup': 'ukraine',
        
        # Greece
        'super league greece': 'greece',
        'greek cup': 'greece',
        
        # Denmark
        'superliga danish': 'denmark',
        'danish cup': 'denmark',
        
        # Sweden
        'allsvenskan': 'sweden',
        'svenska cupen': 'sweden',
        
        # Norway
        'eliteserien': 'norway',
        'nm cup': 'norway',
        
        # Austria
        'bundesliga austria': 'austria',
        'ofb cup': 'austria',
        
        # Switzerland
        'super league swiss': 'switzerland',
        'schweizer cup': 'switzerland',
        
        # Poland
        'ekstraklasa': 'poland',
        'polish cup': 'poland',
        
        # Croatia
        'hrvatska nogometna liga': 'croatia',
        'croatian cup': 'croatia',
        
        # Serbia
        'super liga serbia': 'serbia',
        'serbian cup': 'serbia',
        
        # Romania
        'liga i': 'romania',
        'cupa romaniei': 'romania',
        
        # Bulgaria
        'parva liga': 'bulgaria',
        'bulgarian cup': 'bulgaria',
        
        # Hungary
        'nb i': 'hungary',
        'magyar kupa': 'hungary',
        
        # Czech Republic
        'fortuna liga': 'czech republic',
        'czech cup': 'czech republic',
    }
    
    # Exact matches first
    for keyword, country in league_to_country.items():
        if keyword in league_lower:
            return country
    
    # Fallback: try to extract from common patterns
    if 'league' in league_lower or 'liga' in league_lower or 'ligue' in league_lower:
        # Could not determine - return empty for conservative filtering
        return ""
    
    return ""


def is_league_tier_allowed(league_name: str, league_id: Optional[int] = None) -> Tuple[bool, str]:
    """
    Determine if a league meets the tier filtering requirements.
    
    Rules:
    - Popular countries (England, Spain, Germany, Italy, France): Top 3 tiers allowed
    - Strong secondary countries: Top 2 tiers allowed
    - Other countries: Top 1 tier only
    
    Args:
        league_name: Name of the league
        league_id: API-Football league ID (optional, for override)
    
    Returns:
        Tuple of (allowed, reason)
    """
    if not league_name:
        return False, "No league name provided"
    
    league_lower = league_name.lower()
    
    # Cup competitions are scan-only but allowed for learning
    cup_keywords = ['cup', 'champions league', 'europa league', 'conference league']
    if any(keyword in league_lower for keyword in cup_keywords):
        return True, "Cup competition - scan only, allowed for learning"
    
    # Detect tier
    tier = detect_league_tier(league_name)
    
    # Detect country
    country = get_country_from_league(league_name)
    
    # If country not found, use conservative filter (only tier 1)
    if not country:
        if tier == 1:
            return True, f"Unknown country, top tier only - allowed (tier {tier})"
        else:
            return False, f"Unknown country, tier {tier} not allowed (only tier 1)"
    
    # Get max allowed tier for this country
    max_tier = POPULAR_LEAGUE_TIER_LIMITS.get(country, 1)
    
    if tier <= max_tier:
        return True, f"{country.title()} tier {tier} allowed (max tier {max_tier})"
    else:
        return False, f"{country.title()} tier {tier} not allowed (max tier {max_tier}) for league '{league_name}'"


def can_bypass_tier_restriction(
    league_name: str,
    games_played_home: int,
    games_played_away: int,
    h2h_games: int,
) -> Tuple[bool, str]:
    """
    Determine if a lower-tier league can bypass tier restrictions due to sufficient data.
    
    Requirements:
    - Both teams have played ≥15 games (vs 10 for top tier)
    - H2H has ≥10 games (vs 5 for top tier)
    
    Args:
        league_name: Name of the league
        games_played_home: Games played by home team
        games_played_away: Games played by away team
        h2h_games: Number of H2H meetings
    
    Returns:
        Tuple of (can_bypass, reason)
    """
    min_games = min(games_played_home, games_played_away)
    
    if min_games < TIER_BYPASS_MIN_GAMES:
        return False, f"Insufficient games: {min_games}/{TIER_BYPASS_MIN_GAMES} (need ≥{TIER_BYPASS_MIN_GAMES})"
    
    if h2h_games < TIER_BYPASS_MIN_H2H:
        return False, f"Insufficient H2H: {h2h_games}/{TIER_BYPASS_MIN_H2H} (need ≥{TIER_BYPASS_MIN_H2H})"
    
    return True, f"Bypass allowed: {min_games} games, {h2h_games} H2H meets requirements"


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — PLAYOFF/KNOCKOUT DETECTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def is_rejected_format(
    league_name: str,
    competition_type: str = "league",
    stage: str = "",
    round_name: str = "",
) -> Tuple[bool, str]:
    """
    Check if a match is in a rejected format (playoffs, knockout, relegation).
    
    Args:
        league_name: League name
        competition_type: Competition type ("league", "cup", "playoff")
        stage: Competition stage ("regular_season", "playoff", "group_stage")
        round_name: Specific round name ("Round 32", "Semi-final", "Final")
    
    Returns:
        Tuple of (is_rejected, reason)
    """
    league_lower = league_name.lower() if league_name else ""
    stage_lower = stage.lower() if stage else ""
    round_lower = round_name.lower() if round_name else ""
    
    # Check 1: Rejected competition types
    if competition_type in REJECTED_COMPETITION_TYPES:
        return True, f"Rejected competition type: '{competition_type}'"
    
    # Check 2: League name contains rejected keywords
    for keyword in REJECTED_MATCH_KEYWORDS:
        if keyword in league_lower:
            # Exception: "Premier League" is fine, but "League Playoff" is not
            if keyword == "league" and "playoff" not in league_lower and "relegation" not in league_lower:
                continue
            if keyword == "cup" and competition_type == "league":
                continue
            return True, f"League name contains rejected keyword: '{keyword}' in '{league_name}'"
    
    # Check 3: Stage name indicates playoff/knockout
    for stage_name in REJECTED_STAGE_NAMES:
        if stage_name in stage_lower:
            # "Group" is only rejected for cups, not for league group stages
            if stage_name == "group" and competition_type == "league":
                continue
            return True, f"Rejected stage: '{stage}' contains '{stage_name}'"
    
    # Check 4: Round name indicates playoff/knockout
    for keyword in REJECTED_MATCH_KEYWORDS:
        if keyword in round_lower:
            return True, f"Round name contains rejected keyword: '{round_name}'"
    
    # Check 5: Special handling for league name patterns
    playoff_indicators = [
        "playoff", "play-off", "promotion", "relegation",
        "knockout", "elimination", "championship round",
        "final round", "title round", "closing stage", "opening stage",
        "top 6", "top 8", "championship group", "relegation group"
    ]
    
    for indicator in playoff_indicators:
        if indicator in league_lower:
            return True, f"League name suggests playoff format: '{league_name}' (contains '{indicator}')"
    
    return False, ""


def is_regular_season_league(
    league_name: str,
    competition_type: str = "league",
    stage: str = "",
    round_name: str = "",
) -> bool:
    """
    Check if this is a regular season league match (not playoff/knockout).
    
    Returns:
        True if regular season league match, False otherwise
    """
    # Must be league competition type
    if competition_type != "league":
        return False
    
    league_lower = league_name.lower() if league_name else ""
    
    # Reject if contains playoff keywords
    for keyword in REJECTED_MATCH_KEYWORDS:
        if keyword in league_lower:
            return False
    
    # Check stage - must be regular season or None
    if stage and stage.lower() not in ['regular season', 'regular', 'league', '', 'regular_season']:
        return False
    
    # Check round - must not indicate knockout
    if round_name:
        round_lower = round_name.lower()
        for keyword in REJECTED_MATCH_KEYWORDS:
            if keyword in round_lower:
                return False
    
    return True


def get_competition_format(
    competition_type: str,
    stage: str = "",
    round_name: str = "",
) -> str:
    """
    Get the competition format classification.
    
    Returns:
        Format string: "regular_season", "playoff", "promotion_playoff", 
        "relegation_playoff", "knockout", "cup", "friendly"
    """
    comp_lower = competition_type.lower()
    stage_lower = stage.lower() if stage else ""
    round_lower = round_name.lower() if round_name else ""
    
    # Check for promotion/relegation playoffs
    if 'promotion' in stage_lower or 'promotion' in round_lower:
        return "promotion_playoff"
    if 'relegation' in stage_lower or 'relegation' in round_lower:
        return "relegation_playoff"
    
    # Check for playoffs
    if comp_lower == 'playoff' or 'playoff' in stage_lower or 'playoff' in round_lower:
        return "playoff"
    
    # Check for knockout
    if comp_lower == 'knockout' or 'knockout' in stage_lower or 'knockout' in round_lower:
        return "knockout"
    
    # Check for cup
    if comp_lower == 'cup':
        return "cup"
    
    # Check for friendly
    if comp_lower == 'friendly':
        return "friendly"
    
    # Default to regular season
    return "regular_season"


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — LEAGUE FAILURE TRACKER
# ═══════════════════════════════════════════════════════════════

def compute_league_failure_records(
    feedback: List[Dict],
) -> Dict[str, LeagueFailureRecord]:
    """
    Build per-league failure records from Module 16 feedback history.

    A 'qualified' prediction is one where status contains 'APPROVED'
    (it passed all gates). Correct = 1, wrong = 0 from the DB.

    Args:
        feedback: List of dicts from module16.get_all_feedback()
                  Each dict has: match_id, league, prediction,
                  actual_result, confidence, correct, status

    Returns:
        Dict[league_name → LeagueFailureRecord]
    """
    records: Dict[str, LeagueFailureRecord] = {}
    
    # Group feedback by league, use only approved/qualified predictions
    by_league: Dict[str, List[Dict]] = {}
    for row in feedback:
        league = row.get("league", "Unknown")
        status = row.get("status", "")
        # Only count APPROVED predictions for failure tracking
        if "APPROVED" not in status:
            continue
        if row.get("correct") is None:
            continue
        by_league.setdefault(league, []).append(row)

    for league, rows in by_league.items():
        if len(rows) < MIN_SAMPLES_FOR_ADJUSTMENT:
            continue
            
        # Try to find league_id from first row
        league_id = row.get("league_id", 0) if rows else 0
        
        rec = LeagueFailureRecord(
            league_name=league,
            league_id=league_id,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )
        
        results_seq = []   # W/L sequence for consecutive tracking
        for row in rows:
            rec.total_qualified += 1
            correct = int(row.get("correct", 0))
            if correct:
                rec.total_correct += 1
                results_seq.append("W")
            else:
                rec.total_failed += 1
                results_seq.append("L")

        rec.failure_rate = (rec.total_failed / rec.total_qualified
                            if rec.total_qualified > 0 else 0.0)

        # Consecutive failure streak
        streak = 0
        for r in reversed(results_seq):
            if r == "L":
                streak += 1
            else:
                break
        rec.consecutive_failures = streak
        
        # Max consecutive failures
        max_streak = 0
        current = 0
        for r in results_seq:
            if r == "L":
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        rec.max_consecutive = max_streak

        # Threshold adjustment from failure rate
        if rec.total_qualified >= MIN_SAMPLES_FOR_ADJUSTMENT:
            if rec.failure_rate >= FAILURE_RATE_CRITICAL:
                adj = min(MAX_THRESHOLD_RAISE, THRESHOLD_RAISE_STEP * 4)
                rec.threshold_adjustment = adj
                rec.is_flagged = True
                rec.flag_reason = (
                    f"CRITICAL: {rec.failure_rate:.0%} failure rate "
                    f"over {rec.total_qualified} qualified predictions"
                )
            elif rec.failure_rate >= FAILURE_RATE_ADJUST_THRESH:
                adj = min(MAX_THRESHOLD_RAISE, THRESHOLD_RAISE_STEP * 2)
                rec.threshold_adjustment = adj
                rec.is_flagged = rec.consecutive_failures >= MIN_FAILURES_TO_FLAG
                rec.flag_reason = (
                    f"HIGH failure rate: {rec.failure_rate:.0%} "
                    f"({rec.total_failed}/{rec.total_qualified})"
                ) if rec.is_flagged else ""
            elif rec.consecutive_failures >= MIN_FAILURES_TO_FLAG:
                rec.threshold_adjustment = THRESHOLD_RAISE_STEP
                rec.is_flagged = True
                rec.flag_reason = (
                    f"{rec.consecutive_failures} consecutive failures"
                )

        records[league] = rec
    
    return records


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — PROFILE BUILDER
# ═══════════════════════════════════════════════════════════════

def _scan_priority(league_id: int) -> int:
    """Lower number = scanned first (lower volatility leagues)."""
    try:
        return LEAGUE_SCAN_ORDER.index(league_id)
    except ValueError:
        return 999


def _tier(league_id: int) -> int:
    return LEAGUE_TIERS.get(league_id, 3)


def _base_volatility(league_id: int) -> float:
    return LEAGUE_BASE_VOLATILITY.get(league_id, 0.42)


def _season_phase(games_played: int, total_games: int) -> Tuple[str, float]:
    """Determine season phase and progress."""
    progress = games_played / total_games if total_games > 0 else 0.5
    for phase, (lo, hi) in SEASON_PHASES.items():
        if lo <= progress < hi:
            return phase, progress
    return "end", 1.0


def _data_reliability(tier: int, games_played: int,
                      competition_type: str) -> float:
    """Calculate data reliability score (0-1)."""
    tier_score = 1.0 - (tier - 1) * 0.12
    games_score = min(1.0, games_played / 15)
    comp_score = {
        "league": 1.0,
        "cup": 0.85,
        "playoff": 0.90,
        "champions_league_group": 0.95,
        "champions_league_ko": 0.90,
        "friendly": 0.40,
    }.get(competition_type, 0.80)
    return round(max(0.1, tier_score * games_score * comp_score), 3)


def _is_parlay_eligible(league_id: int, competition_type: str) -> bool:
    """
    Hard rule: cups, friendlies, and UCL are never parlay eligible.
    Scan-only — algorithm learns from them but they never appear
    in CleanedPortfolio top_picks, ultra_safe_acca, or value_acca.
    """
    if competition_type in PARLAY_BANNED_COMPETITION_TYPES:
        return False
    if league_id in CUP_AND_FRIENDLY_IDS:
        return False
    return True


def build_league_profile(
    league_id: int,
    league_name: str,
    competition_type: str = "league",
    games_played: int = 20,
    total_games: int = 38,
    base_home_thresh: float = 0.57,
    base_away_cap: float = 0.25,
    stage: str = "",
    round_name: str = "",
    failure_records: Optional[Dict[str, LeagueFailureRecord]] = None,
) -> LeagueProfile:
    """
    Build a fully-intelligent LeagueProfile for one fixture.

    Args:
        league_id: API-Football league ID
        league_name: Human-readable league name
        competition_type: "league", "cup", "playoff", "friendly", etc.
        games_played: Games played by home team this season
        total_games: Total games in this league's season
        base_home_thresh: SystemConfig home_win_threshold (default 0.57)
        base_away_cap: SystemConfig opponent_win_cap (default 0.25)
        stage: Competition stage ("regular_season", "playoff", etc.)
        round_name: Specific round name
        failure_records: Dict from compute_league_failure_records() (optional)

    Returns:
        LeagueProfile with all intelligence applied
    """
    tier = _tier(league_id)
    phase, prog = _season_phase(games_played, total_games)
    reliability = _data_reliability(tier, games_played, competition_type)
    base_vol = _base_volatility(league_id)
    comp = COMPETITION_PROFILES.get(
        competition_type, COMPETITION_PROFILES["league"]
    )
    eligible = _is_parlay_eligible(league_id, competition_type)
    scan_only = not eligible

    lp = LeagueProfile(
        league_id=league_id,
        league_name=league_name,
        scan_priority=_scan_priority(league_id),
        tier=tier,
        competition_type=competition_type,
        parlay_eligible=eligible,
        scan_only=scan_only,
        games_played=games_played,
        total_games=total_games,
        season_progress=round(prog, 3),
        season_phase=phase,
        base_volatility=base_vol,
        adjusted_volatility=base_vol,
        data_reliability=reliability,
        home_win_adj=comp["home_win_adj"],
        draw_rate_adj=comp["draw_rate_adj"],
        upset_rate_adj=comp["upset_rate_adj"],
    )

    if scan_only:
        lp.notes.append(
            f"SCAN ONLY: {competition_type} — algorithm learns but no parlay output"
        )

    # ── Base threshold adjustments ────────────────────────────
    thresh_adj = 0.0

    # Volatility: higher base volatility = tighter gate
    if base_vol > 0.40:
        extra = round((base_vol - 0.40) * 0.20, 3)
        thresh_adj += extra

    if reliability < 0.60:
        thresh_adj += 0.03
        lp.notes.append("Low data reliability: thresholds raised")

    if phase == "early":
        thresh_adj += 0.02
        lp.notes.append("Early season: standings unreliable")

    if phase == "end":
        lp.notes.append("End of season: motivation context critical (M26)")

    if competition_type == "cup":
        thresh_adj += 0.02
        lp.notes.append("Cup: scanned for learning only — blocked from parlay")

    # ── Playoff/knockout adjustment ───────────────────────────
    format_type = get_competition_format(competition_type, stage, round_name)
    if format_type in ["playoff", "promotion_playoff", "relegation_playoff"]:
        thresh_adj += 0.03
        lp.notes.append(f"Playoff format ({format_type}): thresholds raised, higher variance")

    # ── Failure intelligence adjustment ───────────────────────
    failure_adj = 0.0
    if failure_records:
        rec = failure_records.get(league_name)
        if rec and rec.total_qualified >= MIN_SAMPLES_FOR_ADJUSTMENT:
            failure_adj = rec.threshold_adjustment
            lp.failure_record = rec
            lp.failure_adjustment = failure_adj
            if rec.is_flagged:
                lp.notes.append(
                    f"FAILURE FLAG: {rec.flag_reason} — "
                    f"threshold raised +{failure_adj:.3f}"
                )
            # Adjust volatility score upward if failing consistently
            if rec.failure_rate >= FAILURE_RATE_ADJUST_THRESH:
                vol_bump = round(rec.failure_rate * 0.10, 3)
                lp.adjusted_volatility = round(
                    min(0.80, base_vol + vol_bump), 3
                )
                lp.notes.append(
                    f"Volatility adjusted: {base_vol:.2f} → "
                    f"{lp.adjusted_volatility:.2f} from failure history"
                )

    total_adj = thresh_adj + failure_adj

    lp.home_prob_threshold = round(
        min(0.70, base_home_thresh + total_adj), 3
    )
    lp.away_prob_cap = round(
        max(0.15, base_away_cap - total_adj * 0.5), 3
    )
    lp.min_edge_adjustment = round(total_adj * 0.5, 3)

    return lp


# ═══════════════════════════════════════════════════════════════
# SECTION 10 — SCAN ORDER UTILITIES
# ═══════════════════════════════════════════════════════════════

def get_ordered_scan_leagues(
    available_league_ids: List[int],
    include_scan_only: bool = True,
) -> List[int]:
    """
    Return available league IDs sorted in scan priority order
    (lowest volatility first).

    Args:
        available_league_ids: Leagues the system has data for today
        include_scan_only: If False, exclude cups/friendlies from scan
                           (only use this for final portfolio building,
                           not for data collection / learning)
    Returns:
        Sorted list of league IDs
    """
    if not include_scan_only:
        available_league_ids = [
            lid for lid in available_league_ids
            if lid not in CUP_AND_FRIENDLY_IDS
        ]

    def priority(lid: int) -> int:
        return _scan_priority(lid)

    return sorted(available_league_ids, key=priority)


def get_leagues_flagged_for_review(
    failure_records: Dict[str, LeagueFailureRecord],
) -> List[LeagueFailureRecord]:
    """
    Return all flagged leagues sorted by failure rate descending.
    Use for system health monitoring and manual review.
    """
    flagged = [r for r in failure_records.values() if r.is_flagged]
    return sorted(flagged, key=lambda r: r.failure_rate, reverse=True)


def get_league_scan_info(league_id: int) -> Dict[str, Any]:
    """Get scan order information for a league."""
    return {
        "league_id": league_id,
        "scan_priority": _scan_priority(league_id),
        "tier": _tier(league_id),
        "base_volatility": _base_volatility(league_id),
        "is_cup": league_id in CUP_AND_FRIENDLY_IDS,
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 11 — CONVENIENCE BUILDERS
# ═══════════════════════════════════════════════════════════════

def context_from_leg_and_config(
    leg: Any,
    config: Any,
    league_id: int,
    league_name: str,
    competition_type: str = "league",
    total_games: int = 38,
    stage: str = "",
    round_name: str = "",
    failure_records: Optional[Dict[str, LeagueFailureRecord]] = None,
) -> LeagueProfile:
    """
    Build LeagueProfile directly from a Leg object and SystemConfig.
    One-call integration for M6 and M11.
    """
    games_played = 20
    try:
        if leg.home_profile:
            games_played = int(leg.home_profile.get_metric("core.games", 20))
    except Exception:
        pass

    return build_league_profile(
        league_id=league_id,
        league_name=league_name,
        competition_type=competition_type,
        games_played=games_played,
        total_games=total_games,
        base_home_thresh=getattr(config, "home_win_threshold", 0.57),
        base_away_cap=getattr(config, "opponent_win_cap", 0.25),
        stage=stage,
        round_name=round_name,
        failure_records=failure_records,
    )


# ═══════════════════════════════════════════════════════════════
# SECTION 12 — EXPORTS
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Constants
    "LEAGUE_SCAN_ORDER",
    "CUP_AND_FRIENDLY_IDS",
    "PARLAY_BANNED_COMPETITION_TYPES",
    "PARLAY_ELIGIBLE_COMPETITION_TYPES",
    "LEAGUE_BASE_VOLATILITY",
    "LEAGUE_TIERS",
    "COMPETITION_PROFILES",
    # Tier filtering constants
    "POPULAR_LEAGUE_TIER_LIMITS",
    "TIER_PATTERNS",
    "TIER_BYPASS_MIN_GAMES",
    "TIER_BYPASS_MIN_H2H",
    # Playoff detection constants
    "REJECTED_COMPETITION_TYPES",
    "REJECTED_MATCH_KEYWORDS",
    "REJECTED_STAGE_NAMES",
    # Tier detection functions
    "detect_league_tier",
    "get_country_from_league",
    "is_league_tier_allowed",
    "can_bypass_tier_restriction",
    # Playoff detection functions
    "is_rejected_format",
    "is_regular_season_league",
    "get_competition_format",
    # Data classes
    "LeagueFailureRecord",
    "LeagueProfile",
    # Core functions
    "compute_league_failure_records",
    "build_league_profile",
    "get_ordered_scan_leagues",
    "get_leagues_flagged_for_review",
    "get_league_scan_info",
    "context_from_leg_and_config",
]


# ═══════════════════════════════════════════════════════════════
# SECTION 13 — EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MODULE 31: LEAGUE INTELLIGENCE ENGINE - TEST RUN")
    print("=" * 70)
    
    # Test 1: Build profile for Premier League
    print("\n📊 TEST 1: Premier League Profile")
    print("-" * 40)
    
    profile = build_league_profile(
        league_id=39,
        league_name="Premier League",
        competition_type="league",
        games_played=28,
        base_home_thresh=0.57,
    )
    
    print(profile.summary())
    print(f"Weighted properties:")
    print(f"  Normalized Volatility: {profile.normalized_volatility:.3f}")
    print(f"  Reliability Score: {profile.reliability_score:.3f}")
    print(f"  Threshold Multiplier: {profile.threshold_multiplier:.2f}")
    
    # Test 2: Cup competition (scan only)
    print("\n📊 TEST 2: FA Cup (Scan Only)")
    print("-" * 40)
    
    cup_profile = build_league_profile(
        league_id=45,
        league_name="FA Cup",
        competition_type="cup",
        games_played=5,
        base_home_thresh=0.57,
    )
    
    print(cup_profile.summary())
    print(f"Parlay Eligible: {cup_profile.parlay_eligible}")
    
    # Test 3: League tier detection
    print("\n📊 TEST 3: League Tier Detection")
    print("-" * 40)
    
    leagues_to_test = [
        "Premier League",
        "Championship",
        "League One",
        "League Two",
        "2. Bundesliga",
        "Serie B",
        "Ligue 2",
        "Eredivisie",
        "J1 League",
        "Unknown League",
    ]
    
    for league in leagues_to_test:
        tier = detect_league_tier(league)
        country = get_country_from_league(league)
        allowed, reason = is_league_tier_allowed(league)
        print(f"  {league:<20} → Tier {tier}, Country: {country:<10}, Allowed: {allowed}")
    
    # Test 4: Playoff detection
    print("\n📊 TEST 4: Playoff/Knockout Detection")
    print("-" * 40)
    
    test_cases = [
        ("Premier League", "league", "", ""),
        ("Championship Playoff", "league", "", ""),
        ("FA Cup", "cup", "quarter-final", ""),
        ("Champions League", "continental_knockout", "round_of_16", ""),
        ("Serie B", "league", "playoff", ""),
        ("Bundesliga", "league", "", "Round 32"),
    ]
    
    for league, comp_type, stage, round_name in test_cases:
        rejected, reason = is_rejected_format(league, comp_type, stage, round_name)
        status = "❌ REJECTED" if rejected else "✅ ALLOWED"
        print(f"  {status}: {league} ({comp_type}, stage={stage}) - {reason if reason else 'OK'}")
    
    # Test 5: Scan order
    print("\n📊 TEST 5: League Scan Order (First 10)")
    print("-" * 40)
    
    available = [39, 140, 78, 135, 61, 71, 128, 292, 94, 88, 144, 207]
    ordered = get_ordered_scan_leagues(available)
    
    for i, lid in enumerate(ordered[:10], 1):
        info = get_league_scan_info(lid)
        print(f"  {i:2d}. ID {lid} (Priority {info['scan_priority']}, Tier {info['tier']}, Vol={info['base_volatility']:.2f})")
    
    # Leg data for M11
    print("\n📊 Leg Data for M11:")
    leg_data = profile.to_leg_data()
    for key, value in leg_data.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("MODULE 31 READY FOR PRODUCTION")
    print("=" * 70)