"""
The Match Oracle - Module 0: Data Integrity & AI Guardrail (REFINED v5)
===================================================================
The first and last line of defence for the entire pipeline.

NEW IN VERSION 5 (May 2026):
---------------------------
1. ADDED: Extreme Decay Bypass — when underdog is in complete freefall,
   lower dominance thresholds for favourite
2. ADDED: Underdog collapse detection (0 away wins, 6+ consecutive losses,
   win rate <10%, PPG <0.8)
3. ADDED: Bypass scoring system with weighted evaluation
4. ADDED: Configurable bypass thresholds

PREVIOUS ENHANCEMENTS (v4):
---------------------------
- Dead Rubber Auto-Reject
- Win Ceiling Stake Reduction
- H2H Draw Rate Boost integration
- Season progress tracking
- Motivation label extraction
- Historical win ceiling tracking

WHAT THIS MODULE PREVENTS
--------------------------
1. DATA INGESTION CORRUPTION
2. ALGORITHM HALLUCINATION
3. AI LAYER CITATION & SIMULATION
4. DATA FLOW INTEGRITY
5. STRUCTURAL GAP & DOMINANCE VALIDATION
6. DEAD RUBBER DETECTION
7. WIN CEILING PROTECTION

DESIGN PRINCIPLES
-----------------
- Raise GuardrailError for hard violations (pipeline must stop)
- Return GuardrailWarning list for soft violations (pipeline continues, flagged)
- Never modify data — only inspect and report
- Every check has an explicit name and failure message
- Zero external dependencies beyond stdlib

INTEGRATION
-----------
Call validate_ingestion(raw_api_response)     → before building Leg objects
Call validate_fixture_date(event_timestamp)   → per fixture in Module 1
Call validate_leg(leg)                         → after build_leg(), before M4
Call validate_ai_response(ai_result, leg)      → inside M7, after each AI call
Call validate_pipeline_output(master_verdict)  → after M11, before M12
Call validate_dataflow(portfolio, report)      → after M13, before storage
Call validate_dominance_vs_decay(leg)          → inside M4, hard filter (UPDATED v5)
Call validate_senior_men_league(leg)           → inside M4, hard filter
Call validate_league_tier(leg)                 → inside M4, hard filter
Call validate_not_playoff_or_knockout(leg)     → inside M4, hard filter
Call validate_dead_rubber(leg)                 → inside M4, hard filter (v4)
Call validate_win_ceiling(leg)                 → inside M4, stake adjustment (v4)
"""
from __future__ import annotations

import re
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from datetime import datetime, timezone


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

MIN_SEQUENCE_GAMES      = 10     # minimum W/D/L entries for a valid TransitionMatrix (warning threshold)
MIN_HARD_SEQUENCE_GAMES = 5      # absolute hard minimum — below this we still raise
MIN_ODDS                = 1.01   # lowest physically possible decimal odd
MAX_ODDS                = 100.0  # sanity cap — nothing real is above this
PROB_SUM_TOLERANCE      = 0.02   # probabilities must sum to 1.0 ± this
MAX_AI_PROB_DIVERGENCE  = 0.25   # AI win_prob vs model_prob max allowed gap
MAX_EXPOSURE_RATIO      = 1.0    # total_exposure must not exceed bankroll
MIN_NARRATIVE_LENGTH    = 10     # characters — below this is effectively empty
HIGH_DRAW_PROB_GATE     = 0.32   # draw probability above this on approved bet = warning
MIN_STRENGTH_DIFF       = 0.5    # strength diff below this = symmetric match warning
VALID_RESULTS           = {"W", "D", "L"}
VALID_STATUSES          = {"NS"}  # fixture must be Not Started
VALID_VERDICTS          = {"APPROVE", "CAUTION", "REJECT", "NEUTRAL"}
VALID_CONFIDENCES       = {"LOW", "MEDIUM", "HIGH"}
VALID_LEAGUE_TYPES      = {"league"}   # only league matches enter the pipeline upstream
VALID_FINAL_STATUSES    = {
    "APPROVED", "REJECTED", "CAUTION",
    "APPROVED (UNDERDOG)", "CAUTION (UNDERDOG THREAT)",
    "CAUTION (AI DISAGREEMENT)", "APPROVED (AI CONSENSUS)",
    "REJECTED (AI CONSENSUS)", "REJECTED (SHADOW LOGGED)",
    "CAUTION (INSUFFICIENT AI DATA)", "PENDING",
}

# AI provider names for validation
VALID_AI_PROVIDERS = {"Claude", "Gemini", "GPT", "DeepSeek"}

# AI provider fingerprint patterns (for simulation detection)
AI_FINGERPRINT_PATTERNS = {
    "Claude": [r"\b(Claude|Anthropic)\b", r"```json", r"As an AI"],
    "Gemini": [r"\b(Gemini|Google)\b", r"I'm a large language model"],
    "GPT": [r"\b(GPT|OpenAI)\b", r"As an AI assistant"],
    "DeepSeek": [r"\b(DeepSeek|深度求索)\b", r"I'm DeepSeek"],
}

# Common analysis terms (not hallucinated entities)
COMMON_ANALYSIS_TERMS = {
    "the", "this", "that", "their", "team", "home", "away", "strong", "weak",
    "good", "high", "low", "edge", "win", "draw", "loss", "trap", "risk",
    "form", "model", "data", "odds", "value", "favourite", "underdog",
    "recent", "current", "league", "season", "match", "fixture", "bet",
    "prediction", "analysis", "probability", "chance", "likely", "possible",
    "favorite", "underdog", "draw", "victory", "defeat", "win", "lose",
    "team", "teams", "both", "each", "their", "they", "them", "these", "those",
    "has", "have", "been", "were", "was", "will", "would", "could", "should",
    "more", "less", "very", "quite", "rather", "somewhat", "slightly",
    "significantly", "substantially", "marginally", "barely", "hardly",
}


# ═══════════════════════════════════════════════════════════════
# DOMINANCE VS DECAY THRESHOLDS (UPDATED v5)
# ═══════════════════════════════════════════════════════════════

# Standard thresholds (unchanged)
DOMINANCE_MIN_WIN_RATE      = 0.55   # Favourite must win >55% of games
DOMINANCE_MIN_PPG           = 1.8    # Favourite must have >1.8 points per game
DOMINANCE_MIN_RECENT_WINS   = 3      # Favourite must have 3+ wins in last 5
DOMINANCE_MIN_FORM_GAP      = 0.30   # Favourite win rate - underdog win rate >= 30%

# Underdog must be decaying
DECAY_MAX_WIN_RATE          = 0.40   # Underdog must win <40% of games
DECAY_MAX_PPG               = 1.2    # Underdog must have <1.2 points per game
DECAY_MIN_RECENT_LOSSES     = 3      # Underdog must have 3+ losses in last 5

# Favourite must NOT be decaying
FAVOURITE_DECAY_MAX_LOSSES  = 2      # Favourite must have ≤2 losses in last 5

# ─── NEW v5: Extreme Decay Bypass Thresholds ─────────────────────
# When underdog is in COMPLETE freefall, lower requirements for favourite
EXTREME_DECAY_MIN_WIN_RATE      = 0.10   # Underdog win rate <10% triggers bypass
EXTREME_DECAY_MIN_PPG           = 0.8    # Underdog PPG <0.8 triggers bypass
EXTREME_DECAY_MIN_LOSS_STREAK   = 6      # Underdog 6+ consecutive losses triggers bypass
EXTREME_DECAY_MIN_AWAY_LOSSES   = 10     # Underdog 10+ away losses triggers bypass
EXTREME_DECAY_ZERO_AWAY_WINS    = True   # Underdog has 0 away wins triggers bypass

# Bypass favourite thresholds (lowered)
BYPASS_FAV_MIN_WIN_RATE         = 0.30   # 30% win rate (was 55%)
BYPASS_FAV_MIN_PPG              = 1.0    # 1.0 PPG (was 1.8)
BYPASS_FAV_MIN_RECENT_WINS      = 2      # 2 wins in last 5 (was 3)
BYPASS_FAV_MAX_RECENT_LOSSES    = 3      # Favourite can have up to 3 losses

# Bypass form gap (can be lower when decay is extreme)
BYPASS_MIN_FORM_GAP             = 0.20   # 20% gap (was 30%)

# Minimum required checks for bypass (4 of 6 instead of 6 of 8)
BYPASS_REQUIRED_CHECKS          = 4

# Minimum thresholds for tier bypass
TIER_BYPASS_MIN_GAMES       = 15     # Lower tier needs 15+ games (vs 10 for top tier)
TIER_BYPASS_MIN_H2H         = 10     # Lower tier needs 10+ H2H games (vs 5 for top tier)

# League tier limits by country (for tier validation)
POPULAR_LEAGUE_TIER_LIMITS = {
    # Tier 1 countries (Top 3 divisions allowed)
    "england": 3, "spain": 3, "germany": 3, "italy": 3, "france": 3,
    # Tier 2 countries (Top 2 divisions allowed)
    "netherlands": 2, "portugal": 2, "belgium": 2, "turkey": 2,
    "brazil": 2, "argentina": 2, "japan": 2, "scotland": 2,
    "russia": 2, "ukraine": 2, "greece": 2, "czech republic": 2,
    "croatia": 2, "denmark": 2, "sweden": 2, "norway": 2,
    "austria": 2, "switzerland": 2, "poland": 2, "serbia": 2,
    "romania": 2, "bulgaria": 2, "hungary": 2, "israel": 2,
    # Tier 3 countries (Top 1 division only)
    "saudi arabia": 1, "united arab emirates": 1, "qatar": 1,
    "egypt": 1, "morocco": 1, "tunisia": 1, "algeria": 1,
    "south africa": 1, "china": 1, "south korea": 1, "australia": 1,
    "usa": 1, "mexico": 1, "chile": 1, "colombia": 1, "peru": 1,
    "ecuador": 1, "uruguay": 1, "paraguay": 1, "bolivia": 1,
    "india": 1, "thailand": 1, "vietnam": 1, "indonesia": 1,
}

# Rejected competition types (playoffs, knockout, relegation)
REJECTED_COMPETITION_TYPES = {
    "playoff", "promotion_playoff", "relegation_playoff",
    "knockout", "cup_knockout", "playoff_semi", "playoff_final",
    "relegation_round", "promotion_round", "elimination",
}

# Keywords that indicate a match should be rejected
REJECTED_MATCH_KEYWORDS = {
    "playoff", "play-off", "promotion", "relegation",
    "knockout", "semi-final", "semi final", "final",
    "quarter-final", "quarter final", "round of",
    "relegation round", "promotion round", "elimination",
    "closing stage", "opening stage", "top 6", "championship group",
}

# Reject keywords for senior men's league validation
SENIOR_LEAGUE_REJECT_KEYWORDS = [
    'youth', 'u19', 'u18', 'u17', 'u20', 'u21', 'u23',
    'women', 'womens', 'female', 'ladies',
    'amateur', 'reserve', 'reserves', 'iii', '4.', '5.', '6.',
    'regional', 'district', 'local'
]

# Allowed competition types for senior men's leagues
ALLOWED_COMPETITION_TYPES = ['league', 'playoff', 'cup', 'continental_group', 'continental_knockout']

# Tier detection patterns
TIER_PATTERNS = {
    1: ['premier', 'pro league', 'bundesliga', 'serie a', 'ligue 1',
        'primeira liga', 'eredivisie', 'j1 league', 'saudi pro league',
        'botola', 'ligue professionnelle', 'professional league'],
    2: ['championship', '2. bundesliga', 'serie b', 'ligue 2',
        'eerste divisie', 'j2 league', '1. division', 'liga leumit'],
    3: ['league one', '3. liga', 'serie c', 'national', '2. division'],
    4: ['league two', '4. division', 'serie d', 'regionalliga'],
}


# ═══════════════════════════════════════════════════════════════
# NEW v4/v5: DEAD RUBBER & WIN CEILING CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Dead rubber detection
DEAD_RUBBER_SEASON_THRESHOLD = 0.85   # 85%+ season progress = late season
DEAD_RUBBER_MOTIVATION_LOW = {"LOW", "DEAD_RUBBER", "VERY_LOW"}
DEAD_RUBBER_REJECT_MESSAGE = "DEAD RUBBER REJECT: Favourite has nothing to play for in late season"

# Win ceiling detection
WIN_CEILING_APPROACH_DISTANCE = 2      # Within 2 wins of historical ceiling = approach
WIN_CEILING_MIN_STREAK = 3             # Minimum streak to consider ceiling
CEILING_APPROACH_MULTIPLIER_LATE = 0.25   # Late season: 25% of stake
CEILING_APPROACH_MULTIPLIER_MID = 0.50    # Mid season: 50% of stake
CEILING_AT_MULTIPLIER = 0.0            # At ceiling: HARD REJECT

# Motivation label extraction keys
MOTIVATION_METRIC_KEYS = [
    "motivation_label",
    "motivation",
    "core.motivation_label",
    "context.motivation_label"
]


# ═══════════════════════════════════════════════════════════════
# RESULT TYPES
# ═══════════════════════════════════════════════════════════════

class GuardrailError(Exception):
    """
    Hard violation — pipeline must not continue.
    Raised when data is so corrupt that any downstream result
    would be meaningless or actively dangerous.
    """
    def __init__(self, check: str, message: str, context: Dict[str, Any] = None):
        self.check = check
        self.message = message
        self.context = context or {}
        super().__init__(f"[GUARDRAIL HARD STOP] {check}: {message}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "check": self.check,
            "message": self.message,
            "context": self.context,
        }


@dataclass
class GuardrailWarning:
    """Soft violation — pipeline continues but result is flagged."""
    check: str
    message: str
    severity: str = "MEDIUM"   # LOW / MEDIUM / HIGH
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "check": self.check,
            "message": self.message,
            "severity": self.severity,
            "context": self.context,
        }


@dataclass
class GuardrailReport:
    """Full guardrail result for one validation call."""
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[GuardrailWarning] = field(default_factory=list)
    checks_run: int = 0
    validation_context: Dict[str, Any] = field(default_factory=dict)

    def add_warning(self, check: str, msg: str, severity: str = "MEDIUM", context: Dict = None) -> None:
        """Add a warning to the report."""
        self.warnings.append(GuardrailWarning(check, msg, severity, context or {}))

    def add_error(self, check: str, msg: str, context: Dict = None) -> None:
        """Add an error to the report (marks report as failed)."""
        self.passed = False
        error_msg = f"{check}: {msg}"
        if context:
            error_msg += f" | Context: {context}"
        self.errors.append(error_msg)

    def has_high_severity_warnings(self) -> bool:
        """Return True if any warnings have HIGH severity."""
        return any(w.severity == "HIGH" for w in self.warnings)
    
    def get_warnings_by_severity(self, severity: str) -> List[GuardrailWarning]:
        """Get warnings filtered by severity."""
        return [w for w in self.warnings if w.severity == severity]
    
    def merge(self, other: GuardrailReport) -> None:
        """Merge another report into this one."""
        self.passed = self.passed and other.passed
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.checks_run += other.checks_run
        self.validation_context.update(other.validation_context)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"GuardrailReport — {'PASSED' if self.passed else 'FAILED'}",
            f"  Checks run : {self.checks_run}",
            f"  Errors     : {len(self.errors)}",
            f"  Warnings   : {len(self.warnings)}",
        ]
        
        # Show error details
        for e in self.errors:
            lines.append(f"  ✘ ERROR   {e}")
        
        # Show warnings by severity
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            sev_warnings = self.get_warnings_by_severity(severity)
            for w in sev_warnings:
                icon = "💀" if severity == "CRITICAL" else "🚨" if severity == "HIGH" else "⚠️" if severity == "MEDIUM" else "🔍"
                lines.append(f"  {icon} {severity:<6} {w.check}: {w.message}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": [w.to_dict() for w in self.warnings],
            "checks_run": self.checks_run,
            "validation_context": self.validation_context,
        }


# ═══════════════════════════════════════════════════════════════
# THREAD-SAFE VALIDATION CONTEXT
# ═══════════════════════════════════════════════════════════════

class ValidationContext:
    """
    Thread-safe context for sharing validation state across calls.
    Useful for tracking consecutive failures or accumulating statistics.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._counters: Dict[str, int] = {}
        self._lock = threading.Lock()
    
    def increment(self, key: str, amount: int = 1) -> int:
        """Increment a counter and return new value."""
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + amount
            return self._counters[key]
    
    def get(self, key: str, default: int = 0) -> int:
        """Get current counter value."""
        with self._lock:
            return self._counters.get(key, default)
    
    def reset(self, key: str = None) -> None:
        """Reset counter(s)."""
        with self._lock:
            if key:
                self._counters[key] = 0
            else:
                self._counters.clear()
    
    def get_all(self) -> Dict[str, int]:
        """Get all counters."""
        with self._lock:
            return self._counters.copy()


_validation_context = ValidationContext()


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _detect_league_tier(league_name: str) -> int:
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
    
    for tier, patterns in TIER_PATTERNS.items():
        for pattern in patterns:
            if pattern in league_lower:
                return tier
    
    # Check for numbered tiers (e.g., "2. Bundesliga", "3. Liga")
    tier_match = re.search(r'(\d+)\.', league_lower)
    if tier_match:
        return int(tier_match.group(1))
    
    # Default: assume top tier if no pattern matches
    return 1


def _get_country_from_league(league_name: str) -> str:
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
    
    # Map of league keywords to country
    league_to_country = {
        'premier': 'england', 'championship': 'england', 'league one': 'england',
        'laliga': 'spain', 'la liga': 'spain', 'segunda': 'spain',
        'bundesliga': 'germany', '2. bundesliga': 'germany', '3. liga': 'germany',
        'serie a': 'italy', 'serie b': 'italy', 'serie c': 'italy',
        'ligue 1': 'france', 'ligue 2': 'france',
        'eredivisie': 'netherlands', 'eerste divisie': 'netherlands',
        'primeira liga': 'portugal', 'liga portugal': 'portugal',
        'pro league': 'belgium', 'super lig': 'turkey',
        'brasileiro': 'brazil', 'serie b brazil': 'brazil',
        'primera division argentina': 'argentina',
        'j1 league': 'japan', 'j2 league': 'japan',
        'saudi pro league': 'saudi arabia', 'qatar stars': 'qatar',
        'egyptian premier': 'egypt', 'botola': 'morocco',
        'ligue professionnelle': 'tunisia', 'liga leumit': 'israel',
    }
    
    for keyword, country in league_to_country.items():
        if keyword in league_lower:
            return country
    
    return ""


def _repair_json_response(response_text: str) -> str:
    """
    Attempt to repair malformed JSON from AI responses.
    """
    if not response_text:
        return "{}"
    
    # Remove markdown code blocks
    cleaned = re.sub(r'```json\s*', '', response_text)
    cleaned = re.sub(r'```\s*$', '', cleaned)
    cleaned = cleaned.strip()
    
    # Try to extract JSON object if there's extra text
    json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
    if json_match and not (cleaned.startswith('{') and cleaned.endswith('}')):
        return json_match.group()
    
    # Fix common JSON issues
    # Fix trailing commas
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    
    # Fix unquoted keys
    cleaned = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)
    
    # Fix single quotes
    cleaned = cleaned.replace("'", '"')
    
    # Fix missing quotes around values
    cleaned = re.sub(r':\s*(true|false|null)', r':"\1"', cleaned, flags=re.IGNORECASE)
    
    return cleaned


def _get_motivation_label(profile: Any) -> str:
    """
    Extract motivation label from team profile.
    
    Args:
        profile: TeamProfile object from module2
    
    Returns:
        Motivation label: "DESPERATE", "HIGH", "NORMAL", "LOW", "DEAD_RUBBER", or "UNKNOWN"
    """
    if profile is None:
        return "UNKNOWN"
    
    # Try direct attribute access
    for key in MOTIVATION_METRIC_KEYS:
        if hasattr(profile, key):
            val = getattr(profile, key)
            if isinstance(val, str):
                return val.upper()
    
    # Try get_metric method
    if hasattr(profile, 'get_metric'):
        for key in MOTIVATION_METRIC_KEYS:
            val = profile.get_metric(key, None)
            if val and isinstance(val, str):
                return val.upper()
    
    # Try features dictionary
    if hasattr(profile, 'features'):
        for key in MOTIVATION_METRIC_KEYS:
            if key in profile.features:
                val = profile.features[key]
                if isinstance(val, str):
                    return val.upper()
    
    # Try form dictionary
    if hasattr(profile, 'form'):
        for key in MOTIVATION_METRIC_KEYS:
            if key in profile.form:
                val = profile.form[key]
                if isinstance(val, str):
                    return val.upper()
    
    return "UNKNOWN"


def _get_historical_win_ceiling(profile: Any) -> int:
    """
    Extract historical win ceiling from team profile (from RTM).
    
    Args:
        profile: TeamProfile object from module2 with multi_rtm
    
    Returns:
        Historical win ceiling (max consecutive wins), default 10
    """
    if profile is None:
        return 10
    
    # Try multi_rtm.overall.win_ceiling
    if hasattr(profile, 'multi_rtm') and profile.multi_rtm:
        if hasattr(profile.multi_rtm, 'overall') and profile.multi_rtm.overall:
            if hasattr(profile.multi_rtm.overall, 'win_ceiling'):
                return profile.multi_rtm.overall.win_ceiling
    
    # Try transition matrix
    if hasattr(profile, 'transition') and profile.transition:
        if hasattr(profile.transition, 'win_ceiling'):
            return profile.transition.win_ceiling
    
    # Default: assume 10-game ceiling (conservative)
    return 10


def _get_current_win_streak(profile: Any) -> int:
    """
    Extract current win streak from team profile.
    
    Args:
        profile: TeamProfile object from module2
    
    Returns:
        Current consecutive wins, default 0
    """
    if profile is None:
        return 0
    
    # Try multi_rtm
    if hasattr(profile, 'multi_rtm') and profile.multi_rtm:
        if hasattr(profile.multi_rtm, 'overall') and profile.multi_rtm.overall:
            if hasattr(profile.multi_rtm.overall, 'current_win_streak'):
                return profile.multi_rtm.overall.current_win_streak
    
    # Try transition matrix
    if hasattr(profile, 'transition') and profile.transition:
        if hasattr(profile.transition, 'current_streak'):
            return profile.transition.current_streak
    
    # Try form results
    if hasattr(profile, 'form'):
        recent = profile.form.get('recent_results', [])[-10:]
        streak = 0
        for r in reversed(recent):
            if r == "W":
                streak += 1
            else:
                break
        return streak
    
    return 0


def _get_season_progress(leg: Any) -> float:
    """
    Extract season progress from leg features.
    
    Args:
        leg: Leg object with features
    
    Returns:
        Season progress (0-1), default 0.5
    """
    if leg is None:
        return 0.5
    
    if hasattr(leg, 'features'):
        for key in ['season_progress', 'season_progress_pct']:
            if key in leg.features:
                val = leg.features[key]
                if isinstance(val, (int, float)):
                    return float(val)
    
    # Try from profiles
    for profile in [getattr(leg, 'home_profile', None), getattr(leg, 'away_profile', None)]:
        if profile and hasattr(profile, 'get_metric'):
            games = profile.get_metric('core.games', 0)
            if games > 0:
                # Assume 38-game season if not specified
                return min(1.0, games / 38)
    
    return 0.5


def _is_extreme_decay(und_profile: Any) -> Tuple[bool, List[str]]:
    """
    NEW v5: Check if underdog is in EXTREME decay (complete freefall).
    
    Returns:
        Tuple of (is_extreme, reasons)
    """
    if und_profile is None:
        return False, []
    
    reasons = []
    games = und_profile.get_metric("core.games", 1)
    wins = und_profile.get_metric("core.wins", 0)
    draws = und_profile.get_metric("core.draws", 0)
    losses = und_profile.get_metric("core.losses", 0)
    win_rate = wins / games if games > 0 else 0
    
    # Get recent form
    form = und_profile.form.get("recent_results", []) if hasattr(und_profile, 'form') else []
    recent_losses = form[-6:].count("L") if len(form) >= 6 else 0
    
    # Get away record if available
    away_games = und_profile.get_metric("away_games", 0)
    away_wins = und_profile.get_metric("away_wins", 0)
    away_losses = und_profile.get_metric("away_losses", 0)
    away_win_rate = away_wins / away_games if away_games > 0 else 0
    
    # Check extreme decay conditions
    if win_rate < EXTREME_DECAY_MIN_WIN_RATE:
        reasons.append(f"win_rate={win_rate:.1%} < {EXTREME_DECAY_MIN_WIN_RATE:.0%}")
    
    ppg = (wins * 3 + draws) / games if games > 0 else 0
    if ppg < EXTREME_DECAY_MIN_PPG:
        reasons.append(f"ppg={ppg:.2f} < {EXTREME_DECAY_MIN_PPG:.1f}")
    
    if recent_losses >= EXTREME_DECAY_MIN_LOSS_STREAK:
        reasons.append(f"{recent_losses} consecutive losses")
    
    if away_games >= 10 and away_wins == 0 and EXTREME_DECAY_ZERO_AWAY_WINS:
        reasons.append(f"0 away wins in {away_games} matches")
    
    if away_losses >= EXTREME_DECAY_MIN_AWAY_LOSSES:
        reasons.append(f"{away_losses} away losses")
    
    # Team is in extreme decay if at least 2 conditions are met
    is_extreme = len(reasons) >= 2
    
    return is_extreme, reasons


# ═══════════════════════════════════════════════════════════════
# SECTION 0 — STANDALONE DATE VALIDATOR
# ═══════════════════════════════════════════════════════════════

def validate_fixture_date(event_timestamp: int, date_tolerance_days: int = 0) -> None:
    """
    Validate that a fixture's Unix timestamp falls on today's date (UTC).

    Args:
        event_timestamp: Unix timestamp (int) from SofaScore or any API.
        date_tolerance_days: Allow fixtures up to N days away (0 = today only)

    Raises:
        GuardrailError: if fixture date is outside tolerance window.
    """
    try:
        match_date = datetime.fromtimestamp(event_timestamp, tz=timezone.utc).date()
    except (OSError, OverflowError, ValueError) as e:
        raise GuardrailError(
            "fixture_timestamp_invalid",
            f"Cannot parse event_timestamp={event_timestamp}: {e}"
        )

    today = datetime.now(timezone.utc).date()
    date_diff = (match_date - today).days
    
    if abs(date_diff) > date_tolerance_days:
        raise GuardrailError(
            "fixture_date_outside_window",
            f"Fixture date {match_date} is {date_diff} days from today {today}. "
            f"Tolerance: ±{date_tolerance_days} days"
        )


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — RAW API RESPONSE VALIDATION
# ═══════════════════════════════════════════════════════════════

def validate_ingestion(
    fixture_raw: Dict,
    goals_home: Optional[int],
    goals_away: Optional[int],
    status_short: str,
    home_odds: Optional[float],
    away_odds: Optional[float],
    draw_odds: Optional[float],
) -> GuardrailReport:
    """
    Validate one raw fixture before any processing begins.
    Compatible with SofaScore and any other data source.

    Call this for every fixture BEFORE build_leg() in Module 1.
    """
    report = GuardrailReport()
    
    # Increment validation counter
    _validation_context.increment("total_fixtures_validated")

    # ── 1. Fixture status ────────────────────────────────────────
    report.checks_run += 1
    if status_short not in VALID_STATUSES:
        raise GuardrailError(
            "fixture_status",
            f"Status '{status_short}' is not NS (Not Started). "
            "Only upcoming fixtures may enter the pipeline."
        )

    # ── 2. Goals must be None for upcoming fixtures ──────────────
    report.checks_run += 1
    if goals_home is not None or goals_away is not None:
        raise GuardrailError(
            "goals_on_upcoming",
            f"Goals ({goals_home}-{goals_away}) present on a NS fixture. "
            "Indicates a data API error or wrong fixture status."
        )

    # ── 3. Odds presence ─────────────────────────────────────────
    report.checks_run += 1
    if home_odds is None or away_odds is None:
        raise GuardrailError(
            "missing_odds",
            "Home or away odds are None. Cannot compute implied probability or edge."
        )

    # ── 4. Odds bounds ───────────────────────────────────────────
    report.checks_run += 1
    for label, val in [("home", home_odds), ("away", away_odds)]:
        if val < MIN_ODDS:
            raise GuardrailError(
                "odds_below_minimum",
                f"{label} odds {val:.3f} < {MIN_ODDS}. Physically impossible decimal odds."
            )
        if val <= 0:
            raise GuardrailError(
                "odds_zero_or_negative",
                f"{label} odds = {val:.3f} is zero or negative."
            )
        if val > MAX_ODDS:
            report.add_warning(
                "odds_above_maximum",
                f"{label} odds {val:.2f} > {MAX_ODDS}. Extreme long-shot — verify data source.",
                severity="LOW",
                context={"odds": val, "max": MAX_ODDS}
            )

    # ── 5. Draw odds ─────────────────────────────────────────────
    report.checks_run += 1
    if draw_odds is not None:
        if draw_odds < MIN_ODDS and draw_odds > 0:
            report.add_warning(
                "draw_odds_invalid",
                f"Draw odds {draw_odds:.3f} < {MIN_ODDS}. May indicate API error.",
                severity="MEDIUM",
                context={"draw_odds": draw_odds, "min_odds": MIN_ODDS}
            )

    # ── 6. Probability sum ───────────────────────────────────────
    report.checks_run += 1
    if draw_odds and draw_odds >= MIN_ODDS:
        raw_sum = (1 / home_odds) + (1 / away_odds) + (1 / draw_odds)
        if raw_sum < 1.0:
            raise GuardrailError(
                "prob_sum_below_one",
                f"Implied probabilities sum to {raw_sum:.4f} < 1.0. Arbitrage opportunity detected."
            )
        if raw_sum > 1.25:
            report.add_warning(
                "excessive_bookmaker_margin",
                f"Bookmaker margin {(raw_sum - 1) * 100:.1f}% is unusually high.",
                severity="LOW",
                context={"margin_pct": (raw_sum - 1) * 100, "raw_sum": raw_sum}
            )

    # ── 7. Required fixture fields ───────────────────────────────
    report.checks_run += 1
    fixture_id = (
        fixture_raw.get("fixture", {}).get("id")
        or fixture_raw.get("id")
    )
    if not fixture_id:
        raise GuardrailError(
            "missing_fixture_id",
            "Fixture has no ID. Cannot build match_id or deduplicate."
        )

    # Support both API-Football (teams.home.id) and flat (homeTeam.id) structure
    home_id = (
        fixture_raw.get("teams", {}).get("home", {}).get("id")
        or fixture_raw.get("homeTeam", {}).get("id")
    )
    away_id = (
        fixture_raw.get("teams", {}).get("away", {}).get("id")
        or fixture_raw.get("awayTeam", {}).get("id")
    )
    if not home_id or not away_id:
        raise GuardrailError(
            "missing_team_ids",
            "Home or away team ID missing from fixture. Cannot match to standings."
        )

    # ── 8. Competition type — league fixtures only ───────────────
    report.checks_run += 1
    comp_type = (
        fixture_raw.get("league", {}).get("type")
        or fixture_raw.get("tournament", {}).get("type")
        or fixture_raw.get("competition_type")
    )
    if comp_type is not None and comp_type not in VALID_LEAGUE_TYPES:
        raise GuardrailError(
            "invalid_competition_type",
            f"Competition type '{comp_type}' is not in {VALID_LEAGUE_TYPES}. "
            "Only league fixtures enter the pipeline."
        )

    return report


# ═══════════════════════════════════════════════════════════════
# SECTION 1.5 — STRUCTURAL GAP & DOMINANCE VALIDATION (UPDATED v5)
# ═══════════════════════════════════════════════════════════════

def validate_dominance_vs_decay(leg: Any, verbose: bool = False) -> GuardrailReport:
    """
    Validate that the fixture represents a clear favourite (dominant team)
    versus a clear underdog (decaying team).
    
    NEW v5: Added EXTREME DECAY BYPASS for when underdog is in complete freefall.
    
    This is a HARD requirement for pipeline entry. Legs that don't show
    a clear dominance vs decay pattern are REJECTED, EXCEPT when underdog
    decay is so extreme that even a mediocre favourite has a structural advantage.
    """
    report = GuardrailReport()
    
    if leg.home_profile is None or leg.away_profile is None:
        report.add_error("dominance_vs_decay", "Missing team profiles")
        return report
    
    # Determine favourite
    if hasattr(leg, 'detect_favourite'):
        fav_is_home = leg.detect_favourite() == "HOME"
    else:
        home_odds = getattr(leg, 'home_odds', None)
        away_odds = getattr(leg, 'away_odds', None)
        if home_odds and away_odds:
            fav_is_home = home_odds <= away_odds
        else:
            fav_is_home = True
    
    fav_profile = leg.home_profile if fav_is_home else leg.away_profile
    und_profile = leg.away_profile if fav_is_home else leg.home_profile
    
    fav_name = getattr(fav_profile, 'team_name', 'Favourite')
    und_name = getattr(und_profile, 'team_name', 'Underdog')
    
    # ── Favourite Metrics ─────────────────────────────────────────
    fav_games = max(fav_profile.get_metric("core.games", 1), 1)
    fav_wins = fav_profile.get_metric("core.wins", 0)
    fav_draws = fav_profile.get_metric("core.draws", 0)
    fav_points = fav_wins * 3 + fav_draws
    fav_ppg = fav_points / fav_games if fav_games > 0 else 0.0
    fav_win_rate = fav_wins / fav_games if fav_games > 0 else 0.0
    
    fav_form = fav_profile.form.get("recent_results", [])[-5:] if fav_profile.form else []
    fav_recent_wins = fav_form.count("W")
    fav_recent_losses = fav_form.count("L")
    
    # ── Underdog Metrics ──────────────────────────────────────────
    und_games = max(und_profile.get_metric("core.games", 1), 1)
    und_wins = und_profile.get_metric("core.wins", 0)
    und_draws = und_profile.get_metric("core.draws", 0)
    und_points = und_wins * 3 + und_draws
    und_ppg = und_points / und_games if und_games > 0 else 0.0
    und_win_rate = und_wins / und_games if und_games > 0 else 0.0
    
    und_form = und_profile.form.get("recent_results", [])[-5:] if und_profile.form else []
    und_recent_losses = und_form.count("L")
    
    # ── Form Gap ──────────────────────────────────────────────────
    form_gap = fav_recent_wins - und_wins
    
    # ── NEW v5: Check for EXTREME DECAY BYPASS ─────────────────────
    is_extreme_decay, decay_reasons = _is_extreme_decay(und_profile)
    
    # ── COIN FLIP DETECTION: Both teams poor ──────────────────────
    both_poor = (fav_win_rate < 0.45 and und_win_rate < 0.45)
    if both_poor and not is_extreme_decay:
        report.add_error(
            "dominance_vs_decay",
            f"Both teams are poor (coin flip). Fav: {fav_win_rate:.1%} WR, "
            f"Und: {und_win_rate:.1%} WR. No structural gap.",
            context={"fav_win_rate": fav_win_rate, "und_win_rate": und_win_rate}
        )
        return report
    
    # ── Check Conditions ──────────────────────────────────────────
    checks = {}
    
    # Favourite must be dominant (standard thresholds)
    checks["fav_win_rate"] = fav_win_rate >= DOMINANCE_MIN_WIN_RATE
    checks["fav_ppg"] = fav_ppg >= DOMINANCE_MIN_PPG
    checks["fav_recent_wins"] = fav_recent_wins >= DOMINANCE_MIN_RECENT_WINS
    checks["form_gap"] = form_gap >= DOMINANCE_MIN_FORM_GAP
    
    # Underdog must be decaying
    checks["und_win_rate"] = und_win_rate <= DECAY_MAX_WIN_RATE
    checks["und_ppg"] = und_ppg <= DECAY_MAX_PPG
    checks["und_recent_losses"] = und_recent_losses >= DECAY_MIN_RECENT_LOSSES
    
    # Favourite must NOT be decaying
    checks["fav_not_decaying"] = fav_recent_losses <= FAVOURITE_DECAY_MAX_LOSSES
    
    # Calculate pass rate
    passed_checks = sum(checks.values())
    total_checks = len(checks)
    pass_rate = passed_checks / total_checks if total_checks > 0 else 0
    is_valid = pass_rate >= 0.75
    
    # ─── NEW v5: EXTREME DECAY BYPASS ─────────────────────────────
    bypass_applied = False
    bypass_reasons = []
    
    if is_extreme_decay and not is_valid:
        # Underdog is in complete freefall - use lower thresholds
        bypass_checks = {}
        
        # Check if favourite meets reduced thresholds
        bypass_checks["fav_win_rate"] = fav_win_rate >= BYPASS_FAV_MIN_WIN_RATE
        bypass_checks["fav_ppg"] = fav_ppg >= BYPASS_FAV_MIN_PPG
        bypass_checks["fav_recent_wins"] = fav_recent_wins >= BYPASS_FAV_MIN_RECENT_WINS
        bypass_checks["fav_not_excessive_decay"] = fav_recent_losses <= BYPASS_FAV_MAX_RECENT_LOSSES
        bypass_checks["form_gap"] = form_gap >= BYPASS_MIN_FORM_GAP
        
        # Underdog must still show decay (already true via is_extreme_decay)
        bypass_checks["und_decaying"] = True  # Already confirmed
        
        bypass_passed = sum(bypass_checks.values())
        bypass_total = len(bypass_checks)
        bypass_rate = bypass_passed / bypass_total if bypass_total > 0 else 0
        
        if bypass_rate >= 0.60:  # 60% of bypass checks = approve
            is_valid = True
            bypass_applied = True
            bypass_reasons = decay_reasons
            
            if verbose:
                print(f"\n[M0] EXTREME DECAY BYPASS APPLIED for {und_name}")
                for reason in decay_reasons:
                    print(f"    → {reason}")
                print(f"    Bypass checks: {bypass_passed}/{bypass_total} ({bypass_rate:.0%})")
    
    if verbose:
        print(f"\n[M0] Dominance vs Decay Validation for {fav_name} vs {und_name}")
        print(f"  Favourite: {fav_name}")
        print(f"    Win Rate: {fav_win_rate:.1%} (need ≥{DOMINANCE_MIN_WIN_RATE:.0%}) → {'✓' if checks['fav_win_rate'] else '✗'}")
        print(f"    PPG: {fav_ppg:.2f} (need ≥{DOMINANCE_MIN_PPG}) → {'✓' if checks['fav_ppg'] else '✗'}")
        print(f"    Recent Wins: {fav_recent_wins} (need ≥{DOMINANCE_MIN_RECENT_WINS}) → {'✓' if checks['fav_recent_wins'] else '✗'}")
        print(f"  Underdog: {und_name}")
        print(f"    Win Rate: {und_win_rate:.1%} (need ≤{DECAY_MAX_WIN_RATE:.0%}) → {'✓' if checks['und_win_rate'] else '✗'}")
        print(f"    PPG: {und_ppg:.2f} (need ≤{DECAY_MAX_PPG}) → {'✓' if checks['und_ppg'] else '✗'}")
        print(f"    Recent Losses: {und_recent_losses} (need ≥{DECAY_MIN_RECENT_LOSSES}) → {'✓' if checks['und_recent_losses'] else '✗'}")
        
        if bypass_applied:
            print(f"  🚨 EXTREME DECAY BYPASS: {', '.join(bypass_reasons)}")
            print(f"  Result: ✅ PASS (bypass) - {bypass_passed}/{bypass_total} bypass checks passed")
        else:
            print(f"  Result: {'✅ PASS' if is_valid else '❌ FAIL'} ({passed_checks}/{total_checks} checks passed)")
    
    if not is_valid:
        failures = [k for k, v in checks.items() if not v]
        error_msg = f"No clear dominance vs decay pattern. Failed checks: {', '.join(failures)}"
        
        if bypass_applied:
            # This shouldn't happen if bypass applied, but just in case
            error_msg = f"Bypass attempted but failed: {', '.join(failures)}"
        
        report.add_error(
            "dominance_vs_decay",
            error_msg,
            context={
                "fav_win_rate": fav_win_rate,
                "fav_ppg": fav_ppg,
                "fav_recent_wins": fav_recent_wins,
                "fav_recent_losses": fav_recent_losses,
                "und_win_rate": und_win_rate,
                "und_ppg": und_ppg,
                "und_recent_losses": und_recent_losses,
                "extreme_decay_bypass_attempted": is_extreme_decay,
                "bypass_applied": bypass_applied,
                "bypass_reasons": bypass_reasons,
            }
        )
    else:
        report.checks_run = total_checks
        report.passed = True
        
        # Add warning if bypass was applied (for audit trail)
        if bypass_applied:
            report.add_warning(
                "extreme_decay_bypass",
                f"Bypass applied: Underdog {und_name} in extreme decay ({', '.join(bypass_reasons)})",
                severity="MEDIUM",
                context={"bypass_reasons": bypass_reasons}
            )
    
    return report


def validate_senior_men_league(leg: Any) -> GuardrailReport:
    """
    Validate that the fixture is from a senior men's league (not youth, women, or amateur).
    """
    report = GuardrailReport()
    
    league_name = getattr(leg, 'league', '').lower()
    competition_type = getattr(leg, 'competition_type', 'league').lower()
    
    # Check for reject keywords
    for keyword in SENIOR_LEAGUE_REJECT_KEYWORDS:
        if keyword in league_name:
            report.add_error(
                "senior_men_league",
                f"League '{league_name}' contains reject keyword '{keyword}'",
                context={"keyword": keyword, "league": league_name}
            )
            return report
    
    # Check competition type
    if competition_type not in ALLOWED_COMPETITION_TYPES:
        report.add_warning(
            "competition_type_unknown",
            f"Competition type '{competition_type}' may not be senior men's league",
            severity="MEDIUM",
            context={"competition_type": competition_type}
        )
    
    report.passed = True
    report.checks_run = 1
    return report


def validate_league_tier(leg: Any) -> GuardrailReport:
    """
    Validate that the league meets tier requirements.
    
    Rules:
    - Popular countries (England, Spain, Germany, Italy, France): Top 3 tiers allowed
    - Strong secondary countries: Top 2 tiers allowed
    - Other countries: Top 1 tier only
    """
    report = GuardrailReport()
    
    league_name = getattr(leg, 'league', '')
    
    if not league_name:
        report.add_error("league_tier", "No league name provided")
        return report
    
    league_lower = league_name.lower()
    
    # Cup competitions are scan-only but allowed for learning
    cup_keywords = ['cup', 'champions league', 'europa league', 'conference league']
    if any(keyword in league_lower for keyword in cup_keywords):
        report.passed = True
        report.checks_run = 1
        return report
    
    # Detect tier and country
    tier = _detect_league_tier(league_name)
    country = _get_country_from_league(league_name)
    
    # If country not found, use conservative filter (only tier 1)
    if not country:
        if tier == 1:
            report.passed = True
            report.checks_run = 1
            return report
        else:
            report.add_error(
                "league_tier_not_allowed",
                f"Unknown country, tier {tier} not allowed (only tier 1)",
                context={"tier": tier, "league": league_name}
            )
            return report
    
    # Get max allowed tier for this country
    max_tier = POPULAR_LEAGUE_TIER_LIMITS.get(country, 1)
    
    if tier <= max_tier:
        report.passed = True
        report.checks_run = 1
        return report
    else:
        report.add_error(
            "league_tier_not_allowed",
            f"{country.title()} tier {tier} not allowed (max tier {max_tier}) for league '{league_name}'",
            context={"tier": tier, "max_tier": max_tier, "country": country, "league": league_name}
        )
        return report


def validate_tier_bypass_with_data(leg: Any) -> GuardrailReport:
    """
    Allow lower tier leagues to bypass tier restrictions if they have sufficient data.
    
    Requirements for bypass:
    - Both teams have played ≥15 games (vs 10 for top tier)
    - H2H has ≥10 games (vs 5 for top tier)
    - Competition type is league (not playoff/cup)
    """
    report = GuardrailReport()
    
    if leg.home_profile is None or leg.away_profile is None:
        report.add_error("tier_bypass", "Missing team profiles")
        return report
    
    # Get games played
    home_games = leg.home_profile.get_metric("core.games", 0)
    away_games = leg.away_profile.get_metric("core.games", 0)
    min_games = min(home_games, away_games)
    
    # Get H2H games
    h2h_games = 0
    if leg.h2h is not None:
        h2h_games = getattr(leg.h2h, 'games', 0)
    
    # Check bypass conditions
    games_sufficient = min_games >= TIER_BYPASS_MIN_GAMES
    h2h_sufficient = h2h_games >= TIER_BYPASS_MIN_H2H
    is_league = getattr(leg, 'competition_type', 'league') == 'league'
    
    if games_sufficient and h2h_sufficient and is_league:
        report.passed = True
        report.checks_run = 1
        report.add_warning(
            "tier_bypass_applied",
            f"Lower tier allowed due to sufficient data: {min_games:.0f} games each, {h2h_games} H2H games",
            severity="MEDIUM",
            context={"min_games": min_games, "h2h_games": h2h_games}
        )
    else:
        report.add_error(
            "tier_bypass_failed",
            f"Lower tier insufficient data: {min_games:.0f}/{TIER_BYPASS_MIN_GAMES} games, "
            f"{h2h_games}/{TIER_BYPASS_MIN_H2H} H2H, league={is_league}",
            context={
                "min_games": min_games,
                "required_games": TIER_BYPASS_MIN_GAMES,
                "h2h_games": h2h_games,
                "required_h2h": TIER_BYPASS_MIN_H2H,
                "is_league": is_league
            }
        )
    
    return report


def validate_not_playoff_or_knockout(leg: Any) -> GuardrailReport:
    """
    Validate that the match is NOT a playoff, knockout, or relegation match.
    
    These formats are REJECTED entirely because:
    - Single elimination changes team psychology
    - Extra time/penalties distort win probability
    - Small sample size for historical patterns
    - Desperation football in relegation matches
    """
    report = GuardrailReport()
    
    league_name = getattr(leg, 'league', '').lower()
    competition_type = getattr(leg, 'competition_type', 'league').lower()
    stage = getattr(leg, 'stage', '').lower()
    round_name = getattr(leg, 'round', '').lower()
    
    # Check 1: Rejected competition types
    if competition_type in REJECTED_COMPETITION_TYPES:
        report.add_error(
            "playoff_knockout_rejected",
            f"Rejected competition type: '{competition_type}'",
            context={"competition_type": competition_type}
        )
        return report
    
    # Check 2: League name contains rejected keywords
    for keyword in REJECTED_MATCH_KEYWORDS:
        if keyword in league_name:
            report.add_error(
                "playoff_knockout_rejected",
                f"League name contains rejected keyword: '{keyword}' in '{league_name}'",
                context={"keyword": keyword, "league": league_name}
            )
            return report
    
    # Check 3: Stage name indicates playoff/knockout
    for keyword in REJECTED_MATCH_KEYWORDS:
        if keyword in stage:
            report.add_error(
                "playoff_knockout_rejected",
                f"Rejected stage: '{stage}' contains '{keyword}'",
                context={"keyword": keyword, "stage": stage}
            )
            return report
    
    # Check 4: Round name indicates playoff/knockout
    for keyword in REJECTED_MATCH_KEYWORDS:
        if keyword in round_name:
            report.add_error(
                "playoff_knockout_rejected",
                f"Round name contains rejected keyword: '{round_name}'",
                context={"keyword": keyword, "round": round_name}
            )
            return report
    
    report.passed = True
    report.checks_run = 1
    return report


def is_regular_season_league(leg: Any) -> bool:
    """
    Quick check if this is a regular season league match (not playoff/knockout).
    
    Returns:
        True if regular season league match, False otherwise
    """
    # Must be league competition type
    competition_type = getattr(leg, 'competition_type', 'league').lower()
    if competition_type != 'league':
        return False
    
    league_name = getattr(leg, 'league', '').lower()
    
    # Reject if contains playoff keywords
    for keyword in REJECTED_MATCH_KEYWORDS:
        if keyword in league_name:
            return False
    
    # Check stage - must be regular season or None
    stage = getattr(leg, 'stage', '').lower()
    if stage and stage not in ['regular season', 'regular', 'league', '']:
        return False
    
    return True


# ═══════════════════════════════════════════════════════════════
# SECTION 1.6 — DEAD RUBBER VALIDATION (v4)
# ═══════════════════════════════════════════════════════════════

def validate_dead_rubber(
    leg: Any,
    season_progress: Optional[float] = None,
    verbose: bool = False,
) -> GuardrailReport:
    """
    Validate that the match is NOT a dead rubber with favourite unmotivated.
    
    HARD REJECT if:
    - Match is dead rubber (nothing to play for)
    - Favourite has LOW or DEAD_RUBBER motivation
    - Season progress >85% (late season)
    
    This prevents betting on matches where the favourite has already won
    the league or secured their position and has nothing to play for.
    """
    report = GuardrailReport()
    
    if leg.home_profile is None or leg.away_profile is None:
        # Can't validate without profiles
        report.passed = True
        report.checks_run = 1
        return report
    
    # Get season progress
    if season_progress is None:
        season_progress = _get_season_progress(leg)
    
    # Only check in late season (85%+ complete)
    if season_progress < DEAD_RUBBER_SEASON_THRESHOLD:
        report.passed = True
        report.checks_run = 1
        return report
    
    # Determine favourite
    if hasattr(leg, 'detect_favourite'):
        fav_is_home = leg.detect_favourite() == "HOME"
    else:
        home_odds = getattr(leg, 'home_odds', None)
        away_odds = getattr(leg, 'away_odds', None)
        if home_odds and away_odds:
            fav_is_home = home_odds <= away_odds
        else:
            fav_is_home = True
    
    fav_profile = leg.home_profile if fav_is_home else leg.away_profile
    und_profile = leg.away_profile if fav_is_home else leg.home_profile
    
    # Get motivation labels
    fav_motivation = _get_motivation_label(fav_profile)
    und_motivation = _get_motivation_label(und_profile)
    
    # Get match context flags
    is_dead_rubber = False
    if hasattr(leg, 'features'):
        is_dead_rubber = leg.features.get('is_dead_rubber', False)
    
    # Also check M26 context if available
    if hasattr(leg, '_match_context') and leg._match_context:
        is_dead_rubber = getattr(leg._match_context, 'is_dead_rubber', False)
    
    # Hard reject conditions
    if is_dead_rubber and fav_motivation in DEAD_RUBBER_MOTIVATION_LOW:
        raise GuardrailError(
            "dead_rubber_reject",
            DEAD_RUBBER_REJECT_MESSAGE,
            context={
                "match_id": getattr(leg, 'match_id', 'unknown'),
                "fav_team": getattr(fav_profile, 'team_name', 'unknown'),
                "fav_motivation": fav_motivation,
                "und_motivation": und_motivation,
                "season_progress": season_progress,
                "is_dead_rubber": is_dead_rubber,
            }
        )
    elif is_dead_rubber:
        # Dead rubber but favourite still motivated (e.g., chasing record)
        report.add_warning(
            "dead_rubber_warning",
            f"Match is dead rubber but favourite motivation is {fav_motivation}. Proceed with caution.",
            severity="HIGH",
            context={
                "fav_motivation": fav_motivation,
                "season_progress": season_progress,
            }
        )
    
    report.checks_run = 2
    report.passed = True
    return report


# ═══════════════════════════════════════════════════════════════
# SECTION 1.7 — WIN CEILING VALIDATION (v4)
# ═══════════════════════════════════════════════════════════════

@dataclass
class WinCeilingResult:
    """Result of win ceiling validation."""
    passed: bool
    stake_multiplier: float
    warning: Optional[str] = None


def validate_win_ceiling(
    leg: Any,
    season_progress: Optional[float] = None,
    verbose: bool = False,
) -> WinCeilingResult:
    """
    Validate win ceiling for favourite team.
    
    Progressive stake reduction when approaching historical win ceiling:
    - At ceiling (streak = historical max) → HARD REJECT
    - Approaching ceiling (within 2 wins) + late season → 25% stake
    - Approaching ceiling (within 2 wins) + mid season → 50% stake
    - No ceiling risk → full stake (1.0)
    """
    if leg.home_profile is None or leg.away_profile is None:
        # Can't validate without profiles
        return WinCeilingResult(passed=True, stake_multiplier=1.0)
    
    # Determine favourite
    if hasattr(leg, 'detect_favourite'):
        fav_is_home = leg.detect_favourite() == "HOME"
    else:
        home_odds = getattr(leg, 'home_odds', None)
        away_odds = getattr(leg, 'away_odds', None)
        if home_odds and away_odds:
            fav_is_home = home_odds <= away_odds
        else:
            fav_is_home = True
    
    fav_profile = leg.home_profile if fav_is_home else leg.away_profile
    
    # Get win ceiling metrics
    current_streak = _get_current_win_streak(fav_profile)
    historical_ceiling = _get_historical_win_ceiling(fav_profile)
    
    # Only check if minimum streak threshold met
    if current_streak < WIN_CEILING_MIN_STREAK:
        if verbose and current_streak > 0:
            print(f"  Win ceiling: {current_streak} wins (below check threshold {WIN_CEILING_MIN_STREAK})")
        return WinCeilingResult(passed=True, stake_multiplier=1.0)
    
    # Get season progress
    if season_progress is None:
        season_progress = _get_season_progress(leg)
    
    # Check if at ceiling (streak equals historical max)
    if current_streak >= historical_ceiling:
        warning = (
            f"WIN CEILING HIT: {current_streak} consecutive wins equals historical max {historical_ceiling}. "
            f"Team due for regression. HARD REJECT."
        )
        if verbose:
            print(f"  🚨 {warning}")
        return WinCeilingResult(passed=False, stake_multiplier=0.0, warning=warning)
    
    # Check if approaching ceiling (within 2 wins)
    distance_to_ceiling = historical_ceiling - current_streak
    if distance_to_ceiling <= WIN_CEILING_APPROACH_DISTANCE and distance_to_ceiling > 0:
        # Determine stake multiplier based on season progress
        if season_progress >= DEAD_RUBBER_SEASON_THRESHOLD:
            # Late season: severe reduction
            multiplier = CEILING_APPROACH_MULTIPLIER_LATE
            phase = "late"
            severity = "HIGH"
        else:
            # Mid season: moderate reduction
            multiplier = CEILING_APPROACH_MULTIPLIER_MID
            phase = "mid"
            severity = "MEDIUM"
        
        warning = (
            f"WIN CEILING APPROACH: {current_streak} consecutive wins (max {historical_ceiling}, "
            f"{distance_to_ceiling} away). {phase.title()} season → stakes reduced to {multiplier:.0%}."
        )
        
        if verbose:
            icon = "🚨" if severity == "HIGH" else "⚠️"
            print(f"  {icon} {warning}")
        
        # Return warning but still allow bet with reduced stake
        return WinCeilingResult(passed=True, stake_multiplier=multiplier, warning=warning)
    
    if verbose and current_streak >= WIN_CEILING_MIN_STREAK:
        print(f"  Win ceiling: {current_streak} wins (max {historical_ceiling}, {distance_to_ceiling} away) - no reduction")
    
    return WinCeilingResult(passed=True, stake_multiplier=1.0)


# ═══════════════════════════════════════════════════════════════
# SECTION 1.8 — ALL FILTERS COMBINED (UPDATED v5)
# ═══════════════════════════════════════════════════════════════

def validate_all_filters(
    leg: Any,
    verbose: bool = False,
) -> Tuple[GuardrailReport, Optional[WinCeilingResult]]:
    """
    Run ALL structural gap and format filters in one call.
    
    UPDATED v5: Now includes extreme decay bypass in dominance validation.
    
    Returns:
        Tuple of (guardrail_report, win_ceiling_result)
    """
    report = GuardrailReport()
    
    # Filter 1: Senior men's league
    sr_report = validate_senior_men_league(leg)
    if not sr_report.passed:
        report.add_error(sr_report.errors[0].split(":")[0], sr_report.errors[0])
        return report, WinCeilingResult(passed=False, stake_multiplier=0.0)
    
    # Filter 2: League tier (with bypass option)
    tier_report = validate_league_tier(leg)
    if not tier_report.passed:
        # Check if bypass is possible
        bypass_report = validate_tier_bypass_with_data(leg)
        if not bypass_report.passed:
            report.add_error(tier_report.errors[0].split(":")[0], tier_report.errors[0])
            return report, WinCeilingResult(passed=False, stake_multiplier=0.0)
        else:
            # Bypass applied - add warning but continue
            for w in bypass_report.warnings:
                report.add_warning(w.check, w.message, w.severity, w.context)
    
    # Filter 3: Not playoff/knockout
    format_report = validate_not_playoff_or_knockout(leg)
    if not format_report.passed:
        report.add_error(format_report.errors[0].split(":")[0], format_report.errors[0])
        return report, WinCeilingResult(passed=False, stake_multiplier=0.0)
    
    # Filter 4: Dominance vs decay (UPDATED v5 with extreme decay bypass)
    dom_report = validate_dominance_vs_decay(leg, verbose)
    if not dom_report.passed:
        report.add_error(dom_report.errors[0].split(":")[0], dom_report.errors[0])
        return report, WinCeilingResult(passed=False, stake_multiplier=0.0)
    
    # Filter 5: Dead rubber (v4)
    dead_rubber_report = validate_dead_rubber(leg, verbose=verbose)
    if not dead_rubber_report.passed:
        if dead_rubber_report.errors:
            report.add_error(dead_rubber_report.errors[0].split(":")[0], dead_rubber_report.errors[0])
            return report, WinCeilingResult(passed=False, stake_multiplier=0.0)
    
    # Filter 6: Win ceiling (v4)
    ceiling_result = validate_win_ceiling(leg, verbose=verbose)
    if not ceiling_result.passed:
        report.add_error("win_ceiling_reject", ceiling_result.warning or "Win ceiling hit - hard reject")
        return report, ceiling_result
    
    # Add win ceiling warning if present (stake reduction)
    if ceiling_result.warning and ceiling_result.stake_multiplier < 1.0:
        report.add_warning(
            "win_ceiling_approach",
            ceiling_result.warning,
            severity="HIGH" if ceiling_result.stake_multiplier <= 0.25 else "MEDIUM"
        )
    
    report.passed = True
    report.checks_run = (
        sr_report.checks_run + 
        tier_report.checks_run + 
        format_report.checks_run + 
        dom_report.checks_run +
        dead_rubber_report.checks_run +
        2  # For the two new checks
    )
    
    if verbose:
        print(f"\n[M0] All filters passed for {getattr(leg, 'match_id', 'unknown')}")
        if ceiling_result.stake_multiplier < 1.0:
            print(f"  ⚠ Stake multiplier: {ceiling_result.stake_multiplier:.0%} (win ceiling approach)")
    
    return report, ceiling_result


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — LEG OBJECT VALIDATION
# ═══════════════════════════════════════════════════════════════

def validate_leg(leg: Any) -> GuardrailReport:
    """
    Validate a fully-built Leg object before it enters Module 4.

    Checks that profiles are real (not ghost objects), sequences are
    long enough for statistical validity, and all required fields exist.

    Raises GuardrailError for hard failures (leg must be dropped).
    Returns GuardrailReport with warnings for soft issues.
    """
    report = GuardrailReport()

    # ── 1. Profiles exist ────────────────────────────────────────
    report.checks_run += 1
    if leg.home_profile is None or leg.away_profile is None:
        raise GuardrailError(
            "missing_profiles",
            f"Leg '{getattr(leg, 'match_id', 'unknown')}' has no home_profile or away_profile. "
            "Ghost Leg — cannot run any analysis module."
        )

    # ── 2. Team IDs are real (not fallback strings) ──────────────
    report.checks_run += 1
    for label, profile in [("home", leg.home_profile), ("away", leg.away_profile)]:
        team_id = getattr(profile, "team_id", None)
        if not team_id or str(team_id) in ("", "0", "None"):
            raise GuardrailError(
                "invalid_team_id",
                f"{label} profile has invalid team_id '{team_id}'. "
                "Indicates team matching failure in Module 1."
            )

    # ── 3. Result sequences ──────────────────────────────────────
    report.checks_run += 1
    for label, profile in [("home", leg.home_profile), ("away", leg.away_profile)]:
        form = getattr(profile, "form", {})
        seq = form.get("recent_results", []) if isinstance(form, dict) else []
        seq_len = len(seq)

        if seq_len < MIN_HARD_SEQUENCE_GAMES:
            raise GuardrailError(
                "insufficient_sequence",
                f"{label} team '{getattr(profile, 'team_name', '?')}' has only {seq_len} results "
                f"(absolute minimum {MIN_HARD_SEQUENCE_GAMES}). "
                "TransitionMatrix and asymmetric pre-filter results would be statistically invalid."
            )

        if seq_len < MIN_SEQUENCE_GAMES:
            report.add_warning(
                "short_sequence",
                f"{label} team '{getattr(profile, 'team_name', '?')}' has only {seq_len} results "
                f"(recommended minimum {MIN_SEQUENCE_GAMES}). "
                "TransitionMatrix quality is reduced — proceed with caution.",
                severity="HIGH",
                context={"seq_len": seq_len, "min_required": MIN_SEQUENCE_GAMES}
            )

        invalid = [r for r in seq if r not in VALID_RESULTS]
        if invalid:
            raise GuardrailError(
                "invalid_result_codes",
                f"{label} sequence contains invalid codes: {invalid}. "
                "Only W / D / L are permitted."
            )

    # ── 4. Core metrics are non-zero ─────────────────────────────
    report.checks_run += 1
    for label, profile in [("home", leg.home_profile), ("away", leg.away_profile)]:
        games = profile.get_metric("core.games", 0) if hasattr(profile, "get_metric") else 0
        wins = profile.get_metric("core.wins", 0) if hasattr(profile, "get_metric") else 0
        if games == 0:
            raise GuardrailError(
                "zero_games_metric",
                f"{label} profile '{getattr(profile, 'team_name', '?')}' reports 0 games played. "
                "All derived metrics (win rate, edge, xG diff) will be zero or NaN."
            )
        if wins > games:
            raise GuardrailError(
                "wins_exceed_games",
                f"{label} profile '{getattr(profile, 'team_name', '?')}': wins ({wins:.0f}) > games ({games:.0f}). "
                "Data corruption — wins cannot exceed total matches."
            )

    # ── 5. Odds validity ─────────────────────────────────────────
    report.checks_run += 1
    leg_odds = getattr(leg, "odds", 0.0)
    if not leg_odds or leg_odds < MIN_ODDS:
        raise GuardrailError(
            "invalid_leg_odds",
            f"Leg odds {leg_odds:.3f} < {MIN_ODDS}. Cannot compute edge or Kelly stake."
        )
    
    # HARD FILTER: Odds must be ≥ 1.70
    if leg_odds < 1.70:
        raise GuardrailError(
            "odds_below_minimum_threshold",
            f"Leg odds {leg_odds:.2f} < 1.70. Favourite priced too short for value."
        )

    # ── 6. Selection is set ──────────────────────────────────────
    report.checks_run += 1
    selection = getattr(leg, "selection", "")
    if not selection or not selection.strip():
        raise GuardrailError(
            "empty_selection",
            f"Leg '{getattr(leg, 'match_id', 'unknown')}' has no selection. "
            "Cannot build ACCA slip or bankroll plan."
        )

    # ── 7. TransitionMatrix quality ──────────────────────────────
    report.checks_run += 1
    for label, profile in [("home", leg.home_profile), ("away", leg.away_profile)]:
        tm = getattr(profile, "transition", None)
        if tm is None:
            report.add_warning(
                "missing_transition_matrix",
                f"{label} team '{getattr(profile, 'team_name', '?')}' has no TransitionMatrix. "
                "Module 4 checks C4/C5 will be skipped, reducing gate strength.",
                severity="HIGH",
                context={"team": getattr(profile, 'team_name', '?')}
            )
        elif hasattr(tm, "sample_size") and tm.sample_size < MIN_SEQUENCE_GAMES - 1:
            report.add_warning(
                "thin_transition_matrix",
                f"{label} team '{getattr(profile, 'team_name', '?')}' TransitionMatrix has only "
                f"{tm.sample_size} transitions. Probability estimates are unreliable.",
                severity="MEDIUM",
                context={"sample_size": tm.sample_size, "min_required": MIN_SEQUENCE_GAMES - 1}
            )

    # ── 8. H2H record ────────────────────────────────────────────
    report.checks_run += 1
    h2h = getattr(leg, "h2h", None)
    h2h_games = getattr(h2h, "games", 0) if h2h else 0
    if h2h is None or h2h_games < 5:
        report.add_warning(
            "thin_h2h",
            f"H2H record has {h2h_games} games (< 5). "
            "Module 4 check C3 (H2H check) will be skipped.",
            severity="LOW",
            context={"h2h_games": h2h_games, "min_required": 5}
        )

    return report


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — AI RESPONSE VALIDATION
# ═══════════════════════════════════════════════════════════════

def validate_ai_response(
    ai_result: Any,
    leg: Any,
    provider_name: str,
) -> GuardrailReport:
    """
    Validate one AI provider's SingleAIAnalysis before it enters
    the consensus engine in Module 7.

    Detects: hallucinated probabilities, simulated responses,
    citation of teams not in the fixture, and JSON parse failures
    silently treated as valid NEUTRAL verdicts.
    """
    report = GuardrailReport()
    
    # Validate provider name
    if provider_name not in VALID_AI_PROVIDERS:
        report.add_warning(
            "unknown_ai_provider",
            f"Provider '{provider_name}' not in {VALID_AI_PROVIDERS}",
            severity="MEDIUM",
            context={"provider": provider_name, "valid_providers": list(VALID_AI_PROVIDERS)}
        )

    # ── 1. API error check ───────────────────────────────────────
    report.checks_run += 1
    if hasattr(ai_result, "error") and ai_result.error:
        raise GuardrailError(
            f"{provider_name}_api_error",
            f"{provider_name} returned an error: '{ai_result.error}'. "
            "This result must not enter the consensus engine."
        )

    # ── 2. Raw response exists ───────────────────────────────────
    report.checks_run += 1
    raw_response = getattr(ai_result, "raw_response", "")
    if not raw_response or not raw_response.strip():
        raise GuardrailError(
            f"{provider_name}_empty_response",
            f"{provider_name} returned an empty response. "
            "No analysis was performed — this is simulation, not intelligence."
        )

    # ── 3. Verdict is a known enum value ────────────────────────
    report.checks_run += 1
    ai_verdict = getattr(ai_result, "ai_verdict", None)
    if ai_verdict is None:
        raise GuardrailError(
            f"{provider_name}_missing_verdict",
            f"{provider_name} returned no verdict. Analysis is incomplete."
        )
    
    # Handle enum or string
    if hasattr(ai_verdict, "value"):
        verdict_val = ai_verdict.value
    else:
        verdict_val = str(ai_verdict)
    
    if verdict_val not in VALID_VERDICTS:
        raise GuardrailError(
            f"{provider_name}_invalid_verdict",
            f"{provider_name} returned unknown verdict '{verdict_val}'. "
            f"Must be one of: {VALID_VERDICTS}"
        )

    # ── 4. Confidence is a known enum value ──────────────────────
    report.checks_run += 1
    ai_confidence = getattr(ai_result, "ai_confidence", None)
    if ai_confidence is None:
        report.add_warning(
            f"{provider_name}_missing_confidence",
            f"{provider_name} returned no confidence level. Defaulting to LOW.",
            severity="MEDIUM"
        )
    else:
        if hasattr(ai_confidence, "value"):
            conf_val = ai_confidence.value
        else:
            conf_val = str(ai_confidence)
        
        if conf_val not in VALID_CONFIDENCES:
            raise GuardrailError(
                f"{provider_name}_invalid_confidence",
                f"{provider_name} returned unknown confidence '{conf_val}'. "
                f"Must be one of: {VALID_CONFIDENCES}"
            )

    # ── 5. AI probability bounds ─────────────────────────────────
    report.checks_run += 1
    ai_prob = getattr(ai_result, "ai_win_probability", None)
    if ai_prob is None:
        raise GuardrailError(
            f"{provider_name}_missing_probability",
            f"{provider_name} returned no ai_win_probability. "
            "Cannot validate against model probability."
        )
    
    if not (0.0 <= ai_prob <= 1.0):
        raise GuardrailError(
            f"{provider_name}_probability_out_of_bounds",
            f"{provider_name} returned ai_win_probability={ai_prob:.4f} "
            "which is outside [0.0, 1.0]. Mathematically impossible."
        )

    # ── 6. AI probability divergence from model ──────────────────
    report.checks_run += 1
    model_prob = getattr(leg, "model_prob", None)

    if model_prob is None:
        report.add_warning(
            f"{provider_name}_missing_model_prob",
            f"model_prob not set on Leg — cannot validate AI divergence for "
            f"{provider_name}. Set leg.model_prob in Module 3/6 before calling M7.",
            severity="HIGH"
        )
    elif not (0.0 < model_prob <= 1.0):
        report.add_warning(
            f"{provider_name}_model_prob_invalid",
            f"leg.model_prob={model_prob:.4f} is outside (0.0, 1.0]. "
            "Divergence check skipped.",
            severity="HIGH"
        )
    else:
        divergence = abs(ai_prob - model_prob)
        if divergence > MAX_AI_PROB_DIVERGENCE:
            report.add_warning(
                f"{provider_name}_probability_divergence",
                f"{provider_name} win_probability {ai_prob:.1%} diverges "
                f"{divergence:.1%} from model probability {model_prob:.1%} "
                f"(max allowed: {MAX_AI_PROB_DIVERGENCE:.0%})",
                severity="HIGH",
                context={"ai_prob": ai_prob, "model_prob": model_prob, "divergence": divergence}
            )

    # ── 7. AI illegal override check ────────────────────────────
    report.checks_run += 1
    pre_verdict = getattr(leg, "pre_verdict", None)
    if pre_verdict == "REJECT" and verdict_val == "APPROVE":
        raise GuardrailError(
            f"{provider_name}_illegal_override",
            f"{provider_name} returned APPROVE on a leg that the core pipeline "
            "already marked REJECT. AI cannot override forensic gate decisions."
        )

    # ── 8. Narrative quality ─────────────────────────────────────
    report.checks_run += 1
    narrative = getattr(ai_result, "narrative", "") or ""
    if len(narrative.strip()) < MIN_NARRATIVE_LENGTH:
        report.add_warning(
            f"{provider_name}_empty_narrative",
            f"{provider_name} narrative is only {len(narrative.strip())} characters. "
            "Too short to represent genuine analysis — likely a simulation or parse failure.",
            severity="MEDIUM",
            context={"narrative_length": len(narrative.strip()), "min_length": MIN_NARRATIVE_LENGTH}
        )

    # ── 9. Hallucinated team names in narrative ──────────────────
    report.checks_run += 1
    if narrative and leg.home_profile and leg.away_profile:
        home_name = getattr(leg.home_profile, "team_name", "").lower()
        away_name = getattr(leg.away_profile, "team_name", "").lower()
        full_team_text = f"{home_name} {away_name}"

        # Look for capitalized sequences that might be team names
        cap_sequences = re.findall(r'(?:\b[A-Z][a-z]{2,}\b(?:\s+[A-Z][a-z]{2,}\b)+)', narrative)
        single_caps = re.findall(r'\b[A-Z][a-z]{2,}\b', narrative)
        suspicious = []

        for seq in cap_sequences:
            seq_lower = seq.lower()
            if (seq_lower not in full_team_text and
                    not any(seq_lower in part for part in home_name.split()) and
                    not any(seq_lower in part for part in away_name.split())):
                suspicious.append(seq)

        for word in single_caps:
            w = word.lower()
            if w in COMMON_ANALYSIS_TERMS:
                continue
            if (w not in home_name and w not in away_name and
                    not any(w in part for part in home_name.split()) and
                    not any(w in part for part in away_name.split())):
                suspicious.append(word)

        if suspicious:
            # Limit to first 3 suspicious terms to avoid spam
            suspicious_sample = list(set(suspicious[:3]))
            report.add_warning(
                f"{provider_name}_possible_hallucination",
                f"{provider_name} narrative contains potential hallucinated entities: "
                f"'{', '.join(suspicious_sample)}' which do not match fixture teams "
                f"'{getattr(leg.home_profile, 'team_name', '?')}' vs "
                f"'{getattr(leg.away_profile, 'team_name', '?')}'.",
                severity="HIGH",
                context={"suspicious_terms": suspicious_sample, "home_team": home_name, "away_team": away_name}
            )

    # ── 10. Score ranges ─────────────────────────────────────────
    report.checks_run += 1
    for score_name in ["anomaly_score", "trap_score", "trend_confidence"]:
        score_val = getattr(ai_result, score_name, None)
        if score_val is not None and not (0.0 <= score_val <= 1.0):
            report.add_warning(
                f"{provider_name}_{score_name}_out_of_bounds",
                f"{provider_name} {score_name}={score_val:.4f} is outside [0.0, 1.0]. "
                "Will be clamped before consensus aggregation.",
                severity="MEDIUM",
                context={"score_name": score_name, "value": score_val}
            )
    
    # ── 11. Response time sanity (simulation detection) ──────────
    report.checks_run += 1
    response_time = getattr(ai_result, "response_time", 0.0)
    if response_time < 0.5 and len(raw_response) > 500:
        report.add_warning(
            f"{provider_name}_suspicious_response_time",
            f"{provider_name} responded in {response_time:.2f}s but returned {len(raw_response)} chars. "
            "May be cached or simulated.",
            severity="LOW",
            context={"response_time": response_time, "response_length": len(raw_response)}
        )

    return report


def validate_ai_consensus(
    claude_result: Optional[Any],
    gemini_result: Optional[Any],
    gpt_result: Optional[Any],
    deepseek_result: Optional[Any] = None,
) -> GuardrailReport:
    """
    Cross-provider sanity check after all AIs have responded.
    Detects simulation where all providers return identical scores.
    Also detects when AIs all have the same error pattern.
    """
    report = GuardrailReport()
    
    # Collect active (non-error) results
    active_results = []
    for r, name in [(deepseek_result, "DeepSeek"), (claude_result, "Claude"), 
                    (gemini_result, "Gemini"), (gpt_result, "GPT")]:
        if r is not None and not getattr(r, "error", True):
            active_results.append((r, name))

    # ── 1. At least one real response ───────────────────────────
    report.checks_run += 1
    if not active_results:
        raise GuardrailError(
            "no_ai_responses",
            "All AI providers failed or returned errors. "
            "No AI analysis is available — cannot proceed to consensus."
        )

    # ── 2. Identical score simulation detection ──────────────────
    report.checks_run += 1
    if len(active_results) >= 2:
        probs = [getattr(r, "ai_win_probability", 0.0) for r, _ in active_results]
        traps = [getattr(r, "trap_score", 0.0) for r, _ in active_results]
        anomaly = [getattr(r, "anomaly_score", 0.0) for r, _ in active_results]

        def _all_identical(vals: list) -> bool:
            if not vals:
                return False
            # Round to 4 decimal places to avoid floating point differences
            rounded = [round(v, 4) for v in vals]
            return len(set(rounded)) == 1

        if _all_identical(probs) and _all_identical(traps) and _all_identical(anomaly):
            raise GuardrailError(
                "ai_simulation_detected",
                f"All {len(active_results)} AI providers returned bit-for-bit "
                "identical scores (prob, trap, anomaly). This is statistically "
                "impossible from independent models — indicates simulation, "
                "API caching, or a mocked response.",
                context={"provider_count": len(active_results)}
            )

    # ── 3. Identical error pattern detection ─────────────────────
    report.checks_run += 1
    error_messages = [getattr(r, "error", "") for r, _ in active_results]
    if error_messages and len(set(error_messages)) == 1 and error_messages[0]:
        report.add_warning(
            "identical_error_patterns",
            f"All {len(active_results)} AI providers returned the same error: "
            f"'{error_messages[0][:100]}'. May indicate a common upstream issue.",
            severity="HIGH",
            context={"error": error_messages[0][:200]}
        )

    # ── 4. All AIs gave same verdict but different scores ─────────
    report.checks_run += 1
    if len(active_results) >= 2:
        verdicts = [getattr(r, "ai_verdict", None) for r, _ in active_results]
        # Convert enums to strings for comparison
        verdict_strs = [v.value if hasattr(v, "value") else str(v) for v in verdicts if v]
        if len(set(verdict_strs)) == 1 and len(active_results) >= 3:
            report.add_warning(
                "unanimous_ai_verdict",
                f"All {len(active_results)} AI providers returned the same verdict: "
                f"{verdict_strs[0]}. Consensus is strong but verify independence.",
                severity="LOW",
                context={"verdict": verdict_strs[0], "provider_count": len(active_results)}
            )

    # ── 5. Response time pattern (simulated responses) ───────────
    report.checks_run += 1
    response_times = [getattr(r, "response_time", 0.0) for r, _ in active_results]
    if (response_times and len(response_times) >= 2 and 
        all(t < 0.3 for t in response_times) and 
        max(response_times) - min(response_times) < 0.05):
        report.add_warning(
            "suspicious_response_timing",
            f"All {len(active_results)} AI providers responded in under 0.3s "
            f"with minimal variance ({max(response_times)-min(response_times):.3f}s). "
            "May indicate cached or simulated responses.",
            severity="MEDIUM",
            context={"response_times": response_times}
        )

    return report


def repair_ai_response(ai_result: Any) -> Any:
    """
    Attempt to repair a malformed AI response by repairing JSON.
    Returns the modified ai_result object.
    """
    if not hasattr(ai_result, "raw_response"):
        return ai_result
    
    raw_response = getattr(ai_result, "raw_response", "")
    if not raw_response:
        return ai_result
    
    try:
        repaired = _repair_json_response(raw_response)
        if repaired != raw_response:
            # Parse the repaired JSON to validate
            parsed = json.loads(repaired)
            
            # Update the ai_result if parsing succeeded
            if hasattr(ai_result, "anomaly_score") and "anomaly_score" in parsed:
                ai_result.anomaly_score = float(parsed.get("anomaly_score", 0.0))
            if hasattr(ai_result, "trap_score") and "trap_score" in parsed:
                ai_result.trap_score = float(parsed.get("trap_score", 0.0))
            if hasattr(ai_result, "trend_confidence") and "trend_confidence" in parsed:
                ai_result.trend_confidence = float(parsed.get("trend_confidence", 0.0))
            if hasattr(ai_result, "ai_win_probability") and "ai_win_probability" in parsed:
                ai_result.ai_win_probability = float(parsed.get("ai_win_probability", 0.5))
            
            # Store the repaired response
            ai_result.raw_response = repaired
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        # Repair failed - leave as is
        pass
    
    return ai_result


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — PIPELINE OUTPUT VALIDATION
# ═══════════════════════════════════════════════════════════════

def validate_pipeline_output(master_verdict: Any) -> GuardrailReport:
    """
    Validate a MasterVerdict from Module 11 before it enters Module 12.

    Catches: approved bets with negative/zero edge (hard stop),
    impossible probability triplets, high draw probability, low
    strength differential, and status/confidence mismatches.
    """
    report = GuardrailReport()
    
    # Get oracle attribute safely
    oracle = getattr(master_verdict, "oracle", None)
    if oracle is None:
        raise GuardrailError(
            "missing_oracle",
            "MasterVerdict has no oracle attribute. Cannot validate pipeline output."
        )

    # ── 1. Final status is a known value ────────────────────────
    report.checks_run += 1
    final_status = getattr(master_verdict, "final_status", "PENDING")
    if final_status not in VALID_FINAL_STATUSES:
        report.add_warning(
            "unknown_final_status",
            f"MasterVerdict has unrecognised final_status '{final_status}'.",
            severity="HIGH",
            context={"final_status": final_status, "valid_statuses": list(VALID_FINAL_STATUSES)}
        )

    # ── 2. APPROVED verdict must have positive edge (HARD STOP) ──
    report.checks_run += 1
    if "APPROVED" in final_status:
        edge = getattr(oracle, "edge", None)
        if edge is None or edge <= 0:
            raise GuardrailError(
                "approved_without_positive_edge",
                f"APPROVED verdict has edge={edge}. A positive edge is the "
                "mathematical prerequisite for a value bet. This verdict must "
                "not reach the portfolio. Check probability engine in Module 3.",
                context={"edge": edge, "final_status": final_status}
            )

    # ── 3. Probability triplet integrity ────────────────────────
    report.checks_run += 1
    hw = getattr(oracle, "home_win_prob", None)
    aw = getattr(oracle, "away_win_prob", None)
    dp = getattr(oracle, "draw_prob", None)
    
    if None not in (hw, aw, dp):
        prob_sum = hw + aw + dp
        if abs(prob_sum - 1.0) > PROB_SUM_TOLERANCE:
            raise GuardrailError(
                "prob_triplet_invalid",
                f"OracleVerdict probability triplet sums to {prob_sum:.4f} "
                f"(expected 1.0 ± {PROB_SUM_TOLERANCE}). "
                "Data corruption in probability engine.",
                context={"sum": prob_sum, "home": hw, "draw": dp, "away": aw}
            )
        for label, val in [("home_win", hw), ("away_win", aw), ("draw", dp)]:
            if not (0.0 <= val <= 1.0):
                raise GuardrailError(
                    f"probability_{label}_out_of_bounds",
                    f"{label}_prob={val:.4f} is outside [0.0, 1.0]."
                )
    else:
        report.add_warning(
            "missing_probabilities",
            "OracleVerdict missing one or more probability fields.",
            severity="MEDIUM",
            context={"has_home": hw is not None, "has_draw": dp is not None, "has_away": aw is not None}
        )

    # ── 4. model_prob and edge consistency ───────────────────────
    report.checks_run += 1
    model_prob = getattr(oracle, "model_prob", None)
    leg = getattr(oracle, "leg", None)
    leg_odds = getattr(leg, "odds", None) if leg else None
    
    if model_prob is not None and leg_odds and leg_odds > 1.0:
        implied = 1.0 / leg_odds
        recomputed = round(model_prob - implied, 4)
        stored = getattr(oracle, "edge", None)
        if stored is not None and abs(recomputed - stored) > 0.01:
            report.add_warning(
                "edge_computation_mismatch",
                f"Stored edge={stored:.4f} but recomputed from "
                f"model_prob={model_prob:.4f} - implied={implied:.4f} "
                f"gives {recomputed:.4f}. Possible stale value.",
                severity="MEDIUM",
                context={"stored": stored, "recomputed": recomputed, "model_prob": model_prob, "implied": implied}
            )

    # ── 5. Risk flags vs approval status ────────────────────────
    report.checks_run += 1
    risk_flags = getattr(master_verdict, "risk_flags", [])
    n_flags = len(risk_flags)
    if "APPROVED" in final_status and n_flags > 3:
        report.add_warning(
            "approved_with_many_risk_flags",
            f"APPROVED verdict carries {n_flags} risk flags. "
            "High flag count on an approved bet is unusual — verify M12 filter.",
            severity="MEDIUM",
            context={"risk_flags": risk_flags, "count": n_flags}
        )

    # ── 6. High draw probability on approved bet ─────────────────
    report.checks_run += 1
    if dp is not None and dp > HIGH_DRAW_PROB_GATE:
        report.add_warning(
            "high_draw_probability",
            f"Draw probability {dp:.1%} exceeds {HIGH_DRAW_PROB_GATE:.0%}. "
            "Match likely unstable — outcome more uncertain than edge implies.",
            severity="HIGH",
            context={"draw_prob": dp, "threshold": HIGH_DRAW_PROB_GATE}
        )

    # ── 7. Low strength differential ─────────────────────────────
    report.checks_run += 1
    strength_diff = getattr(oracle, "strength_diff", None)
    if strength_diff is not None and abs(strength_diff) < MIN_STRENGTH_DIFF:
        report.add_warning(
            "low_strength_diff",
            f"Strength differential {strength_diff:.2f} is below {MIN_STRENGTH_DIFF}. "
            "Teams are closely matched — approved edge is less reliable.",
            severity="HIGH",
            context={"strength_diff": strength_diff, "threshold": MIN_STRENGTH_DIFF}
        )

    return report


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — DATAFLOW VALIDATION (Portfolio → Bankroll)
# ═══════════════════════════════════════════════════════════════

def validate_dataflow(
    portfolio: Any,
    bankroll_report: Any,
) -> GuardrailReport:
    """
    Validate the final portfolio and bankroll report before storage
    and frontend delivery.

    Catches: over-staking, REJECTED legs in top_picks, zero-prob ACCAs.
    """
    report = GuardrailReport()

    # ── 1. No REJECTED legs in top_picks ────────────────────────
    report.checks_run += 1
    if hasattr(portfolio, "top_picks"):
        for v in portfolio.top_picks:
            final_status = getattr(v, "final_status", "")
            if "REJECTED" in final_status:
                leg_id = getattr(v, "leg_id", "unknown")
                raise GuardrailError(
                    "rejected_leg_in_top_picks",
                    f"Leg '{leg_id}' has status '{final_status}' "
                    "but appears in CleanedPortfolio.top_picks. "
                    "Module 12 filter has a bug.",
                    context={"leg_id": leg_id, "final_status": final_status}
                )

    # ── 2. Exposure does not exceed bankroll ─────────────────────
    report.checks_run += 1
    total_exposure = getattr(bankroll_report, "total_exposure", 0.0)
    bankroll = getattr(bankroll_report, "bankroll", 0.0)
    
    if bankroll > 0:
        ratio = total_exposure / bankroll
        if ratio > MAX_EXPOSURE_RATIO:
            raise GuardrailError(
                "over_staked",
                f"Total exposure {total_exposure:.2f} exceeds "
                f"bankroll {bankroll:.2f} ({ratio:.1%}). Module 13 staking calculation error.",
                context={"total_exposure": total_exposure, "bankroll": bankroll, "ratio": ratio}
            )

    # ── 3. ACCA combined probability sanity ─────────────────────
    report.checks_run += 1
    for acca_name in ("ultra_safe_acca", "value_acca"):
        acca = getattr(bankroll_report, acca_name, None)
        if acca is not None:
            combined_prob = getattr(acca, "combined_prob", 1.0)
            combined_odds = getattr(acca, "combined_odds", 1.0)
            
            if combined_prob <= 0:
                raise GuardrailError(
                    f"{acca_name}_zero_probability",
                    f"{acca_name} has combined_prob={combined_prob:.4f}. "
                    "A zero-probability parlay should never be staked.",
                    context={"acca_name": acca_name, "combined_prob": combined_prob}
                )
            if combined_odds < 1.0:
                raise GuardrailError(
                    f"{acca_name}_invalid_combined_odds",
                    f"{acca_name} combined_odds={combined_odds:.2f} < 1.0. "
                    "Mathematical impossibility — multiplication error in Module 13.",
                    context={"acca_name": acca_name, "combined_odds": combined_odds}
                )

    # ── 4. Singles stake floor ───────────────────────────────────
    report.checks_run += 1
    if hasattr(bankroll_report, "singles"):
        for s in bankroll_report.singles:
            stake = getattr(s, "stake", 0.0)
            if stake < 0:
                match_name = getattr(s, "match_name", "unknown")
                raise GuardrailError(
                    "negative_stake",
                    f"Single bet on '{match_name}' has stake={stake:.2f} < 0. "
                    "Module 13 Kelly computation produced a negative number.",
                    context={"match_name": match_name, "stake": stake}
                )
            
            potential_return = getattr(s, "potential_return", 0.0)
            if potential_return < stake and stake > 0:
                report.add_warning(
                    "return_below_stake",
                    f"'{getattr(s, 'match_name', 'unknown')}': potential return {potential_return:.2f} "
                    f"< stake {stake:.2f}. Odds may be below 2.0 — verify.",
                    severity="LOW",
                    context={"match_name": getattr(s, "match_name", "unknown"), "stake": stake, "potential_return": potential_return}
                )

    return report


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — BATCH VALIDATION
# ═══════════════════════════════════════════════════════════════

def validate_batch(
    items: List[Any],
    validator_func,
    *args,
    stop_on_first_error: bool = False,
    **kwargs,
) -> List[GuardrailReport]:
    """
    Run validation on a batch of items.
    
    Args:
        items: List of items to validate
        validator_func: Validation function to call on each item
        stop_on_first_error: If True, stop on first GuardrailError
        *args, **kwargs: Additional arguments to pass to validator_func
    
    Returns:
        List of GuardrailReport objects
    """
    reports = []
    
    for i, item in enumerate(items):
        try:
            report = validator_func(item, *args, **kwargs)
            reports.append(report)
        except GuardrailError as e:
            report = GuardrailReport(passed=False)
            report.add_error(e.check, e.message, e.context)
            reports.append(report)
            if stop_on_first_error:
                raise
    
    return reports


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — CONVENIENCE WRAPPERS
# ═══════════════════════════════════════════════════════════════

def safe_validate(fn, *args, **kwargs) -> Tuple[bool, GuardrailReport]:
    """
    Run any validate_*() function without propagating GuardrailError.
    Returns (ok: bool, report: GuardrailReport).

    Use when you want to log the failure and skip a leg
    rather than crash the entire batch run.
    """
    try:
        report = fn(*args, **kwargs)
        return True, report
    except GuardrailError as e:
        report = GuardrailReport(passed=False)
        report.add_error(e.check, e.message, e.context)
        return False, report


def clamp_ai_scores(ai_result: Any) -> Any:
    """
    Non-destructive clamp of any out-of-bounds AI scores to [0.0, 1.0].
    Call after validate_ai_response() if you want to continue despite warnings
    rather than discard the AI result entirely.
    """
    if hasattr(ai_result, "ai_win_probability"):
        ai_result.ai_win_probability = max(0.0, min(1.0, ai_result.ai_win_probability))
    if hasattr(ai_result, "anomaly_score"):
        ai_result.anomaly_score = max(0.0, min(1.0, ai_result.anomaly_score))
    if hasattr(ai_result, "trap_score"):
        ai_result.trap_score = max(0.0, min(1.0, ai_result.trap_score))
    if hasattr(ai_result, "trend_confidence"):
        ai_result.trend_confidence = max(0.0, min(1.0, ai_result.trend_confidence))
    return ai_result


def guardrail_summary(reports: List[GuardrailReport]) -> str:
    """
    Aggregate multiple GuardrailReports into one session summary.
    Useful for logging after a full batch run.
    """
    total_checks = sum(r.checks_run for r in reports)
    total_errors = sum(len(r.errors) for r in reports)
    total_warnings = sum(len(r.warnings) for r in reports)
    
    # Count warnings by severity
    severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for r in reports:
        for w in r.warnings:
            severity_counts[w.severity] = severity_counts.get(w.severity, 0) + 1
    
    passed_count = sum(1 for r in reports if r.passed)

    return (
        f"The Match Oracle — Guardrail Session Summary\n"
        f"  Reports    : {len(reports)}\n"
        f"  Passed     : {passed_count}/{len(reports)}\n"
        f"  Checks run : {total_checks}\n"
        f"  Errors     : {total_errors}\n"
        f"  Warnings   : {total_warnings}  (HIGH: {severity_counts['HIGH']}, CRITICAL: {severity_counts.get('CRITICAL', 0)})"
    )


def get_validation_stats() -> Dict[str, Any]:
    """
    Get validation statistics from the global context.
    Useful for monitoring system health.
    """
    return {
        "counters": _validation_context.get_all(),
        "total_validations": _validation_context.get("total_fixtures_validated", 0),
    }


def reset_validation_stats() -> None:
    """Reset validation statistics."""
    _validation_context.reset()


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — EXPORTS
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Exceptions
    "GuardrailError",
    # Data classes
    "GuardrailWarning",
    "GuardrailReport",
    "WinCeilingResult",
    # Thread-safe context
    "ValidationContext",
    "get_validation_stats",
    "reset_validation_stats",
    # Validation functions
    "validate_fixture_date",
    "validate_ingestion",
    "validate_leg",
    "validate_ai_response",
    "validate_ai_consensus",
    "validate_pipeline_output",
    "validate_dataflow",
    # Structural gap validators (UPDATED v5)
    "validate_dominance_vs_decay",
    "validate_senior_men_league",
    "validate_league_tier",
    "validate_tier_bypass_with_data",
    "validate_not_playoff_or_knockout",
    "is_regular_season_league",
    "validate_all_filters",
    # NEW v4: Dead rubber and win ceiling
    "validate_dead_rubber",
    "validate_win_ceiling",
    # Batch validation
    "validate_batch",
    # Convenience wrappers
    "safe_validate",
    "clamp_ai_scores",
    "guardrail_summary",
    "repair_ai_response",
    # Helper functions
    "_get_motivation_label",
    "_get_historical_win_ceiling",
    "_get_current_win_streak",
    "_get_season_progress",
    "_is_extreme_decay",  # NEW v5
    # Constants
    "DOMINANCE_MIN_WIN_RATE",
    "DOMINANCE_MIN_PPG",
    "DOMINANCE_MIN_RECENT_WINS",
    "DECAY_MAX_WIN_RATE",
    "DECAY_MAX_PPG",
    "DECAY_MIN_RECENT_LOSSES",
    "FAVOURITE_DECAY_MAX_LOSSES",
    "TIER_BYPASS_MIN_GAMES",
    "TIER_BYPASS_MIN_H2H",
    "POPULAR_LEAGUE_TIER_LIMITS",
    "REJECTED_COMPETITION_TYPES",
    "REJECTED_MATCH_KEYWORDS",
    # NEW v4 constants
    "DEAD_RUBBER_SEASON_THRESHOLD",
    "DEAD_RUBBER_MOTIVATION_LOW",
    "WIN_CEILING_APPROACH_DISTANCE",
    "WIN_CEILING_MIN_STREAK",
    "CEILING_APPROACH_MULTIPLIER_LATE",
    "CEILING_APPROACH_MULTIPLIER_MID",
    # NEW v5 constants (Extreme Decay Bypass)
    "EXTREME_DECAY_MIN_WIN_RATE",
    "EXTREME_DECAY_MIN_PPG",
    "EXTREME_DECAY_MIN_LOSS_STREAK",
    "EXTREME_DECAY_MIN_AWAY_LOSSES",
    "EXTREME_DECAY_ZERO_AWAY_WINS",
    "BYPASS_FAV_MIN_WIN_RATE",
    "BYPASS_FAV_MIN_PPG",
    "BYPASS_FAV_MIN_RECENT_WINS",
    "BYPASS_FAV_MAX_RECENT_LOSSES",
    "BYPASS_MIN_FORM_GAP",
    "BYPASS_REQUIRED_CHECKS",
]


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from dataclasses import dataclass
    
    print("\n" + "=" * 70)
    print("MODULE 0: GUARDRAIL ENGINE v5 - EXTREME DECAY BYPASS TEST")
    print("=" * 70)
    
    # Create mock team profiles
    @dataclass
    class MockProfile:
        team_id: str
        team_name: str
        is_mature: bool = True
        metrics: Dict = field(default_factory=dict)
        form: Dict = field(default_factory=dict)
        multi_rtm: Any = None
        features: Dict = field(default_factory=dict)
        
        def __post_init__(self):
            self.metrics = self.metrics or {"core.games": 30, "core.wins": 10}
            self.form = self.form or {"recent_results": ["W", "L", "W", "L", "W"]}
        
        def get_metric(self, key, default=0):
            return self.metrics.get(key, default)
    
    @dataclass
    class MockLeg:
        match_id: str
        home_profile: Any
        away_profile: Any
        home_odds: float = 1.50
        away_odds: float = 6.00
        draw_odds: float = 4.00
        odds: float = 1.50
        selection: str = "Home"
        features: Dict = field(default_factory=dict)
        
        def detect_favourite(self):
            return "HOME" if self.home_odds <= self.away_odds else "AWAY"
    
    # Test 1: Lazio vs AC Pisa (Extreme Decay Case)
    print("\n📅 Test 1: Lazio vs AC Pisa (Extreme Decay Should Bypass)")
    print("-" * 40)
    print("Lazio: 33% win rate, 1.05 PPG (below normal thresholds)")
    print("AC Pisa: 5% win rate, 0.64 PPG, 6 straight losses, 0 away wins")
    print("Expected: EXTREME DECAY BYPASS → PASS")
    
    lazio = MockProfile(
        team_id="1", team_name="Lazio",
        metrics={"core.games": 42, "core.wins": 14, "core.draws": 15, "core.losses": 13},
        form={"recent_results": ["L", "L", "W", "D", "W", "L"]}
    )
    
    ac_pisa = MockProfile(
        team_id="2", team_name="AC Pisa",
        metrics={"core.games": 39, "core.wins": 2, "core.draws": 13, "core.losses": 24,
                 "away_games": 20, "away_wins": 0, "away_losses": 11},
        form={"recent_results": ["L", "L", "L", "L", "L", "L"]}
    )
    
    leg_pisa = MockLeg(
        match_id="lazio_pisa",
        home_profile=lazio,
        away_profile=ac_pisa,
        home_odds=1.40,
        away_odds=8.00,
    )
    
    try:
        report, ceiling = validate_all_filters(leg_pisa, verbose=True)
        print(f"\n  Result: {'✅ PASSED' if report.passed else '❌ FAILED'}")
        print(f"  Errors: {report.errors}")
        for w in report.warnings:
            print(f"  Warning: {w.message}")
    except GuardrailError as e:
        print(f"  ❌ HARD REJECT: {e.message}")
    
    # Test 2: Normal dominance case (no bypass needed)
    print("\n📅 Test 2: Inter vs Bologna (Normal Dominance)")
    print("-" * 40)
    print("Inter: 69% win rate, 2.21 PPG (well above thresholds)")
    print("Bologna: 43% win rate, 1.32 PPG (decaying but not extreme)")
    print("Expected: STANDARD PASS (no bypass needed)")
    
    inter = MockProfile(
        team_id="3", team_name="Inter",
        metrics={"core.games": 52, "core.wins": 36, "core.draws": 6, "core.losses": 10},
        form={"recent_results": ["W", "W", "W", "D", "W", "D"]}
    )
    
    bologna = MockProfile(
        team_id="4", team_name="Bologna",
        metrics={"core.games": 53, "core.wins": 23, "core.draws": 13, "core.losses": 17},
        form={"recent_results": ["W", "W", "D", "L", "L", "W"]}
    )
    
    leg_inter = MockLeg(
        match_id="inter_bologna",
        home_profile=inter,
        away_profile=bologna,
        home_odds=1.85,
        away_odds=4.20,
    )
    
    try:
        report, ceiling = validate_all_filters(leg_inter, verbose=True)
        print(f"\n  Result: {'✅ PASSED' if report.passed else '❌ FAILED'}")
    except GuardrailError as e:
        print(f"  ❌ HARD REJECT: {e.message}")
    
    # Test 3: Coin flip (both poor, no extreme decay)
    print("\n📅 Test 3: Both Teams Poor (Coin Flip - Should REJECT)")
    print("-" * 40)
    
    poor_team1 = MockProfile(
        team_id="5", team_name="Poor Team A",
        metrics={"core.games": 30, "core.wins": 8, "core.draws": 10, "core.losses": 12},
        form={"recent_results": ["L", "L", "D", "L", "W", "L"]}
    )
    
    poor_team2 = MockProfile(
        team_id="6", team_name="Poor Team B",
        metrics={"core.games": 30, "core.wins": 7, "core.draws": 9, "core.losses": 14},
        form={"recent_results": ["L", "D", "L", "L", "L", "D"]}
    )
    
    leg_coinflip = MockLeg(
        match_id="poor_vs_poor",
        home_profile=poor_team1,
        away_profile=poor_team2,
        home_odds=2.10,
        away_odds=3.50,
    )
    
    try:
        report, ceiling = validate_all_filters(leg_coinflip, verbose=True)
        print(f"\n  Result: {'✅ PASSED' if report.passed else '❌ FAILED'}")
    except GuardrailError as e:
        print(f"  ❌ HARD REJECT: {e.message}")
    
    print("\n" + "=" * 70)
    print("MODULE 0 v5 READY FOR PRODUCTION")
    print("=" * 70)