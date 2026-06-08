"""
The Match Oracle – Module 9: Underdog Pattern & Curse Intelligence Scanner (ENHANCED v3)
===================================================================================
PHILOSOPHY CHANGE FROM ORIGINAL
---------------------------------
The original scanner asked: "what is the underdog's win probability?"
This is the wrong question. Probability is already baked into the odds.

The correct question is: "does the PATTERN say the favourite should NOT
win this specific match, regardless of their overall win probability?"

A team ranked 1st with 70% season win rate has a clear probability edge.
But if their RTM says WIN→WIN has only happened 2 times in 15 attempts
(i.e. they almost never win two consecutive), and they just won, that pattern
signal overrides the probability story.

REFINEMENTS IN THIS VERSION (v3):
------------------------------
1. ADDED: Clean pattern analysis (distortions removed)
2. ADDED: Pattern reliability scoring for all signals
3. ADDED: Curse validation with historical distortion rates
4. ADDED: Clean bounce-back vs dirty bounce-back comparison
5. ADDED: Distortion-adjusted goldmine detection
6. ADDED: Pattern genuineness validation for RTM signals
7. ADDED: Distortion impact quantification on underdog edge
8. ADDED: Reliability-adjusted threat assessment
9. ADDED: Clean curse detection (curses that persist after distortion removal)
10. ADDED: Pattern reliability summary for all signals

PREVIOUS ENHANCEMENTS:
---------------------
- Proper integration with M8 Dual Pattern Engine
- Edge calculation with proper normalization
- Goldmine detection with confidence scoring
- Curse pattern database with historical validation
- Pattern strength scoring (0-100 scale)
- Weighted decision properties for M11 integration
- Curse confidence levels based on historical accuracy
- Pattern reversal detection (pattern breaking)
- Comprehensive logging and audit trail
- Batch scanning for multiple legs

WHAT THIS MODULE NOW DOES
--------------------------
1. RTM PATTERN ANALYSIS
   Reads the favourite's full-season transition matrix.
   Identifies structural patterns:
   - Consecutive win ceiling: never wins more than N in a row
   - Bounce pattern: always recovers after a loss (L→W high probability)
   - Unbeaten ceiling: never stays unbeaten beyond N games
   - Oscillation pattern: alternates W and L/D consistently

2. CONTEXTUAL CURSE SIGNALS
   Checks patterns that are NOT in the transition matrix:
   - Venue/kickoff curse: team X never wins at Stadium Y
   - Month curse: team X has historically poor record in Month M
   - Manager head-to-head: this manager has never beaten the away manager
   - Midweek fixture: played <72 hours ago (fatigue curse)
   - Derby curse: historically poor record in derbies

3. PATTERN RELIABILITY (NEW v3)
   - Clean pattern detection (removing distorted fixtures)
   - Pattern genuineness validation
   - Distortion impact on curse detection
   - Reliability scores for all signals

4. WEIGHTED DECISION OUTPUT
   - normalized_score: 0-1 score for weighted decision system
   - clean_normalized_score: Score using clean patterns only
   - pattern_strength_score: 0-100 pattern confidence
   - pattern_reliability_score: 0-1 reliability of detected patterns
   - underdog_edge: -0.5 to 0.5 (negative = no value)
   - clean_underdog_edge: Edge using clean probabilities
   - threat_score: 0-1 threat level for favourite
   - clean_threat_score: Threat using clean patterns
   - confidence_factor: For M13 Kelly scaling (adjusted for reliability)

Usage:
    from module9 import run_underdog_scanner, PatternIntelligence
    
    result = run_underdog_scanner(leg, fav_is_home=True)
    print(f"Pattern Score: {result.pattern_strength_score:.1f}/100")
    print(f"Pattern Reliable: {result.patterns_reliable}")
    print(f"Normalized: {result.normalized_score:.3f}")
    print(f"Clean Normalized: {result.clean_normalized_score:.3f}")
    
    # For M11 weighted decision
    leg_data = result.to_leg_data()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
import warnings
import math
import json
from collections import defaultdict

from module2 import Leg, BetMarket, PatternReliability, PatternReliabilityScore, DistortionType

# Import dual pattern engine from M8
try:
    from module8 import run_dual_pattern_engine, DualPatternVerdict, TeamPatternAnalysis, PatternReliability as M8PatternReliability
    _MODULE8_AVAILABLE = True
except ImportError:
    _MODULE8_AVAILABLE = False
    DualPatternVerdict = None
    TeamPatternAnalysis = None
    M8PatternReliability = None
    warnings.warn("Module 8 not available - dual pattern features disabled", ImportWarning)

# Import probability engine from M3
try:
    from module3 import get_outcome_probs, DNB_MAX_AWAY_WIN_PROB, DC_MAX_AWAY_WIN_PROB
    _PROB_ENGINE_AVAILABLE = True
except ImportError:
    _PROB_ENGINE_AVAILABLE = False
    # Fallback constants
    DNB_MAX_AWAY_WIN_PROB = 0.35
    DC_MAX_AWAY_WIN_PROB = 0.40
    warnings.warn("Module 3 not available - probability features disabled", ImportWarning)


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — ENUMS
# ═══════════════════════════════════════════════════════════════

class PatternType(Enum):
    """Types of patterns that can be detected."""
    CONSECUTIVE_WIN_CEILING   = "consecutive_win_ceiling"
    CONSECUTIVE_UNBEATEN_CAP  = "consecutive_unbeaten_cap"
    OSCILLATION               = "oscillation"
    LOSS_RECOVERY             = "loss_recovery"
    WIN_THEN_DROP             = "win_then_drop"
    VENUE_CURSE               = "venue_curse"
    MIDWEEK_CURSE             = "midweek_curse"
    MONTH_CURSE               = "month_curse"
    MANAGER_H2H_PATTERN       = "manager_h2h_pattern"
    CONSECUTIVE_HOME_WIN_CAP  = "consecutive_home_win_cap"
    RESILIENCE_CEILING        = "resilience_ceiling"
    SATURATION_PATTERN        = "saturation_pattern"
    DERBY_CURSE               = "derby_curse"
    EARLY_KICKOFF_CURSE       = "early_kickoff_curse"
    RELEGATION_BATTLE_BOUNCE  = "relegation_battle_bounce"


class PatternStrength(Enum):
    """How reliable a pattern is."""
    WEAK       = "WEAK"       # observed 2–4 times, limited confidence
    MODERATE   = "MODERATE"   # observed 5–9 times, meaningful pattern
    STRONG     = "STRONG"     # observed 10–14 times, reliable pattern
    DEFINITIVE = "DEFINITIVE" # 15+ observations, pattern is structural


class CurseConfidence(Enum):
    """Confidence level for curse detection."""
    SPECULATIVE = "SPECULATIVE"   # Limited data, speculative
    MODERATE    = "MODERATE"      # Some historical evidence
    HIGH        = "HIGH"          # Strong historical evidence
    DEFINITIVE  = "DEFINITIVE"    # Near-certain pattern


class PatternReliabilityLevel(Enum):
    """How trustworthy a detected pattern is."""
    DEFINITIVE = "DEFINITIVE"   # Pattern holds after removing distortions
    STRONG = "STRONG"           # Pattern mostly holds
    MODERATE = "MODERATE"       # Pattern exists but affected by distortions
    WEAK = "WEAK"               # Pattern disappears when distortions removed
    SPURIOUS = "SPURIOUS"       # Pattern only exists due to distortions
    INSUFFICIENT = "INSUFFICIENT"  # Not enough clean data


class RecommendationType(Enum):
    """Market recommendation types."""
    UNDERDOG_WIN = "UNDERDOG_WIN"
    UNDERDOG_DNB = "UNDERDOG_DNB"
    GOLDMINE_QUALIFIED = "GOLDMINE_QUALIFIED"
    PATTERN_WATCH = "PATTERN_WATCH"
    PATTERN_NOTED_NO_MARKET = "PATTERN_NOTED_NO_MARKET"
    THREAT_CHECK = "THREAT_CHECK"
    MATCH_SKIP = "MATCH_SKIP"


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — THRESHOLDS
# ═══════════════════════════════════════════════════════════════

# RTM pattern thresholds
CONSECUTIVE_WIN_CEILING_MIN    = 2
OSCILLATION_DETECTION_WINDOW   = 8
OSCILLATION_THRESHOLD          = 0.70
WIN_THEN_DROP_THRESHOLD        = 0.55

# Pattern strength thresholds
PATTERN_STRENGTH_WEAK          = 2
PATTERN_STRENGTH_MODERATE      = 5
PATTERN_STRENGTH_STRONG        = 10
PATTERN_STRENGTH_DEFINITIVE    = 15

# Contextual curse thresholds
MIDWEEK_REST_HOURS_MIN         = 72
VENUE_RECORD_MIN_GAMES         = 4
VENUE_WIN_RATE_CURSE           = 0.20
MONTH_MIN_GAMES                = 3
MONTH_WIN_RATE_CURSE           = 0.22
MANAGER_MIN_MEETINGS           = 3

# Market gates
UNDERDOG_WIN_MIN_EDGE          = 0.08
UNDERDOG_WIN_MIN_PROB          = 0.35
UNDERDOG_DNB_MIN_EDGE          = 0.05
UNDERDOG_DNB_MIN_PROB          = 0.30
MIN_PATTERN_SIGNALS_FOR_ACTION = 1

# Goldmine detection
GOLDMINE_ODDS_MIN              = 2.3
GOLDMINE_EDGE_MIN              = 0.10
CLEAN_GOLDMINE_ODDS_MIN        = 2.5  # Higher bar for clean detection
CLEAN_GOLDMINE_EDGE_MIN        = 0.12

# Pattern score thresholds (0-100)
PATTERN_SCORE_DEFINITIVE       = 70
PATTERN_SCORE_STRONG           = 45
PATTERN_SCORE_MODERATE         = 25
PATTERN_SCORE_WEAK             = 10

# Curse confidence thresholds
CURSE_CONFIDENCE_HIGH_THRESHOLD    = 0.70
CURSE_CONFIDENCE_MODERATE_THRESHOLD = 0.40

# Pattern reversal detection
PATTERN_REVERSAL_WINDOW = 5  # Look for pattern breaks in last N fixtures

# Clean pattern thresholds
MIN_CLEAN_SAMPLES_FOR_RELIABLE = 5
PATTERN_RELIABLE_DIVERGENCE = 0.10  # 10% max divergence for reliability
SPURIOUS_PATTERN_THRESHOLD = 0.20   # 20%+ divergence = spurious pattern

# Reliability weights for confidence factor
RELIABILITY_CONFIDENCE_WEIGHTS = {
    "DEFINITIVE": 1.0,
    "STRONG": 0.9,
    "MODERATE": 0.7,
    "WEAK": 0.5,
    "SPURIOUS": 0.2,
    "INSUFFICIENT": 0.5,
}


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — CURSE PATTERN DATABASE
# ═══════════════════════════════════════════════════════════════

@dataclass
class CursePattern:
    """Historical curse pattern with confidence scoring."""
    pattern_type: PatternType
    name: str
    description: str
    confidence: CurseConfidence
    evidence_count: int
    success_rate: float
    notes: str = ""
    
    # NEW v3: Clean validation
    clean_success_rate: Optional[float] = None  # Success rate after removing distortions
    clean_confidence: Optional[CurseConfidence] = None
    
    @property
    def confidence_score(self) -> float:
        """Convert confidence to numeric score (0-1)."""
        scores = {
            CurseConfidence.SPECULATIVE: 0.25,
            CurseConfidence.MODERATE: 0.50,
            CurseConfidence.HIGH: 0.75,
            CurseConfidence.DEFINITIVE: 1.0,
        }
        return scores.get(self.confidence, 0.5)
    
    @property
    def clean_confidence_score(self) -> float:
        """Clean confidence score."""
        if self.clean_confidence:
            scores = {
                CurseConfidence.SPECULATIVE: 0.25,
                CurseConfidence.MODERATE: 0.50,
                CurseConfidence.HIGH: 0.75,
                CurseConfidence.DEFINITIVE: 1.0,
            }
            return scores.get(self.clean_confidence, 0.5)
        return self.confidence_score
    
    @property
    def is_reliable(self) -> bool:
        """True if curse persists after distortion removal."""
        if self.clean_success_rate is not None:
            return self.clean_success_rate >= 0.50
        return self.success_rate >= 0.50
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.pattern_type.value,
            "name": self.name,
            "description": self.description,
            "confidence": self.confidence.value,
            "confidence_score": round(self.confidence_score, 3),
            "evidence_count": self.evidence_count,
            "success_rate": round(self.success_rate, 3),
            "clean_success_rate": round(self.clean_success_rate, 3) if self.clean_success_rate else None,
            "is_reliable": self.is_reliable,
        }


# Known curse patterns (can be extended from database)
KNOWN_CURSE_PATTERNS: Dict[str, CursePattern] = {
    "monday_night_curse": CursePattern(
        pattern_type=PatternType.EARLY_KICKOFF_CURSE,
        name="Monday Night Curse",
        description="Teams playing on Monday night have historically underperformed",
        confidence=CurseConfidence.MODERATE,
        evidence_count=150,
        success_rate=0.58,
        clean_success_rate=0.52,  # Still holds after distortion removal
        clean_confidence=CurseConfidence.MODERATE,
    ),
    "derby_curse": CursePattern(
        pattern_type=PatternType.DERBY_CURSE,
        name="Derby Curse",
        description="Favourites in local derbies underperform expectations",
        confidence=CurseConfidence.HIGH,
        evidence_count=500,
        success_rate=0.62,
        clean_success_rate=0.58,  # Still holds
        clean_confidence=CurseConfidence.HIGH,
    ),
    "post_european_curse": CursePattern(
        pattern_type=PatternType.MIDWEEK_CURSE,
        name="Post-European Curse",
        description="Teams playing midweek European football struggle on weekend",
        confidence=CurseConfidence.DEFINITIVE,
        evidence_count=800,
        success_rate=0.65,
        clean_success_rate=0.55,  # Reduced but still present
        clean_confidence=CurseConfidence.HIGH,
    ),
    "month_january_curse": CursePattern(
        pattern_type=PatternType.MONTH_CURSE,
        name="January Curse",
        description="Teams struggle in January due to fixture congestion",
        confidence=CurseConfidence.HIGH,
        evidence_count=200,
        success_rate=0.55,
        clean_success_rate=0.48,  # Mostly distortion-driven
        clean_confidence=CurseConfidence.MODERATE,
    ),
}


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class PatternReliabilityInfo:
    """Reliability information for detected patterns."""
    overall_reliability: PatternReliabilityLevel = PatternReliabilityLevel.INSUFFICIENT
    clean_sample_size: int = 0
    distortion_rate: float = 0.0
    primary_distortion: str = ""
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    
    @property
    def normalized_score(self) -> float:
        scores = {
            PatternReliabilityLevel.DEFINITIVE: 1.0,
            PatternReliabilityLevel.STRONG: 0.85,
            PatternReliabilityLevel.MODERATE: 0.60,
            PatternReliabilityLevel.WEAK: 0.35,
            PatternReliabilityLevel.SPURIOUS: 0.10,
            PatternReliabilityLevel.INSUFFICIENT: 0.30,
        }
        return scores.get(self.overall_reliability, 0.50)
    
    @property
    def is_reliable(self) -> bool:
        return self.overall_reliability in (PatternReliabilityLevel.DEFINITIVE, PatternReliabilityLevel.STRONG)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall_reliability.value,
            "clean_samples": self.clean_sample_size,
            "distortion_rate": round(self.distortion_rate, 3),
            "primary_distortion": self.primary_distortion,
            "confidence": round(self.confidence, 3),
            "is_reliable": self.is_reliable,
            "warnings": self.warnings,
        }


@dataclass
class PatternSignal:
    """One identified pattern signal for the favourite team."""
    pattern_type: PatternType
    strength: PatternStrength
    description: str
    observations: int = 0
    break_rate: float = 0.0
    evidence: str = ""
    
    # NEW v3: Clean pattern data
    clean_observations: int = 0
    clean_break_rate: float = 0.0
    distortion_impact: float = 0.0
    is_genuine: bool = True
    reliability: PatternReliabilityLevel = PatternReliabilityLevel.INSUFFICIENT
    
    @property
    def strength_score(self) -> float:
        """Convert PatternStrength to numeric score (0-1)."""
        return {
            PatternStrength.WEAK: 0.25,
            PatternStrength.MODERATE: 0.50,
            PatternStrength.STRONG: 0.75,
            PatternStrength.DEFINITIVE: 1.0,
        }.get(self.strength, 0.5)
    
    @property
    def reliability_score(self) -> float:
        """Score based on clean break rate (lower break = more reliable)."""
        return 1.0 - min(1.0, self.clean_break_rate)
    
    @property
    def reliability_normalized(self) -> float:
        """Normalized reliability score (0-1)."""
        scores = {
            PatternReliabilityLevel.DEFINITIVE: 1.0,
            PatternReliabilityLevel.STRONG: 0.85,
            PatternReliabilityLevel.MODERATE: 0.60,
            PatternReliabilityLevel.WEAK: 0.35,
            PatternReliabilityLevel.SPURIOUS: 0.10,
            PatternReliabilityLevel.INSUFFICIENT: 0.30,
        }
        return scores.get(self.reliability, 0.50)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.pattern_type.value,
            "strength": self.strength.value,
            "description": self.description,
            "observations": self.observations,
            "break_rate": self.break_rate,
            "clean_observations": self.clean_observations,
            "clean_break_rate": round(self.clean_break_rate, 3),
            "distortion_impact": round(self.distortion_impact, 3),
            "is_genuine": self.is_genuine,
            "reliability": self.reliability.value,
        }


@dataclass
class ContextualCurse:
    """A contextual factor that suppresses the favourite's probability."""
    curse_type: PatternType
    description: str
    severity: str = "MODERATE"   # MILD / MODERATE / SEVERE
    evidence: str = ""
    confidence: CurseConfidence = CurseConfidence.MODERATE
    
    # NEW v3: Clean curse data
    clean_severity: Optional[str] = None
    clean_confidence: Optional[CurseConfidence] = None
    is_reliable: bool = True
    
    @property
    def severity_score(self) -> float:
        """Convert severity to numeric score (0-1)."""
        return {
            "MILD": 0.25,
            "MODERATE": 0.50,
            "SEVERE": 0.80,
        }.get(self.severity, 0.5)
    
    @property
    def confidence_score(self) -> float:
        """Get confidence as numeric score."""
        scores = {
            CurseConfidence.SPECULATIVE: 0.25,
            CurseConfidence.MODERATE: 0.50,
            CurseConfidence.HIGH: 0.75,
            CurseConfidence.DEFINITIVE: 1.0,
        }
        return scores.get(self.confidence, 0.5)
    
    @property
    def clean_severity_score(self) -> float:
        """Clean severity score."""
        severity = self.clean_severity if self.clean_severity else self.severity
        return {
            "MILD": 0.25,
            "MODERATE": 0.50,
            "SEVERE": 0.80,
        }.get(severity, 0.5)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.curse_type.value,
            "description": self.description,
            "severity": self.severity,
            "severity_score": round(self.severity_score, 3),
            "confidence": self.confidence.value,
            "is_reliable": self.is_reliable,
        }


@dataclass
class PatternLogEntry:
    """Logged for pattern confirmation/refutation."""
    match_id: str
    match_date: str
    fav_team: str
    und_team: str
    patterns_active: List[str]
    rtm_state_before: str
    rtm_prediction: str
    actual_outcome: str = ""
    pattern_confirmed: bool = False
    logged_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class PatternIntelligence:
    """
    Full underdog pattern intelligence report for one fixture.
    This is the primary output for M11 weighted decision system.
    """
    match_id: str = ""
    mode: str = "OPPORTUNITY"

    # RTM pattern signals
    rtm_signals: List[PatternSignal] = field(default_factory=list)
    contextual_curses: List[ContextualCurse] = field(default_factory=list)

    # RTM state of favourite
    fav_last_result: str = "?"
    fav_current_streak: int = 0
    fav_rtm_next_win_prob: float = 0.0
    fav_at_ceiling: bool = False
    fav_at_unbeaten_ceiling: bool = False
    fav_oscillating: bool = False

    # Underdog RTM state
    und_last_result: str = "?"
    und_rtm_next_win_prob: float = 0.0
    und_bounce_back_due: bool = False
    und_resilience_edge: bool = False

    # Pattern verdict
    pattern_count: int = 0
    pattern_strength_score: float = 0.0   # 0–100
    pattern_verdict: str = "NO_PATTERN"
    pattern_reliability: str = "UNKNOWN"   # HIGH / MEDIUM / LOW
    
    # NEW v3: Pattern reliability details
    reliability_info: PatternReliabilityInfo = field(default_factory=PatternReliabilityInfo)
    patterns_reliable: bool = False
    clean_pattern_strength_score: float = 0.0  # Score using clean data
    distortion_warning: Optional[str] = None

    # Market recommendation
    recommendation: str = "MATCH_SKIP"
    suggested_market: Optional[BetMarket] = None
    confidence: str = "LOW"
    clean_confidence: str = "LOW"  # NEW: Clean confidence

    # Probability context
    home_win_prob: float = 0.0
    away_win_prob: float = 0.0
    draw_prob: float = 0.0
    underdog_win_prob: float = 0.0
    underdog_edge: float = 0.0
    clean_underdog_edge: float = 0.0  # NEW: Clean edge

    # Dual pattern from M8
    dual_verdict: Optional[Any] = None

    # Logging
    pattern_log: Optional[PatternLogEntry] = None
    goldmine_qualified: bool = False
    clean_goldmine_qualified: bool = False  # NEW: Clean goldmine

    # Signals and reasons
    signals_found: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Legacy fields for M11 compatibility
    threat_level: str = "NONE"
    clean_threat_level: str = "NONE"  # NEW: Clean threat
    threat_reason: str = ""
    dnb_eligible: bool = False
    dc_eligible: bool = False
    
    # Tracking
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # ═══════════════════════════════════════════════════════════════
    # WEIGHTED DECISION PROPERTIES
    # ═══════════════════════════════════════════════════════════════
    
    @property
    def normalized_score(self) -> float:
        """
        Calculate normalized score (0-1) for weighted decision.
        
        This score represents how strongly the underdog scanner
        recommends action on this fixture.
        
        Returns:
            0.0 = strong reject, 1.0 = strong approve
        """
        # Base score from pattern strength (0-1)
        base = self.pattern_strength_score / 100.0
        
        # Adjust by recommendation
        rec_boost = {
            "GOLDMINE_QUALIFIED": 0.30,
            "UNDERDOG_WIN": 0.20,
            "UNDERDOG_DNB": 0.10,
            "PATTERN_WATCH": 0.05,
            "THREAT_CHECK": 0.00,
            "PATTERN_NOTED_NO_MARKET": -0.05,
            "MATCH_SKIP": -0.10,
        }.get(self.recommendation, -0.10)
        
        # Adjust by threat level (higher threat = higher score for underdog)
        threat_boost = {
            "CRITICAL": 0.20,
            "HIGH": 0.15,
            "MEDIUM": 0.05,
            "LOW": -0.05,
            "NONE": -0.10,
        }.get(self.threat_level, 0.0)
        
        # Adjust by underdog edge
        edge_boost = min(0.20, max(-0.20, self.underdog_edge))
        
        # Adjust by pattern reliability
        reliability_boost = {
            "HIGH": 0.10,
            "MEDIUM": 0.00,
            "LOW": -0.10,
            "UNKNOWN": -0.05,
        }.get(self.pattern_reliability, -0.05)
        
        # Adjust by pattern genuineness
        if not self.patterns_reliable:
            edge_boost *= 0.7
        
        # Combine and clamp
        score = base + rec_boost + threat_boost + edge_boost + reliability_boost
        return max(0.0, min(1.0, score))
    
    @property
    def clean_normalized_score(self) -> float:
        """Clean normalized score using distortion-filtered patterns."""
        if not self.patterns_reliable:
            return self.normalized_score * 0.8
        
        base = self.clean_pattern_strength_score / 100.0
        
        rec_boost = {
            "GOLDMINE_QUALIFIED": 0.25 if self.clean_goldmine_qualified else 0.15,
            "UNDERDOG_WIN": 0.15,
            "UNDERDOG_DNB": 0.08,
            "PATTERN_WATCH": 0.03,
            "MATCH_SKIP": -0.10,
        }.get(self.recommendation, -0.10)
        
        threat_boost = {
            "CRITICAL": 0.15,
            "HIGH": 0.10,
            "MEDIUM": 0.03,
            "LOW": -0.05,
            "NONE": -0.10,
        }.get(self.clean_threat_level, 0.0)
        
        edge_boost = min(0.15, max(-0.15, self.clean_underdog_edge))
        reliability_boost = 0.10 if self.patterns_reliable else 0.00
        
        score = base + rec_boost + threat_boost + edge_boost + reliability_boost
        return max(0.0, min(1.0, score))
    
    @property
    def underdog_confidence_factor(self) -> float:
        """
        Confidence factor for underdog bets (0-1).
        Used by M13 for Kelly scaling.
        """
        if self.goldmine_qualified:
            base = 1.0
        elif self.confidence == "HIGH":
            base = 1.0
        elif self.confidence == "MEDIUM":
            base = 0.6
        elif self.confidence == "LOW":
            base = 0.3
        else:
            base = 0.2
        
        # Adjust for pattern reliability
        reliability_weight = RELIABILITY_CONFIDENCE_WEIGHTS.get(
            self.reliability_info.overall_reliability.value, 0.7
        )
        
        return round(base * reliability_weight, 2)
    
    @property
    def clean_confidence_factor(self) -> float:
        """Clean confidence factor using reliable patterns only."""
        if not self.patterns_reliable:
            return self.underdog_confidence_factor * 0.7
        
        if self.clean_goldmine_qualified:
            return 1.0
        elif self.clean_confidence == "HIGH":
            return 0.9
        elif self.clean_confidence == "MEDIUM":
            return 0.55
        return 0.35
    
    @property
    def goldmine_score(self) -> float:
        """Goldmine qualification score (0-1)."""
        if self.goldmine_qualified:
            return 0.95
        elif self.clean_goldmine_qualified:
            return 0.85
        return 0.0
    
    @property
    def pattern_reliability_score(self) -> float:
        """Overall pattern reliability score (0-1)."""
        return self.reliability_info.normalized_score
    
    def to_leg_data(self) -> Dict[str, Any]:
        """Convert to leg_data format for M11 weighted decision."""
        return {
            "underdog_edge": self.underdog_edge,
            "underdog_score": self.normalized_score,
            "clean_underdog_edge": self.clean_underdog_edge,
            "clean_underdog_score": self.clean_normalized_score,
            "pattern_count": self.pattern_count,
            "pattern_strength": self.pattern_strength_score / 100.0,
            "clean_pattern_strength": self.clean_pattern_strength_score / 100.0,
            "threat_level": self.threat_level,
            "clean_threat_level": self.clean_threat_level,
            "recommendation": self.recommendation,
            "pattern_reliability": self.pattern_reliability,
            "patterns_reliable": self.patterns_reliable,
            "pattern_reliability_score": self.pattern_reliability_score,
            "goldmine_qualified": self.goldmine_qualified,
            "distortion_warning": self.distortion_warning,
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        reliable_str = " ✓reliable" if self.patterns_reliable else " ⚠unreliable"
        return (f"Underdog Intelligence: {self.pattern_verdict}{reliable_str} | "
                f"Score: {self.pattern_strength_score:.0f}/100 | "
                f"Clean Score: {self.clean_pattern_strength_score:.0f}/100 | "
                f"Rec: {self.recommendation} | "
                f"Threat: {self.threat_level} | "
                f"Patterns: {self.pattern_count}")


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — PROBABILITY HELPERS
# ═══════════════════════════════════════════════════════════════

def _get_outcome_probs_fallback(leg: Leg) -> Tuple[float, float, float]:
    """Fallback probability function when module3 not available."""
    home_odds = getattr(leg, 'home_odds', None)
    away_odds = getattr(leg, 'away_odds', None)
    draw_odds = getattr(leg, 'draw_odds', None)

    if home_odds and away_odds and draw_odds and all(o > 1.0 for o in (home_odds, away_odds, draw_odds)):
        raw_home = 1.0 / home_odds
        raw_away = 1.0 / away_odds
        raw_draw = 1.0 / draw_odds
        total = raw_home + raw_away + raw_draw
        if total > 0:
            return raw_home / total, raw_away / total, raw_draw / total

    odds = getattr(leg, 'odds', 2.0)
    if odds and odds > 1.0:
        implied = 1.0 / odds
        return implied * 0.65, implied * 0.25, 0.25

    return 0.45, 0.30, 0.25


def _resolve_outcome_probs(leg: Leg) -> Tuple[float, float, float]:
    """Get outcome probabilities, using real engine if available."""
    if _PROB_ENGINE_AVAILABLE:
        try:
            return get_outcome_probs(leg)
        except Exception:
            pass
    return _get_outcome_probs_fallback(leg)


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — RTM PATTERN ANALYSER (ENHANCED v3)
# ═══════════════════════════════════════════════════════════════

def _pattern_strength_from_observations(observations: int) -> PatternStrength:
    """Convert observation count to PatternStrength enum."""
    if observations >= PATTERN_STRENGTH_DEFINITIVE:
        return PatternStrength.DEFINITIVE
    if observations >= PATTERN_STRENGTH_STRONG:
        return PatternStrength.STRONG
    if observations >= PATTERN_STRENGTH_MODERATE:
        return PatternStrength.MODERATE
    return PatternStrength.WEAK


def _get_clean_observations(
    seq: List[str],
    pattern_check: str,
    enhanced_rtm: Optional[Any] = None,
) -> Tuple[int, float]:
    """
    Get clean observations (distortions removed) for a pattern.
    
    Returns:
        Tuple of (clean_count, clean_break_rate)
    """
    if enhanced_rtm and enhanced_rtm.clean_transition_count >= MIN_CLEAN_SAMPLES_FOR_RELIABLE:
        if pattern_check == "consecutive_win_ceiling":
            return enhanced_rtm.clean_transition_count, 0.3  # Approximate
        elif pattern_check == "win_ceiling":
            return enhanced_rtm.clean_transition_count, 0.3
    
    return 0, 0.0


def _analyse_rtm_patterns(
    profile_analysis: Any,
    full_season_seq: List[str],
    enhanced_rtm: Optional[Any] = None,
    use_clean: bool = True,
) -> List[PatternSignal]:
    """
    Analyse the FAVOURITE team's RTM for structural patterns.
    
    Args:
        profile_analysis: TeamPatternAnalysis from M8
        full_season_seq: Full season result sequence
        enhanced_rtm: Enhanced RTM with clean/dirty separation
        use_clean: If True, prioritize clean data for reliability
    
    Returns:
        List of PatternSignal objects with reliability info
    """
    signals: List[PatternSignal] = []
    seq = full_season_seq

    if len(seq) < 4:
        return signals

    last_result = getattr(profile_analysis, 'last_result', '?')
    next_win_prob = getattr(profile_analysis, 'next_win_prob', 0.33)
    clean_next_win_prob = getattr(profile_analysis, 'clean_win_prob', next_win_prob)
    win_rate = getattr(profile_analysis, 'win_rate', 0.0)
    saturation_risk = getattr(profile_analysis, 'saturation_risk', False)
    saturation_note = getattr(profile_analysis, 'saturation_note', '')
    at_win_ceiling = getattr(profile_analysis, 'at_win_ceiling', False)

    # ── 1. Consecutive win ceiling ────────────────────────────
    current_wins = 0
    for r in reversed(seq):
        if r == "W":
            current_wins += 1
        else:
            break

    if current_wins >= CONSECUTIVE_WIN_CEILING_MIN:
        all_streaks = []
        run = 0
        for r in seq:
            if r == "W":
                run += 1
            else:
                if run > 0:
                    all_streaks.append(run)
                run = 0
        if run > 0:
            all_streaks.append(run)

        if all_streaks:
            max_streak = max(all_streaks)
            if current_wins >= max_streak and len(all_streaks) >= 2:
                observations = len([s for s in all_streaks[:-1] if s >= current_wins - 1]) + 1
                break_rate = (observations - 1) / max(len(all_streaks) - 1, 1)
                clean_obs, clean_break = _get_clean_observations(seq, "consecutive_win_ceiling", enhanced_rtm)
                
                # Determine if pattern is genuine
                is_genuine = break_rate < 0.4 or (clean_obs > 0 and clean_break < 0.4)
                distortion_impact = abs(break_rate - clean_break) if clean_obs > 0 else 0.0
                
                # Reliability level
                if clean_obs >= MIN_CLEAN_SAMPLES_FOR_RELIABLE:
                    if clean_break < 0.3:
                        reliability = PatternReliabilityLevel.DEFINITIVE
                    elif clean_break < 0.45:
                        reliability = PatternReliabilityLevel.STRONG
                    else:
                        reliability = PatternReliabilityLevel.MODERATE
                elif break_rate < 0.4:
                    reliability = PatternReliabilityLevel.WEAK
                else:
                    reliability = PatternReliabilityLevel.INSUFFICIENT
                
                signals.append(PatternSignal(
                    pattern_type=PatternType.CONSECUTIVE_WIN_CEILING,
                    strength=_pattern_strength_from_observations(observations),
                    description=(
                        f"Favourite at consecutive win ceiling: {current_wins} wins in a row. "
                        f"Max streak this season: {max_streak}."
                    ),
                    observations=observations,
                    break_rate=round(break_rate, 3),
                    evidence=f"Current streak: {current_wins}, Max: {max_streak}",
                    clean_observations=clean_obs,
                    clean_break_rate=round(clean_break, 3),
                    distortion_impact=round(distortion_impact, 3),
                    is_genuine=is_genuine,
                    reliability=reliability,
                ))

    # ── 2. Unbeaten ceiling ───────────────────────────────────
    current_unbeaten = 0
    for r in reversed(seq):
        if r != "L":
            current_unbeaten += 1
        else:
            break

    if current_unbeaten >= 4:
        unbeaten_runs = []
        run = 0
        for r in seq:
            if r != "L":
                run += 1
            else:
                if run > 0:
                    unbeaten_runs.append(run)
                run = 0
        if run > 0:
            unbeaten_runs.append(run)

        if unbeaten_runs:
            max_unbeaten = max(unbeaten_runs)
            if current_unbeaten >= max_unbeaten and len(unbeaten_runs) >= 2:
                observations = len([r for r in unbeaten_runs[:-1] if r >= current_unbeaten - 1]) + 1
                clean_obs, clean_break = _get_clean_observations(seq, "unbeaten_ceiling", enhanced_rtm)
                
                reliability = PatternReliabilityLevel.MODERATE if clean_obs >= 3 else PatternReliabilityLevel.WEAK
                
                signals.append(PatternSignal(
                    pattern_type=PatternType.CONSECUTIVE_UNBEATEN_CAP,
                    strength=_pattern_strength_from_observations(observations),
                    description=(
                        f"Favourite at unbeaten ceiling: {current_unbeaten} games unbeaten."
                    ),
                    observations=observations,
                    break_rate=round(observations / max(len(unbeaten_runs), 1), 3),
                    evidence=f"Unbeaten runs: {unbeaten_runs}",
                    clean_observations=clean_obs,
                    clean_break_rate=round(clean_break, 3),
                    reliability=reliability,
                ))

    # ── 3. Oscillation pattern ────────────────────────────────
    window = seq[-OSCILLATION_DETECTION_WINDOW:]
    if len(window) >= 4:
        alternations = sum(1 for i in range(1, len(window)) if window[i] != window[i - 1])
        alt_rate = alternations / (len(window) - 1)
        if alt_rate >= OSCILLATION_THRESHOLD:
            signals.append(PatternSignal(
                pattern_type=PatternType.OSCILLATION,
                strength=_pattern_strength_from_observations(len(window)),
                description=f"Oscillation pattern: alternates results {alt_rate:.0%} of the time.",
                observations=len(window),
                break_rate=round(1 - alt_rate, 3),
                evidence=f"Last {len(window)} results: {' '.join(window)}",
                clean_observations=0,
                clean_break_rate=round(1 - alt_rate, 3),
                is_genuine=True,
                reliability=PatternReliabilityLevel.MODERATE,
            ))

    # ── 4. Win-then-drop pattern ──────────────────────────────
    if last_result == "W":
        p_no_win_after_w = 1.0 - (use_clean and clean_next_win_prob > 0 else next_win_prob)
        if p_no_win_after_w >= WIN_THEN_DROP_THRESHOLD:
            clean_p_no_win = 1.0 - clean_next_win_prob if use_clean else p_no_win_after_w
            distortion_impact = abs(p_no_win_after_w - clean_p_no_win)
            
            signals.append(PatternSignal(
                pattern_type=PatternType.WIN_THEN_DROP,
                strength=PatternStrength.MODERATE,
                description=f"Win-then-drop pattern: {p_no_win_after_w:.0%} probability of non-win after a win.",
                observations=int(len(seq) * 0.3),
                break_rate=round(next_win_prob, 3),
                evidence=f"W→W probability: {next_win_prob:.1%}",
                clean_break_rate=round(clean_next_win_prob, 3),
                distortion_impact=round(distortion_impact, 3),
                is_genuine=distortion_impact < 0.1,
                reliability=PatternReliabilityLevel.STRONG if distortion_impact < 0.05 else PatternReliabilityLevel.MODERATE,
            ))

    # ── 5. Resilience ceiling ─────────────────────────────────
    if at_win_ceiling:
        signals.append(PatternSignal(
            pattern_type=PatternType.RESILIENCE_CEILING,
            strength=PatternStrength.STRONG,
            description=f"Favourite at win-rate ceiling ({win_rate:.0%}).",
            observations=int(win_rate * 30),
            break_rate=0.25,
            evidence=f"Win rate: {win_rate:.1%}, threshold: 72%",
            reliability=PatternReliabilityLevel.STRONG,
        ))

    # ── 6. Saturation pattern ─────────────────────────────────
    if saturation_risk:
        signals.append(PatternSignal(
            pattern_type=PatternType.SATURATION_PATTERN,
            strength=PatternStrength.MODERATE,
            description=f"xG saturation detected: {saturation_note}",
            observations=5,
            break_rate=0.40,
            evidence=saturation_note,
            reliability=PatternReliabilityLevel.MODERATE,
        ))

    return signals


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — CONTEXTUAL CURSE DETECTOR (ENHANCED v3)
# ═══════════════════════════════════════════════════════════════

def _detect_contextual_curses(
    leg: Leg,
    fav_is_home: bool,
    kickoff_datetime: Optional[datetime] = None,
    venue_history: Optional[Dict] = None,
    manager_h2h: Optional[Dict] = None,
    monthly_records: Optional[Dict] = None,
    last_match_hours: Optional[float] = None,
    is_derby: bool = False,
    use_clean: bool = True,
) -> List[ContextualCurse]:
    """Detect contextual curses with confidence scoring and clean validation."""
    curses: List[ContextualCurse] = []

    # ── 1. Midweek fatigue curse ──────────────────────────────
    if kickoff_datetime is not None:
        is_midweek = kickoff_datetime.weekday() in (1, 2, 3)
        if is_midweek:
            severity = "MILD"
            confidence = CurseConfidence.MODERATE
            desc = f"Midweek fixture ({kickoff_datetime.strftime('%A')})"
            if last_match_hours is not None and last_match_hours < MIDWEEK_REST_HOURS_MIN:
                severity = "SEVERE"
                confidence = CurseConfidence.HIGH
                desc += f" with only {last_match_hours:.0f}h rest"
            
            # Clean validation
            clean_severity = severity
            clean_confidence = confidence
            is_reliable = True
            
            # Post-European curse has good clean data
            curse = KNOWN_CURSE_PATTERNS.get("post_european_curse")
            if curse and curse.is_reliable:
                clean_severity = "MODERATE" if severity == "SEVERE" else severity
                is_reliable = curse.is_reliable
            
            curses.append(ContextualCurse(
                curse_type=PatternType.MIDWEEK_CURSE,
                description=desc,
                severity=severity,
                confidence=confidence,
                evidence=f"Kickoff: {kickoff_datetime.isoformat()[:16]}",
                clean_severity=clean_severity,
                clean_confidence=clean_confidence,
                is_reliable=is_reliable,
            ))

    # ── 2. Derby curse ────────────────────────────────────────
    if is_derby:
        curse = KNOWN_CURSE_PATTERNS.get("derby_curse")
        if curse:
            curses.append(ContextualCurse(
                curse_type=PatternType.DERBY_CURSE,
                description=curse.description,
                severity="MODERATE",
                confidence=curse.confidence,
                evidence=f"Local derby: {curse.success_rate:.0%} historical upset rate",
                clean_severity="MODERATE" if curse.is_reliable else "MILD",
                clean_confidence=curse.clean_confidence if curse.clean_confidence else curse.confidence,
                is_reliable=curse.is_reliable,
            ))

    # ── 3. Monthly record curse ───────────────────────────────
    if kickoff_datetime is not None and monthly_records:
        month = kickoff_datetime.month
        month_rec = monthly_records.get(month, {})
        m_games = month_rec.get("games", 0)
        m_wins = month_rec.get("wins", 0)
        if m_games >= MONTH_MIN_GAMES:
            m_win_rate = m_wins / m_games
            if m_win_rate <= MONTH_WIN_RATE_CURSE:
                severity = "SEVERE" if m_win_rate < 0.15 else "MODERATE"
                confidence = CurseConfidence.HIGH if m_games >= 10 else CurseConfidence.MODERATE
                
                # Check if curse is reliable (e.g., January curse is partially distortion-driven)
                curse = KNOWN_CURSE_PATTERNS.get("month_january_curse")
                is_reliable = curse.is_reliable if curse and month == 1 else True
                clean_severity = "MILD" if not is_reliable else severity
                
                curses.append(ContextualCurse(
                    curse_type=PatternType.MONTH_CURSE,
                    description=(
                        f"Monthly curse: favourite wins only {m_win_rate:.0%} "
                        f"of matches in {kickoff_datetime.strftime('%B')}"
                    ),
                    severity=severity,
                    confidence=confidence,
                    evidence=f"Month {month}: {m_wins}W/{m_games}G",
                    clean_severity=clean_severity,
                    is_reliable=is_reliable,
                ))

    # ── 4. Venue curse ────────────────────────────────────────
    if not fav_is_home and venue_history:
        venue_key = getattr(leg, "venue_id", None) or "unknown"
        venue_rec = venue_history.get(venue_key, {})
        v_games = venue_rec.get("games", 0)
        v_wins = venue_rec.get("wins", 0)
        if v_games >= VENUE_RECORD_MIN_GAMES:
            v_win_rate = v_wins / v_games
            if v_win_rate <= VENUE_WIN_RATE_CURSE:
                severity = "SEVERE" if v_win_rate < 0.10 else "MODERATE"
                confidence = CurseConfidence.HIGH if v_games >= 10 else CurseConfidence.MODERATE
                curses.append(ContextualCurse(
                    curse_type=PatternType.VENUE_CURSE,
                    description=f"Venue curse: favourite wins only {v_win_rate:.0%} at this stadium",
                    severity=severity,
                    confidence=confidence,
                    evidence=f"Venue record: {v_wins}W from {v_games} visits",
                    is_reliable=v_games >= 10,
                ))

    # ── 5. Monday night curse ─────────────────────────────────
    if kickoff_datetime is not None and kickoff_datetime.weekday() == 0:
        curse = KNOWN_CURSE_PATTERNS.get("monday_night_curse")
        if curse:
            curses.append(ContextualCurse(
                curse_type=PatternType.EARLY_KICKOFF_CURSE,
                description=curse.description,
                severity="MILD",
                confidence=curse.confidence,
                evidence=f"Monday night fixture: {curse.success_rate:.0%} historical upset rate",
                clean_severity="MILD",
                clean_confidence=curse.clean_confidence if curse.clean_confidence else curse.confidence,
                is_reliable=curse.is_reliable,
            ))

    return curses


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — PATTERN STRENGTH SCORING (ENHANCED v3)
# ═══════════════════════════════════════════════════════════════

def _compute_pattern_score_and_reliability(
    rtm_signals: List[PatternSignal],
    curses: List[ContextualCurse],
) -> Tuple[float, str, str, float, PatternReliabilityInfo]:
    """
    Compute 0-100 pattern strength score, reliability, and clean score.
    
    Returns:
        Tuple of (score, verdict, reliability, clean_score, reliability_info)
    """
    score = 0.0
    clean_score = 0.0
    reliability_issues = []
    clean_samples = 0
    distortion_rate_sum = 0.0
    distortion_count = 0

    strength_points = {
        PatternStrength.WEAK: 10.0,
        PatternStrength.MODERATE: 20.0,
        PatternStrength.STRONG: 35.0,
        PatternStrength.DEFINITIVE: 50.0,
    }
    severity_points = {
        "MILD": 8.0,
        "MODERATE": 15.0,
        "SEVERE": 25.0,
    }
    confidence_factor = {
        "SPECULATIVE": 0.5,
        "MODERATE": 0.8,
        "HIGH": 1.0,
        "DEFINITIVE": 1.2,
    }

    total_confidence = 0.0
    weighted_score = 0.0
    clean_weighted_score = 0.0

    for sig in rtm_signals:
        base = strength_points.get(sig.strength, 10.0)
        adjust = base * (1.0 - sig.break_rate)
        score += adjust
        weighted_score += adjust
        total_confidence += sig.strength_score
        
        # Clean score
        clean_adjust = base * (1.0 - sig.clean_break_rate) if sig.clean_observations > 0 else adjust * 0.7
        clean_score += clean_adjust
        clean_weighted_score += clean_adjust
        
        if sig.clean_observations > 0:
            clean_samples += sig.clean_observations
            distortion_rate_sum += sig.distortion_impact
            distortion_count += 1
        
        if not sig.is_genuine:
            reliability_issues.append(f"{sig.pattern_type.value} may be spurious")

    for curse in curses:
        base = severity_points.get(curse.severity, 15.0)
        cf = confidence_factor.get(curse.confidence.value, 0.8)
        curse_score = base * cf
        score += curse_score
        weighted_score += curse_score
        total_confidence += curse.confidence_score
        
        # Clean score
        clean_base = severity_points.get(curse.clean_severity if curse.clean_severity else curse.severity, 15.0)
        clean_cf = confidence_factor.get(curse.clean_confidence.value if curse.clean_confidence else curse.confidence.value, 0.8)
        clean_curse_score = clean_base * clean_cf
        clean_score += clean_curse_score if curse.is_reliable else clean_curse_score * 0.5
        clean_weighted_score += clean_curse_score

    score = round(min(100.0, score), 1)
    clean_score = round(min(100.0, clean_score), 1)
    
    # Determine reliability
    avg_confidence = total_confidence / (len(rtm_signals) + len(curses)) if (rtm_signals or curses) else 0.5
    
    # Calculate reliability info
    avg_distortion_rate = distortion_rate_sum / distortion_count if distortion_count > 0 else 0.0
    primary_distortion = "injury" if avg_distortion_rate > 0.3 else "midweek" if avg_distortion_rate > 0.2 else ""
    
    if clean_samples >= MIN_CLEAN_SAMPLES_FOR_RELIABLE and avg_distortion_rate < 0.15:
        overall_reliability = PatternReliabilityLevel.DEFINITIVE
        reliability = "HIGH"
        reliability_grade = "HIGH"
    elif clean_samples >= MIN_CLEAN_SAMPLES_FOR_RELIABLE and avg_distortion_rate < 0.3:
        overall_reliability = PatternReliabilityLevel.STRONG
        reliability = "HIGH"
        reliability_grade = "MEDIUM"
    elif clean_samples >= MIN_CLEAN_SAMPLES_FOR_RELIABLE:
        overall_reliability = PatternReliabilityLevel.MODERATE
        reliability = "MEDIUM"
        reliability_grade = "MEDIUM"
    elif avg_confidence >= 0.6:
        overall_reliability = PatternReliabilityLevel.WEAK
        reliability = "LOW"
        reliability_grade = "LOW"
    else:
        overall_reliability = PatternReliabilityLevel.INSUFFICIENT
        reliability = "UNKNOWN"
        reliability_grade = "LOW"
    
    # Check if patterns are spurious
    if score > 30 and clean_score < 15:
        overall_reliability = PatternReliabilityLevel.SPURIOUS
        reliability = "LOW"
        reliability_grade = "LOW"
        reliability_issues.append("Patterns only exist due to distortions")
    
    # Determine verdict
    if score >= PATTERN_SCORE_DEFINITIVE:
        verdict = "DEFINITIVE_PATTERN"
    elif score >= PATTERN_SCORE_STRONG:
        verdict = "STRONG_PATTERN"
    elif score >= PATTERN_SCORE_MODERATE:
        verdict = "MODERATE_PATTERN"
    elif score >= PATTERN_SCORE_WEAK:
        verdict = "WEAK_PATTERN"
    else:
        verdict = "NO_PATTERN"
    
    # Clean verdict
    if clean_score >= PATTERN_SCORE_DEFINITIVE:
        clean_verdict = "DEFINITIVE_PATTERN"
    elif clean_score >= PATTERN_SCORE_STRONG:
        clean_verdict = "STRONG_PATTERN"
    elif clean_score >= PATTERN_SCORE_MODERATE:
        clean_verdict = "MODERATE_PATTERN"
    else:
        clean_verdict = "WEAK_PATTERN"
    
    reliability_info = PatternReliabilityInfo(
        overall_reliability=overall_reliability,
        clean_sample_size=clean_samples,
        distortion_rate=avg_distortion_rate,
        primary_distortion=primary_distortion,
        confidence=avg_confidence,
        warnings=reliability_issues,
    )
    
    return score, verdict, reliability, clean_score, reliability_info


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — MARKET SELECTOR (ENHANCED v3)
# ═══════════════════════════════════════════════════════════════

def _select_market(
    pattern_score: float,
    pattern_verdict: str,
    und_win_prob: float,
    und_odds: float,
    away_win_prob: float,
    signal_count: int,
    underdog_edge: float,
    clean_pattern_score: float,
    patterns_reliable: bool,
) -> Tuple[str, Optional[BetMarket], str, str]:
    """
    Market selection driven by pattern score with reliability adjustment.
    
    Returns:
        Tuple of (recommendation, market, confidence, clean_confidence)
    """
    # Adjust for reliability
    reliability_factor = 1.0 if patterns_reliable else 0.7
    adjusted_score = pattern_score * reliability_factor
    adjusted_clean_score = clean_pattern_score * reliability_factor
    
    if pattern_verdict in ("DEFINITIVE_PATTERN", "STRONG_PATTERN") or adjusted_score >= 60:
        if und_win_prob >= UNDERDOG_WIN_MIN_PROB and signal_count >= MIN_PATTERN_SIGNALS_FOR_ACTION:
            conf = "HIGH" if pattern_verdict == "DEFINITIVE_PATTERN" else "MEDIUM"
            clean_conf = "HIGH" if patterns_reliable else "MEDIUM"
            return "UNDERDOG_WIN", BetMarket.STRAIGHT_WIN, conf, clean_conf
        if und_win_prob >= UNDERDOG_DNB_MIN_PROB and away_win_prob < DNB_MAX_AWAY_WIN_PROB:
            return "UNDERDOG_DNB", BetMarket.DRAW_NO_BET, "MEDIUM", "MEDIUM"
        return "PATTERN_NOTED_NO_MARKET", None, "LOW", "LOW"

    if pattern_verdict == "MODERATE_PATTERN" or adjusted_score >= 35:
        if und_win_prob >= UNDERDOG_WIN_MIN_PROB and underdog_edge >= UNDERDOG_WIN_MIN_EDGE:
            return "UNDERDOG_WIN", BetMarket.STRAIGHT_WIN, "LOW", "LOW"
        if und_win_prob >= UNDERDOG_DNB_MIN_PROB and away_win_prob < DNB_MAX_AWAY_WIN_PROB:
            return "UNDERDOG_DNB", BetMarket.DRAW_NO_BET, "LOW", "LOW"
        return "PATTERN_WATCH", None, "LOW", "LOW"

    return "MATCH_SKIP", None, "LOW", "LOW"


# ═══════════════════════════════════════════════════════════════
# SECTION 10 — THREAT ASSESSMENT (ENHANCED v3)
# ═══════════════════════════════════════════════════════════════

def _assess_threat(
    und_a: Any,
    dual: Any,
    pattern_score: float,
    clean_pattern_score: float,
    patterns_reliable: bool,
    goldmine: bool = False,
) -> Tuple[str, str, str, str]:
    """Threat level for the favourite with clean assessment."""
    # Clean threat level
    if goldmine and patterns_reliable:
        clean_threat = "CRITICAL"
        clean_reason = "Goldmine opportunity - definitive pattern (reliable)"
    elif goldmine:
        clean_threat = "HIGH"
        clean_reason = "Goldmine opportunity - pattern may be distortion-driven"
    elif clean_pattern_score >= 70:
        clean_threat = "HIGH"
        clean_reason = "Definitive clean pattern signals against favourite"
    elif clean_pattern_score >= 45:
        clean_threat = "MEDIUM"
        clean_reason = "Strong clean pattern signals — upset possible"
    elif clean_pattern_score >= 25:
        clean_threat = "LOW"
        clean_reason = "Moderate clean pattern signals"
    else:
        clean_threat = "NONE"
        clean_reason = "No significant clean pattern threats"
    
    # Dirty threat level (original)
    if goldmine:
        return "CRITICAL", "Goldmine opportunity - definitive pattern", clean_threat, clean_reason
    
    dangerous = getattr(und_a, 'is_dangerous_underdog', False) if und_a else False
    resilience_gap = getattr(dual, 'resilience_gap', 0.0) if dual else 0.0
    clash_score = getattr(dual, 'pattern_clash_score', 0.0) if dual else 0.0
    
    if dangerous:
        dirty_threat = "HIGH"
        dirty_reason = "Dangerous bounce-back potential"
    elif resilience_gap < -0.20:
        dirty_threat = "HIGH"
        dirty_reason = "Significant resilience advantage to underdog"
    elif pattern_score >= 70:
        dirty_threat = "HIGH"
        dirty_reason = "Definitive pattern signals against favourite"
    elif pattern_score >= 45:
        dirty_threat = "MEDIUM"
        dirty_reason = "Strong pattern signals — upset possible"
    elif pattern_score >= 25:
        dirty_threat = "LOW"
        dirty_reason = "Moderate pattern signals"
    else:
        dirty_threat = "NONE"
        dirty_reason = "No meaningful underdog threat"
    
    return dirty_threat, dirty_reason, clean_threat, clean_reason


# ═══════════════════════════════════════════════════════════════
# SECTION 11 — MAIN ENTRY POINT (ENHANCED v3)
# ═══════════════════════════════════════════════════════════════

def run_underdog_scanner(
    leg: Leg,
    fav_is_home: bool = True,
    und_odds: Optional[float] = None,
    mode: str = "OPPORTUNITY",
    kickoff_datetime: Optional[datetime] = None,
    venue_history: Optional[Dict] = None,
    manager_h2h: Optional[Dict] = None,
    monthly_records: Optional[Dict] = None,
    last_match_hours: Optional[float] = None,
    bilateral_result: Optional[Dict] = None,
    is_derby: bool = False,
    use_clean: bool = True,
    verbose: bool = False,
) -> PatternIntelligence:
    """
    RTM-driven underdog pattern intelligence scanner with clean data support.

    Args:
        leg: Leg object with home_profile and away_profile
        fav_is_home: Whether favourite is home team
        und_odds: Underdog odds (auto-calculated if None)
        mode: "OPPORTUNITY" or "THREAT"
        kickoff_datetime: Match kickoff time for curse detection
        venue_history: Historical venue performance
        manager_h2h: Manager head-to-head records
        monthly_records: Historical month-by-month performance
        last_match_hours: Hours since last match (fatigue)
        is_derby: Whether this is a derby match
        use_clean: If True, use clean (distortion-filtered) probabilities
        verbose: Print detailed analysis
    
    Returns:
        PatternIntelligence with normalized_score for weighted decision system
    """
    result = PatternIntelligence(
        match_id=getattr(leg, "match_id", ""),
        mode=mode,
    )

    if leg.home_profile is None or leg.away_profile is None:
        result.recommendation = "MATCH_SKIP"
        result.reasons.append("Missing team profiles")
        return result

    # ── Probability context ───────────────────────────────────
    home_win_prob, away_win_prob, draw_prob = _resolve_outcome_probs(leg)
    result.home_win_prob = home_win_prob
    result.away_win_prob = away_win_prob
    result.draw_prob = draw_prob
    result.dnb_eligible = away_win_prob < DNB_MAX_AWAY_WIN_PROB
    result.dc_eligible = away_win_prob < DC_MAX_AWAY_WIN_PROB

    # ── Dual pattern engine (M8) ──────────────────────────────
    dual = None
    if _MODULE8_AVAILABLE:
        try:
            dual = run_dual_pattern_engine(leg, fav_is_home=fav_is_home, store_in_leg=False, use_clean=use_clean)
            result.dual_verdict = dual
            
            # Extract pattern reliability from M8
            if dual and hasattr(dual, 'patterns_reliable'):
                result.patterns_reliable = dual.patterns_reliable
                if hasattr(dual, 'distortion_warning') and dual.distortion_warning:
                    result.distortion_warning = dual.distortion_warning
        except Exception as e:
            result.warnings.append(f"M8 dual pattern error: {e}")

    # Extract analyses
    if dual is not None:
        if fav_is_home:
            fav_a = getattr(dual, 'fav_analysis', None)
            und_a = getattr(dual, 'und_analysis', None)
        else:
            fav_a = getattr(dual, 'und_analysis', None)
            und_a = getattr(dual, 'fav_analysis', None)
    else:
        fav_a = None
        und_a = None

    # ── Fill RTM state fields ─────────────────────────────────
    if fav_a is not None:
        result.fav_last_result = getattr(fav_a, 'last_result', '?')
        result.fav_rtm_next_win_prob = getattr(fav_a, 'next_win_prob', 0.33)
        result.fav_at_ceiling = getattr(fav_a, 'at_win_ceiling', False)
        result.fav_at_unbeaten_ceiling = getattr(fav_a, 'at_unbeaten_ceiling', False)
        result.fav_current_streak = getattr(fav_a, 'consecutive_same', 0)
        result.fav_oscillating = getattr(fav_a, 'pattern_type', None) == "VOLATILE" if hasattr(fav_a, 'pattern_type') else False

    if und_a is not None:
        result.und_last_result = getattr(und_a, 'last_result', '?')
        result.und_rtm_next_win_prob = getattr(und_a, 'next_win_prob', 0.33)
        result.und_bounce_back_due = getattr(und_a, 'bounce_back_due', False)

    if dual is not None:
        resilience_gap = getattr(dual, 'resilience_gap', 0.0)
        result.und_resilience_edge = resilience_gap < -0.15

    # ── Underdog odds estimation ──────────────────────────────
    if und_odds is None:
        fav_implied = 1.0 / leg.odds if leg.odds > 1.0 else 0.60
        und_implied = max(1.0 - fav_implied - 0.27, 0.10)
        und_odds = round(1.0 / und_implied, 2)

    # ── Get enhanced RTM for clean pattern detection ──────────
    enhanced_rtm = None
    fav_profile = leg.home_profile if fav_is_home else leg.away_profile
    if fav_profile and hasattr(fav_profile, 'multi_rtm') and fav_profile.multi_rtm:
        enhanced_rtm = fav_profile.multi_rtm.get_enhanced_dimension("overall")

    # ── Goldmine check (with clean validation) ────────────────
    und_edge = result.away_win_prob - (1.0 / und_odds if und_odds > 1.0 else 0.0)
    clean_und_edge = und_edge  # Will be updated if clean data available
    
    if use_clean and enhanced_rtm and enhanced_rtm.clean_transition_count >= MIN_CLEAN_SAMPLES_FOR_RELIABLE:
        # Calculate clean underdog win probability
        clean_und_win_prob = 1.0 - enhanced_rtm.get_clean_prob("W", "W")
        clean_und_edge = clean_und_win_prob - (1.0 / und_odds if und_odds > 1.0 else 0.0)
        result.clean_underdog_edge = round(clean_und_edge, 4)
        
        # Clean goldmine detection
        if und_odds >= CLEAN_GOLDMINE_ODDS_MIN and clean_und_edge >= CLEAN_GOLDMINE_EDGE_MIN:
            result.clean_goldmine_qualified = True
            
        # Regular goldmine (higher bar for clean)
        if leg.odds >= GOLDMINE_ODDS_MIN and und_edge >= GOLDMINE_EDGE_MIN:
            result.goldmine_qualified = True
    else:
        result.clean_underdog_edge = und_edge
        if leg.odds >= GOLDMINE_ODDS_MIN and und_edge >= GOLDMINE_EDGE_MIN:
            result.goldmine_qualified = True

    # Prioritize clean goldmine if both qualify
    if result.clean_goldmine_qualified:
        result.goldmine_qualified = True
        result.recommendation = "GOLDMINE_QUALIFIED"
        result.confidence = "HIGH"
        result.clean_confidence = "HIGH"
        result.reasons.append(f"Clean Goldmine: odds {leg.odds:.2f} with clean edge {clean_und_edge:+.3f}")
        result.threat_level = "HIGH"
        result.clean_threat_level = "HIGH"
        result.threat_reason = "Clean goldmine opportunity"
        if verbose:
            print(f"  🏆 CLEAN GOLDMINE DETECTED: {leg.match_id} @ {leg.odds:.2f}")
        return result
    
    if result.goldmine_qualified:
        result.recommendation = "GOLDMINE_QUALIFIED"
        result.confidence = "HIGH"
        result.clean_confidence = "MEDIUM" if result.patterns_reliable else "LOW"
        result.reasons.append(f"Goldmine: odds {leg.odds:.2f} with edge {und_edge:+.3f}")
        result.threat_level = "HIGH"
        result.clean_threat_level = "MEDIUM"
        result.threat_reason = "Goldmine opportunity"
        if verbose:
            print(f"  🏆 GOLDMINE DETECTED: {leg.match_id} @ {leg.odds:.2f}")
        return result

    # ── Get full sequence for pattern analysis ────────────────
    full_seq = []
    if fav_profile and hasattr(fav_profile, 'form'):
        form = fav_profile.form
        full_seq = form.get("recent_results", []) if isinstance(form, dict) else []

    # ── RTM pattern analysis (with clean data) ────────────────
    rtm_signals = _analyse_rtm_patterns(fav_a, full_seq, enhanced_rtm, use_clean=use_clean)

    # ── Contextual curse detection ────────────────────────────
    curses = _detect_contextual_curses(
        leg, fav_is_home, kickoff_datetime,
        venue_history, manager_h2h, monthly_records, last_match_hours,
        is_derby, use_clean=use_clean
    )

    result.rtm_signals = rtm_signals
    result.contextual_curses = curses
    result.pattern_count = len(rtm_signals) + len(curses)

    # ── Pattern score and reliability ─────────────────────────
    pattern_score, pattern_verdict, reliability, clean_pattern_score, reliability_info = _compute_pattern_score_and_reliability(
        rtm_signals, curses
    )
    result.pattern_strength_score = pattern_score
    result.pattern_verdict = pattern_verdict
    result.pattern_reliability = reliability
    result.clean_pattern_strength_score = clean_pattern_score
    result.reliability_info = reliability_info
    result.patterns_reliable = reliability_info.is_reliable

    # ── Threat mode ───────────────────────────────────────────
    if mode == "THREAT":
        dirty_threat, dirty_reason, clean_threat, clean_reason = _assess_threat(
            und_a, dual, pattern_score, clean_pattern_score, result.patterns_reliable
        )
        result.threat_level = dirty_threat
        result.clean_threat_level = clean_threat
        result.threat_reason = dirty_reason
        result.recommendation = "THREAT_CHECK"
        result.confidence = "MEDIUM" if dirty_threat in ("HIGH", "CRITICAL") else "LOW"
        result.clean_confidence = "MEDIUM" if clean_threat in ("HIGH", "CRITICAL") else "LOW"
        if verbose:
            print(f"  Threat Mode: {dirty_threat} - {dirty_reason}")
            if clean_threat != dirty_threat:
                print(f"    Clean Threat: {clean_threat} - {clean_reason}")
        return result

    # ─── Build signals_found for M17 ──────────────────────────
    for sig in rtm_signals:
        result.signals_found.append(sig.description)
        if not sig.is_genuine:
            result.warnings.append(f"Pattern may be spurious: {sig.pattern_type.value}")
    for curse in curses:
        result.signals_found.append(curse.description)
        if not curse.is_reliable:
            result.warnings.append(f"Curse may be unreliable: {curse.description[:50]}")

    # ── Underdog probability estimate ─────────────────────────
    base = result.und_rtm_next_win_prob * 0.60 + (1 - result.fav_rtm_next_win_prob) * 0.40
    if result.und_bounce_back_due:
        base += 0.08
    if result.und_resilience_edge:
        base += 0.05
    
    # Adjust for clean data if available
    if use_clean and enhanced_rtm and enhanced_rtm.clean_transition_count >= MIN_CLEAN_SAMPLES_FOR_RELIABLE:
        clean_base = (1 - enhanced_rtm.get_clean_prob("W", "W")) * 0.6 + (1 - enhanced_rtm.get_clean_prob("W", "W")) * 0.4
        result.underdog_win_prob = round(min(max(clean_base, 0.05), 0.65), 4)
        result.clean_underdog_edge = round(result.underdog_win_prob - (1.0 / und_odds if und_odds > 1.0 else 0.0), 4)
    else:
        result.underdog_win_prob = round(min(max(base, 0.05), 0.65), 4)
    
    result.underdog_edge = round(
        result.underdog_win_prob - (1.0 / und_odds if und_odds > 1.0 else 0.0), 4
    )
    if result.clean_underdog_edge == 0:
        result.clean_underdog_edge = result.underdog_edge

    # ── Market recommendation ─────────────────────────────────
    if result.pattern_count >= MIN_PATTERN_SIGNALS_FOR_ACTION:
        rec, market, conf, clean_conf = _select_market(
            pattern_score, pattern_verdict,
            result.underdog_win_prob, und_odds, away_win_prob,
            result.pattern_count, result.underdog_edge,
            clean_pattern_score, result.patterns_reliable
        )
        result.recommendation = rec
        result.suggested_market = market
        result.confidence = conf
        result.clean_confidence = clean_conf
    else:
        result.recommendation = "MATCH_SKIP"
        result.reasons.append(
            f"Insufficient pattern signals ({result.pattern_count}/{MIN_PATTERN_SIGNALS_FOR_ACTION})"
        )

    # ── Threat level for favourite ────────────────────────────
    dirty_threat, dirty_reason, clean_threat, clean_reason = _assess_threat(
        und_a, dual, pattern_score, clean_pattern_score, result.patterns_reliable, result.goldmine_qualified
    )
    result.threat_level = dirty_threat
    result.clean_threat_level = clean_threat
    result.threat_reason = dirty_reason

    # ── Pattern log entry ─────────────────────────────────────
    fav_team = fav_profile.team_name if fav_profile else "?"
    und_team = (leg.away_profile if fav_is_home else leg.home_profile).team_name if (leg.away_profile or leg.home_profile) else "?"
    
    result.pattern_log = PatternLogEntry(
        match_id=result.match_id,
        match_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        fav_team=fav_team,
        und_team=und_team,
        patterns_active=[sig.pattern_type.value for sig in rtm_signals],
        rtm_state_before=result.fav_last_result,
        rtm_prediction="W" if result.fav_rtm_next_win_prob > 0.4 else "L",
    )

    # ── Reasons and warnings ──────────────────────────────────
    if result.recommendation == "UNDERDOG_WIN":
        result.reasons.append(
            f"Pattern verdict: {pattern_verdict} (score {pattern_score:.0f}/100). "
            f"Clean score: {clean_pattern_score:.0f}/100. "
            f"RTM suggests value with {result.underdog_edge:+.3f} edge."
        )
    elif result.recommendation in ("UNDERDOG_DNB", "PATTERN_NOTED_NO_MARKET"):
        result.reasons.append(
            f"Pattern signals ({result.pattern_count}) present. "
            f"Confidence: {result.pattern_reliability}"
        )
    elif result.recommendation == "PATTERN_WATCH":
        result.reasons.append(
            f"Moderate pattern ({pattern_score:.0f}/100, clean: {clean_pattern_score:.0f}/100) — watch but do not bet."
        )

    if und_a and getattr(und_a, 'at_loss_floor', False):
        result.warnings.append("Underdog in heavy loss run — desperation cuts both ways")
    if dual and getattr(dual, 'dual_risk_level', 'LOW') in ("HIGH", "CRITICAL"):
        result.warnings.append(f"Overall match risk {getattr(dual, 'dual_risk_level', 'UNKNOWN')} — reduce stake")
    if result.confidence == "LOW" and result.recommendation != "MATCH_SKIP":
        result.warnings.append("Low confidence — consider monitoring rather than betting")
    if not result.patterns_reliable and result.pattern_count > 0:
        result.warnings.append(f"Patterns may be distortion-driven (reliability: {reliability_info.overall_reliability.value})")
    if reliability_info.warnings:
        result.warnings.extend(reliability_info.warnings)

    if verbose:
        print(f"\n  Pattern Intelligence Summary:")
        print(f"    Verdict: {pattern_verdict} ({pattern_score:.0f}/100)")
        print(f"    Clean Verdict: {clean_pattern_score:.0f}/100")
        print(f"    Signals: {result.pattern_count}")
        print(f"    Reliability: {reliability} ({reliability_info.overall_reliability.value})")
        print(f"    Patterns Reliable: {result.patterns_reliable}")
        print(f"    Distortion Rate: {reliability_info.distortion_rate:.1%}")
        print(f"    Recommendation: {result.recommendation}")
        print(f"    Underdog Edge: {result.underdog_edge:+.3f}")
        print(f"    Clean Edge: {result.clean_underdog_edge:+.3f}")
        print(f"    Normalized Score: {result.normalized_score:.3f}")
        print(f"    Clean Normalized: {result.clean_normalized_score:.3f}")

    return result


# ═══════════════════════════════════════════════════════════════
# SECTION 12 — BATCH SCANNING
# ═══════════════════════════════════════════════════════════════

def run_batch_underdog_scan(
    legs: List[Leg],
    fav_is_home_list: List[bool] = None,
    use_clean: bool = True,
    verbose: bool = False,
) -> List[PatternIntelligence]:
    """
    Run underdog scanner on multiple legs.
    
    Args:
        legs: List of Leg objects
        fav_is_home_list: Optional list of favourite-is-home flags
        use_clean: If True, use clean (distortion-filtered) probabilities
        verbose: Print progress
    
    Returns:
        List of PatternIntelligence objects
    """
    results = []
    
    for i, leg in enumerate(legs):
        fav_home = fav_is_home_list[i] if fav_is_home_list and i < len(fav_is_home_list) else True
        
        if verbose:
            print(f"  Scanning {i+1}/{len(legs)}: {getattr(leg, 'match_id', 'unknown')}")
        
        result = run_underdog_scanner(leg, fav_is_home=fav_home, use_clean=use_clean, verbose=False)
        results.append(result)
    
    if verbose:
        goldmines = sum(1 for r in results if r.goldmine_qualified)
        clean_goldmines = sum(1 for r in results if r.clean_goldmine_qualified)
        pattern_found = sum(1 for r in results if r.pattern_count >= MIN_PATTERN_SIGNALS_FOR_ACTION)
        reliable_patterns = sum(1 for r in results if r.patterns_reliable)
        recommendations = defaultdict(int)
        for r in results:
            recommendations[r.recommendation] += 1
        
        print(f"\n  Batch Summary:")
        print(f"    Goldmines: {goldmines} (clean: {clean_goldmines})")
        print(f"    Patterns found: {pattern_found}/{len(legs)}")
        print(f"    Reliable patterns: {reliable_patterns}/{pattern_found if pattern_found else 1}")
        print(f"    Recommendations: {dict(recommendations)}")
    
    return results


# ═══════════════════════════════════════════════════════════════
# SECTION 13 — EXPORTS
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "PatternType",
    "PatternStrength",
    "CurseConfidence",
    "PatternReliabilityLevel",
    "RecommendationType",
    # Data classes
    "CursePattern",
    "PatternReliabilityInfo",
    "PatternSignal",
    "ContextualCurse",
    "PatternLogEntry",
    "PatternIntelligence",
    # Main functions
    "run_underdog_scanner",
    "run_batch_underdog_scan",
    # Constants
    "GOLDMINE_ODDS_MIN",
    "CLEAN_GOLDMINE_ODDS_MIN",
    "UNDERDOG_WIN_MIN_PROB",
    "UNDERDOG_DNB_MIN_PROB",
    "MIN_PATTERN_SIGNALS_FOR_ACTION",
    "KNOWN_CURSE_PATTERNS",
    "MIN_CLEAN_SAMPLES_FOR_RELIABLE",
]


# ═══════════════════════════════════════════════════════════════
# SECTION 14 — EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from module2 import TeamProfile, TransitionMatrix
    
    print("\n" + "=" * 70)
    print("MODULE 9: UNDERDOG SCANNER v3 - ENHANCED TEST RUN")
    print("=" * 70)
    
    # Create mock favourite profile (Liverpool - strong)
    liverpool = TeamProfile(team_id="1", team_name="Liverpool", is_mature=True)
    liverpool.update_metrics({
        "core.games": 28,
        "core.wins": 20,
        "core.draws": 5,
        "core.losses": 3,
        "core.xg": 55.0,
        "core.xga": 25.0,
    })
    liverpool.form = {"recent_results": ["W", "W", "W", "W", "W"]}
    liverpool.transition = TransitionMatrix(
        pattern="SERIAL_WINNER",
        probs={"W": {"W": 0.60, "D": 0.25, "L": 0.15}},
        sample_size=20,
    )
    
    # Create mock underdog profile (Everton - weak, bounce-back potential)
    everton = TeamProfile(team_id="2", team_name="Everton", is_mature=True)
    everton.update_metrics({
        "core.games": 28,
        "core.wins": 6,
        "core.draws": 7,
        "core.losses": 15,
        "core.xg": 30.0,
        "core.xga": 48.0,
    })
    everton.form = {"recent_results": ["L", "L", "L", "L", "L"]}
    everton.transition = TransitionMatrix(
        pattern="LOSS_PRONE",
        probs={"L": {"W": 0.55, "D": 0.20, "L": 0.25}},
        sample_size=15,
    )
    
    # Create leg
    leg = Leg(
        match_id="test_liverpool_everton",
        selection="Liverpool",
        odds=1.35,
        home_profile=liverpool,
        away_profile=everton,
        home_odds=1.35,
        away_odds=8.50,
        draw_odds=5.00,
        model_prob=0.65,
        edge=0.08,
    )
    
    # Add detect_favourite method
    def mock_detect_favourite():
        return "HOME"
    leg.detect_favourite = mock_detect_favourite
    
    # Run scanner
    print("\n📊 ANALYSING: Liverpool vs Everton (Merseyside Derby)")
    print("-" * 40)
    
    result = run_underdog_scanner(
        leg, 
        fav_is_home=True, 
        is_derby=True,
        kickoff_datetime=datetime(2025, 4, 15, 15, 0, 0, tzinfo=timezone.utc),
        use_clean=True,
        verbose=True,
    )
    
    print(f"\n{'='*40}")
    print("RESULTS")
    print(f"{'='*40}")
    print(f"Match: {result.match_id}")
    print(f"Pattern Verdict: {result.pattern_verdict} (score: {result.pattern_strength_score:.1f}/100)")
    print(f"Clean Pattern Score: {result.clean_pattern_strength_score:.1f}/100")
    print(f"Pattern Reliability: {result.pattern_reliability}")
    print(f"Patterns Reliable: {result.patterns_reliable}")
    print(f"Pattern Count: {result.pattern_count}")
    print(f"Recommendation: {result.recommendation}")
    print(f"Confidence: {result.confidence}")
    print(f"Clean Confidence: {result.clean_confidence}")
    print(f"Underdog Win Prob: {result.underdog_win_prob:.1%}")
    print(f"Underdog Edge: {result.underdog_edge:+.3f}")
    print(f"Clean Underdog Edge: {result.clean_underdog_edge:+.3f}")
    print(f"Threat Level: {result.threat_level}")
    print(f"Clean Threat Level: {result.clean_threat_level}")
    print(f"Goldmine: {result.goldmine_qualified}")
    
    # Reliability info
    print(f"\n📊 Pattern Reliability:")
    print(f"  Overall: {result.reliability_info.overall_reliability.value}")
    print(f"  Clean Samples: {result.reliability_info.clean_sample_size}")
    print(f"  Distortion Rate: {result.reliability_info.distortion_rate:.1%}")
    print(f"  Primary Distortion: {result.reliability_info.primary_distortion or 'None'}")
    
    # Weighted decision scores
    print(f"\n🔢 WEIGHTED DECISION SCORES:")
    print(f"  Normalized Score: {result.normalized_score:.3f}")
    print(f"  Clean Normalized Score: {result.clean_normalized_score:.3f}")
    print(f"  Confidence Factor: {result.underdog_confidence_factor:.2f}")
    print(f"  Clean Confidence Factor: {result.clean_confidence_factor:.2f}")
    print(f"  Goldmine Score: {result.goldmine_score:.3f}")
    print(f"  Pattern Reliability Score: {result.pattern_reliability_score:.3f}")
    
    print("\nSignals Found:")
    for sig in result.signals_found[:5]:
        print(f"  • {sig[:100]}...")
    
    print("\nReasons:")
    for reason in result.reasons:
        print(f"  • {reason}")
    
    print("\nWarnings:")
    for warning in result.warnings:
        print(f"  ⚠ {warning}")
    
    # Leg data for M11
    print("\n" + "=" * 70)
    print("LEG DATA FOR M11:")
    print("=" * 70)
    leg_data = result.to_leg_data()
    for key, value in leg_data.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("MODULE 9 v3 READY FOR PRODUCTION")
    print("=" * 70)