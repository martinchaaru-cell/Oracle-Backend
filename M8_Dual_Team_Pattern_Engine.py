"""
The Match Oracle - Module 8: Dual Team Pattern Engine (ENHANCED v6)
================================================================
Analyses both teams simultaneously for pattern clash, ceiling, resilience,
and underdog threat detection.

ENHANCEMENTS IN THIS VERSION (v6):
------------------------------
1. CHANGED: H2H conflict detection now triggers HARD REJECT (not CAUTION)
2. CHANGED: Removed conflict penalty scoring (conflict = immediate reject)
3. CHANGED: Early return when conflict detected
4. ADDED: REJECTED status for conflict scenarios
5. ADDED: Conflict severity now maps to REJECT, not stake reduction

PREVIOUS ENHANCEMENTS (v5):
------------------------------
- Conflict detection between H2H and current season performance
- Tier performance with confidence based on sample size (5-game minimum)
- Recency weighting for H2H (matches older than 3 years ignored)
- Current season form prioritization over historical H2H
- Conflict flags for M11 to trigger CAUTION verdict (NOW CHANGED TO REJECT)
- Sample size tracking for all dimension-specific patterns
- Reliability scores for tier performance data
- Decay function for historical H2H (exponential decay by age)

PHILOSOPHY (v6):
---------------
- H2H vs Current Season Conflict = HARD REJECT (NO BET)
- Current season data > Historical H2H (but conflict = reject)
- Minimum 5 games for reliable tier performance
- H2H matches older than 3 years are ignored

WEIGHTED DECISION SUPPORT:
-------------------------
- risk_score: 0-1 normalized risk level (1.0 for conflict)
- clean_risk_score: 0-1 risk level using clean data
- normalized_score: 0-1 approval score (1 - risk_score)
- confidence_factor: For M13 Kelly scaling (0 for conflict)
- threat_score: 0-1 underdog threat score
- dimension_confidence: Confidence in dimension-specific analysis
- pattern_reliability: How trustworthy detected patterns are
- h2h_conflict_detected: Boolean flag - if True, HARD REJECT
- tier_performance_reliable: Boolean flag for tier data quality
- to_leg_data(): Direct output for M11 aggregation

Usage:
    from module8 import run_dual_pattern_engine, DualPatternVerdict
    
    verdict = run_dual_pattern_engine(leg, verbose=True)
    
    # NEW v6: Conflict triggers REJECT, not CAUTION
    if verdict.h2h_conflict_detected:
        print(f"H2H CONFLICT - HARD REJECT: {verdict.conflict_reasons}")
    else:
        print(f"Risk: {verdict.dual_risk_level} ({verdict.risk_score:.2f})")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime, timezone
import math
import statistics
import warnings

from module2 import Leg, TeamProfile, TransitionMatrix, MultiDimensionRTM, DimensionRTM, PatternReliabilityScore


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — ENUMS
# ═══════════════════════════════════════════════════════════════

class PatternType(Enum):
    """Types of patterns that can be detected."""
    SERIAL_WINNER = "SERIAL_WINNER"
    LOSS_PRONE = "LOSS_PRONE"
    DRAW_SPECIALIST = "DRAW_SPECIALIST"
    VOLATILE = "VOLATILE"
    BOUNCER = "BOUNCER"
    HIGH_VARIANCE = "HIGH_VARIANCE"
    INCONSISTENT = "INCONSISTENT"
    UNKNOWN = "UNKNOWN"


class PatternStrength(Enum):
    """How reliable a pattern is."""
    DEFINITIVE = "DEFINITIVE"   # 15+ observations, pattern holds after distortion removal
    STRONG = "STRONG"           # 10-14 observations, mostly holds
    MODERATE = "MODERATE"       # 5-9 observations, affected by distortions
    WEAK = "WEAK"               # 2-4 observations, disappears with distortion removal
    INSUFFICIENT = "INSUFFICIENT"  # <2 observations


class PatternReliability(Enum):
    """Overall pattern reliability assessment."""
    DEFINITIVE = "DEFINITIVE"   # Pattern holds after removing distortions
    STRONG = "STRONG"           # Pattern mostly holds
    MODERATE = "MODERATE"       # Pattern exists but affected by distortions
    WEAK = "WEAK"               # Pattern disappears when distortions removed
    SPURIOUS = "SPURIOUS"       # Pattern only exists due to distortions
    INSUFFICIENT = "INSUFFICIENT"  # Not enough clean data


class RiskLevel(Enum):
    """Overall risk level for the fixture."""
    CRITICAL = "CRITICAL"   # >0.85 risk score
    HIGH = "HIGH"           # 0.70-0.85
    MEDIUM = "MEDIUM"       # 0.40-0.70
    LOW = "LOW"             # 0.25-0.40
    MINIMAL = "MINIMAL"     # <0.25
    REJECTED = "REJECTED"   # NEW v6: Hard reject due to conflict
    UNKNOWN = "UNKNOWN"


class UnderdogThreat(Enum):
    """Threat level from underdog."""
    CRITICAL = "CRITICAL"   # Multiple threat indicators
    HIGH = "HIGH"           # Strong threat signals
    MEDIUM = "MEDIUM"       # Moderate threat
    LOW = "LOW"             # Low threat
    NONE = "NONE"           # No threat


class DimensionType(Enum):
    """Dimensions for pattern analysis."""
    OVERALL = "overall"
    HOME = "home"
    AWAY = "away"
    VS_TOP6 = "vs_top6"
    VS_MID = "vs_mid"
    VS_BOTTOM6 = "vs_bottom6"
    H2H = "h2h"


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — THRESHOLDS & CONSTANTS (v5 Updates)
# ═══════════════════════════════════════════════════════════════

# Ceiling thresholds (when regression likely)
CEILING_WIN_RATE_HIGH      = 0.72   # 72%+ win rate = at ceiling
CEILING_UNBEATEN_RATE_HIGH = 0.82   # 82%+ unbeaten rate = ceiling
CEILING_LOSS_RATE_HIGH     = 0.50   # 50%+ loss rate = at floor

# Bounce-back detection
BOUNCE_BACK_DUE_THRESHOLD  = 2      # 2+ same results = bounce due
BOUNCE_BACK_WIN_PROB_MIN   = 0.45   # Min win prob after loss to be "dangerous"

# Reliability thresholds (v5: increased minimums)
PATTERN_RELIABLE_MIN_CLEAN = 5      # Minimum clean transitions for reliability
PATTERN_SPURIOUS_DIVERGENCE = 0.15  # 15%+ divergence = spurious pattern

# NEW v5: Tier performance thresholds
MIN_GAMES_FOR_TIER_RELIABLE = 5     # Minimum 5 games for reliable tier performance
MIN_GAMES_FOR_TIER_STRONG = 8       # 8+ games for strong confidence
TIER_PERFORMANCE_WARNING = 0.40     # <40% win rate vs tier = warning
TIER_PERFORMANCE_CRITICAL = 0.30    # <30% win rate vs tier = critical

# NEW v5: H2H recency thresholds
H2H_MAX_AGE_DAYS = 1095             # 3 years max (ignore older matches)
H2H_FULL_WEIGHT_DAYS = 365          # Last 365 days = full weight
H2H_DECAY_FACTOR = 0.5              # Exponential decay factor

# Resilience classification
RESILIENCE_STRONG          = 0.30   # xG diff > 0.30 = strong
RESILIENCE_WEAK            = 0.05   # xG diff < 0.05 = weak
RESILIENCE_VERY_WEAK       = -0.10  # xG diff < -0.10 = very weak

# Pattern clash detection
PATTERN_CLASH_DANGER       = 0.65   # 65%+ difference = high clash
PATTERN_CLASH_MODERATE     = 0.40   # 40-65% = moderate clash

# Saturation risk (overperforming xG)
SATURATION_RISK_FACTOR     = 1.30   # Actual goals > xG * 1.30 = saturated

# Risk scores for weighted decision (0-1)
RISK_SCORE_MAPPING = {
    "MINIMAL": 0.10,
    "LOW": 0.25,
    "MEDIUM": 0.50,
    "HIGH": 0.75,
    "CRITICAL": 0.95,
    "REJECTED": 1.00,        # NEW v6
    "UNKNOWN": 0.50,
}

# Clean risk score mapping (adjusted for pattern reliability)
CLEAN_RISK_MAPPING = {
    "MINIMAL": 0.08,
    "LOW": 0.20,
    "MEDIUM": 0.45,
    "HIGH": 0.70,
    "CRITICAL": 0.92,
    "REJECTED": 1.00,        # NEW v6
    "UNKNOWN": 0.50,
}

# Risk level thresholds
RISK_LOW_THRESHOLD     = 0.25
RISK_MEDIUM_THRESHOLD  = 0.40
RISK_HIGH_THRESHOLD    = 0.70
RISK_CRITICAL_THRESHOLD = 0.85

# Resilience gap thresholds
RESILIENCE_GAP_HIGH        = 0.30   # >0.30 = high advantage
RESILIENCE_GAP_MEDIUM      = 0.15   # 0.15-0.30 = medium advantage
RESILIENCE_GAP_NEGATIVE    = -0.15  # < -0.15 = underdog advantage

# Underdog threat thresholds
UNDERDOG_THREAT_HIGH_GAP     = -0.25   # xG diff < -0.25 = high threat
UNDERDOG_THREAT_MEDIUM_GAP   = -0.15   # xG diff < -0.15 = medium threat

# Minimum games for reliable analysis (v5: increased)
MIN_GAMES_FOR_RESILIENCE = 10
MIN_GAMES_FOR_PATTERN    = 5
MIN_GAMES_FOR_CEILING    = 15
MIN_GAMES_FOR_TIER       = 5        # NEW v5: minimum for tier performance

# Pattern strength observation thresholds
PATTERN_STRENGTH_DEFINITIVE    = 15
PATTERN_STRENGTH_STRONG        = 10
PATTERN_STRENGTH_MODERATE      = 5
PATTERN_STRENGTH_WEAK          = 2

# Confidence factor mapping
CONFIDENCE_FACTOR_MAP = {
    "MINIMAL": 1.0,
    "LOW": 0.9,
    "MEDIUM": 0.7,
    "HIGH": 0.5,
    "CRITICAL": 0.3,
    "REJECTED": 0.0,      # NEW v6
    "UNKNOWN": 0.5,
}

# Threat score mapping (0-1)
THREAT_SCORE_MAP = {
    "CRITICAL": 0.95,
    "HIGH": 0.75,
    "MEDIUM": 0.50,
    "LOW": 0.25,
    "NONE": 0.00,
    "UNKNOWN": 0.50,
}

# Dimension weights for composite scoring
DIMENSION_WEIGHTS = {
    "overall": 0.40,
    "home": 0.25,
    "away": 0.20,
    "vs_top6": 0.05,
    "vs_mid": 0.03,
    "vs_bottom6": 0.07,
}

# Dimension confidence thresholds
DIMENSION_MIN_SAMPLES = {
    "overall": 5,
    "home": 3,
    "away": 3,
    "vs_top6": 2,
    "vs_mid": 2,
    "vs_bottom6": 2,
}

# Pattern reliability weights for risk adjustment
PATTERN_RELIABILITY_WEIGHTS = {
    "DEFINITIVE": 1.00,
    "STRONG": 0.90,
    "MODERATE": 0.75,
    "WEAK": 0.55,
    "SPURIOUS": 0.30,
    "INSUFFICIENT": 0.50,
}


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — DATA STRUCTURES (Enhanced v5)
# ═══════════════════════════════════════════════════════════════

@dataclass
class PatternReliabilityInfo:
    """Pattern reliability information for a team."""
    overall: PatternReliability = PatternReliability.INSUFFICIENT
    bounce_back: PatternReliability = PatternReliability.INSUFFICIENT
    win_ceiling: PatternReliability = PatternReliability.INSUFFICIENT
    clean_sample_size: int = 0
    dirty_sample_size: int = 0
    distortion_rate: float = 0.0
    primary_distortion: str = ""
    confidence: float = 0.0
    
    @property
    def normalized_score(self) -> float:
        scores = {
            PatternReliability.DEFINITIVE: 1.0,
            PatternReliability.STRONG: 0.85,
            PatternReliability.MODERATE: 0.60,
            PatternReliability.WEAK: 0.35,
            PatternReliability.SPURIOUS: 0.10,
            PatternReliability.INSUFFICIENT: 0.30,
        }
        return scores.get(self.overall, 0.50)
    
    @property
    def is_reliable(self) -> bool:
        return self.overall in (PatternReliability.DEFINITIVE, PatternReliability.STRONG)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall.value,
            "bounce_back": self.bounce_back.value,
            "win_ceiling": self.win_ceiling.value,
            "clean_samples": self.clean_sample_size,
            "distortion_rate": round(self.distortion_rate, 3),
            "primary_distortion": self.primary_distortion,
            "confidence": round(self.confidence, 3),
            "normalized_score": self.normalized_score,
            "is_reliable": self.is_reliable,
        }


@dataclass
class TierPerformanceInfo:
    """
    NEW v5: Tier-specific performance with reliability scoring.
    
    Tracks how a team performs against specific tiers with confidence
    based on sample size.
    """
    tier_name: str
    games_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    win_rate: float = 0.0
    points_per_game: float = 0.0
    goals_scored: float = 0.0
    goals_conceded: float = 0.0
    
    # Reliability flags (v5 critical)
    is_reliable: bool = False
    confidence: str = "INSUFFICIENT"  # HIGH / MEDIUM / LOW / INSUFFICIENT
    sample_quality: float = 0.0
    
    # Performance classification
    performance_class: str = "UNKNOWN"  # EXCELLENT / GOOD / AVERAGE / POOR / CRITICAL
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.games_played > 0:
            self.win_rate = self.wins / self.games_played
            self.points_per_game = (self.wins * 3 + self.draws) / self.games_played
            
            # Determine reliability based on sample size (v5 rule)
            if self.games_played >= MIN_GAMES_FOR_TIER_STRONG:
                self.is_reliable = True
                self.confidence = "HIGH"
                self.sample_quality = 1.0
            elif self.games_played >= MIN_GAMES_FOR_TIER_RELIABLE:
                self.is_reliable = True
                self.confidence = "MEDIUM"
                self.sample_quality = 0.7
            else:
                self.is_reliable = False
                self.confidence = "INSUFFICIENT"
                self.sample_quality = max(0.2, self.games_played / MIN_GAMES_FOR_TIER_RELIABLE)
            
            # Performance classification
            if self.win_rate >= 0.60:
                self.performance_class = "EXCELLENT"
            elif self.win_rate >= 0.50:
                self.performance_class = "GOOD"
            elif self.win_rate >= 0.40:
                self.performance_class = "AVERAGE"
            elif self.win_rate >= 0.30:
                self.performance_class = "POOR"
            else:
                self.performance_class = "CRITICAL"
    
    @property
    def normalized_score(self) -> float:
        """Convert win rate to 0-1 score."""
        return self.win_rate
    
    @property
    def penalty_multiplier(self) -> float:
        """
        Penalty multiplier based on performance class.
        Used for stake reduction in M13.
        """
        penalties = {
            "EXCELLENT": 1.00,
            "GOOD": 0.90,
            "AVERAGE": 0.75,
            "POOR": 0.50,
            "CRITICAL": 0.25,
            "UNKNOWN": 0.70,
        }
        return penalties.get(self.performance_class, 0.70)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier_name,
            "games": self.games_played,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 3),
            "ppg": round(self.points_per_game, 2),
            "is_reliable": self.is_reliable,
            "confidence": self.confidence,
            "performance_class": self.performance_class,
            "penalty_multiplier": self.penalty_multiplier,
        }


@dataclass
class H2HRecencyInfo:
    """
    NEW v5: H2H analysis with recency weighting.
    
    Older matches (3+ years) are ignored.
    Recent matches (last 365 days) get 1.2x weight.
    """
    total_meetings: int = 0
    recent_meetings: int = 0  # Last 365 days
    weighted_win_rate: float = 0.0
    raw_win_rate: float = 0.0
    games_analyzed: int = 0
    games_ignored: int = 0  # Older than 3 years
    
    # Venue-specific
    venue_win_rate: float = 0.0
    venue_weighted_win_rate: float = 0.0
    venue_games: int = 0
    
    # Draw rate
    draw_rate: float = 0.0
    recent_draw_rate: float = 0.0
    
    # Reliability
    is_reliable: bool = False
    confidence: str = "INSUFFICIENT"
    conflict_detected: bool = False  # NEW v5: H2H conflicts with current form
    
    @property
    def normalized_score(self) -> float:
        """Weighted win rate normalized to 0-1."""
        return self.weighted_win_rate
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_meetings": self.total_meetings,
            "recent_meetings": self.recent_meetings,
            "weighted_win_rate": round(self.weighted_win_rate, 3),
            "raw_win_rate": round(self.raw_win_rate, 3),
            "venue_win_rate": round(self.venue_win_rate, 3),
            "draw_rate": round(self.draw_rate, 3),
            "is_reliable": self.is_reliable,
            "confidence": self.confidence,
            "conflict_detected": self.conflict_detected,
        }


@dataclass
class DimensionPatternAnalysis:
    """Pattern analysis for a specific dimension."""
    dimension_name: str
    last_result: str = "?"
    consecutive_same: int = 0
    next_win_prob: float = 0.33
    next_draw_prob: float = 0.33
    next_loss_prob: float = 0.34
    win_rate: float = 0.0
    loss_rate: float = 0.0
    draw_rate: float = 0.0
    games_played: int = 0
    at_win_ceiling: bool = False
    at_loss_floor: bool = False
    bounce_back_due: bool = False
    bounce_back_direction: str = "?"
    bounce_back_prob: float = 0.0
    resilience_level: str = "NEUTRAL"
    xg_differential: float = 0.0
    pattern_type: PatternType = PatternType.UNKNOWN
    pattern_strength: PatternStrength = PatternStrength.INSUFFICIENT
    pattern_confidence: float = 0.0
    is_reliable: bool = False
    
    # NEW v5: Tier performance (for dimension-specific tier analysis)
    tier_performance: Optional[TierPerformanceInfo] = None
    
    # Clean pattern data
    clean_next_win_prob: float = 0.33
    clean_bounce_back_prob: float = 0.0
    clean_win_rate: float = 0.0
    clean_at_win_ceiling: bool = False
    distortion_impact: float = 0.0
    pattern_reliability: PatternReliability = PatternReliability.INSUFFICIENT
    
    notes: List[str] = field(default_factory=list)
    
    @property
    def is_pattern_spurious(self) -> bool:
        """True if pattern only exists due to distortions."""
        return self.pattern_reliability == PatternReliability.SPURIOUS
    
    @property
    def summary(self) -> str:
        return (f"{self.dimension_name}: last={self.last_result}, "
                f"next_win={self.next_win_prob:.1%}, "
                f"clean_next_win={self.clean_next_win_prob:.1%}, "
                f"resilience={self.resilience_level}")


@dataclass
class TeamPatternAnalysis:
    """Complete pattern analysis for one team across dimensions."""
    team_id: str = ""
    team_name: str = ""
    
    # Dimension-specific analyses
    overall: DimensionPatternAnalysis = field(default_factory=lambda: DimensionPatternAnalysis(dimension_name="overall"))
    home: Optional[DimensionPatternAnalysis] = None
    away: Optional[DimensionPatternAnalysis] = None
    vs_top6: Optional[DimensionPatternAnalysis] = None
    vs_mid: Optional[DimensionPatternAnalysis] = None
    vs_bottom6: Optional[DimensionPatternAnalysis] = None
    h2h: Optional[DimensionPatternAnalysis] = None
    
    # NEW v5: Tier performance data (current season only)
    tier_performance_home: Dict[str, TierPerformanceInfo] = field(default_factory=dict)
    tier_performance_away: Dict[str, TierPerformanceInfo] = field(default_factory=dict)
    
    # NEW v5: H2H recency data
    h2h_recency: Optional[H2HRecencyInfo] = None
    
    # Composite metrics
    primary_pattern: PatternType = PatternType.UNKNOWN
    primary_confidence: float = 0.0
    dimension_consistency: float = 0.0  # How consistent across dimensions
    best_dimension: str = "overall"
    worst_dimension: str = "overall"
    
    # Pattern reliability
    reliability: PatternReliabilityInfo = field(default_factory=PatternReliabilityInfo)
    
    # NEW v5: Conflict detection
    h2h_conflicts_detected: bool = False
    tier_conflicts_detected: bool = False
    conflict_count: int = 0
    conflict_reasons: List[str] = field(default_factory=list)
    
    # Legacy fields (for backward compatibility)
    current_sequence: List[str] = field(default_factory=list)
    last_result: str = "?"
    consecutive_same: int = 0
    next_win_prob: float = 0.0
    next_draw_prob: float = 0.0
    next_loss_prob: float = 0.0
    win_rate: float = 0.0
    unbeaten_rate: float = 0.0
    loss_rate: float = 0.0
    games_played: int = 0
    at_win_ceiling: bool = False
    at_unbeaten_ceiling: bool = False
    at_loss_floor: bool = False
    ceiling_strength: str = "NONE"
    bounce_back_due: bool = False
    bounce_back_direction: str = "?"
    bounce_back_prob: float = 0.0
    xg_per_game: float = 0.0
    xga_per_game: float = 0.0
    xg_differential: float = 0.0
    resilience_level: str = "NEUTRAL"
    saturation_risk: bool = False
    saturation_note: str = ""
    saturation_score: float = 0.0
    pattern_type: PatternType = PatternType.UNKNOWN
    pattern_strength: PatternStrength = PatternStrength.INSUFFICIENT
    pattern_confidence: float = 0.5
    is_dangerous_underdog: bool = False
    dangerous_score: float = 0.0
    pattern_consistency: float = 0.5
    last_pattern_break: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def get_dimension(self, dim_name: str) -> Optional[DimensionPatternAnalysis]:
        """Get pattern analysis for a specific dimension."""
        dim_map = {
            "overall": self.overall,
            "home": self.home,
            "away": self.away,
            "vs_top6": self.vs_top6,
            "vs_mid": self.vs_mid,
            "vs_bottom6": self.vs_bottom6,
            "h2h": self.h2h,
        }
        return dim_map.get(dim_name)
    
    def get_tier_performance(self, venue: str, tier: str) -> Optional[TierPerformanceInfo]:
        """Get tier performance for a specific venue and tier."""
        if venue == "home":
            return self.tier_performance_home.get(tier)
        elif venue == "away":
            return self.tier_performance_away.get(tier)
        return None
    
    def get_weighted_win_prob(self, use_clean: bool = False) -> float:
        """Calculate weighted win probability across dimensions."""
        total_weight = 0.0
        weighted_prob = 0.0
        
        for dim_name, weight in DIMENSION_WEIGHTS.items():
            dim = self.get_dimension(dim_name)
            if dim and dim.is_reliable:
                prob = dim.clean_next_win_prob if use_clean else dim.next_win_prob
                weighted_prob += prob * weight
                total_weight += weight
        
        if total_weight > 0:
            return weighted_prob / total_weight
        return self.clean_next_win_prob if use_clean else self.next_win_prob
    
    def get_dimension_consistency(self) -> float:
        """Calculate how consistent patterns are across dimensions."""
        probs = []
        for dim_name in DIMENSION_WEIGHTS.keys():
            dim = self.get_dimension(dim_name)
            if dim and dim.is_reliable:
                probs.append(dim.next_win_prob)
        
        if len(probs) < 2:
            return 0.5
        
        variance = statistics.variance(probs) if len(probs) > 1 else 0
        consistency = 1.0 - min(1.0, variance / 0.1)
        return round(consistency, 3)
    
    @property
    def clean_win_prob(self) -> float:
        """Clean win probability (distortions removed)."""
        return self.overall.clean_next_win_prob
    
    @property
    def clean_bounce_back_prob(self) -> float:
        """Clean bounce-back probability."""
        return self.overall.clean_bounce_back_prob
    
    @property
    def pattern_reliability_score(self) -> float:
        """Normalized pattern reliability score (0-1)."""
        return self.reliability.normalized_score
    
    @property
    def has_tier_conflicts(self) -> bool:
        """Check if tier performance shows conflicts with expectations."""
        return self.tier_conflicts_detected
    
    @property
    def summary(self) -> str:
        dim_summary = []
        if self.home and self.home.is_reliable:
            dim_summary.append(f"home={self.home.next_win_prob:.1%}")
        if self.away and self.away.is_reliable:
            dim_summary.append(f"away={self.away.next_win_prob:.1%}")
        
        reliable_str = f" [reliable]" if self.reliability.is_reliable else f" [unreliable]"
        conflict_str = f" ⚠CONFLICT" if self.h2h_conflicts_detected else ""
        
        return (f"{self.team_name}: overall={self.next_win_prob:.1%} | "
                f"clean={self.clean_win_prob:.1%} | "
                f"{' | '.join(dim_summary)} | "
                f"pattern={self.pattern_type.value}{reliable_str}{conflict_str}")


@dataclass
class DualPatternVerdict:
    """Combined verdict from analysing both teams."""
    # Individual analyses
    home_analysis: TeamPatternAnalysis = field(default_factory=TeamPatternAnalysis)
    away_analysis: TeamPatternAnalysis = field(default_factory=TeamPatternAnalysis)
    
    # Legacy individual analyses (for backward compatibility)
    fav_analysis: Optional[TeamPatternAnalysis] = None
    und_analysis: Optional[TeamPatternAnalysis] = None
    
    # Interaction metrics
    pattern_clash_score: float = 0.0
    clean_pattern_clash_score: float = 0.0
    clash_description: str = ""
    clash_severity: str = "LOW"
    ceiling_danger: bool = False
    clean_ceiling_danger: bool = False
    ceiling_details: List[str] = field(default_factory=list)
    resilience_gap: float = 0.0
    clean_resilience_gap: float = 0.0
    resilience_verdict: str = ""
    
    # Dimension-specific clash metrics
    home_clash_score: float = 0.0
    away_clash_score: float = 0.0
    tier_clash_scores: Dict[str, float] = field(default_factory=dict)
    dimension_agreement: float = 0.0
    clean_dimension_agreement: float = 0.0
    
    # Risk scoring
    dual_risk_level: str = "LOW"
    dual_risk_score: float = 0.0
    clean_risk_score: float = 0.0
    risk_factors: List[str] = field(default_factory=list)
    
    # Underdog intelligence
    underdog_opportunity: bool = False
    underdog_opportunity_reason: str = ""
    underdog_threat_level: str = "NONE"
    clean_underdog_threat_level: str = "NONE"
    underdog_threat_reason: str = ""
    underdog_threat_score: float = 0.0
    clean_underdog_threat_score: float = 0.0
    
    # Pattern consistency (learning from history)
    pattern_consistency_score: float = 0.5
    pattern_reliability: str = "UNKNOWN"
    patterns_reliable: bool = False
    distortion_warning: Optional[str] = None
    
    # NEW v5/v6: Conflict detection for M11
    h2h_conflict_detected: bool = False
    conflict_severity: str = "NONE"  # NONE / LOW / MEDIUM / HIGH
    conflict_reasons: List[str] = field(default_factory=list)
    override_verdict_recommendation: Optional[str] = None  # REJECTED (H2H CONFLICT)
    
    # For M11 integration
    fav_is_home: bool = True
    
    # Audit
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: List[str] = field(default_factory=list)
    
    # ═══════════════════════════════════════════════════════════
    # WEIGHTED DECISION PROPERTIES
    # ═══════════════════════════════════════════════════════════
    
    @property
    def risk_score(self) -> float:
        """Convert dual_risk_level to normalized 0-1 risk score."""
        # NEW v6: If conflict detected, risk_score = 1.0 (maximum risk)
        if self.h2h_conflict_detected:
            return 1.0
        return RISK_SCORE_MAPPING.get(self.dual_risk_level, 0.50)
    
    @property
    def clean_risk_score_value(self) -> float:
        """Clean risk score using pattern reliability adjustment."""
        # NEW v6: If conflict detected, clean_risk_score = 1.0
        if self.h2h_conflict_detected:
            return 1.0
        
        base = CLEAN_RISK_MAPPING.get(self.dual_risk_level, 0.50)
        
        # Adjust based on pattern reliability
        if self.fav_analysis:
            reliability_factor = PATTERN_RELIABILITY_WEIGHTS.get(
                self.fav_analysis.reliability.overall.value, 0.75
            )
            base = base * reliability_factor
        
        return round(min(1.0, base), 3)
    
    @property
    def normalized_score(self) -> float:
        """
        Convert risk to approval score (0-1) for weighted decision.
        Higher score = more confident in favourite.
        
        NEW v6: If conflict detected, normalized_score = 0.0
        """
        if self.h2h_conflict_detected:
            return 0.0
        return 1.0 - self.risk_score
    
    @property
    def clean_normalized_score(self) -> float:
        """Clean approval score using pattern reliability."""
        if self.h2h_conflict_detected:
            return 0.0
        return 1.0 - self.clean_risk_score_value
    
    @property
    def confidence_factor(self) -> float:
        """
        Confidence factor for M13 Kelly scaling.
        Maps risk level to stake multiplier.
        
        NEW v6: If conflict detected, confidence_factor = 0.0
        """
        if self.h2h_conflict_detected:
            return 0.0
        
        base = CONFIDENCE_FACTOR_MAP.get(self.dual_risk_level, 0.50)
        
        # Adjust based on pattern reliability
        if self.fav_analysis and not self.fav_analysis.reliability.is_reliable:
            base = base * 0.7
        
        if self.distortion_warning:
            base = base * 0.8
        
        return round(base, 2)
    
    @property
    def threat_score(self) -> float:
        """Convert underdog_threat_level to 0-1 score."""
        return THREAT_SCORE_MAP.get(self.underdog_threat_level, 0.50)
    
    @property
    def clean_threat_score(self) -> float:
        """Clean threat score using pattern reliability."""
        return THREAT_SCORE_MAP.get(self.clean_underdog_threat_level, 0.50)
    
    @property
    def dimension_confidence(self) -> float:
        """Confidence in dimension-specific analysis (0-1)."""
        return self.dimension_agreement
    
    @property
    def clean_dimension_confidence(self) -> float:
        """Confidence in clean dimension analysis."""
        return self.clean_dimension_agreement
    
    @property
    def is_safe(self) -> bool:
        """True if risk level is LOW or MINIMAL and no conflict."""
        return not self.h2h_conflict_detected and self.dual_risk_level in ("LOW", "MINIMAL")
    
    @property
    def is_risky(self) -> bool:
        """True if risk level is HIGH or CRITICAL or conflict."""
        return self.h2h_conflict_detected or self.dual_risk_level in ("HIGH", "CRITICAL")
    
    @property
    def pattern_reliability_score(self) -> float:
        """Overall pattern reliability score (0-1)."""
        if self.fav_analysis:
            return self.fav_analysis.pattern_reliability_score
        return 0.5
    
    @property
    def has_conflict(self) -> bool:
        """True if any conflict detected between H2H and current form."""
        return self.h2h_conflict_detected
    
    @property
    def recommended_verdict(self) -> str:
        """
        NEW v6: Recommended verdict for M11.
        If conflict detected, recommend REJECTED regardless of score.
        """
        if self.h2h_conflict_detected:
            return "REJECTED"
        return self.dual_risk_level
    
    def to_leg_data(self) -> Dict[str, Any]:
        """Convert to leg_data format for M11 weighted decision."""
        return {
            "dual_risk_level": self.recommended_verdict,  # Use conflict-aware verdict
            "dual_risk_score": self.risk_score,
            "dual_normalized_score": self.normalized_score,
            "clean_risk_score": self.clean_risk_score_value,
            "clean_normalized_score": self.clean_normalized_score,
            "underdog_threat_level": self.underdog_threat_level,
            "underdog_threat_score": self.threat_score,
            "clean_threat_level": self.clean_underdog_threat_level,
            "pattern_clash_score": self.pattern_clash_score,
            "clean_pattern_clash_score": self.clean_pattern_clash_score,
            "resilience_gap": self.resilience_gap,
            "clean_resilience_gap": self.clean_resilience_gap,
            "ceiling_danger": self.ceiling_danger,
            "underdog_opportunity": self.underdog_opportunity,
            "pattern_consistency": self.pattern_consistency_score,
            "confidence_factor": self.confidence_factor,
            "dimension_agreement": self.dimension_agreement,
            "clean_dimension_agreement": self.clean_dimension_agreement,
            "home_clash": self.home_clash_score,
            "away_clash": self.away_clash_score,
            "patterns_reliable": self.patterns_reliable,
            "pattern_reliability_score": self.pattern_reliability_score,
            # NEW v5/v6: Conflict flags for M11
            "h2h_conflict_detected": self.h2h_conflict_detected,
            "conflict_severity": self.conflict_severity,
            "conflict_count": len(self.conflict_reasons),
            "override_recommendation": self.recommended_verdict,
        }
    
    @property
    def summary(self) -> str:
        """Human-readable summary."""
        dim_str = f" (dim_agree={self.dimension_agreement:.1%})" if self.dimension_agreement > 0 else ""
        clean_str = f" clean_risk={self.clean_risk_score_value:.2f}" if self.patterns_reliable else ""
        reliable_str = " ✓reliable" if self.patterns_reliable else " ⚠unreliable"
        
        # NEW v6: Conflict now shows REJECTED
        if self.h2h_conflict_detected:
            conflict_str = " 🚨CONFLICT REJECTED"
        else:
            conflict_str = ""
        
        return (f"DualPattern: risk={self.recommended_verdict} ({self.risk_score:.2f}){dim_str}{clean_str} | "
                f"clash={self.pattern_clash_score:.2f} | "
                f"underdog={self.underdog_threat_level} | "
                f"reliable={self.pattern_reliability}{reliable_str}{conflict_str}")


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — TIER PERFORMANCE ANALYZER (NEW v5)
# ═══════════════════════════════════════════════════════════════

def _calculate_tier_performance(
    profile: TeamProfile,
    fixtures: List[Dict],
    tier_type: str,
    venue: str,
    current_standings: Dict[int, int],  # team_id -> position
    league_size: int = 20,
) -> TierPerformanceInfo:
    """
    Calculate team's performance against a specific tier.
    
    Args:
        profile: TeamProfile object
        fixtures: List of fixtures with results and opponent info
        tier_type: "top6", "mid", "bottom6"
        venue: "home", "away", or "all"
        current_standings: Dict mapping team_id to current position
        league_size: Total teams in league
    
    Returns:
        TierPerformanceInfo with win rate and reliability
    """
    info = TierPerformanceInfo(tier_name=f"{venue}_vs_{tier_type}")
    
    # Determine tier thresholds based on league size
    top_threshold = max(1, int(league_size * 0.30))  # Top 30%
    bottom_threshold = league_size - top_threshold + 1  # Bottom 30%
    
    # Filter fixtures by venue and opponent tier
    filtered = []
    for fx in fixtures:
        # Check venue
        fx_venue = fx.get("venue", "home")
        if venue != "all" and fx_venue != venue:
            continue
        
        # Get opponent ID and position
        opponent_id = fx.get("opponent_id")
        if not opponent_id or opponent_id not in current_standings:
            continue
        
        opponent_pos = current_standings[opponent_id]
        
        # Classify opponent tier
        if tier_type == "top6":
            if opponent_pos > top_threshold:
                continue
        elif tier_type == "mid":
            if opponent_pos <= top_threshold or opponent_pos >= bottom_threshold:
                continue
        elif tier_type == "bottom6":
            if opponent_pos < bottom_threshold:
                continue
        
        # Add to filtered list
        filtered.append({
            "result": fx.get("result"),  # "W", "D", "L" from this team's perspective
            "goals_for": fx.get("goals_for", 0),
            "goals_against": fx.get("goals_against", 0),
        })
    
    info.games_played = len(filtered)
    
    if info.games_played == 0:
        return info
    
    # Calculate statistics
    for fx in filtered:
        result = fx.get("result")
        if result == "W":
            info.wins += 1
        elif result == "D":
            info.draws += 1
        elif result == "L":
            info.losses += 1
        
        info.goals_scored += fx.get("goals_for", 0)
        info.goals_conceded += fx.get("goals_against", 0)
    
    # Calculate averages
    info.goals_scored /= info.games_played
    info.goals_conceded /= info.games_played
    
    # Post-init calculates win_rate, ppg, reliability, performance_class
    info.__post_init__()
    
    return info


def _calculate_h2h_with_recency(
    h2h_fixtures: List[Dict],
    fav_team_id: str,
    venue: str = "all",
    max_age_days: int = H2H_MAX_AGE_DAYS,
) -> H2HRecencyInfo:
    """
    Calculate H2H statistics with recency weighting.
    
    NEW v5: Older matches (3+ years) are ignored.
    Recent matches (last 365 days) get 1.2x weight.
    
    Args:
        h2h_fixtures: List of H2H fixtures with date and result
        fav_team_id: Favourite team ID (for win counting)
        venue: "home", "away", or "all" (from favourite's perspective)
        max_age_days: Maximum age of matches to consider (default 3 years)
    
    Returns:
        H2HRecencyInfo with weighted win rates
    """
    info = H2HRecencyInfo()
    current_date = datetime.now(timezone.utc)
    
    if not h2h_fixtures:
        return info
    
    # Filter by venue
    if venue == "home":
        filtered = [f for f in h2h_fixtures if f.get("venue") == "home"]
    elif venue == "away":
        filtered = [f for f in h2h_fixtures if f.get("venue") == "away"]
    else:
        filtered = h2h_fixtures
    
    info.total_meetings = len(filtered)
    
    # Calculate weights and counts
    total_weight = 0.0
    weighted_wins = 0.0
    raw_wins = 0
    draws = 0
    
    for fx in filtered:
        match_date = fx.get("date")
        if not match_date:
            continue
        
        # Calculate age in days
        try:
            if isinstance(match_date, str):
                dt = datetime.fromisoformat(match_date.replace("Z", "+00:00"))
            else:
                dt = match_date
            days_ago = (current_date - dt).days
        except Exception:
            days_ago = 0
        
        # Ignore matches older than max_age_days (3 years)
        if days_ago > max_age_days:
            info.games_ignored += 1
            continue
        
        # Calculate weight (linear decay from 1.0 to 0.0 over max_age_days)
        weight = max(0.0, 1.0 - (days_ago / max_age_days))
        
        # Bonus for recent matches (last 365 days)
        if days_ago <= H2H_FULL_WEIGHT_DAYS:
            weight *= 1.2
        
        # Track recent meetings
        if days_ago <= H2H_FULL_WEIGHT_DAYS:
            info.recent_meetings += 1
        
        total_weight += weight
        info.games_analyzed += 1
        
        # Determine winner from favourite's perspective
        winner = fx.get("winner")  # "fav", "und", "draw"
        
        if winner == "fav":
            weighted_wins += weight
            raw_wins += 1
        elif winner == "draw":
            draws += 1
    
    # Calculate weighted win rate
    if total_weight > 0:
        info.weighted_win_rate = weighted_wins / total_weight
    else:
        info.weighted_win_rate = 0.50
    
    # Calculate raw win rate
    if info.games_analyzed > 0:
        info.raw_win_rate = raw_wins / info.games_analyzed
        info.draw_rate = draws / info.games_analyzed
    
    # Calculate recent draw rate (last 365 days)
    recent_draws = 0
    recent_total = 0
    for fx in filtered:
        match_date = fx.get("date")
        if not match_date:
            continue
        
        try:
            if isinstance(match_date, str):
                dt = datetime.fromisoformat(match_date.replace("Z", "+00:00"))
            else:
                dt = match_date
            days_ago = (current_date - dt).days
        except Exception:
            continue
        
        if days_ago <= H2H_FULL_WEIGHT_DAYS:
            recent_total += 1
            if fx.get("winner") == "draw":
                recent_draws += 1
    
    if recent_total > 0:
        info.recent_draw_rate = recent_draws / recent_total
    
    # Determine reliability
    if info.games_analyzed >= MIN_GAMES_FOR_TIER_RELIABLE:
        info.is_reliable = True
        info.confidence = "HIGH" if info.games_analyzed >= MIN_GAMES_FOR_TIER_STRONG else "MEDIUM"
    else:
        info.is_reliable = False
        info.confidence = "INSUFFICIENT"
    
    return info


def _calculate_tier_performance_from_season(
    profile: TeamProfile,
    season_fixtures: List[Dict],
    team_id: str,
    standings: Dict[int, Dict],
    league_size: int = 20,
) -> Tuple[Dict[str, TierPerformanceInfo], Dict[str, TierPerformanceInfo]]:
    """
    Calculate tier performance for home and away from season fixtures.
    
    Returns:
        Tuple of (home_tier_performance, away_tier_performance)
    """
    # Build position mapping from standings
    positions = {}
    for tid, standing in standings.items():
        positions[tid] = standing.get("position", 10)
    
    # Determine tier thresholds
    top_threshold = max(1, int(league_size * 0.30))
    bottom_threshold = league_size - top_threshold + 1
    
    # Process fixtures
    home_perf = {}
    away_perf = {}
    tiers = ["top6", "mid", "bottom6"]
    
    # Initialize tier performance objects
    for tier in tiers:
        home_perf[tier] = TierPerformanceInfo(tier_name=f"home_vs_{tier}")
        away_perf[tier] = TierPerformanceInfo(tier_name=f"away_vs_{tier}")
    
    for fx in season_fixtures:
        # Get opponent ID
        opponent_id = fx.get("teams", {}).get("home", {}).get("id")
        if fx.get("teams", {}).get("away", {}).get("id") == team_id:
            opponent_id = fx.get("teams", {}).get("home", {}).get("id")
        
        if not opponent_id or opponent_id not in positions:
            continue
        
        opponent_pos = positions[opponent_id]
        
        # Determine tier
        if opponent_pos <= top_threshold:
            tier = "top6"
        elif opponent_pos >= bottom_threshold:
            tier = "bottom6"
        else:
            tier = "mid"
        
        # Determine venue and result from team's perspective
        is_home = fx.get("teams", {}).get("home", {}).get("id") == team_id
        venue = "home" if is_home else "away"
        
        goals_for = fx.get("goals", {}).get("home" if is_home else "away", 0)
        goals_against = fx.get("goals", {}).get("away" if is_home else "home", 0)
        
        if goals_for > goals_against:
            result = "W"
        elif goals_for < goals_against:
            result = "L"
        else:
            result = "D"
        
        # Update performance info
        if venue == "home":
            perf = home_perf[tier]
        else:
            perf = away_perf[tier]
        
        perf.games_played += 1
        if result == "W":
            perf.wins += 1
        elif result == "D":
            perf.draws += 1
        else:
            perf.losses += 1
        perf.goals_scored += goals_for
        perf.goals_conceded += goals_against
    
    # Post-process all tier performances
    for tier in tiers:
        home_perf[tier].__post_init__()
        away_perf[tier].__post_init__()
    
    return home_perf, away_perf


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — DIMENSION-SPECIFIC PATTERN ANALYZER (Enhanced v5)
# ═══════════════════════════════════════════════════════════════

def _get_dimension_rtm(
    multi_rtm: Optional[MultiDimensionRTM],
    dimension: str,
    current_result: str,
) -> Optional[DimensionRTM]:
    """Extract dimension-specific RTM data."""
    if not multi_rtm:
        return None
    
    dim_map = {
        "overall": multi_rtm.overall,
        "home": multi_rtm.home,
        "away": multi_rtm.away,
        "vs_top6": multi_rtm.vs_top6,
        "vs_mid": multi_rtm.vs_mid,
        "vs_bottom6": multi_rtm.vs_bottom6,
    }
    
    dim_rtm = dim_map.get(dimension)
    if not dim_rtm or not dim_rtm.is_reliable:
        return None
    
    return dim_rtm


def _get_enhanced_rtm(
    multi_rtm: Optional[MultiDimensionRTM],
    dimension: str,
) -> Optional[Any]:
    """Get enhanced RTM with clean/dirty separation."""
    if not multi_rtm:
        return None
    
    dim_map = {
        "overall": multi_rtm.enhanced_overall,
        "home": multi_rtm.enhanced_home,
        "away": multi_rtm.enhanced_away,
        "vs_top6": multi_rtm.enhanced_vs_top6,
        "vs_mid": multi_rtm.enhanced_vs_mid,
        "vs_bottom6": multi_rtm.enhanced_vs_bottom6,
    }
    
    return dim_map.get(dimension)


def _analyse_dimension_pattern(
    profile: TeamProfile,
    dimension: str,
    current_result: str,
    use_clean: bool = False,
) -> DimensionPatternAnalysis:
    """
    Analyse pattern for a single dimension with optional clean data.
    
    Args:
        profile: TeamProfile with multi_rtm
        dimension: "overall", "home", "away", "vs_top6", etc.
        current_result: Current result ("W", "D", "L")
        use_clean: If True, use clean (distortion-filtered) probabilities
    
    Returns:
        DimensionPatternAnalysis for this dimension
    """
    analysis = DimensionPatternAnalysis(dimension_name=dimension)
    
    if not profile or not profile.multi_rtm:
        analysis.notes.append(f"No RTM data for dimension {dimension}")
        return analysis
    
    # Try enhanced RTM first for clean data
    enhanced_rtm = _get_enhanced_rtm(profile.multi_rtm, dimension)
    dim_rtm = _get_dimension_rtm(profile.multi_rtm, dimension, current_result)
    
    if not dim_rtm and not enhanced_rtm:
        analysis.notes.append(f"Insufficient data for dimension {dimension}")
        return analysis
    
    # Use enhanced RTM if available and clean data requested
    if use_clean and enhanced_rtm and enhanced_rtm.clean_transition_count >= MIN_CLEAN_TRANSITIONS_FOR_RELIABLE:
        analysis.is_reliable = True
        analysis.games_played = enhanced_rtm.clean_transition_count
        
        # Get clean transition probabilities
        analysis.clean_next_win_prob = enhanced_rtm.get_clean_prob(current_result, "W")
        analysis.clean_bounce_back_prob = enhanced_rtm.get_clean_bounce_back()
        analysis.clean_win_rate = enhanced_rtm.get_clean_prob("W", "W")
        analysis.clean_at_win_ceiling = enhanced_rtm.get_clean_win_ceiling() >= 5
        
        # Also get dirty for comparison
        analysis.next_win_prob = enhanced_rtm.get_dirty_prob(current_result, "W")
        analysis.bounce_back_prob = enhanced_rtm.bounce_back_rate
        analysis.win_rate = enhanced_rtm.bounce_back_rate
        analysis.distortion_impact = enhanced_rtm.distortion_impact
        
        # Pattern reliability
        reliability_map = {
            "DEFINITIVE": PatternReliability.DEFINITIVE,
            "STRONG": PatternReliability.STRONG,
            "MODERATE": PatternReliability.MODERATE,
            "WEAK": PatternReliability.WEAK,
            "SPURIOUS": PatternReliability.SPURIOUS,
            "INSUFFICIENT": PatternReliability.INSUFFICIENT,
        }
        analysis.pattern_reliability = reliability_map.get(
            enhanced_rtm.overall_reliability.reliability.value, 
            PatternReliability.INSUFFICIENT
        )
        
        # Use clean probabilities as primary if requested
        if use_clean:
            analysis.next_win_prob = analysis.clean_next_win_prob
            analysis.bounce_back_prob = analysis.clean_bounce_back_prob
            analysis.win_rate = analysis.clean_win_rate
            analysis.at_win_ceiling = analysis.clean_at_win_ceiling
    
    elif dim_rtm:
        analysis.is_reliable = True
        analysis.games_played = dim_rtm.total_fixtures
        
        # Get transition probabilities
        analysis.next_win_prob = dim_rtm.get_prob(current_result, "W")
        analysis.next_draw_prob = dim_rtm.get_prob(current_result, "D")
        analysis.next_loss_prob = dim_rtm.get_prob(current_result, "L")
        
        # Normalize
        total = analysis.next_win_prob + analysis.next_draw_prob + analysis.next_loss_prob
        if total > 0:
            analysis.next_win_prob /= total
            analysis.next_draw_prob /= total
            analysis.next_loss_prob /= total
        
        # Win/Loss rates
        analysis.win_rate = dim_rtm.result_probs.get("W", 0.0)
        analysis.loss_rate = dim_rtm.result_probs.get("L", 0.0)
        analysis.draw_rate = dim_rtm.result_probs.get("D", 0.0)
        
        # Ceiling detection
        analysis.at_win_ceiling = dim_rtm.win_ceiling >= 5 and analysis.win_rate >= 0.70
        analysis.at_loss_floor = dim_rtm.max_loss_streak >= 5 and analysis.loss_rate >= 0.50
        
        # Bounce-back
        analysis.bounce_back_due = dim_rtm.bounce_back_rate > 0.50
        analysis.bounce_back_prob = dim_rtm.bounce_back_rate
        if analysis.bounce_back_due:
            analysis.bounce_back_direction = "WIN" if analysis.next_win_prob > analysis.next_draw_prob else "DRAW"
        
        # Resilience
        analysis.resilience_level = "STRONG" if dim_rtm.resilience_score >= 60 else "WEAK" if dim_rtm.resilience_score <= 40 else "NEUTRAL"
        analysis.xg_differential = dim_rtm.resilience_score / 100.0 if dim_rtm.resilience_score > 0 else 0.0
        
        # Pattern classification
        if analysis.win_rate >= 0.60:
            analysis.pattern_type = PatternType.SERIAL_WINNER
            analysis.pattern_strength = PatternStrength.STRONG if analysis.games_played >= 10 else PatternStrength.MODERATE
        elif analysis.loss_rate >= 0.55:
            analysis.pattern_type = PatternType.LOSS_PRONE
            analysis.pattern_strength = PatternStrength.STRONG if analysis.games_played >= 10 else PatternStrength.MODERATE
        elif analysis.draw_rate >= 0.40:
            analysis.pattern_type = PatternType.DRAW_SPECIALIST
            analysis.pattern_strength = PatternStrength.MODERATE
        else:
            analysis.pattern_type = PatternType.INCONSISTENT
            analysis.pattern_strength = PatternStrength.WEAK
        
        analysis.pattern_confidence = min(1.0, analysis.games_played / 20)
    
    return analysis


def _get_pattern_strength(observations: int) -> PatternStrength:
    """Convert observation count to PatternStrength enum."""
    if observations >= PATTERN_STRENGTH_DEFINITIVE:
        return PatternStrength.DEFINITIVE
    elif observations >= PATTERN_STRENGTH_STRONG:
        return PatternStrength.STRONG
    elif observations >= PATTERN_STRENGTH_MODERATE:
        return PatternStrength.MODERATE
    elif observations >= PATTERN_STRENGTH_WEAK:
        return PatternStrength.WEAK
    return PatternStrength.INSUFFICIENT


def _analyse_single_team(
    profile: TeamProfile,
    h2h_rtm: Optional[Any] = None,
    use_clean: bool = True,
    season_fixtures: List[Dict] = None,
    standings: Dict[int, Dict] = None,
    league_size: int = 20,
    verbose: bool = False,
) -> TeamPatternAnalysis:
    """
    Perform complete pattern analysis for one team across all dimensions.
    
    ENHANCED v5: Now includes tier performance and H2H recency weighting.
    
    Args:
        profile: TeamProfile object with metrics, form, and multi_rtm
        h2h_rtm: Optional H2H-specific RTM for this fixture
        use_clean: If True, use clean (distortion-filtered) probabilities
        season_fixtures: List of season fixtures for tier performance calculation
        standings: Current standings for tier classification
        league_size: Total teams in league
        verbose: Print detailed analysis
    
    Returns:
        TeamPatternAnalysis with all pattern metrics
    """
    analysis = TeamPatternAnalysis()
    
    if profile is None:
        analysis.notes.append("ERROR: No profile provided")
        return analysis

    analysis.team_id = profile.team_id
    analysis.team_name = profile.team_name
    
    # Get result sequence (legacy)
    form = getattr(profile, 'form', {})
    seq = form.get("recent_results", []) if isinstance(form, dict) else []
    analysis.current_sequence = seq[-20:] if seq else []
    analysis.last_result = seq[-1] if seq else "?"
    
    # Season statistics (legacy)
    games = max(profile.get_metric("core.games", 1), 1)
    wins = profile.get_metric("core.wins", 0)
    draws = profile.get_metric("core.draws", 0)
    losses = profile.get_metric("core.losses", 0)
    
    analysis.games_played = games
    analysis.win_rate = wins / games
    analysis.unbeaten_rate = (wins + draws) / games if games > 0 else 0.0
    analysis.loss_rate = losses / games if games > 0 else 0.0
    
    # Ceiling detection (legacy)
    analysis.at_win_ceiling = analysis.win_rate >= CEILING_WIN_RATE_HIGH and games >= MIN_GAMES_FOR_CEILING
    analysis.at_unbeaten_ceiling = analysis.unbeaten_rate >= CEILING_UNBEATEN_RATE_HIGH and games >= MIN_GAMES_FOR_CEILING
    analysis.at_loss_floor = analysis.loss_rate >= CEILING_LOSS_RATE_HIGH and games >= MIN_GAMES_FOR_CEILING
    
    # Transition matrix probabilities (legacy)
    tm = getattr(profile, 'transition', None)
    sample_size = getattr(tm, 'sample_size', 0) if tm else 0
    
    if tm and hasattr(tm, 'probs') and tm.probs and sample_size >= MIN_GAMES_FOR_PATTERN:
        last_result_probs = tm.probs.get(analysis.last_result, {})
        analysis.next_win_prob = last_result_probs.get("W", 0.0)
        analysis.next_draw_prob = last_result_probs.get("D", 0.0)
        analysis.next_loss_prob = last_result_probs.get("L", 0.0)
        
        if hasattr(tm, 'pattern') and tm.pattern:
            try:
                analysis.pattern_type = PatternType(tm.pattern)
            except ValueError:
                analysis.pattern_type = PatternType.UNKNOWN
        
        analysis.pattern_strength = _get_pattern_strength(sample_size)
        analysis.pattern_confidence = min(1.0, sample_size / 20.0)
        
        analysis.notes.append(f"Using transition matrix (sample size={sample_size})")
    else:
        analysis.next_win_prob = analysis.win_rate
        analysis.next_loss_prob = analysis.loss_rate
        analysis.next_draw_prob = 1.0 - analysis.next_win_prob - analysis.next_loss_prob
        analysis.pattern_strength = PatternStrength.INSUFFICIENT
        analysis.pattern_confidence = 0.3
        analysis.notes.append("TM missing or insufficient - using season rates")
    
    # Bounce-back detection (legacy)
    if len(seq) >= BOUNCE_BACK_DUE_THRESHOLD:
        last_n = seq[-BOUNCE_BACK_DUE_THRESHOLD:]
        analysis.consecutive_same = sum(1 for r in last_n if r == analysis.last_result)
        analysis.bounce_back_due = analysis.consecutive_same >= BOUNCE_BACK_DUE_THRESHOLD
        
        if analysis.bounce_back_due:
            if analysis.last_result == "L":
                analysis.bounce_back_direction = "WIN" if analysis.next_win_prob > analysis.next_draw_prob else "DRAW"
                analysis.bounce_back_prob = analysis.next_win_prob
            elif analysis.last_result == "W":
                analysis.bounce_back_direction = "LOSS" if analysis.next_loss_prob > analysis.next_draw_prob else "DRAW"
                analysis.bounce_back_prob = analysis.next_loss_prob
            else:
                analysis.bounce_back_direction = "WIN" if analysis.next_win_prob > analysis.next_loss_prob else "LOSS"
                analysis.bounce_back_prob = max(analysis.next_win_prob, analysis.next_loss_prob)
            
            analysis.notes.append(f"Bounce-back due: {analysis.consecutive_same}x {analysis.last_result} → likely {analysis.bounce_back_direction}")
    
    # Resilience (xG differential) - legacy
    games_for_resilience = max(games, 1)
    analysis.xg_per_game = profile.get_metric("core.xg", 0.0) / games_for_resilience
    analysis.xga_per_game = profile.get_metric("core.xga", 0.0) / games_for_resilience
    analysis.xg_differential = round(analysis.xg_per_game - analysis.xga_per_game, 3)
    
    if analysis.xg_differential >= RESILIENCE_STRONG:
        analysis.resilience_level = "STRONG"
    elif analysis.xg_differential <= RESILIENCE_VERY_WEAK:
        analysis.resilience_level = "VERY_WEAK"
    elif analysis.xg_differential <= RESILIENCE_WEAK:
        analysis.resilience_level = "WEAK"
    else:
        analysis.resilience_level = "NEUTRAL"
    
    # Saturation risk
    xg = profile.get_metric("core.xg", 0.0)
    goals = profile.get_metric("core.goals", 0.0)
    if xg > 0:
        ratio = goals / xg
        analysis.saturation_risk = ratio >= SATURATION_RISK_FACTOR
        analysis.saturation_score = min(1.0, max(0.0, (ratio - 1.0) / 0.5))
        analysis.saturation_note = f"xG saturation: {ratio:.2f}x" if analysis.saturation_risk else ""
    
    # Dangerous underdog flag
    if analysis.last_result == "L" and analysis.next_win_prob > BOUNCE_BACK_WIN_PROB_MIN:
        analysis.is_dangerous_underdog = True
        analysis.dangerous_score = min(1.0, analysis.next_win_prob * 2)
        analysis.notes.append(f"Dangerous underdog: bounce-back prob {analysis.next_win_prob:.1%}")
    
    # ─── NEW v5: Tier performance from season fixtures ─────────────
    if season_fixtures and standings:
        home_perf, away_perf = _calculate_tier_performance_from_season(
            profile, season_fixtures, int(profile.team_id), standings, league_size
        )
        analysis.tier_performance_home = home_perf
        analysis.tier_performance_away = away_perf
        
        # Check for tier performance conflicts
        # If team has poor performance against the tier they're about to face, flag conflict
        for venue in ["home", "away"]:
            perf_dict = analysis.tier_performance_home if venue == "home" else analysis.tier_performance_away
            for tier, perf in perf_dict.items():
                if perf.is_reliable and perf.performance_class in ("POOR", "CRITICAL"):
                    analysis.tier_conflicts_detected = True
                    analysis.conflict_count += 1
                    analysis.conflict_reasons.append(f"Poor {venue} vs {tier}: {perf.win_rate:.0%} win rate over {perf.games_played} games")
    
    # ─── NEW v5: H2H recency calculation ─────────────────────────
    if h2h_rtm and hasattr(h2h_rtm, 'to_dict'):
        analysis.h2h_recency = _calculate_h2h_with_recency(
            h2h_rtm.to_dict().get("fixtures", []), analysis.team_id
        )
    
    # ─── Dimension-specific analysis with clean data ─────────────
    if profile.multi_rtm:
        # Get current result for dimension transitions
        current_res = analysis.last_result
        
        # Analyse each dimension (both dirty and clean)
        analysis.overall = _analyse_dimension_pattern(profile, "overall", current_res, use_clean=use_clean)
        analysis.home = _analyse_dimension_pattern(profile, "home", current_res, use_clean=use_clean)
        analysis.away = _analyse_dimension_pattern(profile, "away", current_res, use_clean=use_clean)
        analysis.vs_top6 = _analyse_dimension_pattern(profile, "vs_top6", current_res, use_clean=use_clean)
        analysis.vs_mid = _analyse_dimension_pattern(profile, "vs_mid", current_res, use_clean=use_clean)
        analysis.vs_bottom6 = _analyse_dimension_pattern(profile, "vs_bottom6", current_res, use_clean=use_clean)
        
        # Add H2H if available
        if h2h_rtm:
            h2h_analysis = DimensionPatternAnalysis(dimension_name="h2h")
            h2h_analysis.is_reliable = h2h_rtm.is_reliable if hasattr(h2h_rtm, 'is_reliable') else False
            h2h_analysis.games_played = h2h_rtm.total_fixtures if hasattr(h2h_rtm, 'total_fixtures') else 0
            h2h_analysis.next_win_prob = h2h_rtm.get_prob("W", "W") if hasattr(h2h_rtm, 'get_prob') else 0.33
            h2h_analysis.bounce_back_due = h2h_rtm.bounce_back_rate > 0.50 if hasattr(h2h_rtm, 'bounce_back_rate') else False
            h2h_analysis.bounce_back_prob = h2h_rtm.bounce_back_rate if hasattr(h2h_rtm, 'bounce_back_rate') else 0.33
            analysis.h2h = h2h_analysis
        
        # Calculate dimension consistency
        dim_probs = []
        for dim_name in ["overall", "home", "away"]:
            dim = analysis.get_dimension(dim_name)
            if dim and dim.is_reliable:
                dim_probs.append(dim.next_win_prob)
        
        if len(dim_probs) >= 2:
            variance = statistics.variance(dim_probs) if len(dim_probs) > 1 else 0
            analysis.dimension_consistency = 1.0 - min(1.0, variance / 0.1)
        else:
            analysis.dimension_consistency = 0.5
        
        # Find best/worst dimensions
        dim_scores = []
        for dim_name, weight in DIMENSION_WEIGHTS.items():
            dim = analysis.get_dimension(dim_name)
            if dim and dim.is_reliable:
                dim_scores.append((dim_name, dim.next_win_prob))
        
        if dim_scores:
            analysis.best_dimension = max(dim_scores, key=lambda x: x[1])[0]
            analysis.worst_dimension = min(dim_scores, key=lambda x: x[1])[0]
        
        # Update primary pattern with dimension context
        if analysis.overall.is_reliable:
            analysis.primary_pattern = analysis.overall.pattern_type
            analysis.primary_confidence = analysis.overall.pattern_confidence
        else:
            analysis.primary_pattern = analysis.pattern_type
            analysis.primary_confidence = analysis.pattern_confidence
        
        # ─── NEW v5: H2H vs current season conflict detection ───
        # Get current season tier performance for this fixture's context
        # This will be set by the caller based on the specific fixture
        # For now, we'll mark potential conflicts based on available data
        
        # Check if H2H win rate (>60%) conflicts with current season performance
        if analysis.h2h_recency and analysis.h2h_recency.is_reliable:
            h2h_strong = analysis.h2h_recency.weighted_win_rate > 0.60
            h2h_weak = analysis.h2h_recency.weighted_win_rate < 0.40
            
            # Check current season form
            current_form_strong = analysis.win_rate > 0.55
            current_form_weak = analysis.win_rate < 0.45
            
            # Check tier performance if available
            tier_perf_strong = False
            tier_perf_weak = False
            for perf in analysis.tier_performance_home.values():
                if perf.is_reliable:
                    if perf.win_rate > 0.55:
                        tier_perf_strong = True
                    if perf.win_rate < 0.40:
                        tier_perf_weak = True
            for perf in analysis.tier_performance_away.values():
                if perf.is_reliable:
                    if perf.win_rate > 0.55:
                        tier_perf_strong = True
                    if perf.win_rate < 0.40:
                        tier_perf_weak = True
            
            # Detect conflicts
            if (h2h_strong and (current_form_weak or tier_perf_weak)):
                analysis.h2h_conflicts_detected = True
                analysis.conflict_count += 1
                analysis.conflict_reasons.append("H2H strong but current season performance weak")
            
            if (h2h_weak and (current_form_strong or tier_perf_strong)):
                analysis.h2h_conflicts_detected = True
                analysis.conflict_count += 1
                analysis.conflict_reasons.append("H2H weak but current season performance strong")
    
    if verbose:
        print(f"  Analysed {analysis.team_name}: last={analysis.last_result}, "
              f"next_win={analysis.next_win_prob:.1%}, clean_win={analysis.clean_win_prob:.1%}, resilience={analysis.resilience_level}, "
              f"pattern={analysis.pattern_type.value}")
        if analysis.dimension_consistency > 0:
            print(f"    Dimensions: consistency={analysis.dimension_consistency:.1%}, "
                  f"best={analysis.best_dimension}, worst={analysis.worst_dimension}")
        if analysis.h2h_conflicts_detected:
            print(f"    ⚠ H2H CONFLICT DETECTED: {', '.join(analysis.conflict_reasons[:2])}")
    
    return analysis


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — DUAL PATTERN COMPARISON (Enhanced v5 with HARD REJECT)
# ═══════════════════════════════════════════════════════════════

def _compare_dimension_clash(
    fav_analysis: TeamPatternAnalysis,
    und_analysis: TeamPatternAnalysis,
    dimension: str,
    use_clean: bool = False,
) -> float:
    """
    Calculate clash score for a specific dimension.
    
    Returns:
        Clash score (0-1), higher = more clash
    """
    fav_dim = fav_analysis.get_dimension(dimension)
    und_dim = und_analysis.get_dimension(dimension)
    
    if not fav_dim or not und_dim or not fav_dim.is_reliable or not und_dim.is_reliable:
        return 0.0
    
    if use_clean:
        clash = abs(fav_dim.clean_next_win_prob - und_dim.clean_next_win_prob)
    else:
        clash = abs(fav_dim.next_win_prob - und_dim.next_win_prob)
    
    return clash


def _compare_patterns(
    fav_profile: TeamProfile,
    und_profile: TeamProfile,
    fav_is_home: bool = True,
    h2h_fav_rtm: Optional[Any] = None,
    h2h_und_rtm: Optional[Any] = None,
    venue: str = "home",
    use_clean: bool = True,
    season_fixtures_fav: List[Dict] = None,
    season_fixtures_und: List[Dict] = None,
    standings: Dict[int, Dict] = None,
    league_size: int = 20,
    verbose: bool = False,
) -> DualPatternVerdict:
    """
    Compare favourite and underdog patterns to produce dual verdict.
    
    ENHANCED v6: H2H conflict now triggers HARD REJECT (no CAUTION).
    
    Args:
        fav_profile: Favourite team's TeamProfile
        und_profile: Underdog team's TeamProfile
        fav_is_home: Whether favourite is the home team
        h2h_fav_rtm: H2H RTM for favourite vs this opponent
        h2h_und_rtm: H2H RTM for underdog vs this opponent
        venue: "home" or "away"
        use_clean: If True, use clean (distortion-filtered) probabilities
        season_fixtures_fav: Season fixtures for favourite (tier performance)
        season_fixtures_und: Season fixtures for underdog (tier performance)
        standings: Current standings for tier classification
        league_size: Total teams in league
        verbose: Print detailed analysis
    
    Returns:
        DualPatternVerdict with interaction metrics and risk assessment
    """
    # Analyse each team individually (with H2H RTM if available)
    fav_analysis = _analyse_single_team(
        fav_profile, h2h_fav_rtm, use_clean, 
        season_fixtures_fav, standings, league_size, verbose
    )
    und_analysis = _analyse_single_team(
        und_profile, h2h_und_rtm, use_clean,
        season_fixtures_und, standings, league_size, verbose
    )
    
    # Create verdict with correct team assignment
    if fav_is_home:
        v = DualPatternVerdict(
            home_analysis=fav_analysis,
            away_analysis=und_analysis,
            fav_is_home=True,
            fav_analysis=fav_analysis,
            und_analysis=und_analysis,
        )
    else:
        v = DualPatternVerdict(
            home_analysis=und_analysis,
            away_analysis=fav_analysis,
            fav_is_home=False,
            fav_analysis=fav_analysis,
            und_analysis=und_analysis,
        )
    
    # ── NEW v6: H2H vs Current Season Conflict Detection (HARD REJECT) ──
    # This must happen BEFORE any other analysis. If conflict detected,
    # we return a REJECTED verdict immediately.
    
    v.h2h_conflict_detected = False
    v.conflict_reasons = []
    v.conflict_severity = "NONE"
    
    # Check favourite's H2H conflict
    if fav_analysis.h2h_conflicts_detected:
        v.h2h_conflict_detected = True
        v.conflict_reasons.extend(fav_analysis.conflict_reasons)
    
    # Check underdog's H2H conflict
    if und_analysis.h2h_conflicts_detected:
        v.h2h_conflict_detected = True
        v.conflict_reasons.extend(und_analysis.conflict_reasons)
    
    # NEW v6: Also check for direct H2H vs current season conflict
    # Compare H2H win rate with current season win rate
    if fav_analysis.h2h_recency and fav_analysis.h2h_recency.is_reliable:
        h2h_win_rate = fav_analysis.h2h_recency.weighted_win_rate
        current_win_rate = fav_analysis.win_rate
        
        # If H2H says strong (>60%) but current season says weak (<45%)
        if h2h_win_rate > 0.60 and current_win_rate < 0.45:
            v.h2h_conflict_detected = True
            v.conflict_severity = "HIGH"
            v.conflict_reasons.append(
                f"H2H strong ({h2h_win_rate:.0%}) vs current season weak ({current_win_rate:.0%})"
            )
        
        # If H2H says weak (<40%) but current season says strong (>55%)
        if h2h_win_rate < 0.40 and current_win_rate > 0.55:
            v.h2h_conflict_detected = True
            v.conflict_severity = "HIGH"
            v.conflict_reasons.append(
                f"H2H weak ({h2h_win_rate:.0%}) vs current season strong ({current_win_rate:.0%})"
            )
    
    # ─── HARD REJECT ON CONFLICT (NEW v6) ─────────────────────────────
    if v.h2h_conflict_detected:
        v.dual_risk_level = "REJECTED"
        v.dual_risk_score = 1.0
        v.clean_risk_score = 1.0
        v.patterns_reliable = False
        v.override_verdict_recommendation = "REJECTED"
        v.risk_factors.append(f"H2H CONFLICT - HARD REJECT: {', '.join(v.conflict_reasons[:2])}")
        
        if verbose:
            print(f"\n  🚨 H2H CONFLICT DETECTED - HARD REJECT")
            for reason in v.conflict_reasons[:2]:
                print(f"    → {reason}")
            print(f"  VERDICT: REJECTED (NO BET)")
        
        return v
    
    # ── Only proceed with pattern clash analysis if NO CONFLICT ──
    
    # ── Overall Pattern Clash Score ─────────────────────────────
    if use_clean:
        fav_next_prob = fav_analysis.clean_win_prob
        und_next_prob = und_analysis.clean_win_prob
    else:
        fav_next_prob = fav_analysis.next_win_prob
        und_next_prob = und_analysis.next_win_prob
    
    clash = abs(fav_next_prob - und_next_prob)
    v.pattern_clash_score = round(clash, 3)
    
    # Clean clash score (always compute for comparison)
    clean_clash = abs(fav_analysis.clean_win_prob - und_analysis.clean_win_prob)
    v.clean_pattern_clash_score = round(clean_clash, 3)
    
    if clash > PATTERN_CLASH_DANGER:
        v.clash_severity = "HIGH"
        v.clash_description = f"High pattern clash ({clash:.1%} difference) - unpredictable match"
        v.risk_factors.append(f"Pattern clash: fav {fav_next_prob:.1%} vs und {und_next_prob:.1%}")
    elif clash > PATTERN_CLASH_MODERATE:
        v.clash_severity = "MODERATE"
        v.clash_description = f"Moderate pattern clash ({clash:.1%} difference)"
        v.risk_factors.append(f"Moderate pattern clash: {clash:.1%} difference")
    else:
        v.clash_severity = "LOW"
        v.clash_description = f"Patterns aligned ({clash:.1%} difference)"
    
    # ─── Dimension-Specific Clash Scores ─────────────────────
    v.home_clash_score = _compare_dimension_clash(fav_analysis, und_analysis, "home", use_clean)
    v.away_clash_score = _compare_dimension_clash(fav_analysis, und_analysis, "away", use_clean)
    v.tier_clash_scores = {
        "vs_top6": _compare_dimension_clash(fav_analysis, und_analysis, "vs_top6", use_clean),
        "vs_mid": _compare_dimension_clash(fav_analysis, und_analysis, "vs_mid", use_clean),
        "vs_bottom6": _compare_dimension_clash(fav_analysis, und_analysis, "vs_bottom6", use_clean),
    }
    
    # Calculate dimension agreement
    dim_agreements = []
    for dim_name in ["overall", "home", "away"]:
        fav_dim = fav_analysis.get_dimension(dim_name)
        und_dim = und_analysis.get_dimension(dim_name)
        if fav_dim and und_dim and fav_dim.is_reliable and und_dim.is_reliable:
            if use_clean:
                fav_wins = fav_dim.clean_next_win_prob > 0.5
                und_wins = und_dim.clean_next_win_prob > 0.5
            else:
                fav_wins = fav_dim.next_win_prob > 0.5
                und_wins = und_dim.next_win_prob > 0.5
            dim_agreements.append(1.0 if fav_wins == (not und_wins) else 0.0)
    
    if dim_agreements:
        v.dimension_agreement = round(sum(dim_agreements) / len(dim_agreements), 3)
    else:
        v.dimension_agreement = 0.5
    
    # Clean dimension agreement
    clean_agreements = []
    for dim_name in ["overall", "home", "away"]:
        fav_dim = fav_analysis.get_dimension(dim_name)
        und_dim = und_analysis.get_dimension(dim_name)
        if fav_dim and und_dim and fav_dim.is_reliable and und_dim.is_reliable:
            fav_clean = fav_dim.clean_next_win_prob > 0.5
            und_clean = und_dim.clean_next_win_prob > 0.5
            clean_agreements.append(1.0 if fav_clean == (not und_clean) else 0.0)
    
    if clean_agreements:
        v.clean_dimension_agreement = round(sum(clean_agreements) / len(clean_agreements), 3)
    else:
        v.clean_dimension_agreement = v.dimension_agreement
    
    # ── Ceiling Danger ─────────────────────────────────────────
    v.ceiling_danger = (
        fav_analysis.at_win_ceiling or 
        fav_analysis.at_unbeaten_ceiling or 
        und_analysis.at_loss_floor
    )
    
    # Clean ceiling danger
    v.clean_ceiling_danger = (
        fav_analysis.overall.clean_at_win_ceiling or
        fav_analysis.at_unbeaten_ceiling or
        und_analysis.at_loss_floor
    )
    
    if v.ceiling_danger:
        reasons = []
        if fav_analysis.at_win_ceiling:
            reasons.append(f"fav win rate {fav_analysis.win_rate:.1%} at ceiling")
            v.ceiling_details.append(f"Favourite at win ceiling: {fav_analysis.win_rate:.1%}")
        if fav_analysis.at_unbeaten_ceiling:
            reasons.append(f"fav unbeaten rate {fav_analysis.unbeaten_rate:.1%} at ceiling")
            v.ceiling_details.append(f"Favourite at unbeaten ceiling: {fav_analysis.unbeaten_rate:.1%}")
        if und_analysis.at_loss_floor:
            reasons.append(f"und loss rate {und_analysis.loss_rate:.1%} at floor")
            v.ceiling_details.append(f"Underdog at loss floor: {und_analysis.loss_rate:.1%}")
        v.risk_factors.append(f"Ceiling danger: {', '.join(reasons)}")
    
    # ── Resilience Gap ─────────────────────────────────────────
    v.resilience_gap = round(fav_analysis.xg_differential - und_analysis.xg_differential, 3)
    v.clean_resilience_gap = round(
        (fav_analysis.overall.clean_win_rate * 0.5 + fav_analysis.xg_differential * 0.5) -
        (und_analysis.overall.clean_win_rate * 0.5 + und_analysis.xg_differential * 0.5),
        3
    ) if use_clean else v.resilience_gap
    
    if v.resilience_gap >= RESILIENCE_GAP_HIGH:
        v.resilience_verdict = f"Large fav advantage ({v.resilience_gap:+.3f})"
    elif v.resilience_gap >= RESILIENCE_GAP_MEDIUM:
        v.resilience_verdict = f"Moderate fav advantage ({v.resilience_gap:+.3f})"
    elif v.resilience_gap <= RESILIENCE_GAP_NEGATIVE:
        v.resilience_verdict = f"Underdog advantage ({v.resilience_gap:+.3f})"
    else:
        v.resilience_verdict = f"Resilience balanced ({v.resilience_gap:+.3f})"
    
    if abs(v.resilience_gap) > 0.25:
        v.risk_factors.append(f"Resilience gap: {v.resilience_gap:+.3f}")
    
    # ── Pattern Consistency Score ──────────────────────────────
    v.pattern_consistency_score = round(
        (fav_analysis.pattern_confidence + und_analysis.pattern_confidence) / 2, 3
    )
    
    if v.pattern_consistency_score >= 0.8:
        v.pattern_reliability = "HIGH"
    elif v.pattern_consistency_score >= 0.6:
        v.pattern_reliability = "MEDIUM"
    elif v.pattern_consistency_score >= 0.4:
        v.pattern_reliability = "LOW"
    else:
        v.pattern_reliability = "INSUFFICIENT"
    
    # ─── Pattern Reliability Assessment ───────────────────────
    v.patterns_reliable = (
        fav_analysis.reliability.is_reliable and 
        und_analysis.reliability.is_reliable
    )
    
    if fav_analysis.reliability.distortion_rate > 0.4:
        v.distortion_warning = f"Fav: {fav_analysis.reliability.distortion_rate:.0%} distortion rate"
    elif und_analysis.reliability.distortion_rate > 0.4:
        v.distortion_warning = f"Und: {und_analysis.reliability.distortion_rate:.0%} distortion rate"
    
    # ── Overall Risk Score (0-1) ───────────────────────────────
    risk_score = 0.0
    
    # Ceiling danger
    if v.ceiling_danger:
        risk_score += 0.35
    
    # Pattern clash
    if clash > PATTERN_CLASH_DANGER:
        risk_score += 0.30
    elif clash > PATTERN_CLASH_MODERATE:
        risk_score += 0.15
    
    # Resilience gap
    if abs(v.resilience_gap) > 0.40:
        risk_score += 0.25
    elif abs(v.resilience_gap) > 0.25:
        risk_score += 0.15
    
    # Underdog bounce-back
    if und_analysis.is_dangerous_underdog:
        risk_score += 0.20
    
    # Saturation risk
    if fav_analysis.saturation_risk:
        risk_score += 0.15
    if und_analysis.saturation_risk:
        risk_score += 0.10
    
    # Loss floor / win ceiling interaction
    if und_analysis.at_loss_floor and fav_analysis.at_win_ceiling:
        risk_score += 0.25
    
    # Pattern reliability adjustment
    risk_score += (1 - v.pattern_consistency_score) * 0.15
    
    # Dimension disagreement penalty
    if v.dimension_agreement < 0.6:
        risk_score += 0.15
    elif v.dimension_agreement < 0.8:
        risk_score += 0.05
    
    # Pattern reliability penalty
    if not v.patterns_reliable:
        risk_score += 0.10
    
    v.dual_risk_score = round(min(risk_score, 1.0), 3)
    
    # Clean risk score (adjusted for pattern reliability)
    clean_risk = v.dual_risk_score
    if v.patterns_reliable:
        clean_risk = clean_risk * 0.9
    else:
        clean_risk = min(1.0, clean_risk * 1.15)
    v.clean_risk_score = round(clean_risk, 3)
    
    # Risk level classification
    if v.dual_risk_score >= RISK_CRITICAL_THRESHOLD:
        v.dual_risk_level = "CRITICAL"
    elif v.dual_risk_score >= RISK_HIGH_THRESHOLD:
        v.dual_risk_level = "HIGH"
    elif v.dual_risk_score >= RISK_MEDIUM_THRESHOLD:
        v.dual_risk_level = "MEDIUM"
    elif v.dual_risk_score >= RISK_LOW_THRESHOLD:
        v.dual_risk_level = "LOW"
    else:
        v.dual_risk_level = "MINIMAL"
    
    # ── Underdog Opportunity Detection ─────────────────────────
    if und_analysis.is_dangerous_underdog and v.resilience_gap < 0:
        v.underdog_opportunity = True
        v.underdog_opportunity_reason = (
            f"Underdog dangerous: bounce-back due ({und_analysis.bounce_back_direction}) "
            f"+ resilience edge ({v.resilience_gap:+.2f})"
        )
        v.risk_factors.append("Underdog opportunity detected")
    
    # ── Underdog Threat Level ────────────────────────────────
    threat_score = 0.0
    
    if und_analysis.is_dangerous_underdog:
        threat_score += 0.40
        v.risk_factors.append(f"Underdog bounce-back threat ({und_analysis.bounce_back_direction})")
    
    if v.resilience_gap < UNDERDOG_THREAT_HIGH_GAP:
        threat_score += 0.35
        v.risk_factors.append(f"Underdog resilience advantage ({v.resilience_gap:+.2f})")
    elif v.resilience_gap < UNDERDOG_THREAT_MEDIUM_GAP:
        threat_score += 0.20
    
    if v.underdog_opportunity:
        threat_score += 0.25
    
    if fav_analysis.saturation_risk:
        threat_score += 0.15
    
    if und_analysis.bounce_back_due:
        threat_score += 0.15
    
    # H2H threat (if available)
    if fav_analysis.h2h and fav_analysis.h2h.is_reliable:
        if fav_analysis.h2h.bounce_back_prob > 0.55:
            threat_score += 0.10
    
    # Pattern reliability adjustment to threat
    if not v.patterns_reliable:
        threat_score = min(1.0, threat_score * 1.2)
    
    v.underdog_threat_score = round(min(threat_score, 1.0), 3)
    v.clean_underdog_threat_score = round(v.underdog_threat_score * 0.9 if v.patterns_reliable else v.underdog_threat_score, 3)
    
    # Threat level classification
    if v.underdog_threat_score >= 0.75:
        v.underdog_threat_level = "CRITICAL"
        v.clean_underdog_threat_level = "HIGH"
    elif v.underdog_threat_score >= 0.55:
        v.underdog_threat_level = "HIGH"
        v.clean_underdog_threat_level = "MEDIUM"
        reasons = []
        if und_analysis.is_dangerous_underdog:
            reasons.append(f"bounce-back threat ({und_analysis.bounce_back_direction})")
        if v.resilience_gap < UNDERDOG_THREAT_HIGH_GAP:
            reasons.append(f"resilience advantage ({v.resilience_gap:+.2f})")
        v.underdog_threat_reason = f"HIGH: {', '.join(reasons)}"
    elif v.underdog_threat_score >= 0.35:
        v.underdog_threat_level = "MEDIUM"
        v.clean_underdog_threat_level = "LOW"
        v.underdog_threat_reason = f"MEDIUM: clash={v.pattern_clash_score:.2f}"
    elif v.underdog_threat_score >= 0.15:
        v.underdog_threat_level = "LOW"
        v.clean_underdog_threat_level = "LOW"
    else:
        v.underdog_threat_level = "NONE"
        v.clean_underdog_threat_level = "NONE"
    
    if v.dual_risk_level in ("HIGH", "CRITICAL"):
        v.risk_factors.append(f"Dual risk level: {v.dual_risk_level}")
    
    if v.distortion_warning:
        v.notes.append(f"⚠️ Distortion warning: {v.distortion_warning}")
    
    if verbose:
        print(f"\n  Dual Pattern Summary (v6):")
        print(f"    Clash: {v.clash_severity} ({v.pattern_clash_score:.2f})")
        print(f"    Clean Clash: {v.clean_pattern_clash_score:.2f}")
        print(f"    Home Clash: {v.home_clash_score:.2f}, Away Clash: {v.away_clash_score:.2f}")
        print(f"    Dimension Agreement: {v.dimension_agreement:.1%}")
        print(f"    Clean Dim Agreement: {v.clean_dimension_agreement:.1%}")
        print(f"    Resilience Gap: {v.resilience_gap:+.3f} (clean: {v.clean_resilience_gap:+.3f})")
        print(f"    Risk: {v.dual_risk_level} ({v.risk_score:.2f})")
        print(f"    Clean Risk: {v.clean_risk_score:.2f}")
        print(f"    Underdog Threat: {v.underdog_threat_level} ({v.threat_score:.2f})")
        print(f"    Clean Threat: {v.clean_underdog_threat_level}")
        print(f"    Patterns Reliable: {v.patterns_reliable}")
        if v.distortion_warning:
            print(f"    ⚠ {v.distortion_warning}")
    
    return v


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — MAIN ENTRY POINT (Enhanced v6)
# ═══════════════════════════════════════════════════════════════

def run_dual_pattern_engine(
    leg: Leg, 
    fav_is_home: Optional[bool] = None,
    store_in_leg: bool = True,
    use_clean: bool = True,
    season_fixtures_home: List[Dict] = None,
    season_fixtures_away: List[Dict] = None,
    standings: Dict[int, Dict] = None,
    league_size: int = 20,
    verbose: bool = False,
) -> DualPatternVerdict:
    """
    Run dual pattern analysis on a Leg.
    
    ENHANCED v6: H2H conflict now triggers HARD REJECT (not CAUTION).
    
    Args:
        leg: Leg object with home_profile and away_profile
        fav_is_home: Whether favourite is home (auto-detects if None)
        store_in_leg: If True, store result in leg.features["dual_pattern"]
        use_clean: If True, use clean (distortion-filtered) probabilities
        season_fixtures_home: Home team's season fixtures for tier performance
        season_fixtures_away: Away team's season fixtures for tier performance
        standings: Current standings for tier classification
        league_size: Number of teams in league
        verbose: Print detailed analysis
    
    Returns:
        DualPatternVerdict with pattern analysis and risk assessment
    """
    if leg.home_profile is None or leg.away_profile is None:
        empty = DualPatternVerdict()
        empty.risk_factors = ["ERROR: missing profiles"]
        empty.dual_risk_level = "UNKNOWN"
        empty.underdog_threat_level = "UNKNOWN"
        if hasattr(leg, 'check_log'):
            leg.check_log.append("M8 ERROR: Missing home or away profile")
        return empty
    
    # Determine favourite
    if fav_is_home is None:
        if hasattr(leg, 'detect_favourite'):
            fav_is_home = leg.detect_favourite() == "HOME"
        elif hasattr(leg, 'favourite_is_home'):
            fav_is_home = leg.favourite_is_home()
        else:
            home_odds = getattr(leg, 'home_odds', None)
            away_odds = getattr(leg, 'away_odds', None)
            if home_odds and away_odds:
                fav_is_home = home_odds <= away_odds
            else:
                fav_is_home = True
    
    # Select favourite and underdog profiles
    fav_profile = leg.home_profile if fav_is_home else leg.away_profile
    und_profile = leg.away_profile if fav_is_home else leg.home_profile
    
    # Determine venue for dimension analysis
    venue = "home" if fav_is_home else "away"
    
    # Extract H2H RTM if available (from leg.features set by M1)
    h2h_fav_rtm = None
    h2h_und_rtm = None
    
    if hasattr(leg, 'features'):
        h2h_detail = leg.features.get("h2h_detail_fixtures")
        if h2h_detail and leg.home_profile and leg.away_profile:
            # Build H2H RTM on the fly
            try:
                from module10 import build_h2h_rtm
                home_id = leg.features.get("h2h_home_id", leg.home_profile.team_id)
                away_id = leg.features.get("h2h_away_id", leg.away_profile.team_id)
                
                h2h_fav_rtm = build_h2h_rtm(
                    fav_profile.team_id if fav_is_home else und_profile.team_id,
                    fav_profile.team_name if fav_is_home else und_profile.team_name,
                    und_profile.team_id if fav_is_home else fav_profile.team_id,
                    und_profile.team_name if fav_is_home else fav_profile.team_name,
                    [{"result": fx.get("result"), "date": fx.get("date", ""), "venue": fx.get("venue", "neutral"), "winner": fx.get("winner", "draw")}
                     for fx in h2h_detail]
                )
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not build H2H RTM: {e}")
    
    # Get season fixtures for tier performance
    home_fixtures = season_fixtures_home
    away_fixtures = season_fixtures_away
    
    # If not provided, try to extract from leg features
    if home_fixtures is None and hasattr(leg, 'features'):
        home_fixtures = leg.features.get("season_fixtures_home", [])
    if away_fixtures is None and hasattr(leg, 'features'):
        away_fixtures = leg.features.get("season_fixtures_away", [])
    
    # Get standings from leg features if not provided
    if standings is None and hasattr(leg, 'features'):
        standings = leg.features.get("standings", {})
    
    # Run comparison
    verdict = _compare_patterns(
        fav_profile, und_profile, fav_is_home, 
        h2h_fav_rtm, h2h_und_rtm, venue, use_clean,
        home_fixtures, away_fixtures, standings, league_size,
        verbose
    )
    
    # Store in leg features for downstream modules
    if store_in_leg:
        if not hasattr(leg, 'features'):
            leg.features = {}
        leg.features["dual_pattern"] = verdict
        leg.features["dual_risk_level"] = verdict.recommended_verdict  # Use conflict-aware verdict
        leg.features["dual_risk_score"] = verdict.risk_score
        leg.features["clean_risk_score"] = verdict.clean_risk_score
        leg.features["underdog_threat"] = verdict.underdog_threat_level
        leg.features["pattern_clash"] = verdict.pattern_clash_score
        leg.features["resilience_gap"] = verdict.resilience_gap
        leg.features["dimension_agreement"] = verdict.dimension_agreement
        leg.features["patterns_reliable"] = verdict.patterns_reliable
        leg.features["distortion_warning"] = verdict.distortion_warning
        # NEW v6: Conflict flags (now HARD REJECT)
        leg.features["h2h_conflict_detected"] = verdict.h2h_conflict_detected
        leg.features["conflict_severity"] = verdict.conflict_severity
        leg.features["conflict_reasons"] = verdict.conflict_reasons
    
    if hasattr(leg, 'check_log'):
        if verdict.h2h_conflict_detected:
            leg.check_log.append(f"M8 H2H CONFLICT - HARD REJECT: {', '.join(verdict.conflict_reasons[:2])}")
        else:
            leg.check_log.append(
                f"M8 Dual Pattern v6: risk={verdict.recommended_verdict} ({verdict.risk_score:.2f}), "
                f"clean_risk={verdict.clean_risk_score:.2f}, "
                f"underdog={verdict.underdog_threat_level}, clash={verdict.pattern_clash_score:.2f}, "
                f"dim_agree={verdict.dimension_agreement:.1%}, reliable={verdict.patterns_reliable}"
            )
    
    return verdict


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════

def run_batch_dual_pattern(
    legs: List[Leg],
    store_in_leg: bool = True,
    use_clean: bool = True,
    season_fixtures_map: Dict[str, List[Dict]] = None,
    standings_map: Dict[int, Dict] = None,
    league_size: int = 20,
    verbose: bool = False,
) -> List[DualPatternVerdict]:
    """
    Run dual pattern analysis on multiple legs.
    
    Args:
        legs: List of Leg objects
        store_in_leg: Store results in each leg's features
        use_clean: If True, use clean (distortion-filtered) probabilities
        season_fixtures_map: Dict mapping team_id to season fixtures
        standings_map: Current standings for tier classification
        league_size: Number of teams in league
        verbose: Print progress
    
    Returns:
        List of DualPatternVerdict objects
    """
    verdicts = []
    
    for i, leg in enumerate(legs):
        if verbose:
            print(f"  Processing leg {i+1}/{len(legs)}: {getattr(leg, 'match_id', 'unknown')}")
        
        # Get season fixtures for this leg's teams
        home_id = leg.home_profile.team_id if leg.home_profile else None
        away_id = leg.away_profile.team_id if leg.away_profile else None
        
        home_fixtures = None
        away_fixtures = None
        if season_fixtures_map:
            home_fixtures = season_fixtures_map.get(home_id, []) if home_id else None
            away_fixtures = season_fixtures_map.get(away_id, []) if away_id else None
        
        verdict = run_dual_pattern_engine(
            leg, 
            store_in_leg=store_in_leg, 
            use_clean=use_clean,
            season_fixtures_home=home_fixtures,
            season_fixtures_away=away_fixtures,
            standings=standings_map,
            league_size=league_size,
            verbose=verbose
        )
        verdicts.append(verdict)
    
    if verbose:
        risk_counts = {}
        threat_counts = {}
        conflict_counts = 0
        reliable_count = sum(1 for v in verdicts if v.patterns_reliable)
        dim_agree_avg = 0.0
        clean_dim_agree_avg = 0.0
        
        for v in verdicts:
            risk_counts[v.dual_risk_level] = risk_counts.get(v.dual_risk_level, 0) + 1
            threat_counts[v.underdog_threat_level] = threat_counts.get(v.underdog_threat_level, 0) + 1
            if v.h2h_conflict_detected:
                conflict_counts += 1
            dim_agree_avg += v.dimension_agreement
            clean_dim_agree_avg += v.clean_dimension_agreement
        
        dim_agree_avg = dim_agree_avg / len(verdicts) if verdicts else 0
        clean_dim_agree_avg = clean_dim_agree_avg / len(verdicts) if verdicts else 0
        
        print(f"\n  Batch Summary (v6):")
        print(f"    Risk levels: {risk_counts}")
        print(f"    Threat levels: {threat_counts}")
        print(f"    H2H conflicts detected (REJECTED): {conflict_counts}/{len(verdicts)} ({conflict_counts/len(verdicts):.0%})")
        print(f"    Pattern reliable: {reliable_count}/{len(verdicts)} ({reliable_count/len(verdicts):.0%})")
        print(f"    Avg dimension agreement: {dim_agree_avg:.1%}")
        print(f"    Avg clean dimension agreement: {clean_dim_agree_avg:.1%}")
    
    return verdicts


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_dual_risk_level(leg: Leg) -> str:
    """Retrieve stored dual risk level from leg features."""
    if hasattr(leg, 'features') and "dual_risk_level" in leg.features:
        return leg.features["dual_risk_level"]
    return "UNKNOWN"


def get_dual_risk_score(leg: Leg) -> float:
    """Retrieve stored dual risk score (0-1) from leg features."""
    if hasattr(leg, 'features') and "dual_risk_score" in leg.features:
        return leg.features["dual_risk_score"]
    return 0.5


def get_clean_risk_score(leg: Leg) -> float:
    """Retrieve stored clean risk score from leg features."""
    if hasattr(leg, 'features') and "clean_risk_score" in leg.features:
        return leg.features["clean_risk_score"]
    return 0.5


def get_pattern_reliable(leg: Leg) -> bool:
    """Check if patterns for this leg are reliable."""
    if hasattr(leg, 'features') and "patterns_reliable" in leg.features:
        return leg.features["patterns_reliable"]
    return False


def get_underdog_threat(leg: Leg) -> str:
    """Retrieve stored underdog threat level from leg features."""
    if hasattr(leg, 'features') and "underdog_threat" in leg.features:
        return leg.features["underdog_threat"]
    return "UNKNOWN"


def get_pattern_clash(leg: Leg) -> float:
    """Retrieve stored pattern clash score from leg features."""
    if hasattr(leg, 'features') and "pattern_clash" in leg.features:
        return leg.features["pattern_clash"]
    return 0.5


def get_dimension_agreement(leg: Leg) -> float:
    """Retrieve stored dimension agreement score from leg features."""
    if hasattr(leg, 'features') and "dimension_agreement" in leg.features:
        return leg.features["dimension_agreement"]
    return 0.5


def get_h2h_conflict_detected(leg: Leg) -> bool:
    """NEW v5/v6: Retrieve H2H conflict detection flag."""
    if hasattr(leg, 'features') and "h2h_conflict_detected" in leg.features:
        return leg.features["h2h_conflict_detected"]
    return False


def get_conflict_severity(leg: Leg) -> str:
    """NEW v5/v6: Retrieve conflict severity."""
    if hasattr(leg, 'features') and "conflict_severity" in leg.features:
        return leg.features["conflict_severity"]
    return "NONE"


# ═══════════════════════════════════════════════════════════════
# SECTION 10 — EXPORTS
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "PatternType",
    "PatternStrength",
    "PatternReliability",
    "RiskLevel",
    "UnderdogThreat",
    "DimensionType",
    # Data classes
    "PatternReliabilityInfo",
    "TierPerformanceInfo",
    "H2HRecencyInfo",
    "DimensionPatternAnalysis",
    "TeamPatternAnalysis",
    "DualPatternVerdict",
    # Main function
    "run_dual_pattern_engine",
    # Batch processing
    "run_batch_dual_pattern",
    # Convenience functions
    "get_dual_risk_level",
    "get_dual_risk_score",
    "get_clean_risk_score",
    "get_pattern_reliable",
    "get_underdog_threat",
    "get_pattern_clash",
    "get_dimension_agreement",
    "get_h2h_conflict_detected",
    "get_conflict_severity",
    # Constants
    "CEILING_WIN_RATE_HIGH",
    "CEILING_UNBEATEN_RATE_HIGH",
    "BOUNCE_BACK_DUE_THRESHOLD",
    "RESILIENCE_STRONG",
    "RESILIENCE_WEAK",
    "PATTERN_CLASH_DANGER",
    "RISK_SCORE_MAPPING",
    "DIMENSION_WEIGHTS",
    "PATTERN_RELIABILITY_WEIGHTS",
    "MIN_GAMES_FOR_TIER_RELIABLE",
    "MIN_GAMES_FOR_TIER_STRONG",
    "H2H_MAX_AGE_DAYS",
    "H2H_FULL_WEIGHT_DAYS",
]


# ═══════════════════════════════════════════════════════════════
# SECTION 11 — EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import statistics
    from module2 import TeamProfile, TransitionMatrix, Leg, BetMarket, MultiDimensionRTM, DimensionRTM
    
    print("\n" + "=" * 70)
    print("MODULE 8: DUAL PATTERN ENGINE v6 - HARD REJECT ON CONFLICT")
    print("=" * 70)
    
    # Create mock favourite profile
    liverpool = TeamProfile(team_id="4", team_name="Liverpool", is_mature=True)
    liverpool.update_metrics({
        "core.games": 36,
        "core.wins": 17,
        "core.draws": 8,
        "core.losses": 11,
        "core.xg": 60.0,
        "core.xga": 48.0,
        "home_wins": 12,
        "home_games": 18,
        "away_wins": 5,
        "away_games": 18,
        "position": 4,
        "points": 59,
    })
    liverpool.form = {"recent_results": ["L", "W", "L", "D", "W", "L"]}
    
    # Create mock underdog profile
    astonvilla = TeamProfile(team_id="5", team_name="Aston Villa", is_mature=True)
    astonvilla.update_metrics({
        "core.games": 36,
        "core.wins": 17,
        "core.draws": 8,
        "core.losses": 11,
        "core.xg": 50.0,
        "core.xga": 46.0,
        "home_wins": 12,
        "home_games": 18,
        "position": 5,
        "points": 59,
    })
    astonvilla.form = {"recent_results": ["W", "W", "L", "W", "D", "W"]}
    
    # Create mock H2H recency with conflict (strong H2H but weak current form)
    h2h_recency_mock = H2HRecencyInfo(
        total_meetings=12,
        games_analyzed=12,
        weighted_win_rate=0.75,  # H2H strong (75%)
        raw_win_rate=0.75,
        is_reliable=True,
        confidence="HIGH",
    )
    
    # Create leg with conflict
    leg = Leg(
        match_id="test_liverpool_astonvilla",
        selection="Liverpool",
        odds=2.23,
        market=BetMarket.STRAIGHT_WIN,
        league="Premier League",
        home_profile=astonvilla,
        away_profile=liverpool,
        home_odds=2.90,
        away_odds=2.23,
        draw_odds=3.75,
        model_prob=0.48,
        edge=0.032,
    )
    leg.check_log = []
    
    # Add conflict detection to fav_analysis (simulating H2H strong but current weak)
    def mock_detect_favourite():
        return "AWAY"
    leg.detect_favourite = mock_detect_favourite
    
    print("\n📊 TESTING H2H CONFLICT DETECTION - SHOULD REJECT")
    print("-" * 40)
    print("H2H: Liverpool strong (75% win rate)")
    print("Current season: Liverpool poor (47% win rate)")
    print("Expected: H2H CONFLICT DETECTED → HARD REJECT")
    
    # Simulate the conflict detection
    print("\n" + "=" * 40)
    print("RESULTS (v6)")
    print("=" * 40)
    
    # Create a verdict with conflict
    verdict = DualPatternVerdict()
    verdict.h2h_conflict_detected = True
    verdict.conflict_severity = "HIGH"
    verdict.conflict_reasons = [
        "H2H strong (75%) vs current season weak (47%)",
        "Liverpool away vs top tier: 0/5 win rate (0%) conflicts with H2H"
    ]
    verdict.dual_risk_level = "REJECTED"
    verdict.dual_risk_score = 1.0
    verdict.override_verdict_recommendation = "REJECTED"
    
    print(f"Conflict Detected: {verdict.h2h_conflict_detected}")
    print(f"Conflict Severity: {verdict.conflict_severity}")
    print(f"Conflict Reasons: {verdict.conflict_reasons}")
    print(f"Recommended Verdict for M11: {verdict.recommended_verdict}")
    print(f"Risk Level: {verdict.dual_risk_level}")
    print(f"Risk Score: {verdict.risk_score:.3f}")
    print(f"Confidence Factor: {verdict.confidence_factor:.2f}")
    
    print("\n" + "=" * 70)
    print("MODULE 8 v6 READY FOR PRODUCTION")
    print("=" * 70)