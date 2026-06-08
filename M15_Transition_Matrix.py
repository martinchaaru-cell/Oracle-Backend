"""
The Match Oracle - Module 15: Results Transition Matrix (RTM) (ENHANCED v3)
=======================================================================
Tracks 9-outcome state transitions across all season results.
Validates pattern resilience, bounce-back capability, and win-streak ceiling.

ENHANCEMENTS IN THIS VERSION (v3):
------------------------------
1. ADDED: Distortion factor tracking for all transitions
2. ADDED: Clean matrix (undistorted transitions only)
3. ADDED: EnhancedDimensionMatrix with clean/dirty separation
4. ADDED: Pattern reliability scoring (DEFINITIVE → SPURIOUS)
5. ADDED: DistortionStats for tracking distortion frequencies
6. ADDED: Clean transition probability getters
7. ADDED: Pattern genuineness validation
8. ADDED: Distortion-adjusted predictions
9. ADDED: Reliability warnings for suspicious patterns

PREVIOUS ENHANCEMENTS:
---------------------
- Dimension-specific transition matrices (home, away, vs_tier)
- Venue-specific RTM for home/away pattern analysis
- Tier-specific RTM (vs Top 6, vs Mid, vs Bottom 6)
- H2H-specific RTM for rivalry analysis
- Dimension comparison utilities
- Multi-dimension streak tracking
- Dimension resilience scoring

States (3×3 matrix):
├─ Previous Result (3 states): WIN | DRAW | LOSS
├─ Current Result (3 states): WIN | DRAW | LOSS
└─ Transition: 9 possible outcomes

Dimensions supported:
├─ Overall (all fixtures)
├─ Home (home fixtures only)
├─ Away (away fixtures only)
├─ vs Top 6 (vs top 6 opponents)
├─ vs Mid (vs mid-table opponents)
├─ vs Bottom 6 (vs bottom 6 opponents)
└─ H2H (vs specific opponent)

NEW v3 Features:
├─ Clean vs Dirty matrix separation
├─ Distortion impact quantification
├─ Pattern reliability scoring
├─ Spurious pattern detection
└─ Distortion-adjusted predictions

Usage:
    from module15 import ResultsTransitionMatrix, OutcomeState, EnhancedDimensionMatrix
    
    rtm = ResultsTransitionMatrix()
    
    # Build from historical fixtures (with dimensions)
    analysis = rtm.build_season_matrix("2024-25", fixtures)
    
    # Get dimension-specific probability
    prob = rtm.get_dimension_probability("2024-25", "home", "WIN", "WIN")
    
    # Get clean probability (distortions removed)
    clean_prob = rtm.get_clean_probability("2024-25", "home", "WIN", "WIN")
    
    # Check if pattern is genuine
    is_genuine = rtm.is_pattern_genuine("2024-25", "bounce_back")
    
    # Validate system resilience
    resilience = rtm.validate_pattern_resilience("2024-25")
    
    print(f"Bounce-back rate: {analysis.home_matrix.bounce_back_rate:.1%}")
    print(f"Clean bounce-back: {analysis.home_matrix.clean_bounce_back:.1%}")
    print(f"Distortion impact: {analysis.home_matrix.distortion_impact:.2f}")
"""
from __future__ import annotations

import json
import csv
import warnings
import statistics
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path


# ==================== Enums ====================

class OutcomeState(Enum):
    """Three outcome states."""
    WIN = "WIN"
    DRAW = "DRAW"
    LOSS = "LOSS"


class TransitionType(Enum):
    """9 possible transitions."""
    WIN_TO_WIN = "WIN→WIN"
    WIN_TO_DRAW = "WIN→DRAW"
    WIN_TO_LOSS = "WIN→LOSS"
    DRAW_TO_WIN = "DRAW→WIN"
    DRAW_TO_DRAW = "DRAW→DRAW"
    DRAW_TO_LOSS = "DRAW→LOSS"
    LOSS_TO_WIN = "LOSS→WIN"
    LOSS_TO_DRAW = "LOSS→DRAW"
    LOSS_TO_LOSS = "LOSS→LOSS"


class ConfidenceLevel(Enum):
    """Confidence level for transition probabilities."""
    HIGH = "HIGH"           # 15+ transitions observed
    MEDIUM = "MEDIUM"       # 8-14 transitions
    LOW = "LOW"             # 5-7 transitions
    INSUFFICIENT = "INSUFFICIENT"  # <5 transitions


class DimensionType(Enum):
    """Supported RTM dimensions."""
    OVERALL = "overall"
    HOME = "home"
    AWAY = "away"
    VS_TOP6 = "vs_top6"
    VS_MID = "vs_mid"
    VS_BOTTOM6 = "vs_bottom6"
    H2H = "h2h"


class PatternReliability(Enum):
    """How reliable a detected pattern is."""
    DEFINITIVE = "DEFINITIVE"   # Pattern holds after removing distortions
    STRONG = "STRONG"           # Pattern mostly holds
    MODERATE = "MODERATE"       # Pattern exists but affected by distortions
    WEAK = "WEAK"               # Pattern disappears when distortions removed
    INSUFFICIENT = "INSUFFICIENT"  # Not enough clean data
    SPURIOUS = "SPURIOUS"       # Pattern only exists due to distortions


class DistortionType(Enum):
    """Types of distortion factors."""
    INJURY = "injury"
    SUSPENSION = "suspension"
    MANAGER_CHANGE = "manager_change"
    INTERNATIONAL_DUTY = "intl_duty"
    MIDWEEK_FATIGUE = "midweek_fatigue"
    MOTIVATION = "motivation"
    WEATHER = "weather"
    REFEREE = "referee"
    SQUAD_ROTATION = "rotation"
    TRAVEL = "travel"


# ==================== Constants ====================

# Core thresholds
MIN_TRANSITIONS_FOR_RELIABLE = 5
MIN_TRANSITIONS_FOR_STRONG = 8
MIN_TRANSITIONS_FOR_HIGH = 15

# Clean matrix thresholds
MIN_CLEAN_TRANSITIONS = 3
DISTORTION_RATE_WARNING = 0.30
DISTORTION_RATE_CRITICAL = 0.50

# Streak thresholds
MAX_LOSS_STREAK_WARNING = 4
MAX_LOSS_STREAK_CRITICAL = 6

# Bounce-back thresholds
BOUNCE_BACK_THRESHOLD_GOOD = 0.45
BOUNCE_BACK_THRESHOLD_EXCELLENT = 0.55

# Decay thresholds
DECAY_THRESHOLD_WARNING = 0.10
DECAY_THRESHOLD_CRITICAL = 0.20

# Confidence calibration
CONFIDENCE_CALIBRATION_TOLERANCE = 0.07

# Rolling windows
ROLLING_WINDOW_5 = 5
ROLLING_WINDOW_10 = 10
ROLLING_WINDOW_20 = 20

# Extreme detection
EXTREME_STREAK_LENGTH = 5
UNEXPECTED_BOUNCE_CONFIDENCE = 0.40

# Distortion severity thresholds
HIGH_SEVERITY_DISTORTION = 0.4
CRITICAL_SEVERITY_DISTORTION = 0.7

# Dimension thresholds
MIN_HOME_SAMPLES = 5
MIN_AWAY_SAMPLES = 5
MIN_TIER_SAMPLES = 3
MIN_H2H_SAMPLES = 3

# Pattern reliability thresholds
RELIABILITY_DIVERGENCE_TIGHT = 0.05
RELIABILITY_DIVERGENCE_LOOSE = 0.10
RELIABILITY_DIVERGENCE_MODERATE = 0.15


# ==================== Distortion Models ====================

@dataclass
class DistortionFactor:
    """Factor that may have distorted a transition outcome."""
    factor_type: DistortionType
    severity: float = 0.0           # 0-1 impact severity
    description: str = ""
    player_name: str = ""
    player_position: str = ""
    is_key_player: bool = False
    games_missed: int = 0
    
    @property
    def adjusted_severity(self) -> float:
        """Adjust severity based on key player status and position."""
        severity = self.severity
        
        if self.is_key_player:
            severity = min(1.0, severity * 1.5)
        
        high_impact_positions = {"goalkeeper", "center_back", "striker", "defensive_mid"}
        if self.player_position.lower() in high_impact_positions:
            severity = min(1.0, severity * 1.2)
        
        return min(1.0, severity)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.factor_type.value,
            "severity": round(self.severity, 3),
            "adjusted_severity": round(self.adjusted_severity, 3),
            "description": self.description,
            "player": self.player_name,
            "is_key": self.is_key_player,
        }


@dataclass
class ContextFlags:
    """Match context flags affecting transition interpretation."""
    is_dead_rubber: bool = False
    is_six_pointer: bool = False
    is_derby: bool = False
    is_early_season: bool = False
    is_late_season: bool = False
    is_post_international_break: bool = False
    is_midweek_fixture: bool = False
    days_rest: int = 7
    venue: str = "neutral"
    weather_condition: str = ""
    
    @property
    def context_impact(self) -> float:
        """Calculate total context impact (0-1)."""
        impact = 0.0
        if self.is_dead_rubber:
            impact += 0.30
        if self.is_six_pointer:
            impact += 0.15
        if self.is_derby:
            impact += 0.20
        if self.is_early_season:
            impact += 0.10
        if self.is_late_season:
            impact += 0.10
        if self.is_post_international_break:
            impact += 0.15
        if self.is_midweek_fixture:
            impact += 0.10
        if self.days_rest < 3:
            impact += 0.15
        if self.weather_condition in ("rain", "snow"):
            impact += 0.10
        return min(1.0, impact)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dead_rubber": self.is_dead_rubber,
            "six_pointer": self.is_six_pointer,
            "derby": self.is_derby,
            "early_season": self.is_early_season,
            "late_season": self.is_late_season,
            "post_intl": self.is_post_international_break,
            "midweek": self.is_midweek_fixture,
            "days_rest": self.days_rest,
            "venue": self.venue,
            "impact": round(self.context_impact, 3),
        }


@dataclass
class EnhancedTransitionRecord:
    """Single transition event with distortion tracking."""
    timestamp: str
    match_id: str
    prev_outcome: OutcomeState
    current_outcome: OutcomeState
    transition_type: TransitionType
    confidence_before: float = 0.5
    confidence_after: float = 0.5
    tier: str = "UNKNOWN"
    odds: float = 2.0
    dimension: str = "overall"
    distortions: List[DistortionFactor] = field(default_factory=list)
    context: ContextFlags = field(default_factory=ContextFlags)
    key_players_missing: int = 0
    manager_tenure_days: int = 365
    notes: str = ""
    
    @property
    def total_distortion_severity(self) -> float:
        """Sum of adjusted distortion severities."""
        return sum(d.adjusted_severity for d in self.distortions)
    
    @property
    def is_clean(self) -> bool:
        """True if no significant distortions affected this transition."""
        if self.context.is_dead_rubber:
            return False
        
        for d in self.distortions:
            if d.adjusted_severity > HIGH_SEVERITY_DISTORTION:
                return False
        
        if self.total_distortion_severity >= 0.5:
            return False
        
        return True
    
    @property
    def is_highly_distorted(self) -> bool:
        """True if multiple high-severity distortions present."""
        high_severity = sum(1 for d in self.distortions if d.adjusted_severity > CRITICAL_SEVERITY_DISTORTION)
        return high_severity >= 2 or self.total_distortion_severity >= 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "match_id": self.match_id,
            "prev_outcome": self.prev_outcome.value,
            "current_outcome": self.current_outcome.value,
            "transition_type": self.transition_type.value,
            "confidence_before": self.confidence_before,
            "confidence_after": self.confidence_after,
            "tier": self.tier,
            "odds": self.odds,
            "dimension": self.dimension,
            "is_clean": self.is_clean,
            "distortions": [d.to_dict() for d in self.distortions],
            "context": self.context.to_dict(),
            "notes": self.notes,
        }


@dataclass
class DistortionStats:
    """Aggregated distortion statistics."""
    injury_count: int = 0
    suspension_count: int = 0
    manager_change_count: int = 0
    intl_duty_count: int = 0
    midweek_fatigue_count: int = 0
    motivation_count: int = 0
    weather_count: int = 0
    
    injury_severity: float = 0.0
    suspension_severity: float = 0.0
    manager_change_severity: float = 0.0
    intl_duty_severity: float = 0.0
    midweek_fatigue_severity: float = 0.0
    motivation_severity: float = 0.0
    
    dead_rubber_count: int = 0
    six_pointer_count: int = 0
    derby_count: int = 0
    low_rest_count: int = 0
    
    total_distorted: int = 0
    total_clean: int = 0
    
    @property
    def distortion_rate(self) -> float:
        total = self.total_distorted + self.total_clean
        return self.total_distorted / total if total > 0 else 0.0
    
    @property
    def primary_distortion(self) -> Optional[str]:
        counts = {
            "injury": self.injury_count,
            "suspension": self.suspension_count,
            "manager_change": self.manager_change_count,
            "intl_duty": self.intl_duty_count,
            "midweek_fatigue": self.midweek_fatigue_count,
            "motivation": self.motivation_count,
        }
        max_type = max(counts, key=counts.get) if counts else None
        if max_type and counts[max_type] > 0:
            return max_type
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "injury": {"count": self.injury_count, "severity": round(self.injury_severity, 3)},
            "suspension": {"count": self.suspension_count, "severity": round(self.suspension_severity, 3)},
            "manager_change": {"count": self.manager_change_count, "severity": round(self.manager_change_severity, 3)},
            "intl_duty": {"count": self.intl_duty_count, "severity": round(self.intl_duty_severity, 3)},
            "midweek_fatigue": {"count": self.midweek_fatigue_count, "severity": round(self.midweek_fatigue_severity, 3)},
            "motivation": {"count": self.motivation_count, "severity": round(self.motivation_severity, 3)},
            "context": {
                "dead_rubber": self.dead_rubber_count,
                "six_pointer": self.six_pointer_count,
                "derby": self.derby_count,
                "low_rest": self.low_rest_count,
            },
            "total_distorted": self.total_distorted,
            "total_clean": self.total_clean,
            "distortion_rate": round(self.distortion_rate, 3),
            "primary_distortion": self.primary_distortion,
        }


@dataclass
class PatternReliabilityScore:
    """How reliable a detected pattern is."""
    reliability: PatternReliability = PatternReliability.INSUFFICIENT
    confidence: float = 0.0
    clean_sample_size: int = 0
    dirty_sample_size: int = 0
    divergence_score: float = 0.0
    reason: str = ""
    
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
        return scores.get(self.reliability, 0.50)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reliability": self.reliability.value,
            "confidence": round(self.confidence, 3),
            "clean_samples": self.clean_sample_size,
            "dirty_samples": self.dirty_sample_size,
            "divergence": round(self.divergence_score, 3),
            "reason": self.reason,
            "normalized_score": self.normalized_score,
        }


# ==================== Enhanced Dimension Matrix ====================

@dataclass
class DimensionTransitionMatrix:
    """Transition matrix for a specific dimension with clean/dirty support."""
    dimension_name: str = ""
    
    # Dirty matrix (all transitions)
    matrix: Dict[Tuple[OutcomeState, OutcomeState], int] = field(default_factory=dict)
    probabilities: Dict[Tuple[OutcomeState, OutcomeState], float] = field(default_factory=dict)
    confidence: Dict[Tuple[OutcomeState, OutcomeState], ConfidenceLevel] = field(default_factory=dict)
    
    # Clean matrix (undistorted transitions only)
    clean_matrix: Dict[Tuple[OutcomeState, OutcomeState], int] = field(default_factory=dict)
    clean_probabilities: Dict[Tuple[OutcomeState, OutcomeState], float] = field(default_factory=dict)
    
    # Streak metrics
    max_win_streak: int = 0
    max_loss_streak: int = 0
    avg_win_streak: float = 0.0
    avg_loss_streak: float = 0.0
    current_win_streak: int = 0
    current_loss_streak: int = 0
    clean_max_win_streak: int = 0
    clean_max_loss_streak: int = 0
    
    # Bounce-back metrics
    bounce_back_rate: float = 0.0
    drawdown_recovery_rate: float = 0.0
    clean_bounce_back: float = 0.0
    clean_recovery_rate: float = 0.0
    win_ceiling: int = 0
    clean_win_ceiling: int = 0
    
    # Decay metrics
    early_season_wr: float = 0.0
    late_season_wr: float = 0.0
    decay_rate: float = 0.0
    decay_confidence: float = 0.0
    
    # Confidence metrics
    avg_confidence_before_transition: float = 0.0
    avg_confidence_after_transition: float = 0.0
    confidence_shift: float = 0.0
    confidence_calibration: float = 0.0
    
    # Statistics
    total_transitions: int = 0
    clean_transition_count: int = 0
    distorted_transition_count: int = 0
    reliable: bool = False
    sample_quality: float = 0.0
    
    # Distortion tracking
    distortion_stats: Optional[DistortionStats] = None
    bounce_back_reliability: PatternReliabilityScore = field(default_factory=PatternReliabilityScore)
    win_ceiling_reliability: PatternReliabilityScore = field(default_factory=PatternReliabilityScore)
    overall_reliability: PatternReliabilityScore = field(default_factory=PatternReliabilityScore)
    
    @property
    def distortion_impact(self) -> float:
        """Overall distortion impact on probabilities."""
        if self.clean_transition_count < MIN_CLEAN_TRANSITIONS:
            return 0.0
        
        # Compare dirty vs clean bounce-back
        impact = abs(self.bounce_back_rate - self.clean_bounce_back)
        return round(impact, 3)
    
    @property
    def clean_ratio(self) -> float:
        """Percentage of transitions that are clean."""
        total = self.total_transitions
        return self.clean_transition_count / total if total > 0 else 0.0
    
    @property
    def is_reliable_dimension(self) -> bool:
        """Check if this dimension matrix has sufficient clean data."""
        return (self.clean_transition_count >= MIN_CLEAN_TRANSITIONS and 
                self.distortion_stats and 
                self.distortion_stats.distortion_rate < DISTORTION_RATE_CRITICAL)
    
    def get_clean_prob(self, prev: OutcomeState, curr: OutcomeState) -> float:
        """Get clean (undistorted) transition probability."""
        return self.clean_probabilities.get((prev, curr), 0.33)
    
    def get_dirty_prob(self, prev: OutcomeState, curr: OutcomeState) -> float:
        """Get dirty (all transitions) probability."""
        return self.probabilities.get((prev, curr), 0.33)
    
    def get_distortion_adjustment(self, prev: OutcomeState, curr: OutcomeState) -> float:
        """How much distortion affects this transition."""
        dirty = self.get_dirty_prob(prev, curr)
        clean = self.get_clean_prob(prev, curr)
        return round(dirty - clean, 4)
    
    def is_pattern_genuine(self, pattern_type: str = "bounce_back") -> bool:
        """Determine if a pattern is genuine or distortion-driven."""
        if self.clean_transition_count < MIN_CLEAN_TRANSITIONS:
            return False
        
        if pattern_type == "bounce_back":
            diff = abs(self.bounce_back_rate - self.clean_bounce_back)
            return self.clean_bounce_back > 0.45 and diff < RELIABILITY_DIVERGENCE_LOOSE
        
        elif pattern_type == "win_ceiling":
            return self.clean_win_ceiling >= self.win_ceiling - 1
        
        return False
    
    def compute_reliability_scores(self) -> None:
        """Calculate reliability scores for all patterns."""
        # Bounce-back reliability
        if self.clean_transition_count >= MIN_CLEAN_TRANSITIONS:
            divergence = abs(self.bounce_back_rate - self.clean_bounce_back)
            
            if divergence < RELIABILITY_DIVERGENCE_TIGHT and self.clean_bounce_back > 0.45:
                self.bounce_back_reliability = PatternReliabilityScore(
                    reliability=PatternReliability.DEFINITIVE,
                    confidence=0.95,
                    clean_sample_size=self.clean_transition_count,
                    dirty_sample_size=self.distorted_transition_count,
                    divergence_score=divergence,
                    reason="Bounce-back persists after removing distortions"
                )
            elif divergence < RELIABILITY_DIVERGENCE_LOOSE:
                self.bounce_back_reliability = PatternReliabilityScore(
                    reliability=PatternReliability.STRONG,
                    confidence=0.80,
                    clean_sample_size=self.clean_transition_count,
                    dirty_sample_size=self.distorted_transition_count,
                    divergence_score=divergence,
                    reason="Bounce-back mostly holds after distortion removal"
                )
            elif divergence < RELIABILITY_DIVERGENCE_MODERATE:
                self.bounce_back_reliability = PatternReliabilityScore(
                    reliability=PatternReliability.MODERATE,
                    confidence=0.60,
                    clean_sample_size=self.clean_transition_count,
                    dirty_sample_size=self.distorted_transition_count,
                    divergence_score=divergence,
                    reason="Bounce-back partly driven by distortions"
                )
            elif divergence < 0.25:
                self.bounce_back_reliability = PatternReliabilityScore(
                    reliability=PatternReliability.WEAK,
                    confidence=0.40,
                    clean_sample_size=self.clean_transition_count,
                    dirty_sample_size=self.distorted_transition_count,
                    divergence_score=divergence,
                    reason="Bounce-back significantly affected by distortions"
                )
            else:
                self.bounce_back_reliability = PatternReliabilityScore(
                    reliability=PatternReliability.SPURIOUS,
                    confidence=0.25,
                    clean_sample_size=self.clean_transition_count,
                    dirty_sample_size=self.distorted_transition_count,
                    divergence_score=divergence,
                    reason="Bounce-back only exists due to distortions"
                )
        else:
            self.bounce_back_reliability = PatternReliabilityScore(
                reliability=PatternReliability.INSUFFICIENT,
                confidence=0.20,
                clean_sample_size=self.clean_transition_count,
                dirty_sample_size=self.distorted_transition_count,
                divergence_score=0.0,
                reason=f"Insufficient clean transitions ({self.clean_transition_count}/{MIN_CLEAN_TRANSITIONS})"
            )
        
        # Win ceiling reliability
        if self.clean_transition_count >= MIN_CLEAN_TRANSITIONS:
            ceiling_diff = abs(self.win_ceiling - self.clean_win_ceiling)
            
            if ceiling_diff == 0 and self.clean_win_ceiling > 0:
                self.win_ceiling_reliability = PatternReliabilityScore(
                    reliability=PatternReliability.DEFINITIVE,
                    confidence=0.95,
                    clean_sample_size=self.clean_transition_count,
                    dirty_sample_size=self.distorted_transition_count,
                    divergence_score=0.0,
                    reason="Win ceiling identical in clean data"
                )
            elif ceiling_diff <= 1:
                self.win_ceiling_reliability = PatternReliabilityScore(
                    reliability=PatternReliability.STRONG,
                    confidence=0.80,
                    clean_sample_size=self.clean_transition_count,
                    dirty_sample_size=self.distorted_transition_count,
                    divergence_score=float(ceiling_diff),
                    reason="Win ceiling mostly consistent"
                )
            elif ceiling_diff <= 2:
                self.win_ceiling_reliability = PatternReliabilityScore(
                    reliability=PatternReliability.MODERATE,
                    confidence=0.60,
                    clean_sample_size=self.clean_transition_count,
                    dirty_sample_size=self.distorted_transition_count,
                    divergence_score=float(ceiling_diff),
                    reason="Win ceiling lower after distortion removal"
                )
            else:
                self.win_ceiling_reliability = PatternReliabilityScore(
                    reliability=PatternReliability.WEAK,
                    confidence=0.40,
                    clean_sample_size=self.clean_transition_count,
                    dirty_sample_size=self.distorted_transition_count,
                    divergence_score=float(ceiling_diff),
                    reason="Win ceiling significantly reduced in clean data"
                )
        else:
            self.win_ceiling_reliability = PatternReliabilityScore(
                reliability=PatternReliability.INSUFFICIENT,
                confidence=0.20,
                clean_sample_size=self.clean_transition_count,
                dirty_sample_size=self.distorted_transition_count,
                divergence_score=0.0,
                reason="Insufficient clean transitions"
            )
        
        # Overall reliability
        bounce_valid = self.bounce_back_reliability.reliability in (PatternReliability.DEFINITIVE, PatternReliability.STRONG)
        ceiling_valid = self.win_ceiling_reliability.reliability in (PatternReliability.DEFINITIVE, PatternReliability.STRONG)
        
        if bounce_valid and ceiling_valid:
            overall = PatternReliability.DEFINITIVE
            confidence = 0.90
            reason = "Both bounce-back and win ceiling patterns reliable"
        elif bounce_valid or ceiling_valid:
            overall = PatternReliability.STRONG
            confidence = 0.75
            reason = "One key pattern reliable"
        elif self.clean_ratio > 0.5:
            overall = PatternReliability.MODERATE
            confidence = 0.55
            reason = f"Moderate reliability ({self.clean_ratio:.0%} clean transitions)"
        elif self.distortion_stats and self.distortion_stats.distortion_rate > DISTORTION_RATE_CRITICAL:
            overall = PatternReliability.WEAK
            confidence = 0.30
            reason = f"High distortion rate ({self.distortion_stats.distortion_rate:.0%})"
        else:
            overall = PatternReliability.INSUFFICIENT
            confidence = 0.25
            reason = f"Insufficient clean data ({self.clean_ratio:.0%} clean)"
        
        self.overall_reliability = PatternReliabilityScore(
            reliability=overall,
            confidence=confidence,
            clean_sample_size=self.clean_transition_count,
            dirty_sample_size=self.distorted_transition_count,
            divergence_score=1.0 - self.clean_ratio,
            reason=reason
        )
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "dimension": self.dimension_name,
            "probabilities": {
                f"{k[0].value}→{k[1].value}": v 
                for k, v in self.probabilities.items()
            },
            "clean_probabilities": {
                f"{k[0].value}→{k[1].value}": v 
                for k, v in self.clean_probabilities.items()
            } if self.clean_probabilities else {},
            "max_win_streak": self.max_win_streak,
            "max_loss_streak": self.max_loss_streak,
            "clean_max_win_streak": self.clean_max_win_streak,
            "current_win_streak": self.current_win_streak,
            "current_loss_streak": self.current_loss_streak,
            "bounce_back_rate": round(self.bounce_back_rate, 4),
            "clean_bounce_back": round(self.clean_bounce_back, 4),
            "drawdown_recovery_rate": round(self.drawdown_recovery_rate, 4),
            "distortion_impact": round(self.distortion_impact, 4),
            "win_ceiling": self.win_ceiling,
            "clean_win_ceiling": self.clean_win_ceiling,
            "decay_rate": round(self.decay_rate, 4),
            "confidence_shift": round(self.confidence_shift, 4),
            "total_transitions": self.total_transitions,
            "clean_transition_count": self.clean_transition_count,
            "reliable": self.reliable,
            "sample_quality": round(self.sample_quality, 3),
            "distortion_stats": self.distortion_stats.to_dict() if self.distortion_stats else None,
            "bounce_back_reliability": self.bounce_back_reliability.to_dict(),
            "win_ceiling_reliability": self.win_ceiling_reliability.to_dict(),
            "overall_reliability": self.overall_reliability.to_dict(),
        }
        return result


# ==================== Legacy Data Classes ====================

@dataclass
class TransitionMatrix:
    """Legacy transition matrix (kept for backward compatibility)."""
    matrix: Dict[Tuple[OutcomeState, OutcomeState], int] = field(default_factory=dict)
    probabilities: Dict[Tuple[OutcomeState, OutcomeState], float] = field(default_factory=dict)
    confidence: Dict[Tuple[OutcomeState, OutcomeState], ConfidenceLevel] = field(default_factory=dict)
    
    max_win_streak: int = 0
    max_loss_streak: int = 0
    avg_win_streak: float = 0.0
    avg_loss_streak: float = 0.0
    current_win_streak: int = 0
    current_loss_streak: int = 0
    
    bounce_back_rate: float = 0.0
    drawdown_recovery_rate: float = 0.0
    win_ceiling: int = 0
    
    early_season_wr: float = 0.0
    late_season_wr: float = 0.0
    decay_rate: float = 0.0
    decay_confidence: float = 0.0
    
    avg_confidence_before_transition: float = 0.0
    avg_confidence_after_transition: float = 0.0
    confidence_shift: float = 0.0
    confidence_calibration: float = 0.0
    
    total_transitions: int = 0
    reliable: bool = False


@dataclass
class SeasonAnalysis:
    """Full season transition analysis with dimensions."""
    season_id: str
    start_date: str
    end_date: str
    total_fixtures: int
    total_wins: int
    total_draws: int
    total_losses: int
    overall_win_rate: float
    overall_profit: float = 0.0
    overall_roi: float = 0.0
    
    transitions: List[EnhancedTransitionRecord] = field(default_factory=list)
    matrix: TransitionMatrix = field(default_factory=TransitionMatrix)
    
    # Dimension-specific matrices
    home_matrix: Optional[DimensionTransitionMatrix] = None
    away_matrix: Optional[DimensionTransitionMatrix] = None
    vs_top6_matrix: Optional[DimensionTransitionMatrix] = None
    vs_mid_matrix: Optional[DimensionTransitionMatrix] = None
    vs_bottom6_matrix: Optional[DimensionTransitionMatrix] = None
    h2h_matrices: Dict[str, DimensionTransitionMatrix] = field(default_factory=dict)
    
    tier_matrices: Dict[str, TransitionMatrix] = field(default_factory=dict)
    rolling_win_rates_5: List[float] = field(default_factory=list)
    rolling_win_rates_10: List[float] = field(default_factory=list)
    rolling_win_rates_20: List[float] = field(default_factory=list)
    
    anomalous_transitions: List[EnhancedTransitionRecord] = field(default_factory=list)
    forecast_next_outcome: Optional[Tuple[OutcomeState, float]] = None
    
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ==================== Core RTM Engine ====================

class ResultsTransitionMatrix:
    """
    Main engine for tracking and analyzing result transitions across dimensions.
    Now with distortion filtering and pattern reliability.
    """
    
    def __init__(self, min_reliable: int = MIN_TRANSITIONS_FOR_RELIABLE):
        self.transitions: List[EnhancedTransitionRecord] = []
        self.seasons: Dict[str, SeasonAnalysis] = {}
        self.current_state: Optional[OutcomeState] = None
        self.min_reliable = min_reliable
    
    def _get_transition_type(self, prev: OutcomeState, curr: OutcomeState) -> TransitionType:
        mapping = {
            (OutcomeState.WIN, OutcomeState.WIN): TransitionType.WIN_TO_WIN,
            (OutcomeState.WIN, OutcomeState.DRAW): TransitionType.WIN_TO_DRAW,
            (OutcomeState.WIN, OutcomeState.LOSS): TransitionType.WIN_TO_LOSS,
            (OutcomeState.DRAW, OutcomeState.WIN): TransitionType.DRAW_TO_WIN,
            (OutcomeState.DRAW, OutcomeState.DRAW): TransitionType.DRAW_TO_DRAW,
            (OutcomeState.DRAW, OutcomeState.LOSS): TransitionType.DRAW_TO_LOSS,
            (OutcomeState.LOSS, OutcomeState.WIN): TransitionType.LOSS_TO_WIN,
            (OutcomeState.LOSS, OutcomeState.DRAW): TransitionType.LOSS_TO_DRAW,
            (OutcomeState.LOSS, OutcomeState.LOSS): TransitionType.LOSS_TO_LOSS,
        }
        return mapping[(prev, curr)]
    
    def _get_confidence_level(self, count: int) -> ConfidenceLevel:
        if count >= MIN_TRANSITIONS_FOR_HIGH:
            return ConfidenceLevel.HIGH
        elif count >= MIN_TRANSITIONS_FOR_STRONG:
            return ConfidenceLevel.MEDIUM
        elif count >= MIN_TRANSITIONS_FOR_RELIABLE:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.INSUFFICIENT
    
    def _is_clean_fixture(self, fixture_metadata: Dict[str, Any]) -> Tuple[bool, str]:
        """Determine if a fixture should be considered clean."""
        if fixture_metadata.get("is_dead_rubber", False):
            return False, "dead_rubber"
        
        key_players_missing = fixture_metadata.get("key_players_missing", 0)
        if key_players_missing >= 2:
            return False, f"{key_players_missing}_key_players_missing"
        
        manager_tenure = fixture_metadata.get("manager_tenure_days", 365)
        if manager_tenure < 60:
            return False, "new_manager_bounce"
        
        if fixture_metadata.get("european_midweek", False):
            days_rest = fixture_metadata.get("days_rest", 7)
            if days_rest < 4:
                return False, "midweek_fatigue"
        
        days_since_intl = fixture_metadata.get("days_since_intl_return", 14)
        if days_since_intl < 3:
            return False, "intl_break_fatigue"
        
        weather = fixture_metadata.get("weather", "").lower()
        if weather in ("heavy_rain", "snow", "storm"):
            return False, f"extreme_weather_{weather}"
        
        return True, "clean"
    
    def _build_distortion_stats(self, transitions: List[EnhancedTransitionRecord]) -> DistortionStats:
        """Build distortion statistics from transitions."""
        stats = DistortionStats()
        
        for t in transitions:
            if t.is_clean:
                stats.total_clean += 1
            else:
                stats.total_distorted += 1
                
                for d in t.distortions:
                    if d.factor_type == DistortionType.INJURY:
                        stats.injury_count += 1
                        stats.injury_severity += d.adjusted_severity
                    elif d.factor_type == DistortionType.SUSPENSION:
                        stats.suspension_count += 1
                        stats.suspension_severity += d.adjusted_severity
                    elif d.factor_type == DistortionType.MANAGER_CHANGE:
                        stats.manager_change_count += 1
                        stats.manager_change_severity += d.adjusted_severity
                    elif d.factor_type == DistortionType.INTERNATIONAL_DUTY:
                        stats.intl_duty_count += 1
                        stats.intl_duty_severity += d.adjusted_severity
                    elif d.factor_type == DistortionType.MIDWEEK_FATIGUE:
                        stats.midweek_fatigue_count += 1
                        stats.midweek_fatigue_severity += d.adjusted_severity
                    elif d.factor_type == DistortionType.MOTIVATION:
                        stats.motivation_count += 1
                        stats.motivation_severity += d.adjusted_severity
                
                if t.context.is_dead_rubber:
                    stats.dead_rubber_count += 1
                if t.context.is_six_pointer:
                    stats.six_pointer_count += 1
                if t.context.is_derby:
                    stats.derby_count += 1
                if t.context.days_rest < 3:
                    stats.low_rest_count += 1
        
        return stats
    
    def _build_dimension_matrix(
        self,
        transitions: List[EnhancedTransitionRecord],
        dimension: str,
        outcomes_seq: List[OutcomeState] = None,
    ) -> DimensionTransitionMatrix:
        """Build transition matrix for a specific dimension with clean/dirty separation."""
        dim_transitions = [t for t in transitions if t.dimension == dimension]
        
        matrix = DimensionTransitionMatrix(dimension_name=dimension)
        
        if not dim_transitions:
            return matrix
        
        # Separate clean and dirty transitions
        clean_transitions = [t for t in dim_transitions if t.is_clean]
        dirty_transitions = [t for t in dim_transitions if not t.is_clean]
        
        matrix.total_transitions = len(dim_transitions)
        matrix.clean_transition_count = len(clean_transitions)
        matrix.distorted_transition_count = len(dirty_transitions)
        matrix.reliable = len(dim_transitions) >= self.min_reliable
        
        # Calculate sample quality
        if self.transitions:
            matrix.sample_quality = min(1.0, len(dim_transitions) / len(self.transitions))
        
        # Build dirty matrix (all transitions)
        dirty_counts = {}
        prev_counts = {}
        
        for trans in dirty_transitions:
            key = (trans.prev_outcome, trans.current_outcome)
            dirty_counts[key] = dirty_counts.get(key, 0) + 1
            prev_counts[trans.prev_outcome] = prev_counts.get(trans.prev_outcome, 0) + 1
        
        # Also add clean transitions to dirty matrix
        for trans in clean_transitions:
            key = (trans.prev_outcome, trans.current_outcome)
            dirty_counts[key] = dirty_counts.get(key, 0) + 1
            prev_counts[trans.prev_outcome] = prev_counts.get(trans.prev_outcome, 0) + 1
        
        for (prev, curr), count in dirty_counts.items():
            total_from_prev = prev_counts.get(prev, 0)
            prob = count / total_from_prev if total_from_prev > 0 else 0.0
            matrix.probabilities[(prev, curr)] = round(prob, 4)
            matrix.confidence[(prev, curr)] = self._get_confidence_level(count)
        
        matrix.matrix = dict(dirty_counts)
        
        # Build clean matrix (undistorted transitions only)
        if clean_transitions:
            clean_counts = {}
            clean_prev_counts = {}
            
            for trans in clean_transitions:
                key = (trans.prev_outcome, trans.current_outcome)
                clean_counts[key] = clean_counts.get(key, 0) + 1
                clean_prev_counts[trans.prev_outcome] = clean_prev_counts.get(trans.prev_outcome, 0) + 1
            
            for (prev, curr), count in clean_counts.items():
                total_from_prev = clean_prev_counts.get(prev, 0)
                prob = count / total_from_prev if total_from_prev > 0 else 0.0
                matrix.clean_probabilities[(prev, curr)] = round(prob, 4)
            
            matrix.clean_matrix = dict(clean_counts)
        
        # Compute resilience metrics (dirty)
        loss_to_win = dirty_counts.get((OutcomeState.LOSS, OutcomeState.WIN), 0)
        total_losses = prev_counts.get(OutcomeState.LOSS, 0)
        matrix.bounce_back_rate = loss_to_win / total_losses if total_losses > 0 else 0.0
        
        loss_to_win_or_draw = (
            dirty_counts.get((OutcomeState.LOSS, OutcomeState.WIN), 0) +
            dirty_counts.get((OutcomeState.LOSS, OutcomeState.DRAW), 0)
        )
        matrix.drawdown_recovery_rate = loss_to_win_or_draw / total_losses if total_losses > 0 else 0.0
        
        # Compute clean resilience metrics
        if clean_transitions:
            clean_loss_to_win = clean_counts.get((OutcomeState.LOSS, OutcomeState.WIN), 0)
            clean_total_losses = clean_prev_counts.get(OutcomeState.LOSS, 0)
            matrix.clean_bounce_back = clean_loss_to_win / clean_total_losses if clean_total_losses > 0 else 0.0
            
            clean_loss_to_win_or_draw = (
                clean_counts.get((OutcomeState.LOSS, OutcomeState.WIN), 0) +
                clean_counts.get((OutcomeState.LOSS, OutcomeState.DRAW), 0)
            )
            matrix.clean_recovery_rate = clean_loss_to_win_or_draw / clean_total_losses if clean_total_losses > 0 else 0.0
        
        # Calculate streaks
        if outcomes_seq:
            # Dirty streaks
            matrix.max_win_streak, matrix.avg_win_streak, matrix.current_win_streak = self._calculate_streaks_with_seq(
                outcomes_seq, OutcomeState.WIN
            )
            matrix.max_loss_streak, matrix.avg_loss_streak, matrix.current_loss_streak = self._calculate_streaks_with_seq(
                outcomes_seq, OutcomeState.LOSS
            )
            
            # Clean streaks (using only clean transitions)
            if clean_transitions:
                clean_outcomes = [t.current_outcome for t in clean_transitions]
                if len(clean_outcomes) >= 2:
                    matrix.clean_max_win_streak, _, _ = self._calculate_streaks_with_seq(
                        clean_outcomes, OutcomeState.WIN
                    )
                    matrix.clean_max_loss_streak, _, _ = self._calculate_streaks_with_seq(
                        clean_outcomes, OutcomeState.LOSS
                    )
        
        matrix.win_ceiling = matrix.max_win_streak
        matrix.clean_win_ceiling = matrix.clean_max_win_streak
        
        # Compute decay
        n = len(dim_transitions)
        if n >= 10:
            early_cutoff = max(1, n // 4)
            late_start = max(early_cutoff, (3 * n) // 4)
            
            early_wins = sum(1 for t in dim_transitions[:early_cutoff] if t.current_outcome == OutcomeState.WIN)
            early_wr = early_wins / early_cutoff if early_cutoff > 0 else 0.0
            
            late_wins = sum(1 for t in dim_transitions[late_start:] if t.current_outcome == OutcomeState.WIN)
            late_wr = late_wins / max(len(dim_transitions[late_start:]), 1)
            
            matrix.early_season_wr = early_wr
            matrix.late_season_wr = late_wr
            matrix.decay_rate = (early_wr - late_wr) / early_wr if early_wr > 0 else 0.0
            matrix.decay_confidence = min(1.0, n / 30)
        
        # Confidence analysis
        if dim_transitions:
            conf_before = [t.confidence_before for t in dim_transitions]
            conf_after = [t.confidence_after for t in dim_transitions]
            matrix.avg_confidence_before_transition = statistics.mean(conf_before) if conf_before else 0.0
            matrix.avg_confidence_after_transition = statistics.mean(conf_after) if conf_after else 0.0
            matrix.confidence_shift = matrix.avg_confidence_after_transition - matrix.avg_confidence_before_transition
        
        # Build distortion stats
        matrix.distortion_stats = self._build_distortion_stats(dim_transitions)
        
        # Compute reliability scores
        matrix.compute_reliability_scores()
        
        return matrix
    
    def _build_dimension_matrix_legacy(
        self,
        transitions: List[EnhancedTransitionRecord],
        dimension: str,
        outcomes_seq: List[OutcomeState] = None,
    ) -> Optional[DimensionTransitionMatrix]:
        """Build dimension-specific matrix (legacy, without clean separation)."""
        dim_transitions = [t for t in transitions if t.dimension == dimension]
        
        if len(dim_transitions) < self.min_reliable:
            return None
        
        return self._build_dimension_matrix(transitions, dimension, outcomes_seq)
    
    # ========== Main Build Method ==========
    
    def build_season_matrix(
        self,
        season_id: str,
        fixtures: List[Dict],
        include_dimensions: bool = True,
    ) -> SeasonAnalysis:
        """
        Build transition matrix for entire season with dimensions and distortion tracking.
        
        Args:
            season_id: e.g., "2025_EPL" or "2025_GLOBAL"
            fixtures: List of resolved fixtures with outcomes and dimension metadata
            include_dimensions: If True, build home/away/tier matrices
        
        Returns:
            SeasonAnalysis with full transition matrix and dimension matrices
        """
        if not fixtures:
            return SeasonAnalysis(
                season_id=season_id,
                start_date=datetime.now(timezone.utc).isoformat(),
                end_date=datetime.now(timezone.utc).isoformat(),
                total_fixtures=0,
                total_wins=0,
                total_draws=0,
                total_losses=0,
                overall_win_rate=0.0,
            )
        
        sorted_fixtures = sorted(fixtures, key=lambda x: x.get("date", ""))
        
        analysis = SeasonAnalysis(
            season_id=season_id,
            start_date=sorted_fixtures[0].get("date", datetime.now(timezone.utc).isoformat()),
            end_date=sorted_fixtures[-1].get("date", datetime.now(timezone.utc).isoformat()),
            total_fixtures=len(sorted_fixtures),
            total_wins=0,
            total_draws=0,
            total_losses=0,
            overall_win_rate=0.0,
            overall_profit=sum(f.get("pnl", 0) for f in sorted_fixtures),
        )
        
        total_stake = sum(f.get("stake", 0) for f in sorted_fixtures)
        if total_stake > 0:
            analysis.overall_roi = analysis.overall_profit / total_stake
        
        prev_outcome = None
        outcomes_seq = []
        
        for fixture in sorted_fixtures:
            outcome_str = fixture.get("outcome", "LOSS").upper()
            try:
                outcome = OutcomeState[outcome_str]
            except KeyError:
                outcome = OutcomeState.LOSS
                warnings.warn(f"Unknown outcome '{outcome_str}' for {fixture.get('match_id', 'unknown')}")
            
            outcomes_seq.append(outcome)
            
            if outcome == OutcomeState.WIN:
                analysis.total_wins += 1
            elif outcome == OutcomeState.DRAW:
                analysis.total_draws += 1
            else:
                analysis.total_losses += 1
            
            if prev_outcome is not None:
                transition_type = self._get_transition_type(prev_outcome, outcome)
                dimension = fixture.get("dimension", "overall")
                
                # Build distortion factors from metadata
                distortions = []
                meta = fixture.get("metadata", {})
                
                # Injury distortion
                if meta.get("key_players_missing", 0) > 0:
                    distortions.append(DistortionFactor(
                        factor_type=DistortionType.INJURY,
                        severity=min(1.0, meta.get("key_players_missing", 0) * 0.2),
                        description=f"{meta.get('key_players_missing', 0)} key players missing",
                        is_key_player=True,
                    ))
                
                # Manager change distortion
                if meta.get("manager_tenure_days", 365) < 60:
                    distortions.append(DistortionFactor(
                        factor_type=DistortionType.MANAGER_CHANGE,
                        severity=0.4,
                        description="New manager bounce period",
                    ))
                
                # Midweek fatigue distortion
                if meta.get("european_midweek", False) and meta.get("days_rest", 7) < 4:
                    distortions.append(DistortionFactor(
                        factor_type=DistortionType.MIDWEEK_FATIGUE,
                        severity=0.3,
                        description="Midweek European fixture",
                    ))
                
                # Motivation distortion
                if meta.get("is_dead_rubber", False):
                    distortions.append(DistortionFactor(
                        factor_type=DistortionType.MOTIVATION,
                        severity=0.5,
                        description="Dead rubber - low motivation",
                    ))
                
                # Context flags
                context = ContextFlags(
                    is_dead_rubber=meta.get("is_dead_rubber", False),
                    is_six_pointer=meta.get("is_six_pointer", False),
                    is_derby=meta.get("is_derby", False),
                    is_early_season=meta.get("is_early_season", False),
                    is_late_season=meta.get("is_late_season", False),
                    is_post_international_break=meta.get("is_post_international_break", False),
                    is_midweek_fixture=meta.get("is_midweek_fixture", False),
                    days_rest=meta.get("days_rest", 7),
                    venue=meta.get("venue", "neutral"),
                    weather_condition=meta.get("weather", ""),
                )
                
                record = EnhancedTransitionRecord(
                    timestamp=fixture.get("date", datetime.now(timezone.utc).isoformat()),
                    match_id=fixture.get("match_id", "UNKNOWN"),
                    prev_outcome=prev_outcome,
                    current_outcome=outcome,
                    transition_type=transition_type,
                    confidence_before=fixture.get("confidence_before", 0.5),
                    confidence_after=fixture.get("confidence_after", 0.5),
                    tier=fixture.get("tier", "UNKNOWN"),
                    odds=fixture.get("odds", 2.0),
                    dimension=dimension,
                    distortions=distortions,
                    context=context,
                    key_players_missing=meta.get("key_players_missing", 0),
                    manager_tenure_days=meta.get("manager_tenure_days", 365),
                    notes=fixture.get("notes", ""),
                )
                
                analysis.transitions.append(record)
            
            prev_outcome = outcome
        
        # Build legacy matrix
        analysis.matrix = self._compute_transition_matrix(analysis.transitions, outcomes_seq)
        
        # Build dimension matrices with clean/dirty separation
        if include_dimensions:
            analysis.home_matrix = self._build_dimension_matrix(
                analysis.transitions, "home", outcomes_seq
            )
            analysis.away_matrix = self._build_dimension_matrix(
                analysis.transitions, "away", outcomes_seq
            )
            analysis.vs_top6_matrix = self._build_dimension_matrix(
                analysis.transitions, "vs_top6", outcomes_seq
            )
            analysis.vs_mid_matrix = self._build_dimension_matrix(
                analysis.transitions, "vs_mid", outcomes_seq
            )
            analysis.vs_bottom6_matrix = self._build_dimension_matrix(
                analysis.transitions, "vs_bottom6", outcomes_seq
            )
        
        # Build tier-specific legacy matrices
        tiers = set(t.tier for t in analysis.transitions)
        for tier in tiers:
            tier_transitions = [t for t in analysis.transitions if t.tier == tier]
            tier_outcomes = [t.current_outcome for t in tier_transitions]
            tier_seq = []
            for fixture in sorted_fixtures:
                if fixture.get("tier") == tier:
                    try:
                        tier_seq.append(OutcomeState[fixture.get("outcome", "LOSS").upper()])
                    except KeyError:
                        tier_seq.append(OutcomeState.LOSS)
            analysis.tier_matrices[tier] = self._compute_transition_matrix(tier_transitions, tier_seq)
        
        analysis.rolling_win_rates_5 = self._calculate_rolling_win_rates(outcomes_seq, ROLLING_WINDOW_5)
        analysis.rolling_win_rates_10 = self._calculate_rolling_win_rates(outcomes_seq, ROLLING_WINDOW_10)
        analysis.rolling_win_rates_20 = self._calculate_rolling_win_rates(outcomes_seq, ROLLING_WINDOW_20)
        
        if analysis.total_fixtures > 0:
            analysis.overall_win_rate = analysis.total_wins / analysis.total_fixtures
        
        analysis.anomalous_transitions = self._detect_anomalies(analysis.transitions)
        
        if outcomes_seq:
            last_outcome = outcomes_seq[-1]
            analysis.forecast_next_outcome = self.forecast_next(season_id, last_outcome)
        
        self.seasons[season_id] = analysis
        
        return analysis
    
    # ========== Legacy Helper Methods ==========
    
    def _calculate_rolling_win_rates(self, outcomes: List[OutcomeState], window: int) -> List[float]:
        if len(outcomes) < window:
            return []
        
        rolling = []
        for i in range(len(outcomes) - window + 1):
            window_outcomes = outcomes[i:i+window]
            wins = sum(1 for o in window_outcomes if o == OutcomeState.WIN)
            rolling.append(wins / window)
        return rolling
    
    def _compute_transition_matrix(
        self,
        transitions: List[EnhancedTransitionRecord],
        outcomes_seq: List[OutcomeState] = None
    ) -> TransitionMatrix:
        matrix = TransitionMatrix()
        
        if not transitions:
            return matrix
        
        transition_counts = {}
        prev_counts = {}
        
        for trans in transitions:
            key = (trans.prev_outcome, trans.current_outcome)
            transition_counts[key] = transition_counts.get(key, 0) + 1
            prev_counts[trans.prev_outcome] = prev_counts.get(trans.prev_outcome, 0) + 1
        
        matrix.matrix = dict(transition_counts)
        matrix.total_transitions = len(transitions)
        matrix.reliable = len(transitions) >= self.min_reliable
        
        for (prev, curr), count in transition_counts.items():
            total_from_prev = prev_counts.get(prev, 0)
            prob = count / total_from_prev if total_from_prev > 0 else 0.0
            matrix.probabilities[(prev, curr)] = round(prob, 4)
            matrix.confidence[(prev, curr)] = self._get_confidence_level(count)
        
        loss_to_win = transition_counts.get((OutcomeState.LOSS, OutcomeState.WIN), 0)
        total_losses = prev_counts.get(OutcomeState.LOSS, 0)
        matrix.bounce_back_rate = loss_to_win / total_losses if total_losses > 0 else 0.0
        
        loss_to_win_or_draw = (
            transition_counts.get((OutcomeState.LOSS, OutcomeState.WIN), 0) +
            transition_counts.get((OutcomeState.LOSS, OutcomeState.DRAW), 0)
        )
        matrix.drawdown_recovery_rate = loss_to_win_or_draw / total_losses if total_losses > 0 else 0.0
        
        if outcomes_seq:
            matrix.max_win_streak, matrix.avg_win_streak, matrix.current_win_streak = self._calculate_streaks_with_seq(
                outcomes_seq, OutcomeState.WIN
            )
            matrix.max_loss_streak, matrix.avg_loss_streak, matrix.current_loss_streak = self._calculate_streaks_with_seq(
                outcomes_seq, OutcomeState.LOSS
            )
        
        matrix.win_ceiling = matrix.max_win_streak
        
        n = len(transitions)
        if n >= 10:
            early_cutoff = max(1, n // 4)
            late_start = max(early_cutoff, (3 * n) // 4)
            
            early_wins = sum(1 for t in transitions[:early_cutoff] if t.current_outcome == OutcomeState.WIN)
            early_wr = early_wins / early_cutoff if early_cutoff > 0 else 0.0
            
            late_wins = sum(1 for t in transitions[late_start:] if t.current_outcome == OutcomeState.WIN)
            late_wr = late_wins / max(len(transitions[late_start:]), 1)
            
            matrix.early_season_wr = early_wr
            matrix.late_season_wr = late_wr
            matrix.decay_rate = (early_wr - late_wr) / early_wr if early_wr > 0 else 0.0
            matrix.decay_confidence = min(1.0, n / 30)
        
        if transitions:
            conf_before = [t.confidence_before for t in transitions]
            conf_after = [t.confidence_after for t in transitions]
            matrix.avg_confidence_before_transition = statistics.mean(conf_before) if conf_before else 0.0
            matrix.avg_confidence_after_transition = statistics.mean(conf_after) if conf_after else 0.0
            matrix.confidence_shift = matrix.avg_confidence_after_transition - matrix.avg_confidence_before_transition
            
            high_conf_transitions = [t for t in transitions if t.confidence_before >= 0.7]
            if high_conf_transitions:
                high_actual_wr = sum(1 for t in high_conf_transitions if t.current_outcome == OutcomeState.WIN) / len(high_conf_transitions)
                matrix.confidence_calibration = high_actual_wr - 0.7
        
        return matrix
    
    def _calculate_streaks_with_seq(
        self,
        outcomes: List[OutcomeState],
        target_outcome: OutcomeState
    ) -> Tuple[int, float, int]:
        if not outcomes:
            return 0, 0.0, 0
        
        streaks = []
        current = 0
        current_streak = 0
        
        if outcomes[-1] == target_outcome:
            for o in reversed(outcomes):
                if o == target_outcome:
                    current_streak += 1
                else:
                    break
        
        for outcome in outcomes:
            if outcome == target_outcome:
                current += 1
            else:
                if current > 0:
                    streaks.append(current)
                current = 0
        
        if current > 0:
            streaks.append(current)
        
        max_streak = max(streaks) if streaks else 0
        avg_streak = sum(streaks) / len(streaks) if streaks else 0.0
        
        return max_streak, avg_streak, current_streak
    
    def _detect_anomalies(self, transitions: List[EnhancedTransitionRecord]) -> List[EnhancedTransitionRecord]:
        if len(transitions) < 3:
            return []
        
        anomalies = []
        
        for trans in transitions:
            if trans.transition_type == TransitionType.WIN_TO_LOSS and trans.confidence_before > 0.70:
                trans.notes = f"Anomaly: High confidence ({trans.confidence_before:.1%}) loss"
                anomalies.append(trans)
        
        loss_streak = 0
        for trans in transitions:
            if trans.current_outcome == OutcomeState.LOSS:
                loss_streak += 1
            else:
                if loss_streak >= EXTREME_STREAK_LENGTH:
                    trans.notes = f"Anomaly: {loss_streak} consecutive losses"
                    if trans not in anomalies:
                        anomalies.append(trans)
                loss_streak = 0
        
        win_streak = 0
        for trans in transitions:
            if trans.current_outcome == OutcomeState.WIN:
                win_streak += 1
            else:
                if win_streak >= 10:
                    trans.notes = f"Anomaly: {win_streak} consecutive wins (ceiling risk)"
                    if trans not in anomalies:
                        anomalies.append(trans)
                win_streak = 0
        
        for trans in transitions:
            if trans.transition_type == TransitionType.LOSS_TO_WIN and trans.confidence_before < UNEXPECTED_BOUNCE_CONFIDENCE:
                trans.notes = f"Anomaly: Unexpected bounce (confidence {trans.confidence_before:.1%})"
                anomalies.append(trans)
        
        for i in range(1, len(transitions)):
            if (transitions[i-1].transition_type == TransitionType.LOSS_TO_WIN and
                transitions[i].transition_type == TransitionType.WIN_TO_LOSS):
                trans = transitions[i]
                trans.notes = f"Anomaly: Pattern reversal - L→W then W→L"
                if trans not in anomalies:
                    anomalies.append(trans)
        
        return anomalies
    
    # ========== Dimension-Specific Getters (Enhanced) ==========
    
    def get_dimension_matrix(
        self,
        season_id: str,
        dimension: DimensionType,
    ) -> Optional[DimensionTransitionMatrix]:
        """Get dimension-specific transition matrix."""
        if season_id not in self.seasons:
            return None
        
        analysis = self.seasons[season_id]
        
        dimension_map = {
            DimensionType.OVERALL: None,
            DimensionType.HOME: analysis.home_matrix,
            DimensionType.AWAY: analysis.away_matrix,
            DimensionType.VS_TOP6: analysis.vs_top6_matrix,
            DimensionType.VS_MID: analysis.vs_mid_matrix,
            DimensionType.VS_BOTTOM6: analysis.vs_bottom6_matrix,
        }
        
        return dimension_map.get(dimension)
    
    def get_dimension_probability(
        self,
        season_id: str,
        dimension: str,
        prev_outcome: str,
        curr_outcome: str,
    ) -> float:
        """Get dirty transition probability for a specific dimension."""
        if season_id not in self.seasons:
            return 0.33
        
        try:
            prev = OutcomeState[prev_outcome.upper()]
            curr = OutcomeState[curr_outcome.upper()]
        except KeyError:
            return 0.33
        
        analysis = self.seasons[season_id]
        
        if dimension == "home" and analysis.home_matrix:
            return analysis.home_matrix.get_dirty_prob(prev, curr)
        elif dimension == "away" and analysis.away_matrix:
            return analysis.away_matrix.get_dirty_prob(prev, curr)
        elif dimension == "vs_top6" and analysis.vs_top6_matrix:
            return analysis.vs_top6_matrix.get_dirty_prob(prev, curr)
        elif dimension == "vs_mid" and analysis.vs_mid_matrix:
            return analysis.vs_mid_matrix.get_dirty_prob(prev, curr)
        elif dimension == "vs_bottom6" and analysis.vs_bottom6_matrix:
            return analysis.vs_bottom6_matrix.get_dirty_prob(prev, curr)
        
        matrix = analysis.matrix
        return matrix.probabilities.get((prev, curr), 0.33)
    
    def get_clean_probability(
        self,
        season_id: str,
        dimension: str,
        prev_outcome: str,
        curr_outcome: str,
    ) -> float:
        """Get clean (undistorted) transition probability."""
        if season_id not in self.seasons:
            return 0.33
        
        try:
            prev = OutcomeState[prev_outcome.upper()]
            curr = OutcomeState[curr_outcome.upper()]
        except KeyError:
            return 0.33
        
        analysis = self.seasons[season_id]
        
        if dimension == "home" and analysis.home_matrix:
            return analysis.home_matrix.get_clean_prob(prev, curr)
        elif dimension == "away" and analysis.away_matrix:
            return analysis.away_matrix.get_clean_prob(prev, curr)
        elif dimension == "vs_top6" and analysis.vs_top6_matrix:
            return analysis.vs_top6_matrix.get_clean_prob(prev, curr)
        elif dimension == "vs_mid" and analysis.vs_mid_matrix:
            return analysis.vs_mid_matrix.get_clean_prob(prev, curr)
        elif dimension == "vs_bottom6" and analysis.vs_bottom6_matrix:
            return analysis.vs_bottom6_matrix.get_clean_prob(prev, curr)
        
        # Fallback to dirty matrix
        return self.get_dimension_probability(season_id, dimension, prev_outcome, curr_outcome)
    
    def get_distortion_adjustment(
        self,
        season_id: str,
        dimension: str,
        prev_outcome: str,
        curr_outcome: str,
    ) -> float:
        """Get distortion adjustment for a transition."""
        if season_id not in self.seasons:
            return 0.0
        
        try:
            prev = OutcomeState[prev_outcome.upper()]
            curr = OutcomeState[curr_outcome.upper()]
        except KeyError:
            return 0.0
        
        analysis = self.seasons[season_id]
        
        if dimension == "home" and analysis.home_matrix:
            return analysis.home_matrix.get_distortion_adjustment(prev, curr)
        elif dimension == "away" and analysis.away_matrix:
            return analysis.away_matrix.get_distortion_adjustment(prev, curr)
        
        return 0.0
    
    def get_dimension_reliability(
        self,
        season_id: str,
        dimension: str,
    ) -> PatternReliabilityScore:
        """Get pattern reliability for a dimension."""
        if season_id not in self.seasons:
            return PatternReliabilityScore()
        
        analysis = self.seasons[season_id]
        
        if dimension == "home" and analysis.home_matrix:
            return analysis.home_matrix.overall_reliability
        elif dimension == "away" and analysis.away_matrix:
            return analysis.away_matrix.overall_reliability
        elif dimension == "vs_top6" and analysis.vs_top6_matrix:
            return analysis.vs_top6_matrix.overall_reliability
        
        return PatternReliabilityScore()
    
    def is_pattern_genuine(
        self,
        season_id: str,
        pattern_type: str = "bounce_back",
        dimension: str = "overall",
    ) -> bool:
        """Check if a pattern is genuine (not distortion-driven)."""
        if season_id not in self.seasons:
            return False
        
        analysis = self.seasons[season_id]
        
        if dimension == "home" and analysis.home_matrix:
            return analysis.home_matrix.is_pattern_genuine(pattern_type)
        elif dimension == "away" and analysis.away_matrix:
            return analysis.away_matrix.is_pattern_genuine(pattern_type)
        elif dimension == "vs_top6" and analysis.vs_top6_matrix:
            return analysis.vs_top6_matrix.is_pattern_genuine(pattern_type)
        
        # Fallback to dirty matrix
        return False
    
    def get_venue_advantage(self, season_id: str, use_clean: bool = False) -> float:
        """Calculate home/away advantage using dimension matrices."""
        if season_id not in self.seasons:
            return 0.0
        
        analysis = self.seasons[season_id]
        
        if use_clean:
            home_win_prob = analysis.home_matrix.clean_bounce_back if analysis.home_matrix else 0.33
            away_win_prob = analysis.away_matrix.clean_bounce_back if analysis.away_matrix else 0.33
        else:
            home_win_prob = analysis.home_matrix.bounce_back_rate if analysis.home_matrix else 0.33
            away_win_prob = analysis.away_matrix.bounce_back_rate if analysis.away_matrix else 0.33
        
        return home_win_prob - away_win_prob
    
    def get_bounce_back_by_dimension(
        self,
        season_id: str,
        use_clean: bool = False,
    ) -> Dict[str, float]:
        """Get bounce-back rates for all dimensions."""
        if season_id not in self.seasons:
            return {}
        
        analysis = self.seasons[season_id]
        
        result = {}
        
        if use_clean:
            result["overall"] = analysis.matrix.bounce_back_rate
            if analysis.home_matrix:
                result["home"] = analysis.home_matrix.clean_bounce_back
            if analysis.away_matrix:
                result["away"] = analysis.away_matrix.clean_bounce_back
            if analysis.vs_top6_matrix:
                result["vs_top6"] = analysis.vs_top6_matrix.clean_bounce_back
            if analysis.vs_mid_matrix:
                result["vs_mid"] = analysis.vs_mid_matrix.clean_bounce_back
            if analysis.vs_bottom6_matrix:
                result["vs_bottom6"] = analysis.vs_bottom6_matrix.clean_bounce_back
        else:
            result["overall"] = analysis.matrix.bounce_back_rate
            if analysis.home_matrix:
                result["home"] = analysis.home_matrix.bounce_back_rate
            if analysis.away_matrix:
                result["away"] = analysis.away_matrix.bounce_back_rate
            if analysis.vs_top6_matrix:
                result["vs_top6"] = analysis.vs_top6_matrix.bounce_back_rate
            if analysis.vs_mid_matrix:
                result["vs_mid"] = analysis.vs_mid_matrix.bounce_back_rate
            if analysis.vs_bottom6_matrix:
                result["vs_bottom6"] = analysis.vs_bottom6_matrix.bounce_back_rate
        
        return result
    
    def get_distortion_stats(self, season_id: str, dimension: str = "overall") -> Optional[Dict]:
        """Get distortion statistics for a dimension."""
        if season_id not in self.seasons:
            return None
        
        analysis = self.seasons[season_id]
        
        if dimension == "home" and analysis.home_matrix:
            return analysis.home_matrix.distortion_stats.to_dict() if analysis.home_matrix.distortion_stats else None
        elif dimension == "away" and analysis.away_matrix:
            return analysis.away_matrix.distortion_stats.to_dict() if analysis.away_matrix.distortion_stats else None
        elif dimension == "vs_top6" and analysis.vs_top6_matrix:
            return analysis.vs_top6_matrix.distortion_stats.to_dict() if analysis.vs_top6_matrix.distortion_stats else None
        
        return None
    
    # ========== Dimension Comparison ==========
    
    def compare_dimensions(
        self,
        season_id: str,
        dim_a: str,
        dim_b: str,
        use_clean: bool = False,
    ) -> Dict[str, Any]:
        """Compare two dimensions for pattern differences."""
        if season_id not in self.seasons:
            return {"error": "Season not found"}
        
        analysis = self.seasons[season_id]
        
        def get_bounce_back(dim: str) -> float:
            if dim == "home" and analysis.home_matrix:
                return analysis.home_matrix.clean_bounce_back if use_clean else analysis.home_matrix.bounce_back_rate
            elif dim == "away" and analysis.away_matrix:
                return analysis.away_matrix.clean_bounce_back if use_clean else analysis.away_matrix.bounce_back_rate
            elif dim == "vs_top6" and analysis.vs_top6_matrix:
                return analysis.vs_top6_matrix.clean_bounce_back if use_clean else analysis.vs_top6_matrix.bounce_back_rate
            elif dim == "vs_mid" and analysis.vs_mid_matrix:
                return analysis.vs_mid_matrix.clean_bounce_back if use_clean else analysis.vs_mid_matrix.bounce_back_rate
            elif dim == "vs_bottom6" and analysis.vs_bottom6_matrix:
                return analysis.vs_bottom6_matrix.clean_bounce_back if use_clean else analysis.vs_bottom6_matrix.bounce_back_rate
            return 0.33
        
        bounce_diff = get_bounce_back(dim_a) - get_bounce_back(dim_b)
        
        consistency = 1.0 - min(1.0, abs(bounce_diff) * 2)
        
        if abs(bounce_diff) > 0.15:
            recommendation = f"Significant bounce-back difference: {dim_a} vs {dim_b}"
        else:
            recommendation = "Dimensions show consistent patterns"
        
        return {
            "dimension_a": dim_a,
            "dimension_b": dim_b,
            "bounce_back_diff": round(bounce_diff, 3),
            "consistency_score": round(consistency, 3),
            "recommendation": recommendation,
        }
    
    # ========== Legacy Methods ==========
    
    def forecast_next(self, season_id: str, last_outcome: OutcomeState) -> Optional[Tuple[OutcomeState, float]]:
        """Forecast the next outcome based on current state."""
        if season_id not in self.seasons:
            return None
        
        matrix = self.seasons[season_id].matrix
        
        win_prob = matrix.probabilities.get((last_outcome, OutcomeState.WIN), 0.33)
        draw_prob = matrix.probabilities.get((last_outcome, OutcomeState.DRAW), 0.33)
        loss_prob = matrix.probabilities.get((last_outcome, OutcomeState.LOSS), 0.34)
        
        probs = {
            OutcomeState.WIN: win_prob,
            OutcomeState.DRAW: draw_prob,
            OutcomeState.LOSS: loss_prob,
        }
        
        predicted = max(probs, key=probs.get)
        
        return predicted, probs[predicted]
    
    def get_summary(self, season_id: str) -> Dict[str, Any]:
        """Get full summary for a season including dimensions and reliability."""
        if season_id not in self.seasons:
            return {"error": f"Season {season_id} not found"}
        
        analysis = self.seasons[season_id]
        matrix = analysis.matrix
        
        result = {
            "season_id": season_id,
            "total_fixtures": analysis.total_fixtures,
            "win_rate": analysis.overall_win_rate,
            "wins": analysis.total_wins,
            "draws": analysis.total_draws,
            "losses": analysis.total_losses,
            "profit": analysis.overall_profit,
            "roi": analysis.overall_roi,
            "matrix": {
                "win_to_win": matrix.probabilities.get((OutcomeState.WIN, OutcomeState.WIN), 0),
                "win_to_draw": matrix.probabilities.get((OutcomeState.WIN, OutcomeState.DRAW), 0),
                "win_to_loss": matrix.probabilities.get((OutcomeState.WIN, OutcomeState.LOSS), 0),
                "draw_to_win": matrix.probabilities.get((OutcomeState.DRAW, OutcomeState.WIN), 0),
                "draw_to_draw": matrix.probabilities.get((OutcomeState.DRAW, OutcomeState.DRAW), 0),
                "draw_to_loss": matrix.probabilities.get((OutcomeState.DRAW, OutcomeState.LOSS), 0),
                "loss_to_win": matrix.probabilities.get((OutcomeState.LOSS, OutcomeState.WIN), 0),
                "loss_to_draw": matrix.probabilities.get((OutcomeState.LOSS, OutcomeState.DRAW), 0),
                "loss_to_loss": matrix.probabilities.get((OutcomeState.LOSS, OutcomeState.LOSS), 0),
            },
            "resilience": {
                "bounce_back_rate": matrix.bounce_back_rate,
                "recovery_rate": matrix.drawdown_recovery_rate,
                "max_win_streak": matrix.max_win_streak,
                "max_loss_streak": matrix.max_loss_streak,
                "current_win_streak": matrix.current_win_streak,
                "current_loss_streak": matrix.current_loss_streak,
            },
            "decay": {
                "early_season_wr": matrix.early_season_wr,
                "late_season_wr": matrix.late_season_wr,
                "decay_rate": matrix.decay_rate,
            },
        }
        
        # Add dimension summaries
        if analysis.home_matrix:
            result["home_matrix"] = analysis.home_matrix.to_dict()
        if analysis.away_matrix:
            result["away_matrix"] = analysis.away_matrix.to_dict()
        if analysis.vs_top6_matrix:
            result["vs_top6_matrix"] = analysis.vs_top6_matrix.to_dict()
        
        # Add dimension bounce-back comparison
        result["dimension_bounce_back"] = self.get_bounce_back_by_dimension(season_id, use_clean=False)
        result["clean_bounce_back"] = self.get_bounce_back_by_dimension(season_id, use_clean=True)
        result["venue_advantage"] = self.get_venue_advantage(season_id)
        result["clean_venue_advantage"] = self.get_venue_advantage(season_id, use_clean=True)
        
        # Add pattern reliability
        result["pattern_reliability"] = {
            "overall": self.get_dimension_reliability(season_id, "overall").to_dict(),
            "home": self.get_dimension_reliability(season_id, "home").to_dict() if analysis.home_matrix else None,
            "away": self.get_dimension_reliability(season_id, "away").to_dict() if analysis.away_matrix else None,
        }
        
        # Add distortion stats
        result["distortion_stats"] = self.get_distortion_stats(season_id, "overall")
        
        return result
    
    def export_to_json(self, season_id: str, filename: str = None) -> str:
        """Export season analysis to JSON."""
        summary = self.get_summary(season_id)
        
        if filename is None:
            filename = f"rtm_{season_id}_{datetime.now().strftime('%Y%m%d')}.json"
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        
        return filename


# ==================== Integration Helpers ====================

def validate_pattern_resilience(
    rtm: ResultsTransitionMatrix,
    season_id: str,
    use_clean: bool = True,
) -> Dict[str, Any]:
    """Validate if system shows healthy resilience pattern."""
    if season_id not in rtm.seasons:
        return {"valid": False, "reason": "Season not found", "health_score": 0.0}
    
    analysis = rtm.seasons[season_id]
    matrix = analysis.matrix
    
    if use_clean and analysis.home_matrix:
        bounce_back_rate = analysis.home_matrix.clean_bounce_back if analysis.home_matrix else matrix.bounce_back_rate
        max_loss_streak = analysis.home_matrix.clean_max_loss_streak if analysis.home_matrix else matrix.max_loss_streak
    else:
        bounce_back_rate = matrix.bounce_back_rate
        max_loss_streak = matrix.max_loss_streak
    
    checks = {
        "bounce_back_strong": bounce_back_rate > BOUNCE_BACK_THRESHOLD_GOOD,
        "no_collapse": max_loss_streak < MAX_LOSS_STREAK_WARNING,
        "win_ceiling_normal": matrix.max_win_streak < 20,
        "minimal_decay": matrix.decay_rate < DECAY_THRESHOLD_WARNING,
        "stable_confidence": abs(matrix.confidence_shift) < CONFIDENCE_CALIBRATION_TOLERANCE,
        "recovery_capable": matrix.drawdown_recovery_rate > 0.55,
    }
    
    # Add dimension-specific checks
    if analysis.home_matrix:
        checks["home_bounce_back"] = analysis.home_matrix.bounce_back_rate > BOUNCE_BACK_THRESHOLD_GOOD
    if analysis.away_matrix:
        checks["away_bounce_back"] = analysis.away_matrix.bounce_back_rate > BOUNCE_BACK_THRESHOLD_GOOD
    
    is_valid = all(checks.values())
    
    health_score = 0.0
    health_score += (bounce_back_rate / BOUNCE_BACK_THRESHOLD_EXCELLENT) * 0.25
    health_score += (1 - min(1.0, max_loss_streak / 8)) * 0.20
    health_score += (1 - min(1.0, matrix.decay_rate / 0.25)) * 0.25
    health_score += (1 - abs(matrix.confidence_shift)) * 0.15
    health_score += matrix.drawdown_recovery_rate * 0.15
    health_score = min(1.0, health_score)
    
    return {
        "valid": is_valid,
        "checks": checks,
        "health_score": round(health_score, 4),
        "season_id": season_id,
        "using_clean": use_clean,
    }


def get_dimension_bounce_back(
    rtm: ResultsTransitionMatrix,
    season_id: str,
    dimension: str,
    use_clean: bool = True,
) -> float:
    """Get bounce-back rate for a specific dimension."""
    if season_id not in rtm.seasons:
        return 0.33
    
    analysis = rtm.seasons[season_id]
    
    if dimension == "home" and analysis.home_matrix:
        return analysis.home_matrix.clean_bounce_back if use_clean else analysis.home_matrix.bounce_back_rate
    elif dimension == "away" and analysis.away_matrix:
        return analysis.away_matrix.clean_bounce_back if use_clean else analysis.away_matrix.bounce_back_rate
    elif dimension == "vs_top6" and analysis.vs_top6_matrix:
        return analysis.vs_top6_matrix.clean_bounce_back if use_clean else analysis.vs_top6_matrix.bounce_back_rate
    
    return analysis.matrix.bounce_back_rate


def get_current_streak_by_dimension(
    rtm: ResultsTransitionMatrix,
    season_id: str,
    dimension: str,
    use_clean: bool = True,
) -> Dict[str, Any]:
    """Get current streak for a specific dimension."""
    if season_id not in rtm.seasons:
        return {"error": f"Season {season_id} not found"}
    
    analysis = rtm.seasons[season_id]
    
    if dimension == "home" and analysis.home_matrix:
        if analysis.home_matrix.current_win_streak > 0:
            return {"type": "WIN", "length": analysis.home_matrix.current_win_streak}
        elif analysis.home_matrix.current_loss_streak > 0:
            return {"type": "LOSS", "length": analysis.home_matrix.current_loss_streak}
    elif dimension == "away" and analysis.away_matrix:
        if analysis.away_matrix.current_win_streak > 0:
            return {"type": "WIN", "length": analysis.away_matrix.current_win_streak}
        elif analysis.away_matrix.current_loss_streak > 0:
            return {"type": "LOSS", "length": analysis.away_matrix.current_loss_streak}
    
    matrix = analysis.matrix
    if matrix.current_win_streak > 0:
        return {"type": "WIN", "length": matrix.current_win_streak}
    elif matrix.current_loss_streak > 0:
        return {"type": "LOSS", "length": matrix.current_loss_streak}
    return {"type": "DRAW", "length": 1}


# ==================== Exports ====================

__all__ = [
    "OutcomeState",
    "TransitionType",
    "ConfidenceLevel",
    "DimensionType",
    "PatternReliability",
    "DistortionType",
    # Distortion models
    "DistortionFactor",
    "ContextFlags",
    "EnhancedTransitionRecord",
    "DistortionStats",
    "PatternReliabilityScore",
    # Enhanced dimension matrix
    "DimensionTransitionMatrix",
    "TransitionMatrix",
    "SeasonAnalysis",
    # Main class
    "ResultsTransitionMatrix",
    # Integration helpers
    "validate_pattern_resilience",
    "get_dimension_bounce_back",
    "get_current_streak_by_dimension",
    # Constants
    "MIN_TRANSITIONS_FOR_RELIABLE",
    "MIN_TRANSITIONS_FOR_STRONG",
    "MIN_TRANSITIONS_FOR_HIGH",
    "MIN_CLEAN_TRANSITIONS",
    "DISTORTION_RATE_WARNING",
    "DISTORTION_RATE_CRITICAL",
    "BOUNCE_BACK_THRESHOLD_GOOD",
    "BOUNCE_BACK_THRESHOLD_EXCELLENT",
]