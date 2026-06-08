"""
The Match Oracle - Module 2: Core Data Models & Types (ENHANCED v3)
==============================================================
Defines the foundational data structures used across all 34+ modules.

ENHANCEMENTS IN THIS VERSION (v3):
---------------------------
1. ADDED: DistortionFactor for tracking external factors affecting transitions
2. ADDED: EnhancedTransitionRecord with distortion tracking
3. ADDED: EnhancedDimensionRTM with clean/dirty matrix comparison
4. ADDED: PatternReliabilityScore for trust assessment
5. ADDED: DistortionStats for tracking distortion frequencies
6. ADDED: ContextFlags for match context in transitions
7. ADDED: AllWeightedDecisionProperties for M11 integration
8. ADDED: to_leg_data() methods for all major classes

PREVIOUS ENHANCEMENTS:
---------------------
- DimensionRTM for venue/tier/H2H specific transition matrices
- MultiDimensionRTM container for all RTM dimensions
- MultiDimensionStreak for streak tracking across dimensions
- DimensionBounceBack for bounce-back analysis by dimension
- DimensionResilience for resilience scoring by dimension
- TierClassification enum for opponent tier detection
- VenueType enum for home/away/neutral

DATA MODELS INCLUDED:
-------------------
- BetMarket: Enum for bet types
- TransitionPattern: Pattern types from RTM
- CompetitionFormat: Competition classification
- LeagueTier: League tier classification
- QualificationStatus: Pipeline tracking
- MarketBias: Odds market bias
- TierClassification: TOP6/MID/BOTTOM6/UNKNOWN
- VenueType: HOME/AWAY/NEUTRAL
- DimensionType: OVERALL/HOME/AWAY/VS_TOP6/VS_MID/VS_BOTTOM6/H2H

NEW DISTORTION MODELS:
--------------------
- DistortionFactor: Tracks injuries, suspensions, manager changes, etc.
- EnhancedTransitionRecord: Transition with distortion tracking
- DistortionStats: Aggregated distortion statistics
- EnhancedDimensionRTM: Clean/dirty matrix comparison
- PatternReliabilityScore: How trustworthy a pattern is

Usage:
    from module2 import Leg, TeamProfile, MultiDimensionRTM, EnhancedDimensionRTM
    
    leg = Leg(match_id="...", ...)
    
    # Get dimension-specific RTM
    home_rtm = leg.home_profile.multi_rtm.home
    bounce_rate = home_rtm.bounce_back_rate
    
    # Check pattern reliability
    if enhanced_rtm.is_pattern_reliable():
        prob = enhanced_rtm.get_clean_prob("W", "W")
    
    # Get weighted decision scores for M11
    leg_data = leg.to_leg_data()
"""
from __future__ import annotations

import math
import statistics
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from datetime import datetime, timezone


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — ENUMS
# ═══════════════════════════════════════════════════════════════

class BetMarket(Enum):
    """Supported betting markets."""
    STRAIGHT_WIN = "straight_win"
    DRAW_NO_BET = "draw_no_bet"
    DOUBLE_CHANCE = "double_chance"
    OVER_UNDER = "over_under"
    BOTH_TEAMS_TO_SCORE = "btts"


class TransitionPattern(Enum):
    """Pattern types inferred from transition matrix."""
    SERIAL_WINNER = "SERIAL_WINNER"
    LOSS_PRONE = "LOSS_PRONE"
    DRAW_SPECIALIST = "DRAW_SPECIALIST"
    VOLATILE = "VOLATILE"
    BOUNCER = "BOUNCER"
    HIGH_VARIANCE = "HIGH_VARIANCE"
    INCONSISTENT = "INCONSISTENT"
    UNKNOWN = "UNKNOWN"


class CompetitionFormat(Enum):
    """Competition format classification for filtering."""
    REGULAR_SEASON = "regular_season"
    PLAYOFF = "playoff"
    PROMOTION_PLAYOFF = "promotion_playoff"
    RELEGATION_PLAYOFF = "relegation_playoff"
    KNOCKOUT = "knockout"
    CUP = "cup"
    FRIENDLY = "friendly"


class LeagueTier(Enum):
    """League tier classification for filtering."""
    TIER_1_ELITE = 1      # Top 5 European leagues
    TIER_2_STRONG = 2     # Championship, Eredivisie, Primeira Liga, etc.
    TIER_3_MID = 3        # Lower divisions, smaller European leagues
    TIER_4_LOW = 4        # Regional leagues, lower tiers
    UNKNOWN = 5


class QualificationStatus(Enum):
    """Pipeline qualification status tracking."""
    PENDING = "PENDING"
    PASSED_M4 = "PASSED_M4"
    REJECTED_M4 = "REJECTED_M4"
    PASSED_M5 = "PASSED_M5"
    REJECTED_M5 = "REJECTED_M5"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    APPROVED_AI = "APPROVED_AI"
    REJECTED_AI = "REJECTED_AI"


class MarketBias(Enum):
    """Market bias detection for odds."""
    NEUTRAL = "NEUTRAL"
    HOME_BIASED = "HOME_BIASED"
    AWAY_BIASED = "AWAY_BIASED"
    SHARP_MONEY_HOME = "SHARP_MONEY_HOME"
    SHARP_MONEY_AWAY = "SHARP_MONEY_AWAY"


class TierClassification(Enum):
    """Opponent tier classification based on league position."""
    TOP6 = "top6"           # Top 6 in league (champions/Europa spots)
    MID = "mid"             # Middle of the table
    BOTTOM6 = "bottom6"     # Bottom 6 (relegation battlers)
    UNKNOWN = "unknown"


class VenueType(Enum):
    """Venue classification for fixtures."""
    HOME = "home"
    AWAY = "away"
    NEUTRAL = "neutral"


class DimensionType(Enum):
    """Types of dimensions for RTM analysis."""
    OVERALL = "overall"
    HOME = "home"
    AWAY = "away"
    VS_TOP6 = "vs_top6"
    VS_MID = "vs_mid"
    VS_BOTTOM6 = "vs_bottom6"
    H2H_OVERALL = "h2h_overall"
    H2H_HOME = "h2h_home"
    H2H_AWAY = "h2h_away"


class DistortionType(Enum):
    """Types of distortion factors that can affect transitions."""
    INJURY = "injury"
    SUSPENSION = "suspension"
    MANAGER_CHANGE = "manager_change"
    INTERNATIONAL_DUTY = "intl_duty"
    MIDWEEK_FATIGUE = "midweek_fatigue"
    MOTIVATION = "motivation"        # Dead rubber, six-pointer
    WEATHER = "weather"
    REFEREE = "referee"
    SQUAD_ROTATION = "rotation"
    TRAVEL = "travel"                 # Long travel distance
    INTERNATIONAL_BREAK = "intl_break"


class PatternReliability(Enum):
    """How reliable a detected pattern is."""
    DEFINITIVE = "DEFINITIVE"   # Pattern holds after removing distortions
    STRONG = "STRONG"           # Pattern mostly holds
    MODERATE = "MODERATE"       # Pattern exists but affected by distortions
    WEAK = "WEAK"               # Pattern disappears when distortions removed
    INSUFFICIENT = "INSUFFICIENT"  # Not enough clean data
    SPURIOUS = "SPURIOUS"       # Pattern only exists due to distortions


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — CONSTANTS
# ═══════════════════════════════════════════════════════════════

MIN_CLEAN_TRANSITIONS_FOR_RELIABLE = 5
MIN_TOTAL_TRANSITIONS_FOR_ANALYSIS = 10
RELIABILITY_THRESHOLD_DIFFERENCE = 0.10  # 10% max difference for pattern reliability

# Outcome states as strings for matrix keys
W, D, L = "W", "D", "L"
ALL_OUTCOMES = [W, D, L]


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — DISTORTION MODELS (NEW v3)
# ═══════════════════════════════════════════════════════════════

@dataclass
class DistortionFactor:
    """
    Factor that may have distorted a transition outcome.
    
    Tracks external influences that could make a result
    non-representative of the team's true underlying strength.
    """
    factor_type: DistortionType
    severity: float = 0.0           # 0-1 impact severity
    description: str = ""
    player_name: str = ""           # For injuries/suspensions
    player_position: str = ""       # GK, CB, ST, etc.
    games_missed: int = 0
    days_affected: int = 0
    is_key_player: bool = False
    
    @property
    def adjusted_severity(self) -> float:
        """Adjust severity based on key player status and position."""
        severity = self.severity
        
        # Key player doubles the impact
        if self.is_key_player:
            severity = min(1.0, severity * 1.5)
        
        # Position importance adjustment
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
    """
    Match context flags that affect transition interpretation.
    
    These flags help distinguish between genuine pattern
    and context-driven anomalies.
    """
    is_dead_rubber: bool = False
    is_six_pointer: bool = False
    is_derby: bool = False
    is_early_season: bool = False
    is_late_season: bool = False
    is_post_international_break: bool = False
    is_midweek_fixture: bool = False
    days_rest: int = 7
    venue: VenueType = VenueType.NEUTRAL
    weather_condition: str = ""      # rain, snow, wind
    temperature_celsius: Optional[float] = None
    
    @property
    def context_impact(self) -> float:
        """Calculate total context impact (0-1, higher = more distortion)."""
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
            "venue": self.venue.value,
            "impact": round(self.context_impact, 3),
        }


@dataclass
class EnhancedTransitionRecord:
    """
    Single transition event with distortion tracking.
    
    Includes all factors that might have influenced
    the outcome, enabling clean/dirty matrix separation.
    """
    timestamp: str
    match_id: str
    prev_outcome: str           # W, D, L
    current_outcome: str        # W, D, L
    transition_type: str        # W→W, W→D, etc.
    
    # Standard fields
    confidence_before: float = 0.5
    confidence_after: float = 0.5
    tier: str = "UNKNOWN"
    odds: float = 2.0
    
    # NEW: Distortion factors
    distortions: List[DistortionFactor] = field(default_factory=list)
    
    # NEW: Context flags
    context: ContextFlags = field(default_factory=ContextFlags)
    
    # NEW: Team state at transition
    key_players_missing: int = 0
    manager_tenure_days: int = 365
    squad_rotation_count: int = 0
    is_new_manager: bool = False
    
    @property
    def total_distortion_severity(self) -> float:
        """Sum of adjusted distortion severities."""
        return sum(d.adjusted_severity for d in self.distortions)
    
    @property
    def is_clean(self) -> bool:
        """
        True if this transition is clean (no significant distortions).
        
        A transition is clean if:
        - No distortions with adjusted severity > 0.3
        - Total distortion severity < 0.5
        - Not a dead rubber or early/late season context that distorts motivation
        """
        if self.context.is_dead_rubber:
            return False
        
        if self.total_distortion_severity >= 0.5:
            return False
        
        for d in self.distortions:
            if d.adjusted_severity > 0.3:
                return False
        
        # Early/late season less than 5 games played
        if self.context.is_early_season and self.context.days_rest < 3:
            return False
        
        return True
    
    @property
    def is_highly_distorted(self) -> bool:
        """True if multiple high-severity distortions present."""
        high_severity = sum(1 for d in self.distortions if d.adjusted_severity > 0.4)
        return high_severity >= 2 or self.total_distortion_severity >= 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "match_id": self.match_id,
            "prev": self.prev_outcome,
            "current": self.current_outcome,
            "transition": self.transition_type,
            "is_clean": self.is_clean,
            "distortions": [d.to_dict() for d in self.distortions],
            "context": self.context.to_dict(),
            "key_players_missing": self.key_players_missing,
            "manager_tenure_days": self.manager_tenure_days,
        }


@dataclass
class DistortionStats:
    """
    Aggregated distortion statistics for a team or dimension.
    
    Helps identify patterns in what distorts results.
    """
    # Counts by distortion type
    injury_count: int = 0
    suspension_count: int = 0
    manager_change_count: int = 0
    intl_duty_count: int = 0
    midweek_fatigue_count: int = 0
    motivation_count: int = 0
    weather_count: int = 0
    
    # Severity totals by type
    injury_severity: float = 0.0
    suspension_severity: float = 0.0
    manager_change_severity: float = 0.0
    intl_duty_severity: float = 0.0
    midweek_fatigue_severity: float = 0.0
    motivation_severity: float = 0.0
    
    # Context counts
    dead_rubber_count: int = 0
    six_pointer_count: int = 0
    derby_count: int = 0
    low_rest_count: int = 0  # <3 days rest
    
    # Totals
    total_distorted: int = 0
    total_clean: int = 0
    
    @property
    def distortion_rate(self) -> float:
        """Percentage of transitions that were distorted."""
        total = self.total_distorted + self.total_clean
        return self.total_distorted / total if total > 0 else 0.0
    
    @property
    def primary_distortion(self) -> Optional[str]:
        """Most common distortion type."""
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


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — ENHANCED DIMENSION RTM (NEW v3)
# ═══════════════════════════════════════════════════════════════

@dataclass
class PatternReliabilityScore:
    """
    How reliable a detected pattern is.
    
    Compares clean vs dirty matrices to determine
    if a pattern is genuine or distortion-driven.
    """
    reliability: PatternReliability = PatternReliability.INSUFFICIENT
    confidence: float = 0.0          # 0-1 confidence in assessment
    clean_sample_size: int = 0
    dirty_sample_size: int = 0
    divergence_score: float = 0.0    # How much clean/dirty differ
    reason: str = ""
    
    @property
    def normalized_score(self) -> float:
        """Convert to 0-1 normalized score."""
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


@dataclass
class EnhancedDimensionRTM:
    """
    RTM with distortion filtering capability.
    
    Maintains separate matrices for:
    - Clean transitions (no significant distortions)
    - Dirty transitions (all transitions, current standard)
    
    This allows comparison to identify genuine patterns
    vs patterns that only exist due to external factors.
    """
    dimension_name: str = ""
    
    # Core matrices
    clean_matrix: Optional[DimensionRTM] = None
    dirty_matrix: Optional[DimensionRTM] = None
    
    # Statistics
    clean_transition_count: int = 0
    distorted_transition_count: int = 0
    distortion_stats: DistortionStats = field(default_factory=DistortionStats)
    
    # Pattern reliability
    bounce_back_reliability: PatternReliabilityScore = field(default_factory=PatternReliabilityScore)
    win_ceiling_reliability: PatternReliabilityScore = field(default_factory=PatternReliabilityScore)
    overall_reliability: PatternReliabilityScore = field(default_factory=PatternReliabilityScore)
    
    # Metadata
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    @property
    def total_transitions(self) -> int:
        return self.clean_transition_count + self.distorted_transition_count
    
    @property
    def clean_ratio(self) -> float:
        """Percentage of transitions that are clean."""
        total = self.total_transitions
        return self.clean_transition_count / total if total > 0 else 0.0
    
    @property
    def is_reliable(self) -> bool:
        """True if clean sample size is sufficient and patterns agree."""
        return (self.clean_transition_count >= MIN_CLEAN_TRANSITIONS_FOR_RELIABLE and
                self.overall_reliability.reliability in (PatternReliability.DEFINITIVE, PatternReliability.STRONG))
    
    def get_clean_prob(self, from_result: str, to_result: str) -> float:
        """Probability from undistorted transitions only."""
        if self.clean_matrix:
            return self.clean_matrix.get_prob(from_result, to_result)
        return 0.33
    
    def get_dirty_prob(self, from_result: str, to_result: str) -> float:
        """Probability from all transitions (traditional RTM)."""
        if self.dirty_matrix:
            return self.dirty_matrix.get_prob(from_result, to_result)
        return 0.33
    
    def get_distortion_adjustment(self, from_result: str, to_result: str) -> float:
        """
        How much distortion affects this transition.
        Positive = distortion inflates probability (overestimates)
        Negative = distortion deflates probability (underestimates)
        """
        dirty = self.get_dirty_prob(from_result, to_result)
        clean = self.get_clean_prob(from_result, to_result)
        return round(dirty - clean, 4)
    
    def get_clean_bounce_back(self) -> float:
        """Bounce-back rate from clean transitions only."""
        if self.clean_matrix:
            return self.clean_matrix.bounce_back_rate
        return 0.33
    
    def get_dirty_bounce_back(self) -> float:
        """Bounce-back rate from all transitions."""
        if self.dirty_matrix:
            return self.dirty_matrix.bounce_back_rate
        return 0.33
    
    def get_clean_win_ceiling(self) -> int:
        """Win ceiling from clean transitions."""
        if self.clean_matrix:
            return self.clean_matrix.win_ceiling
        return 0
    
    def get_dirty_win_ceiling(self) -> int:
        """Win ceiling from all transitions."""
        if self.dirty_matrix:
            return self.dirty_matrix.win_ceiling
        return 0
    
    def is_pattern_genuine(self, pattern_type: str = "bounce_back") -> bool:
        """
        Determine if a pattern is genuine or distortion-driven.
        
        Args:
            pattern_type: "bounce_back", "win_ceiling", "streak", "alternation"
        
        Returns:
            True if pattern holds in clean data
        """
        if self.clean_transition_count < MIN_CLEAN_TRANSITIONS_FOR_RELIABLE:
            return False
        
        if pattern_type == "bounce_back":
            clean_prob = self.get_clean_bounce_back()
            dirty_prob = self.get_dirty_bounce_back()
            diff = abs(clean_prob - dirty_prob)
            
            # Pattern is genuine if bounce-back persists in clean data
            return (clean_prob > 0.45 and diff < RELIABILITY_THRESHOLD_DIFFERENCE)
        
        elif pattern_type == "win_ceiling":
            clean_ceiling = self.get_clean_win_ceiling()
            dirty_ceiling = self.get_dirty_win_ceiling()
            
            # Pattern is genuine if ceiling is same or lower in clean data
            return clean_ceiling >= dirty_ceiling - 1
        
        return False
    
    def compute_reliability_scores(self) -> None:
        """Calculate reliability scores for all patterns."""
        # Bounce-back reliability
        if self.clean_transition_count >= MIN_CLEAN_TRANSITIONS_FOR_RELIABLE:
            clean_bb = self.get_clean_bounce_back()
            dirty_bb = self.get_dirty_bounce_back()
            divergence = abs(clean_bb - dirty_bb)
            
            if divergence < 0.05 and clean_bb > 0.45:
                self.bounce_back_reliability = PatternReliabilityScore(
                    reliability=PatternReliability.DEFINITIVE,
                    confidence=0.95,
                    clean_sample_size=self.clean_transition_count,
                    dirty_sample_size=self.distorted_transition_count,
                    divergence_score=divergence,
                    reason="Bounce-back pattern persists after removing distortions"
                )
            elif divergence < 0.10:
                self.bounce_back_reliability = PatternReliabilityScore(
                    reliability=PatternReliability.STRONG,
                    confidence=0.80,
                    clean_sample_size=self.clean_transition_count,
                    dirty_sample_size=self.distorted_transition_count,
                    divergence_score=divergence,
                    reason="Bounce-back pattern mostly holds after distortion removal"
                )
            elif divergence < 0.15:
                self.bounce_back_reliability = PatternReliabilityScore(
                    reliability=PatternReliability.MODERATE,
                    confidence=0.65,
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
                    confidence=0.30,
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
                reason=f"Insufficient clean transitions ({self.clean_transition_count}/{MIN_CLEAN_TRANSITIONS_FOR_RELIABLE})"
            )
        
        # Win ceiling reliability
        if self.clean_transition_count >= MIN_CLEAN_TRANSITIONS_FOR_RELIABLE:
            clean_ceiling = self.get_clean_win_ceiling()
            dirty_ceiling = self.get_dirty_win_ceiling()
            ceiling_diff = abs(clean_ceiling - dirty_ceiling)
            
            if ceiling_diff == 0 and clean_ceiling > 0:
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
        else:
            overall = PatternReliability.WEAK
            confidence = 0.35
            reason = f"Limited clean data ({self.clean_ratio:.0%} clean transitions)"
        
        self.overall_reliability = PatternReliabilityScore(
            reliability=overall,
            confidence=confidence,
            clean_sample_size=self.clean_transition_count,
            dirty_sample_size=self.distorted_transition_count,
            divergence_score=1.0 - self.clean_ratio,
            reason=reason
        )
    
    def summary(self) -> Dict[str, Any]:
        """Get comprehensive summary."""
        return {
            "dimension": self.dimension_name,
            "clean_transitions": self.clean_transition_count,
            "distorted_transitions": self.distorted_transition_count,
            "clean_ratio": round(self.clean_ratio, 3),
            "is_reliable": self.is_reliable,
            "bounce_back": {
                "clean": round(self.get_clean_bounce_back(), 3),
                "dirty": round(self.get_dirty_bounce_back(), 3),
                "adjustment": round(self.get_distortion_adjustment("L", "W"), 3),
                "reliability": self.bounce_back_reliability.to_dict(),
            },
            "win_ceiling": {
                "clean": self.get_clean_win_ceiling(),
                "dirty": self.get_dirty_win_ceiling(),
                "reliability": self.win_ceiling_reliability.to_dict(),
            },
            "distortion_stats": self.distortion_stats.to_dict(),
            "overall_reliability": self.overall_reliability.to_dict(),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension_name": self.dimension_name,
            "clean_transition_count": self.clean_transition_count,
            "distorted_transition_count": self.distorted_transition_count,
            "clean_ratio": round(self.clean_ratio, 3),
            "is_reliable": self.is_reliable,
            "distortion_stats": self.distortion_stats.to_dict(),
            "bounce_back_reliability": self.bounce_back_reliability.to_dict(),
            "win_ceiling_reliability": self.win_ceiling_reliability.to_dict(),
            "overall_reliability": self.overall_reliability.to_dict(),
            "last_updated": self.last_updated,
        }
    
    def to_leg_data(self) -> Dict[str, Any]:
        """Convert to leg_data format for M11 weighted decision."""
        return {
            f"{self.dimension_name}_clean_transitions": self.clean_transition_count,
            f"{self.dimension_name}_distortion_rate": round(self.distortion_stats.distortion_rate, 3),
            f"{self.dimension_name}_reliability": self.overall_reliability.normalized_score,
            f"{self.dimension_name}_bounce_back_reliable": self.is_pattern_genuine("bounce_back"),
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — DIMENSION-SPECIFIC DATA STRUCTURES (Existing)
# ═══════════════════════════════════════════════════════════════

@dataclass
class DimensionRTM:
    """
    Results Transition Matrix for a specific dimension.
    
    Tracks 3x3 transitions (W/D/L → W/D/L) for filtered fixtures:
    - Venue-specific (home/away)
    - Tier-specific (vs Top6, vs Mid, vs Bottom6)
    - H2H-specific (overall, home, away)
    """
    dimension_name: str = ""
    sample_size: int = 0
    total_fixtures: int = 0
    
    # Raw transition counts
    matrix: Dict[Tuple[str, str], int] = field(default_factory=dict)
    
    # Transition probabilities
    probabilities: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    # Result probabilities (P(W), P(D), P(L))
    result_probs: Dict[str, float] = field(default_factory=dict)
    
    # Streak metrics
    max_win_streak: int = 0
    max_loss_streak: int = 0
    avg_win_streak: float = 0.0
    avg_loss_streak: float = 0.0
    current_win_streak: int = 0
    current_loss_streak: int = 0
    
    # Bounce-back metrics
    bounce_back_rate: float = 0.0          # P(WIN | LOSS)
    bounce_back_confidence: float = 0.0    # 0-1 based on sample size
    bounce_back_qualifies: bool = False    # Whether bounce-back is significant
    
    # Ceiling detection
    win_ceiling: int = 0                   # Max consecutive wins before regression
    unbeaten_ceiling: int = 0              # Max unbeaten streak before loss
    at_ceiling: bool = False               # Currently at ceiling risk
    
    # Resilience metrics
    resilience_score: float = 0.0          # 0-100 resilience score
    recovery_rate: float = 0.0             # P(WIN or DRAW | LOSS)
    drawdown_resistance: float = 0.0       # Ability to avoid consecutive losses
    
    # Confidence flags
    is_reliable: bool = False
    confidence_level: str = "LOW"          # LOW / MEDIUM / HIGH
    
    # NEW v3: Enhanced RTM reference (optional)
    enhanced_rtm: Optional[EnhancedDimensionRTM] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        self.is_reliable = self.sample_size >= 5
        if self.sample_size >= 10:
            self.confidence_level = "HIGH"
        elif self.sample_size >= 5:
            self.confidence_level = "MEDIUM"
    
    def get_prob(self, from_result: str, to_result: str) -> float:
        """Get transition probability."""
        return self.probabilities.get((from_result, to_result), 0.33)
    
    def get_next_prob(self, current_result: str) -> Dict[str, float]:
        """Get probabilities for next result given current."""
        return {
            "W": self.get_prob(current_result, "W"),
            "D": self.get_prob(current_result, "D"),
            "L": self.get_prob(current_result, "L"),
        }
    
    def get_win_probability(self) -> float:
        """Get overall win probability from result distribution."""
        return self.result_probs.get("W", 0.33)
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "dimension": self.dimension_name,
            "sample_size": self.sample_size,
            "bounce_back_rate": round(self.bounce_back_rate, 3),
            "bounce_back_confidence": round(self.bounce_back_confidence, 3),
            "win_ceiling": self.win_ceiling,
            "resilience_score": round(self.resilience_score, 1),
            "current_streak": f"{self.current_win_streak}W" if self.current_win_streak > 0 else f"{self.current_loss_streak}L",
            "reliable": self.is_reliable,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "dimension_name": self.dimension_name,
            "sample_size": self.sample_size,
            "total_fixtures": self.total_fixtures,
            "probabilities": {
                f"{k[0]}→{k[1]}": v for k, v in self.probabilities.items()
            },
            "result_probs": {k: round(v, 3) for k, v in self.result_probs.items()},
            "max_win_streak": self.max_win_streak,
            "max_loss_streak": self.max_loss_streak,
            "current_win_streak": self.current_win_streak,
            "current_loss_streak": self.current_loss_streak,
            "bounce_back_rate": round(self.bounce_back_rate, 3),
            "bounce_back_confidence": round(self.bounce_back_confidence, 3),
            "bounce_back_qualifies": self.bounce_back_qualifies,
            "win_ceiling": self.win_ceiling,
            "unbeaten_ceiling": self.unbeaten_ceiling,
            "at_ceiling": self.at_ceiling,
            "resilience_score": round(self.resilience_score, 1),
            "recovery_rate": round(self.recovery_rate, 3),
            "drawdown_resistance": round(self.drawdown_resistance, 3),
            "is_reliable": self.is_reliable,
            "confidence_level": self.confidence_level,
        }


@dataclass
class DimensionBounceBack:
    """
    Bounce-back analysis across different dimensions.
    
    Compares how a team responds to losses in different contexts.
    """
    overall_rate: float = 0.0
    home_rate: float = 0.0
    away_rate: float = 0.0
    vs_top6_rate: float = 0.0
    vs_mid_rate: float = 0.0
    vs_bottom6_rate: float = 0.0
    h2h_rate: float = 0.0
    
    @property
    def home_advantage(self) -> float:
        """Home bounce-back advantage over away."""
        return self.home_rate - self.away_rate
    
    @property
    def tier_variance(self) -> float:
        """Variance in bounce-back across tiers."""
        rates = [self.vs_top6_rate, self.vs_mid_rate, self.vs_bottom6_rate]
        valid = [r for r in rates if r > 0]
        if not valid:
            return 0.0
        mean = sum(valid) / len(valid)
        return sum((r - mean) ** 2 for r in valid) / len(valid)
    
    def strongest_bounce_back(self) -> Tuple[str, float]:
        """Return (dimension, rate) with highest bounce-back."""
        dimensions = [
            ("overall", self.overall_rate),
            ("home", self.home_rate),
            ("away", self.away_rate),
            ("vs_bottom6", self.vs_bottom6_rate),
        ]
        return max(dimensions, key=lambda x: x[1])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": round(self.overall_rate, 3),
            "home": round(self.home_rate, 3),
            "away": round(self.away_rate, 3),
            "vs_top6": round(self.vs_top6_rate, 3),
            "vs_mid": round(self.vs_mid_rate, 3),
            "vs_bottom6": round(self.vs_bottom6_rate, 3),
            "h2h": round(self.h2h_rate, 3),
            "home_advantage": round(self.home_advantage, 3),
            "tier_variance": round(self.tier_variance, 3),
        }


@dataclass
class DimensionResilience:
    """
    Resilience analysis across different dimensions.
    
    Measures ability to recover from setbacks in different contexts.
    """
    overall_score: float = 0.0
    home_score: float = 0.0
    away_score: float = 0.0
    vs_top6_score: float = 0.0
    vs_mid_score: float = 0.0
    vs_bottom6_score: float = 0.0
    
    # Recovery rates (P(Win or Draw | Loss))
    recovery_rate_overall: float = 0.0
    recovery_rate_home: float = 0.0
    recovery_rate_away: float = 0.0
    
    # Drawdown resistance (ability to avoid consecutive losses)
    drawdown_resistance_overall: float = 0.0
    drawdown_resistance_home: float = 0.0
    drawdown_resistance_away: float = 0.0
    
    @property
    def best_dimension(self) -> Tuple[str, float]:
        """Return dimension with highest resilience."""
        dimensions = [
            ("overall", self.overall_score),
            ("home", self.home_score),
            ("away", self.away_score),
            ("vs_bottom6", self.vs_bottom6_score),
        ]
        return max(dimensions, key=lambda x: x[1])
    
    @property
    def worst_dimension(self) -> Tuple[str, float]:
        """Return dimension with lowest resilience."""
        dimensions = [
            ("overall", self.overall_score),
            ("home", self.home_score),
            ("away", self.away_score),
            ("vs_top6", self.vs_top6_score),
        ]
        return min(dimensions, key=lambda x: x[1])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 1),
            "home_score": round(self.home_score, 1),
            "away_score": round(self.away_score, 1),
            "vs_top6_score": round(self.vs_top6_score, 1),
            "vs_mid_score": round(self.vs_mid_score, 1),
            "vs_bottom6_score": round(self.vs_bottom6_score, 1),
            "recovery_rate_overall": round(self.recovery_rate_overall, 3),
            "recovery_rate_home": round(self.recovery_rate_home, 3),
            "recovery_rate_away": round(self.recovery_rate_away, 3),
            "drawdown_resistance_overall": round(self.drawdown_resistance_overall, 3),
            "drawdown_resistance_home": round(self.drawdown_resistance_home, 3),
            "drawdown_resistance_away": round(self.drawdown_resistance_away, 3),
        }


@dataclass
class MultiDimensionRTM:
    """
    Container for all RTM dimensions for a single team.
    
    Holds separate transition matrices for:
    - Overall (all fixtures)
    - Home (home fixtures only)
    - Away (away fixtures only)
    - Vs Top 6 (vs top 6 opponents)
    - Vs Mid (vs mid-table opponents)
    - Vs Bottom 6 (vs bottom 6 opponents)
    - H2H Overall (vs specific opponent, all venues)
    - H2H Home (vs specific opponent at home)
    - H2H Away (vs specific opponent away)
    
    NEW v3: Now includes enhanced RTM variants for pattern reliability.
    """
    team_id: str = ""
    team_name: str = ""
    
    # Core dimensions
    overall: DimensionRTM = field(default_factory=lambda: DimensionRTM(dimension_name="overall"))
    home: DimensionRTM = field(default_factory=lambda: DimensionRTM(dimension_name="home"))
    away: DimensionRTM = field(default_factory=lambda: DimensionRTM(dimension_name="away"))
    
    # Tier dimensions
    vs_top6: DimensionRTM = field(default_factory=lambda: DimensionRTM(dimension_name="vs_top6"))
    vs_mid: DimensionRTM = field(default_factory=lambda: DimensionRTM(dimension_name="vs_mid"))
    vs_bottom6: DimensionRTM = field(default_factory=lambda: DimensionRTM(dimension_name="vs_bottom6"))
    
    # H2H dimensions (populated dynamically for specific opponents)
    h2h_overall: Optional[DimensionRTM] = None
    h2h_home: Optional[DimensionRTM] = None
    h2h_away: Optional[DimensionRTM] = None
    
    # NEW v3: Enhanced dimensions (clean/dirty separation)
    enhanced_overall: Optional[EnhancedDimensionRTM] = None
    enhanced_home: Optional[EnhancedDimensionRTM] = None
    enhanced_away: Optional[EnhancedDimensionRTM] = None
    enhanced_vs_top6: Optional[EnhancedDimensionRTM] = None
    enhanced_vs_mid: Optional[EnhancedDimensionRTM] = None
    enhanced_vs_bottom6: Optional[EnhancedDimensionRTM] = None
    enhanced_h2h_overall: Optional[EnhancedDimensionRTM] = None
    
    # Aggregate analysis
    bounce_back: DimensionBounceBack = field(default_factory=DimensionBounceBack)
    resilience: DimensionResilience = field(default_factory=DimensionResilience)
    
    # Metadata
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def get_dimension(self, dim: Union[str, DimensionType]) -> Optional[DimensionRTM]:
        """Get RTM for a specific dimension."""
        if isinstance(dim, DimensionType):
            dim = dim.value
        
        mapping = {
            "overall": self.overall,
            "home": self.home,
            "away": self.away,
            "vs_top6": self.vs_top6,
            "vs_mid": self.vs_mid,
            "vs_bottom6": self.vs_bottom6,
            "h2h_overall": self.h2h_overall,
            "h2h_home": self.h2h_home,
            "h2h_away": self.h2h_away,
        }
        return mapping.get(dim)
    
    def get_enhanced_dimension(self, dim: str) -> Optional[EnhancedDimensionRTM]:
        """Get enhanced RTM for a specific dimension."""
        mapping = {
            "overall": self.enhanced_overall,
            "home": self.enhanced_home,
            "away": self.enhanced_away,
            "vs_top6": self.enhanced_vs_top6,
            "vs_mid": self.enhanced_vs_mid,
            "vs_bottom6": self.enhanced_vs_bottom6,
        }
        return mapping.get(dim)
    
    def has_h2h_data(self) -> bool:
        """Check if H2H RTM data is available."""
        return self.h2h_overall is not None and self.h2h_overall.sample_size >= 3
    
    def get_venue_advantage(self) -> float:
        """
        Calculate venue advantage based on RTM.
        Positive = performs better at home.
        """
        home_win_prob = self.home.get_win_probability()
        away_win_prob = self.away.get_win_probability()
        return home_win_prob - away_win_prob
    
    def get_tier_weakness(self) -> Optional[str]:
        """
        Identify if team has weakness against specific tier.
        Returns tier name if weakness detected.
        """
        if self.vs_top6.sample_size >= 3:
            if self.vs_top6.get_win_probability() < self.overall.get_win_probability() - 0.15:
                return "top6"
        if self.vs_bottom6.sample_size >= 3:
            if self.vs_bottom6.get_win_probability() > self.overall.get_win_probability() + 0.10:
                return "bottom6"
        return None
    
    def get_bounce_back_rating(self) -> Tuple[float, str]:
        """
        Get bounce-back rating (0-100) and label.
        Higher = more likely to bounce back after loss.
        """
        rate = self.bounce_back.overall_rate
        if rate >= 0.55:
            return rate * 100, "ELITE"
        elif rate >= 0.45:
            return rate * 100, "GOOD"
        elif rate >= 0.35:
            return rate * 100, "AVERAGE"
        return rate * 100, "POOR"
    
    def get_resilience_rating(self) -> Tuple[float, str]:
        """
        Get resilience rating (0-100) and label.
        Higher = more resilient to setbacks.
        """
        score = self.resilience.overall_score
        if score >= 75:
            return score, "ELITE"
        elif score >= 60:
            return score, "GOOD"
        elif score >= 45:
            return score, "AVERAGE"
        return score, "POOR"
    
    def get_pattern_reliability(self, pattern: str = "overall") -> PatternReliabilityScore:
        """Get pattern reliability score for a dimension."""
        enhanced = self.get_enhanced_dimension(pattern)
        if enhanced:
            return enhanced.overall_reliability
        return PatternReliabilityScore()
    
    def is_pattern_genuine(self, pattern_type: str = "bounce_back", dimension: str = "overall") -> bool:
        """Check if a pattern is genuine (not distortion-driven)."""
        enhanced = self.get_enhanced_dimension(dimension)
        if enhanced:
            return enhanced.is_pattern_genuine(pattern_type)
        return False
    
    def summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all dimensions."""
        result = {
            "team": self.team_name,
            "overall": self.overall.summary(),
            "home": self.home.summary(),
            "away": self.away.summary(),
            "vs_top6": self.vs_top6.summary(),
            "vs_bottom6": self.vs_bottom6.summary(),
            "venue_advantage": round(self.get_venue_advantage(), 3),
            "bounce_back_rating": self.get_bounce_back_rating()[1],
            "resilience_rating": self.get_resilience_rating()[1],
            "has_h2h": self.has_h2h_data(),
        }
        
        # Add reliability info if enhanced RTM available
        if self.enhanced_overall:
            result["pattern_reliability"] = self.enhanced_overall.overall_reliability.to_dict()
            result["clean_transition_ratio"] = round(self.enhanced_overall.clean_ratio, 3)
            result["primary_distortion"] = self.enhanced_overall.distortion_stats.primary_distortion
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "team_id": self.team_id,
            "team_name": self.team_name,
            "overall": self.overall.to_dict(),
            "home": self.home.to_dict(),
            "away": self.away.to_dict(),
            "vs_top6": self.vs_top6.to_dict(),
            "vs_mid": self.vs_mid.to_dict(),
            "vs_bottom6": self.vs_bottom6.to_dict(),
            "bounce_back": self.bounce_back.to_dict(),
            "resilience": self.resilience.to_dict(),
            "enhanced_overall": self.enhanced_overall.to_dict() if self.enhanced_overall else None,
            "last_updated": self.last_updated,
        }
    
    def to_leg_data(self) -> Dict[str, Any]:
        """Convert to leg_data format for M11 weighted decision."""
        return {
            "team_id": self.team_id,
            "team_name": self.team_name,
            "overall_win_prob": self.overall.get_win_probability(),
            "home_win_prob": self.home.get_win_probability(),
            "away_win_prob": self.away.get_win_probability(),
            "bounce_back_rate": self.bounce_back.overall_rate,
            "resilience_score": self.resilience.overall_score,
            "venue_advantage": self.get_venue_advantage(),
            "has_h2h": self.has_h2h_data(),
            "clean_transition_ratio": self.enhanced_overall.clean_ratio if self.enhanced_overall else 0.0,
            "pattern_reliable": self.enhanced_overall.is_reliable if self.enhanced_overall else False,
        }


@dataclass
class MultiDimensionStreak:
    """
    Streak tracking across multiple dimensions.
    
    Tracks current and historical streaks by venue and opponent tier.
    """
    overall: Tuple[int, str] = (0, "NONE")      # (length, direction)
    home: Tuple[int, str] = (0, "NONE")
    away: Tuple[int, str] = (0, "NONE")
    vs_top6: Tuple[int, str] = (0, "NONE")
    vs_mid: Tuple[int, str] = (0, "NONE")
    vs_bottom6: Tuple[int, str] = (0, "NONE")
    
    @property
    def longest_streak(self) -> Tuple[str, int, str]:
        """Return (dimension, length, direction) of longest streak."""
        streaks = [
            ("overall", self.overall[0], self.overall[1]),
            ("home", self.home[0], self.home[1]),
            ("away", self.away[0], self.away[1]),
        ]
        return max(streaks, key=lambda x: x[1])
    
    @property
    def has_momentum(self) -> bool:
        """Check if team has positive momentum in any dimension."""
        return any(
            length >= 3 and direction == "W"
            for length, direction in [self.overall, self.home, self.away]
        )
    
    @property
    def is_collapsing(self) -> bool:
        """Check if team is collapsing (losses across dimensions)."""
        return any(
            length >= 3 and direction == "L"
            for length, direction in [self.overall, self.home, self.away]
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": f"{self.overall[0]}x{self.overall[1]}" if self.overall[0] > 0 else "none",
            "home": f"{self.home[0]}x{self.home[1]}" if self.home[0] > 0 else "none",
            "away": f"{self.away[0]}x{self.away[1]}" if self.away[0] > 0 else "none",
            "longest": f"{self.longest_streak[1]}x{self.longest_streak[2]} ({self.longest_streak[0]})",
            "has_momentum": self.has_momentum,
            "is_collapsing": self.is_collapsing,
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — CORE DATA MODELS (Existing, Enhanced)
# ═══════════════════════════════════════════════════════════════

@dataclass
class StructuralDominance:
    """Favourite's structural dominance metrics."""
    win_rate: float = 0.0
    points_per_game: float = 0.0
    recent_wins: int = 0
    form_gap: float = 0.0
    xg_differential: float = 0.0
    games_played: int = 0
    
    @property
    def passes_win_rate(self) -> bool:
        return self.win_rate >= 0.55
    
    @property
    def passes_ppg(self) -> bool:
        return self.points_per_game >= 1.8
    
    @property
    def passes_recent_wins(self) -> bool:
        return self.recent_wins >= 3
    
    @property
    def passes_form_gap(self) -> bool:
        return self.form_gap >= 0.30
    
    @property
    def dominance_score(self) -> float:
        scores = [
            self.win_rate,
            min(1.0, self.points_per_game / 2.5),
            min(1.0, self.recent_wins / 5),
            0.5 + (self.form_gap / 0.5) if self.form_gap > 0 else 0.5,
        ]
        return round(sum(scores) / len(scores), 3)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "win_rate": round(self.win_rate, 3),
            "points_per_game": round(self.points_per_game, 2),
            "recent_wins": self.recent_wins,
            "form_gap": round(self.form_gap, 3),
            "dominance_score": self.dominance_score,
        }


@dataclass
class DecayMetrics:
    """Underdog's decay metrics."""
    win_rate: float = 0.0
    points_per_game: float = 0.0
    recent_losses: int = 0
    xg_differential: float = 0.0
    games_played: int = 0
    
    @property
    def passes_win_rate(self) -> bool:
        return self.win_rate <= 0.40
    
    @property
    def passes_ppg(self) -> bool:
        return self.points_per_game <= 1.2
    
    @property
    def passes_recent_losses(self) -> bool:
        return self.recent_losses >= 3
    
    @property
    def decay_score(self) -> float:
        scores = [
            1.0 - min(1.0, self.win_rate / 0.40),
            1.0 - min(1.0, self.points_per_game / 1.2),
            min(1.0, self.recent_losses / 5),
        ]
        return round(sum(scores) / len(scores), 3)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "win_rate": round(self.win_rate, 3),
            "points_per_game": round(self.points_per_game, 2),
            "recent_losses": self.recent_losses,
            "decay_score": self.decay_score,
        }


@dataclass
class RTMPattern:
    """Results Transition Matrix pattern classification."""
    pattern_type: TransitionPattern = TransitionPattern.UNKNOWN
    confidence: float = 0.0
    sample_size: int = 0
    next_win_prob: float = 0.33
    next_draw_prob: float = 0.33
    next_loss_prob: float = 0.34
    current_streak: int = 0
    streak_direction: str = "NONE"
    bounce_back_prob: float = 0.0
    bounce_back_due: bool = False
    win_ceiling: int = 0
    loss_floor: int = 0
    
    @property
    def is_reliable(self) -> bool:
        return self.sample_size >= 10
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type.value,
            "confidence": round(self.confidence, 3),
            "sample_size": self.sample_size,
            "next_win_prob": round(self.next_win_prob, 3),
            "next_draw_prob": round(self.next_draw_prob, 3),
            "next_loss_prob": round(self.next_loss_prob, 3),
            "bounce_back_prob": round(self.bounce_back_prob, 3),
            "bounce_back_due": self.bounce_back_due,
            "win_ceiling": self.win_ceiling,
        }


@dataclass
class TransitionMatrix:
    """Results transition matrix (W/D/L → next result probabilities)."""
    pattern: str = "UNKNOWN"
    probs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    sample_size: int = 0
    
    def get_next_prob(self, current_result: str, next_result: str) -> float:
        if current_result not in self.probs:
            return 0.33
        return self.probs[current_result].get(next_result, 0.33)
    
    def most_likely_next(self, current_result: str) -> Tuple[str, float]:
        if current_result not in self.probs:
            return "D", 0.34
        probs = self.probs[current_result]
        return max(probs.items(), key=lambda x: x[1])
    
    def is_reliable(self) -> bool:
        return self.sample_size >= 5
    
    def to_rtm_pattern(self, current_result: str) -> RTMPattern:
        next_win = self.get_next_prob(current_result, "W")
        next_draw = self.get_next_prob(current_result, "D")
        next_loss = self.get_next_prob(current_result, "L")
        
        try:
            pattern_type = TransitionPattern(self.pattern)
        except ValueError:
            pattern_type = TransitionPattern.UNKNOWN
        
        return RTMPattern(
            pattern_type=pattern_type,
            confidence=min(1.0, self.sample_size / 20),
            sample_size=self.sample_size,
            next_win_prob=next_win,
            next_draw_prob=next_draw,
            next_loss_prob=next_loss,
            bounce_back_prob=self.get_next_prob("L", "W"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern": self.pattern,
            "probs": self.probs,
            "sample_size": self.sample_size,
        }


@dataclass
class H2HRecord:
    """Head-to-head record between two teams."""
    games: int = 0
    fav_wins: int = 0
    draws: int = 0
    und_wins: int = 0
    
    @property
    def fav_win_rate(self) -> float:
        return self.fav_wins / self.games if self.games > 0 else 0.0
    
    @property
    def draw_rate(self) -> float:
        return self.draws / self.games if self.games > 0 else 0.0
    
    @property
    def und_win_rate(self) -> float:
        return self.und_wins / self.games if self.games > 0 else 0.0
    
    @property
    def is_one_sided(self) -> bool:
        if self.games < 5:
            return False
        dominant_rate = max(self.fav_win_rate, self.und_win_rate)
        return dominant_rate >= 0.70
    
    @property
    def h2h_score(self) -> float:
        if self.games < 5:
            return 0.5
        return self.fav_win_rate
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "games": self.games,
            "fav_wins": self.fav_wins,
            "draws": self.draws,
            "und_wins": self.und_wins,
            "fav_win_rate": round(self.fav_win_rate, 3),
            "draw_rate": round(self.draw_rate, 3),
            "is_one_sided": self.is_one_sided,
        }


@dataclass
class TeamProfile:
    """Complete statistical profile for one team."""
    team_id: str = ""
    team_name: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    form: Dict[str, Any] = field(default_factory=dict)
    transition: Optional[TransitionMatrix] = None
    multi_rtm: Optional[MultiDimensionRTM] = None
    is_mature: bool = False
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def __post_init__(self):
        games = self.get_metric("core.games", 0)
        self.is_mature = games >= 10
    
    def get_metric(self, key: str, default: float = 0.0) -> float:
        return self.metrics.get(key, default)
    
    def update_metrics(self, updates: Dict[str, float]) -> None:
        self.metrics.update(updates)
        if "core.games" in updates:
            self.is_mature = self.get_metric("core.games", 0) >= 10
        self.last_updated = datetime.now(timezone.utc).isoformat()
    
    def win_rate(self) -> float:
        games = self.get_metric("core.games", 1)
        wins = self.get_metric("core.wins", 0)
        return wins / games if games > 0 else 0.0
    
    def loss_rate(self) -> float:
        games = self.get_metric("core.games", 1)
        losses = self.get_metric("core.losses", 0)
        return losses / games if games > 0 else 0.0
    
    def points_per_game(self) -> float:
        wins = self.get_metric("core.wins", 0)
        draws = self.get_metric("core.draws", 0)
        games = self.get_metric("core.games", 1)
        return (wins * 3 + draws) / games if games > 0 else 0.0
    
    def home_win_rate(self) -> float:
        home_games = self.get_metric("home_games", 1)
        home_wins = self.get_metric("home_wins", 0)
        return home_wins / home_games if home_games > 0 else 0.0
    
    def away_win_rate(self) -> float:
        away_games = self.get_metric("away_games", 1)
        away_wins = self.get_metric("away_wins", 0)
        return away_wins / away_games if away_games > 0 else 0.0
    
    def recent_wins(self, n_games: int = 5) -> int:
        recent = self.form.get("recent_results", [])[-n_games:]
        return recent.count("W")
    
    def recent_losses(self, n_games: int = 5) -> int:
        recent = self.form.get("recent_results", [])[-n_games:]
        return recent.count("L")
    
    def xg_differential(self) -> float:
        xg = self.get_metric("core.xg", 0.0)
        xga = self.get_metric("core.xga", 0.0)
        games = self.get_metric("core.games", 1)
        return (xg - xga) / games if games > 0 else 0.0
    
    def get_venue_advantage(self) -> float:
        """Get venue advantage from multi-dimension RTM."""
        if self.multi_rtm:
            return self.multi_rtm.get_venue_advantage()
        return self.home_win_rate() - self.away_win_rate()
    
    def get_bounce_back_rate(self, dimension: str = "overall") -> float:
        """Get bounce-back rate for a specific dimension."""
        if self.multi_rtm:
            dim_rtm = self.multi_rtm.get_dimension(dimension)
            if dim_rtm:
                return dim_rtm.bounce_back_rate
        return self.transition.get_next_prob("L", "W") if self.transition else 0.33
    
    def get_clean_bounce_back_rate(self, dimension: str = "overall") -> float:
        """Get clean bounce-back rate (distortions removed)."""
        if self.multi_rtm:
            enhanced = self.multi_rtm.get_enhanced_dimension(dimension)
            if enhanced:
                return enhanced.get_clean_bounce_back()
        return self.get_bounce_back_rate(dimension)
    
    def get_bounce_back_rating(self) -> Tuple[float, str]:
        """Get bounce-back rating (0-100) and label."""
        if self.multi_rtm:
            return self.multi_rtm.get_bounce_back_rating()
        rate = self.get_bounce_back_rate()
        if rate >= 0.55:
            return rate * 100, "ELITE"
        elif rate >= 0.45:
            return rate * 100, "GOOD"
        elif rate >= 0.35:
            return rate * 100, "AVERAGE"
        return rate * 100, "POOR"
    
    def get_pattern_reliability(self) -> PatternReliabilityScore:
        """Get overall pattern reliability score."""
        if self.multi_rtm and self.multi_rtm.enhanced_overall:
            return self.multi_rtm.enhanced_overall.overall_reliability
        return PatternReliabilityScore()
    
    def to_dominance(self) -> StructuralDominance:
        games = self.get_metric("core.games", 0)
        return StructuralDominance(
            win_rate=self.win_rate(),
            points_per_game=self.points_per_game(),
            recent_wins=self.recent_wins(5),
            form_gap=0.0,
            xg_differential=self.xg_differential(),
            games_played=int(games),
        )
    
    def to_decay(self) -> DecayMetrics:
        games = self.get_metric("core.games", 0)
        return DecayMetrics(
            win_rate=self.win_rate(),
            points_per_game=self.points_per_game(),
            recent_losses=self.recent_losses(5),
            xg_differential=self.xg_differential(),
            games_played=int(games),
        )
    
    def summary(self) -> str:
        games = self.get_metric("core.games", 0)
        wins = self.get_metric("core.wins", 0)
        draws = self.get_metric("core.draws", 0)
        losses = self.get_metric("core.losses", 0)
        pos = self.get_metric("position", 0)
        
        form_5 = self.form.get("recent_results", [])[-5:]
        form_str = "".join(form_5) if form_5 else "?"
        
        return (f"{self.team_name} (ID:{self.team_id}) | "
                f"Pos {int(pos)} | {wins:.0f}-{draws:.0f}-{losses:.0f} "
                f"({self.win_rate():.0%}) | Form: {form_str} | "
                f"Mature: {self.is_mature}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "team_id": self.team_id,
            "team_name": self.team_name,
            "metrics": self.metrics,
            "form": self.form,
            "is_mature": self.is_mature,
            "win_rate": round(self.win_rate(), 3),
            "ppg": round(self.points_per_game(), 2),
            "xg_diff": round(self.xg_differential(), 3),
            "venue_advantage": round(self.get_venue_advantage(), 3),
            "pattern_reliability": self.get_pattern_reliability().to_dict() if self.multi_rtm else None,
        }


@dataclass
class Leg:
    """
    Complete fixture representation with both teams and all metadata.
    
    ENHANCED v3: Now includes distortion tracking and pattern reliability.
    """
    # ── Core fields ────────────────────────────────────────────────────
    match_id: str = ""
    selection: str = ""
    odds: float = 0.0
    market: BetMarket = BetMarket.STRAIGHT_WIN
    league: str = ""
    kickoff: str = ""
    
    # ── Team data ──────────────────────────────────────────────────────
    home_profile: Optional[TeamProfile] = None
    away_profile: Optional[TeamProfile] = None
    h2h: Optional[H2HRecord] = None
    
    # ── Odds for all outcomes ─────────────────────────────────────────
    home_odds: Optional[float] = None
    away_odds: Optional[float] = None
    draw_odds: Optional[float] = None
    
    # ── Filter enforcement fields ─────────────────────────────────────
    league_id: Optional[int] = None
    league_tier: int = 3
    league_country: str = "unknown"
    competition_type: str = "league"
    stage: str = ""
    round: str = ""
    is_playoff: bool = False
    is_knockout: bool = False
    competition_format: CompetitionFormat = CompetitionFormat.REGULAR_SEASON
    
    # ── Probability fields ────────────────────────────────────────────
    model_prob: float = 0.0
    adjusted_prob: float = 0.0
    edge: float = 0.0
    pre_verdict: str = "PENDING"
    
    # ── Qualification tracking ────────────────────────────────────────
    qualification_status: QualificationStatus = QualificationStatus.PENDING
    dominance_score: float = 0.0
    decay_score: float = 0.0
    rtm_state: str = "?"
    market_bias: MarketBias = MarketBias.NEUTRAL
    
    # ── Module-specific storage ───────────────────────────────────────
    features: Dict[str, Any] = field(default_factory=dict)
    check_log: List[str] = field(default_factory=list)
    
    # ── NEW v3: Context for this fixture (for distortion tracking) ───
    context_flags: Optional[ContextFlags] = None
    fixture_date: Optional[str] = None
    venue: VenueType = VenueType.NEUTRAL
    
    def __post_init__(self):
        playoff_keywords = [
            'playoff', 'play-off', 'promotion', 'relegation',
            'semi', 'semifinal', 'final', 'quarter', 'knockout',
            'elimination', 'closing stage', 'opening stage', 'top 6'
        ]
        stage_lower = self.stage.lower()
        round_lower = self.round.lower()
        
        for keyword in playoff_keywords:
            if keyword in stage_lower or keyword in round_lower:
                self.is_playoff = True
                break
        
        if self.competition_type in ['cup', 'knockout']:
            self.is_knockout = True
        if self.is_playoff:
            self.is_knockout = True
    
    # ─── Team selection helpers ────────────────────────────────────────
    
    def get_favourite(self) -> Tuple[TeamProfile, TeamProfile]:
        if self.home_odds and self.away_odds:
            fav_is_home = self.home_odds <= self.away_odds
        else:
            fav_is_home = self.selection == getattr(self.home_profile, "team_name", "")
        
        if fav_is_home:
            return self.home_profile, self.away_profile
        return self.away_profile, self.home_profile
    
    def favourite_is_home(self) -> bool:
        if self.home_odds and self.away_odds:
            return self.home_odds <= self.away_odds
        return self.selection == getattr(self.home_profile, "team_name", "")
    
    def implied_probability(self) -> float:
        return 1.0 / self.odds if self.odds > 1.0 else 0.0
    
    # ─── Dimension-specific RTM methods ───────────────────────────────
    
    def get_dimension_rtm(self, team: str, dimension: Union[str, DimensionType]) -> Optional[DimensionRTM]:
        """
        Get dimension-specific RTM for a team.
        
        Args:
            team: "home" or "away"
            dimension: DimensionType or dimension name string
        """
        profile = self.home_profile if team == "home" else self.away_profile
        if not profile or not profile.multi_rtm:
            return None
        return profile.multi_rtm.get_dimension(dimension)
    
    def get_enhanced_rtm(self, team: str, dimension: str = "overall") -> Optional[EnhancedDimensionRTM]:
        """Get enhanced RTM for a team and dimension."""
        profile = self.home_profile if team == "home" else self.away_profile
        if not profile or not profile.multi_rtm:
            return None
        return profile.multi_rtm.get_enhanced_dimension(dimension)
    
    def get_venue_advantage(self, team: str = "home") -> float:
        """Get venue advantage for specified team."""
        profile = self.home_profile if team == "home" else self.away_profile
        if profile:
            return profile.get_venue_advantage()
        return 0.0
    
    def get_venue_advantage_gap(self) -> float:
        """Get difference in venue advantage between home and away."""
        return self.get_venue_advantage("home") - self.get_venue_advantage("away")
    
    def get_bounce_back_rate(self, team: str = "home", dimension: str = "overall") -> float:
        """Get bounce-back rate for specified team and dimension."""
        profile = self.home_profile if team == "home" else self.away_profile
        if profile:
            return profile.get_bounce_back_rate(dimension)
        return 0.33
    
    def get_clean_bounce_back_rate(self, team: str = "home", dimension: str = "overall") -> float:
        """Get clean bounce-back rate (distortions removed)."""
        profile = self.home_profile if team == "home" else self.away_profile
        if profile:
            return profile.get_clean_bounce_back_rate(dimension)
        return 0.33
    
    def get_bounce_back_gap(self, dimension: str = "overall") -> float:
        """Get difference in bounce-back rates between teams."""
        return self.get_bounce_back_rate("home", dimension) - self.get_bounce_back_rate("away", dimension)
    
    def get_tier_performance(self, tier: str, team: str = "home") -> float:
        """
        Get performance against specific tier.
        
        Args:
            tier: "top6", "mid", or "bottom6"
            team: "home" or "away"
        """
        profile = self.home_profile if team == "home" else self.away_profile
        if not profile or not profile.multi_rtm:
            return 0.5
        
        dim_map = {
            "top6": profile.multi_rtm.vs_top6,
            "mid": profile.multi_rtm.vs_mid,
            "bottom6": profile.multi_rtm.vs_bottom6,
        }
        dim_rtm = dim_map.get(tier)
        if dim_rtm:
            return dim_rtm.get_win_probability()
        return 0.5
    
    def get_pattern_reliability(self, team: str = "home") -> PatternReliabilityScore:
        """Get pattern reliability for a team."""
        profile = self.home_profile if team == "home" else self.away_profile
        if profile:
            return profile.get_pattern_reliability()
        return PatternReliabilityScore()
    
    def has_h2h_rtm(self) -> bool:
        """Check if H2H RTM data is available for this fixture."""
        if self.home_profile and self.home_profile.multi_rtm:
            return self.home_profile.multi_rtm.has_h2h_data()
        return False
    
    def get_h2h_bounce_back(self) -> float:
        """Get H2H-specific bounce-back rate."""
        if self.home_profile and self.home_profile.multi_rtm and self.home_profile.multi_rtm.h2h_overall:
            return self.home_profile.multi_rtm.h2h_overall.bounce_back_rate
        return 0.33
    
    def get_h2h_clean_bounce_back(self) -> float:
        """Get clean H2H bounce-back rate."""
        if self.home_profile and self.home_profile.multi_rtm and self.home_profile.multi_rtm.enhanced_h2h_overall:
            return self.home_profile.multi_rtm.enhanced_h2h_overall.get_clean_bounce_back()
        return self.get_h2h_bounce_back()
    
    # ─── Distortion detection methods (NEW v3) ─────────────────────────
    
    def has_distortion_risk(self) -> bool:
        """Check if current fixture has high distortion risk."""
        if not self.context_flags:
            return False
        return self.context_flags.context_impact > 0.4
    
    def get_distortion_adjustment(self, team: str = "home", transition: str = "L→W") -> float:
        """
        Get distortion adjustment for a specific transition.
        
        Args:
            team: "home" or "away"
            transition: e.g., "L→W" (loss to win)
        """
        profile = self.home_profile if team == "home" else self.away_profile
        if not profile or not profile.multi_rtm:
            return 0.0
        
        enhanced = profile.multi_rtm.get_enhanced_dimension("overall")
        if enhanced:
            from_result, to_result = transition.split("→")
            return enhanced.get_distortion_adjustment(from_result, to_result)
        return 0.0
    
    def is_pattern_reliable(self, team: str = "home", pattern: str = "bounce_back") -> bool:
        """Check if a pattern is reliable for this team."""
        profile = self.home_profile if team == "home" else self.away_profile
        if profile and profile.multi_rtm:
            return profile.multi_rtm.is_pattern_genuine(pattern)
        return False
    
    # ─── Dominance vs Decay validation ────────────────────────────────
    
    def passes_dominance_vs_decay(self) -> Tuple[bool, int, int]:
        fav, und = self.get_favourite()
        
        fav_dom = fav.to_dominance()
        und_dec = und.to_decay()
        
        fav_recent_wins = fav.recent_wins(5)
        und_recent_wins = und.recent_wins(5)
        form_gap = fav_recent_wins - und_recent_wins
        form_gap_normalized = min(1.0, form_gap / 5.0)
        
        checks = {
            "fav_win_rate": fav_dom.win_rate >= 0.55,
            "fav_ppg": fav_dom.points_per_game >= 1.8,
            "fav_recent_wins": fav_dom.recent_wins >= 3,
            "fav_form_gap": form_gap_normalized >= 0.30,
            "und_win_rate": und_dec.win_rate <= 0.40,
            "und_ppg": und_dec.points_per_game <= 1.2,
            "und_recent_losses": und_dec.recent_losses >= 3,
            "fav_not_decaying": fav.recent_losses(5) <= 2,
        }
        
        passed = sum(1 for v in checks.values() if v)
        return passed >= 6, passed, len(checks)
    
    # ─── RTM State management ─────────────────────────────────────────
    
    def update_rtm_state(self, result: str) -> None:
        if result.upper() in ("W", "D", "L"):
            self.rtm_state = result.upper()
            self.log_check(f"RTM state updated to {self.rtm_state}")
    
    def get_rtm_pattern(self, team: str = "home") -> Optional[RTMPattern]:
        profile = self.home_profile if team == "home" else self.away_profile
        if profile and profile.transition and self.rtm_state != "?":
            return profile.transition.to_rtm_pattern(self.rtm_state)
        return None
    
    # ─── Pipeline tracking ─────────────────────────────────────────────
    
    def mark_passed_m4(self) -> None:
        self.qualification_status = QualificationStatus.PASSED_M4
        self.log_check("M4: Passed asymmetric pre-filter")
    
    def mark_rejected_m4(self, reason: str) -> None:
        self.qualification_status = QualificationStatus.REJECTED_M4
        self.log_check(f"M4: REJECTED - {reason}")
    
    def mark_approved(self) -> None:
        self.qualification_status = QualificationStatus.APPROVED
        self.log_check("Final: APPROVED")
    
    def is_qualified(self) -> bool:
        return self.qualification_status in (QualificationStatus.APPROVED, QualificationStatus.APPROVED_AI)
    
    # ─── Validation helpers ───────────────────────────────────────────
    
    def is_valid(self) -> bool:
        if not self.match_id or not self.selection:
            return False
        if self.odds < 1.01:
            return False
        if self.home_profile is None or self.away_profile is None:
            return False
        if not self.home_profile.is_mature or not self.away_profile.is_mature:
            return False
        return True
    
    def is_eligible_for_pipeline(self) -> bool:
        if self.competition_type != "league":
            return False
        if self.is_playoff:
            return False
        if self.odds < 1.70:
            return False
        if not self.home_profile or not self.away_profile:
            return False
        if not self.home_profile.is_mature or not self.away_profile.is_mature:
            return False
        return True
    
    # ─── Summary and logging ───────────────────────────────────────────
    
    def summary(self) -> str:
        home_name = self.home_profile.team_name if self.home_profile else "?"
        away_name = self.away_profile.team_name if self.away_profile else "?"
        format_str = ""
        if self.is_playoff:
            format_str = " [PLAYOFF]"
        elif self.competition_type != "league":
            format_str = f" [{self.competition_type.upper()}]"
        
        status_str = f" [{self.qualification_status.value}]" if self.qualification_status != QualificationStatus.PENDING else ""
        
        return (f"{self.match_id}{format_str}{status_str} | {home_name} vs {away_name} | "
                f"Pick: {self.selection} @ {self.odds:.2f} | "
                f"Edge: {self.edge:+.3f} | Pre: {self.pre_verdict}")
    
    def dimension_summary(self) -> str:
        """Summary of dimension-specific metrics."""
        home_venue_adv = self.get_venue_advantage("home")
        away_venue_adv = self.get_venue_advantage("away")
        bounce_gap = self.get_bounce_back_gap()
        venue_gap = self.get_venue_advantage_gap()
        
        return (f"Dimensions: H_venue={home_venue_adv:+.2f}, A_venue={away_venue_adv:+.2f}, "
                f"venue_gap={venue_gap:+.2f}, bounce_gap={bounce_gap:+.2f}")
    
    def reliability_summary(self) -> str:
        """Summary of pattern reliability scores."""
        home_reliability = self.get_pattern_reliability("home")
        away_reliability = self.get_pattern_reliability("away")
        
        return (f"Reliability: H={home_reliability.reliability.value} ({home_reliability.normalized_score:.2f}), "
                f"A={away_reliability.reliability.value} ({away_reliability.normalized_score:.2f})")
    
    def log_check(self, message: str) -> None:
        self.check_log.append(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}")
    
    def log_feature(self, key: str, value: Any) -> None:
        self.features[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "selection": self.selection,
            "odds": self.odds,
            "league": self.league,
            "league_id": self.league_id,
            "league_tier": self.league_tier,
            "competition_type": self.competition_type,
            "is_playoff": self.is_playoff,
            "qualification_status": self.qualification_status.value,
            "dominance_score": self.dominance_score,
            "decay_score": self.decay_score,
            "model_prob": self.model_prob,
            "edge": self.edge,
            "pre_verdict": self.pre_verdict,
            "venue_advantage_home": self.get_venue_advantage("home"),
            "venue_advantage_away": self.get_venue_advantage("away"),
        }
    
    def to_leg_data(self) -> Dict[str, Any]:
        """Convert to leg_data format for M11 weighted decision."""
        return {
            "match_id": self.match_id,
            "selection": self.selection,
            "odds": self.odds,
            "model_prob": self.model_prob,
            "edge": self.edge,
            "venue": self.venue.value,
            "is_playoff": self.is_playoff,
            "is_knockout": self.is_knockout,
            "league_tier": self.league_tier,
            "home_win_rate": self.home_profile.win_rate() if self.home_profile else 0.0,
            "away_win_rate": self.away_profile.win_rate() if self.away_profile else 0.0,
            "home_venue_advantage": self.get_venue_advantage("home"),
            "away_venue_advantage": self.get_venue_advantage("away"),
            "pattern_reliable_home": self.is_pattern_reliable("home"),
            "pattern_reliable_away": self.is_pattern_reliable("away"),
            "distortion_risk": self.has_distortion_risk(),
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def create_empty_team_profile(team_id: str, team_name: str) -> TeamProfile:
    profile = TeamProfile(team_id=team_id, team_name=team_name)
    profile.update_metrics({
        "core.games": 0,
        "core.wins": 0,
        "core.draws": 0,
        "core.losses": 0,
        "core.xg": 0.0,
        "core.xga": 0.0,
        "core.goals": 0,
        "core.goals_against": 0,
        "core.implied_prob": 0.0,
        "position": 10.0,
        "points": 0.0,
        "draw_rate": 0.0,
        "pts_to_first": 0.0,
        "motivation.fatigue_days": 7.0,
        "motivation.relegation_pressure": 0.0,
        "motivation.desperation_phase": 0.0,
        "new_manager": 0.0,
    })
    profile.form = {"recent_results": []}
    profile.multi_rtm = MultiDimensionRTM(team_id=team_id, team_name=team_name)
    return profile


def create_basic_leg(
    match_id: str,
    home_team: str,
    away_team: str,
    selection: str,
    odds: float,
    league: str = "Unknown",
    league_id: Optional[int] = None,
    league_tier: int = 3,
    league_country: str = "unknown",
) -> Leg:
    home_profile = create_empty_team_profile("0", home_team)
    away_profile = create_empty_team_profile("0", away_team)
    
    return Leg(
        match_id=match_id,
        selection=selection,
        odds=odds,
        market=BetMarket.STRAIGHT_WIN,
        league=league,
        league_id=league_id,
        league_tier=league_tier,
        league_country=league_country,
        home_profile=home_profile,
        away_profile=away_profile,
        h2h=None,
    )


def classify_opponent_tier(position: int, league_size: int = 20) -> TierClassification:
    """Classify opponent tier based on league position."""
    top6_threshold = 6
    bottom6_threshold = league_size - 5  # e.g., 15 for 20-team league
    
    if position <= top6_threshold:
        return TierClassification.TOP6
    elif position >= bottom6_threshold:
        return TierClassification.BOTTOM6
    else:
        return TierClassification.MID


def create_distortion_factor(
    factor_type: DistortionType,
    severity: float,
    description: str,
    player_name: str = "",
    player_position: str = "",
    is_key_player: bool = False,
    games_missed: int = 0,
) -> DistortionFactor:
    """Factory function for creating distortion factors."""
    return DistortionFactor(
        factor_type=factor_type,
        severity=severity,
        description=description,
        player_name=player_name,
        player_position=player_position,
        is_key_player=is_key_player,
        games_missed=games_missed,
    )


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — EXPORTS
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "BetMarket",
    "TransitionPattern",
    "CompetitionFormat",
    "LeagueTier",
    "QualificationStatus",
    "MarketBias",
    "TierClassification",
    "VenueType",
    "DimensionType",
    "DistortionType",
    "PatternReliability",
    
    # Distortion models (NEW)
    "DistortionFactor",
    "ContextFlags",
    "EnhancedTransitionRecord",
    "DistortionStats",
    "PatternReliabilityScore",
    "EnhancedDimensionRTM",
    
    # Dimension-specific data classes
    "DimensionRTM",
    "DimensionBounceBack",
    "DimensionResilience",
    "MultiDimensionRTM",
    "MultiDimensionStreak",
    
    # Core data classes
    "StructuralDominance",
    "DecayMetrics",
    "RTMPattern",
    "TransitionMatrix",
    "H2HRecord",
    "TeamProfile",
    "Leg",
    
    # Factory functions
    "create_empty_team_profile",
    "create_basic_leg",
    "classify_opponent_tier",
    "create_distortion_factor",
    
    # Constants
    "MIN_CLEAN_TRANSITIONS_FOR_RELIABLE",
    "MIN_TOTAL_TRANSITIONS_FOR_ANALYSIS",
    "RELIABILITY_THRESHOLD_DIFFERENCE",
    "W", "D", "L",
    "ALL_OUTCOMES",
]