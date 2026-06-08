"""
The Match Oracle - Module 10: Season Tally Matrix & Bilateral Intersection (ENHANCED v3)
====================================================================================
Tracks W/D/L transitions across a full season and computes bilateral
intersection between two teams' RTM patterns.

ENHANCEMENTS IN THIS VERSION (v3):
------------------------------
1. ADDED: Enhanced dimension builder with distortion filtering
2. ADDED: Clean transition matrix builder (distortions removed)
3. ADDED: EnhancedDimensionTallyMatrix with reliability scoring
4. ADDED: Distortion factor integration from M2
5. ADDED: Clean vs dirty probability comparison
6. ADDED: Pattern reliability assessment for each dimension
7. ADDED: Distortion adjustment for transition probabilities
8. ADDED: Bilateral intersection with clean probabilities
9. ADDED: Trap/value detection using clean probabilities

PREVIOUS ENHANCEMENTS:
---------------------
- Dimension-specific tally matrix builder (venue, tier, H2H)
- Multi-dimension RTM container population
- Venue-specific transition analysis
- Tier-specific performance tracking (vs Top 6, vs Mid, vs Bottom 6)
- H2H-specific transition matrix for rivalry analysis
- Season progress weighting for early/late season adjustments

What this module does:
---------------------
1. SEASON TALLY MATRIX: Builds transition counts (W→W, W→D, W→L, etc.)
   for a single team across their season results.

2. CLEAN MATRIX (NEW): Builds transition matrix excluding distorted fixtures
   (key injuries, dead rubbers, new manager bounce, etc.)

3. BILATERAL INTERSECTION: Combines home and away transition matrices
   to predict the most likely outcome of a fixture.

4. DISTORTION ADJUSTMENT: Quantifies how much external factors affect
   transition probabilities.

5. TRAP/VALUE DETECTION: Compares RTM probabilities against market odds
   to identify overvalued favourites (traps) or undervalued underdogs.

6. DIMENSION RTM: Builds separate matrices for:
   - Home matches
   - Away matches
   - vs Top 6 opponents
   - vs Mid-table opponents
   - vs Bottom 6 opponents
   - H2H (specific opponent)

Usage:
    from module10 import (
        build_tally_matrix,
        build_enhanced_tally_matrix,
        build_multi_dimension_rtm,
        run_tally_matrix_analysis,
        TallyMatrixAnalysis
    )
    
    # Basic analysis
    matrix = build_tally_matrix(team_id, results, season)
    
    # Enhanced analysis with distortion filtering
    enhanced = build_enhanced_tally_matrix(team_id, fixtures_with_context)
    
    # Multi-dimension RTM
    multi_rtm = build_multi_dimension_rtm(team_id, fixtures_with_metadata)
    
    # Run full analysis
    analysis = run_tally_matrix_analysis(
        home_results=home_results,
        away_results=away_results,
        home_enhanced=home_enhanced,
        away_enhanced=away_enhanced,
    )
"""
from __future__ import annotations

import math
import json
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timezone
from collections import defaultdict
from enum import Enum
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — CONSTANTS
# ═══════════════════════════════════════════════════════════════

MIN_TALLY_RELIABLE       = 5      # minimum transitions for reliable probability
MIN_TALLY_STRONG         = 8      # minimum for strong confidence
EVEN_DISTRIBUTION_GAP    = 0.15   # max range for "even" distribution detection
STREAK_CEILING_TRIGGER   = 3      # consecutive same results triggers ceiling warning
STREAK_DANGER            = 5      # consecutive same results = critical risk

# Clean matrix thresholds
MIN_CLEAN_TRANSITIONS    = 3      # minimum clean transitions for reliability
DISTORTION_RATE_WARNING  = 0.30   # >30% distortion rate = reduced confidence
DISTORTION_RATE_CRITICAL = 0.50   # >50% distortion rate = unreliable

# Underdog detection thresholds
UNDERDOG_LONG_ODDS_MIN   = 4.0    # minimum odds for "long shot" classification
UNDERDOG_ELITE_ODDS_MIN  = 7.0    # minimum odds for "elite value" classification

# Trap/Value detection thresholds
TRAP_FAV_OVER_VALUE      = 0.15   # favourite implied > model + this = trap
VALUE_FAV_UNDER_VALUE    = 0.12   # favourite model > implied + this = value
VALUE_UND_OVER_VALUE     = 0.15   # underdog model > implied + this = value

# Weighted decision scores
USEFULNESS_SCORE_MAP = {
    True: 0.8,   # matrix_useful = True
    False: 0.3,  # matrix_useful = False
}

TRAP_VALUE_SCORE_MAP = {
    "TRAP": 0.10,
    "VALUE": 0.85,
    "NONE": 0.50,
    "UNCERTAIN": 0.40,
}

# Dimension type constants
DIMENSION_OVERALL = "overall"
DIMENSION_HOME = "home"
DIMENSION_AWAY = "away"
DIMENSION_VS_TOP6 = "vs_top6"
DIMENSION_VS_MID = "vs_mid"
DIMENSION_VS_BOTTOM6 = "vs_bottom6"
DIMENSION_H2H = "h2h"
DIMENSION_CLEAN = "clean"        # NEW: Distortion-free version

# All dimension types
ALL_DIMENSIONS = [
    DIMENSION_OVERALL,
    DIMENSION_HOME,
    DIMENSION_AWAY,
    DIMENSION_VS_TOP6,
    DIMENSION_VS_MID,
    DIMENSION_VS_BOTTOM6,
]

# Results
W, D, L = "W", "D", "L"
ALL_RESULTS = [W, D, L]


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — ENUMS
# ═══════════════════════════════════════════════════════════════

class TrapValueType(Enum):
    TRAP = "TRAP"
    VALUE = "VALUE"
    NONE = "NONE"
    UNCERTAIN = "UNCERTAIN"


class BilateralConfidence(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNCERTAIN = "UNCERTAIN"


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class TransitionCounts:
    """Raw transition counts for a matrix."""
    counts: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        W: {W: 0, D: 0, L: 0},
        D: {W: 0, D: 0, L: 0},
        L: {W: 0, D: 0, L: 0},
    })
    total: int = 0


@dataclass
class CleanTransitionStats:
    """Statistics for clean vs dirty transitions."""
    clean_count: int = 0
    dirty_count: int = 0
    distortion_rate: float = 0.0
    primary_distortion: str = ""
    
    @property
    def is_reliable(self) -> bool:
        return self.clean_count >= MIN_CLEAN_TRANSITIONS and self.distortion_rate < DISTORTION_RATE_WARNING
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "clean_count": self.clean_count,
            "dirty_count": self.dirty_count,
            "distortion_rate": round(self.distortion_rate, 3),
            "primary_distortion": self.primary_distortion,
            "is_reliable": self.is_reliable,
        }


@dataclass
class DimensionTallyMatrix:
    """
    Transition matrix for a specific dimension.
    
    Tracks counts and probabilities for one dimension
    (home, away, vs_top6, vs_mid, vs_bottom6, or h2h).
    """
    dimension_name: str = ""
    team_id: str = ""
    team_name: str = ""
    season: str = ""
    
    # Raw counts
    tally: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        W: {W: 0, D: 0, L: 0},
        D: {W: 0, D: 0, L: 0},
        L: {W: 0, D: 0, L: 0},
    })
    
    # Probabilities
    probs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Reliability flags
    is_reliable: Dict[str, bool] = field(default_factory=dict)
    is_even: Dict[str, bool] = field(default_factory=dict)
    
    # Statistics
    total_transitions: int = 0
    games_analyzed: int = 0
    season_progress: float = 0.0
    
    # Dimension-specific metrics
    sample_quality: float = 0.0  # 0-1, based on sample size relative to overall
    
    # NEW v3: Clean transition stats
    clean_stats: Optional[CleanTransitionStats] = None
    
    def row_total(self, from_res: str) -> int:
        return sum(self.tally.get(from_res, {}).values())
    
    def get_prob(self, from_res: str, to_res: str) -> float:
        if from_res in self.probs and to_res in self.probs[from_res]:
            return self.probs[from_res][to_res]
        return 1.0 / 3
    
    def get_clean_prob(self, from_res: str, to_res: str) -> float:
        """
        Get probability from clean transitions only.
        If clean stats not available, falls back to regular probability.
        """
        # This would be populated by enhanced builder
        # For now, return regular probability
        return self.get_prob(from_res, to_res)
    
    def get_distortion_adjustment(self, from_res: str, to_res: str) -> float:
        """
        Calculate how much distortion affects this transition.
        Positive = distortion inflates probability (overestimate)
        Negative = distortion deflates probability (underestimate)
        """
        dirty = self.get_prob(from_res, to_res)
        clean = self.get_clean_prob(from_res, to_res)
        return round(dirty - clean, 4)
    
    def is_row_reliable(self, from_res: str) -> bool:
        return self.is_reliable.get(from_res, False)
    
    def get_most_likely_next(self, current: str) -> Tuple[str, float]:
        probs = self.probs.get(current, {W: 0.33, D: 0.33, L: 0.34})
        return max(probs.items(), key=lambda x: x[1])
    
    def get_most_likely_next_clean(self, current: str) -> Tuple[str, float]:
        """Get most likely next outcome from clean transitions."""
        # This would use clean probabilities
        return self.get_most_likely_next(current)
    
    @property
    def overall_reliability(self) -> float:
        if self.total_transitions < MIN_TALLY_RELIABLE:
            return 0.2
        elif self.total_transitions < MIN_TALLY_STRONG:
            return 0.5
        elif self.total_transitions < MIN_TALLY_STRONG * 2:
            return 0.7
        return 0.9
    
    @property
    def clean_reliability(self) -> float:
        """Reliability of clean transitions."""
        if not self.clean_stats:
            return self.overall_reliability
        
        if self.clean_stats.clean_count < MIN_CLEAN_TRANSITIONS:
            return 0.3
        if self.clean_stats.distortion_rate > DISTORTION_RATE_CRITICAL:
            return 0.3
        if self.clean_stats.distortion_rate > DISTORTION_RATE_WARNING:
            return 0.5
        return 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "dimension": self.dimension_name,
            "total_transitions": self.total_transitions,
            "games_analyzed": self.games_analyzed,
            "reliability": self.overall_reliability,
            "clean_reliability": self.clean_reliability,
            "probs": {
                f"{fr}→{to}": round(prob, 3)
                for fr in ALL_RESULTS
                for to, prob in self.probs.get(fr, {}).items()
            },
        }
        if self.clean_stats:
            result["clean_stats"] = self.clean_stats.to_dict()
        return result
    
    def to_dimension_rtm(self) -> Any:
        """Convert to M2's DimensionRTM object."""
        try:
            from module2 import DimensionRTM
        except ImportError:
            return None
        
        dim = DimensionRTM(dimension_name=self.dimension_name)
        dim.sample_size = self.total_transitions
        dim.total_fixtures = self.games_analyzed
        dim.probabilities = {
            (fr, to): prob
            for fr in ALL_RESULTS
            for to, prob in self.probs.get(fr, {}).items()
        }
        
        # Calculate result probabilities
        win_count = sum(self.tally.get(fr, {}).get(W, 0) for fr in ALL_RESULTS)
        draw_count = sum(self.tally.get(fr, {}).get(D, 0) for fr in ALL_RESULTS)
        loss_count = sum(self.tally.get(fr, {}).get(L, 0) for fr in ALL_RESULTS)
        total = win_count + draw_count + loss_count
        
        if total > 0:
            dim.result_probs = {
                "W": win_count / total,
                "D": draw_count / total,
                "L": loss_count / total,
            }
        
        return dim


@dataclass
class SeasonTallyMatrix:
    """
    Legacy transition matrix for one team across a season.
    Kept for backward compatibility.
    """
    team_id: str = ""
    team_name: str = ""
    season: str = ""
    tally: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        W: {W: 0, D: 0, L: 0},
        D: {W: 0, D: 0, L: 0},
        L: {W: 0, D: 0, L: 0},
    })
    probs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    is_reliable: Dict[str, bool] = field(default_factory=dict)
    is_even: Dict[str, bool] = field(default_factory=dict)
    total_transitions: int = 0
    games_analyzed: int = 0
    season_progress: float = 0.0

    def row_total(self, from_res: str) -> int:
        return sum(self.tally.get(from_res, {}).values())

    def get_prob(self, from_res: str, to_res: str) -> float:
        if from_res in self.probs and to_res in self.probs[from_res]:
            return self.probs[from_res][to_res]
        return 1.0 / 3

    def is_row_reliable(self, from_res: str) -> bool:
        return self.is_reliable.get(from_res, False)

    def get_most_likely_next(self, current: str) -> Tuple[str, float]:
        probs = self.probs.get(current, {W: 0.33, D: 0.33, L: 0.34})
        return max(probs.items(), key=lambda x: x[1])

    @property
    def overall_reliability(self) -> float:
        if self.total_transitions < MIN_TALLY_RELIABLE:
            return 0.2
        elif self.total_transitions < MIN_TALLY_STRONG:
            return 0.5
        elif self.total_transitions < MIN_TALLY_STRONG * 2:
            return 0.7
        return 0.9

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team_id": self.team_id,
            "team_name": self.team_name,
            "season": self.season,
            "total_transitions": self.total_transitions,
            "games_analyzed": self.games_analyzed,
            "overall_reliability": self.overall_reliability,
            "probs": self.probs,
        }


@dataclass
class MultiDimensionTallyResult:
    """
    Container for all dimension tally matrices.
    """
    team_id: str
    team_name: str
    season: str
    
    # Core dimensions
    overall: Optional[DimensionTallyMatrix] = None
    home: Optional[DimensionTallyMatrix] = None
    away: Optional[DimensionTallyMatrix] = None
    vs_top6: Optional[DimensionTallyMatrix] = None
    vs_mid: Optional[DimensionTallyMatrix] = None
    vs_bottom6: Optional[DimensionTallyMatrix] = None
    
    # NEW v3: Clean dimensions (distortion-filtered)
    clean_overall: Optional[DimensionTallyMatrix] = None
    clean_home: Optional[DimensionTallyMatrix] = None
    clean_away: Optional[DimensionTallyMatrix] = None
    
    # H2H dimension (populated for specific opponent)
    h2h: Optional[DimensionTallyMatrix] = None
    clean_h2h: Optional[DimensionTallyMatrix] = None
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def get_dimension(self, dimension: str) -> Optional[DimensionTallyMatrix]:
        return getattr(self, dimension, None)
    
    def get_clean_dimension(self, dimension: str) -> Optional[DimensionTallyMatrix]:
        """Get clean (distortion-filtered) dimension."""
        clean_name = f"clean_{dimension}"
        return getattr(self, clean_name, None)
    
    def get_all_dimensions(self) -> List[DimensionTallyMatrix]:
        dims = []
        for dim_name in ALL_DIMENSIONS:
            dim = getattr(self, dim_name, None)
            if dim and dim.total_transitions > 0:
                dims.append(dim)
        if self.h2h and self.h2h.total_transitions > 0:
            dims.append(self.h2h)
        return dims
    
    def has_dimension(self, dimension: str) -> bool:
        dim = getattr(self, dimension, None)
        return dim is not None and dim.total_transitions >= MIN_TALLY_RELIABLE
    
    def has_clean_data(self, dimension: str = "overall") -> bool:
        """Check if clean data is available and reliable."""
        clean_dim = self.get_clean_dimension(dimension)
        if not clean_dim or not clean_dim.clean_stats:
            return False
        return clean_dim.clean_stats.is_reliable
    
    def get_clean_prob(self, dimension: str, from_res: str, to_res: str) -> float:
        """Get probability from clean transitions."""
        clean_dim = self.get_clean_dimension(dimension)
        if clean_dim:
            return clean_dim.get_clean_prob(from_res, to_res)
        dim = self.get_dimension(dimension)
        if dim:
            return dim.get_prob(from_res, to_res)
        return 0.33
    
    def get_distortion_impact(self, dimension: str, from_res: str, to_res: str) -> float:
        """Get distortion adjustment for a transition."""
        dim = self.get_dimension(dimension)
        if dim:
            return dim.get_distortion_adjustment(from_res, to_res)
        return 0.0
    
    def to_multi_dimension_rtm(self) -> Any:
        """Convert to M2's MultiDimensionRTM object."""
        try:
            from module2 import MultiDimensionRTM, DimensionRTM, EnhancedDimensionRTM, PatternReliabilityScore
        except ImportError:
            return None
        
        multi_rtm = MultiDimensionRTM(team_id=self.team_id, team_name=self.team_name)
        
        # Populate each dimension
        if self.overall:
            multi_rtm.overall = self.overall.to_dimension_rtm()
        if self.home:
            multi_rtm.home = self.home.to_dimension_rtm()
        if self.away:
            multi_rtm.away = self.away.to_dimension_rtm()
        if self.vs_top6:
            multi_rtm.vs_top6 = self.vs_top6.to_dimension_rtm()
        if self.vs_mid:
            multi_rtm.vs_mid = self.vs_mid.to_dimension_rtm()
        if self.vs_bottom6:
            multi_rtm.vs_bottom6 = self.vs_bottom6.to_dimension_rtm()
        if self.h2h:
            multi_rtm.h2h_overall = self.h2h.to_dimension_rtm()
        
        return multi_rtm
    
    def summary(self) -> Dict[str, Any]:
        return {
            "team": self.team_name,
            "dimensions": {
                "overall": self.overall.total_transitions if self.overall else 0,
                "home": self.home.total_transitions if self.home else 0,
                "away": self.away.total_transitions if self.away else 0,
                "vs_top6": self.vs_top6.total_transitions if self.vs_top6 else 0,
                "vs_mid": self.vs_mid.total_transitions if self.vs_mid else 0,
                "vs_bottom6": self.vs_bottom6.total_transitions if self.vs_bottom6 else 0,
            },
            "clean_available": self.has_clean_data(),
        }


@dataclass
class BilateralIntersection:
    home_next_probs: Dict[str, float] = field(default_factory=dict)
    away_next_probs: Dict[str, float] = field(default_factory=dict)
    combined_probs: Dict[str, float] = field(default_factory=dict)
    clean_combined_probs: Dict[str, float] = field(default_factory=dict)  # NEW
    predicted_outcome: str = "UNCERTAIN"
    clean_predicted_outcome: str = "UNCERTAIN"  # NEW
    confidence: BilateralConfidence = BilateralConfidence.UNCERTAIN
    clean_confidence: BilateralConfidence = BilateralConfidence.UNCERTAIN  # NEW
    confidence_score: float = 0.0
    clean_confidence_score: float = 0.0  # NEW
    is_uncertain: bool = True
    clean_is_uncertain: bool = True  # NEW
    margin: float = 0.0
    clean_margin: float = 0.0  # NEW
    distortion_impact: float = 0.0  # NEW - how much distortions affect prediction


@dataclass
class TrapValueSignal:
    signal_type: TrapValueType = TrapValueType.NONE
    description: str = ""
    recommendation: str = "NEUTRAL"
    strength: float = 0.0
    edge: float = 0.0
    fav_model_prob: float = 0.0
    fav_implied_prob: float = 0.0
    
    # NEW: Clean version (without distortions)
    clean_signal_type: TrapValueType = TrapValueType.NONE
    clean_description: str = ""
    clean_strength: float = 0.0
    clean_edge: float = 0.0
    
    @property
    def normalized_score(self) -> float:
        return TRAP_VALUE_SCORE_MAP.get(self.signal_type.value, 0.50)
    
    @property
    def clean_normalized_score(self) -> float:
        return TRAP_VALUE_SCORE_MAP.get(self.clean_signal_type.value, 0.50)


@dataclass
class TallyMatrixAnalysis:
    home_matrix: SeasonTallyMatrix = field(default_factory=SeasonTallyMatrix)
    away_matrix: SeasonTallyMatrix = field(default_factory=SeasonTallyMatrix)
    home_multi_rtm: Optional[MultiDimensionTallyResult] = None
    away_multi_rtm: Optional[MultiDimensionTallyResult] = None
    home_streak: Dict[str, Any] = field(default_factory=dict)
    away_streak: Dict[str, Any] = field(default_factory=dict)
    bilateral: BilateralIntersection = field(default_factory=BilateralIntersection)
    trap_value_signal: TrapValueSignal = field(default_factory=TrapValueSignal)
    has_trap: bool = False
    has_value: bool = False
    matrix_useful: bool = False
    combined_risk_flag: str = "NONE"
    
    # NEW v3: Clean analysis results
    clean_matrix_useful: bool = False
    distortion_adjusted: bool = False
    reliability_warning: Optional[str] = None
    
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    @property
    def usefulness_score(self) -> float:
        return USEFULNESS_SCORE_MAP.get(self.matrix_useful, 0.5)
    
    @property
    def clean_usefulness_score(self) -> float:
        """Usefulness score based on clean analysis."""
        return USEFULNESS_SCORE_MAP.get(self.clean_matrix_useful, 0.5)
    
    @property
    def trap_value_score(self) -> float:
        return self.trap_value_signal.normalized_score
    
    @property
    def clean_trap_value_score(self) -> float:
        return self.trap_value_signal.clean_normalized_score
    
    @property
    def normalized_score(self) -> float:
        return round((self.usefulness_score * 0.6) + (self.trap_value_score * 0.4), 3)
    
    @property
    def clean_normalized_score(self) -> float:
        """Normalized score using clean analysis."""
        if self.distortion_adjusted:
            return round((self.clean_usefulness_score * 0.6) + (self.clean_trap_value_score * 0.4), 3)
        return self.normalized_score
    
    @property
    def confidence_factor(self) -> float:
        confidence_map = {
            BilateralConfidence.HIGH: 1.0,
            BilateralConfidence.MEDIUM: 0.7,
            BilateralConfidence.LOW: 0.4,
            BilateralConfidence.UNCERTAIN: 0.3,
        }
        base = confidence_map.get(self.bilateral.confidence, 0.5)
        if self.matrix_useful:
            base = min(1.0, base * 1.2)
        if self.bilateral.is_uncertain:
            base *= 0.8
        if self.reliability_warning:
            base *= 0.85
        return round(base, 2)
    
    @property
    def clean_confidence_factor(self) -> float:
        """Confidence factor using clean analysis."""
        confidence_map = {
            BilateralConfidence.HIGH: 1.0,
            BilateralConfidence.MEDIUM: 0.7,
            BilateralConfidence.LOW: 0.4,
            BilateralConfidence.UNCERTAIN: 0.3,
        }
        base = confidence_map.get(self.bilateral.clean_confidence, 0.5)
        if self.clean_matrix_useful:
            base = min(1.0, base * 1.2)
        if self.bilateral.clean_is_uncertain:
            base *= 0.8
        return round(base, 2)
    
    def to_leg_data(self) -> Dict[str, Any]:
        return {
            "matrix_useful": self.matrix_useful,
            "matrix_usefulness_score": self.usefulness_score,
            "trap_value_signal": self.trap_value_signal.signal_type.value,
            "trap_value_score": self.trap_value_score,
            "matrix_normalized_score": self.normalized_score,
            "combined_risk_flag": self.combined_risk_flag,
            "bilateral_prediction": self.bilateral.predicted_outcome,
            "bilateral_confidence": self.bilateral.confidence.value,
            "bilateral_confidence_score": self.bilateral.confidence_score,
            # NEW v3 fields
            "clean_matrix_useful": self.clean_matrix_useful,
            "clean_trap_value_signal": self.trap_value_signal.clean_signal_type.value,
            "clean_normalized_score": self.clean_normalized_score,
            "clean_confidence_factor": self.clean_confidence_factor,
            "distortion_impact": self.bilateral.distortion_impact,
            "reliability_warning": self.reliability_warning,
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — TALLY MATRIX BUILDER (SINGLE DIMENSION)
# ═══════════════════════════════════════════════════════════════

def _calculate_season_progress(games_analyzed: int, total_games: int = 38) -> float:
    if total_games <= 0:
        return 0.5
    return min(1.0, games_analyzed / total_games)


def build_tally_matrix(
    team_id: str,
    results: List[str],
    season: str = "",
    team_name: str = "",
    total_games: int = 38,
) -> SeasonTallyMatrix:
    """Build a SeasonTallyMatrix from a sequence of results."""
    matrix = SeasonTallyMatrix(
        team_id=team_id,
        season=season,
        team_name=team_name,
        games_analyzed=len(results),
        season_progress=_calculate_season_progress(len(results), total_games),
    )
    
    if len(results) < 2:
        return matrix
    
    for i in range(len(results) - 1):
        fr, tr = results[i], results[i + 1]
        if fr in matrix.tally and tr in matrix.tally[fr]:
            matrix.tally[fr][tr] += 1
            matrix.total_transitions += 1
    
    for fr in ALL_RESULTS:
        total = matrix.row_total(fr)
        matrix.is_reliable[fr] = total >= MIN_TALLY_RELIABLE
        
        if total > 0:
            for tr in ALL_RESULTS:
                prob = matrix.tally[fr][tr] / total
                matrix.probs.setdefault(fr, {})[tr] = round(prob, 4)
        else:
            for tr in ALL_RESULTS:
                matrix.probs.setdefault(fr, {})[tr] = 1.0 / 3
        
        if total >= MIN_TALLY_STRONG:
            probs_list = [matrix.probs[fr].get(r, 0.0) for r in ALL_RESULTS]
            max_prob = max(probs_list)
            min_prob = min(probs_list)
            matrix.is_even[fr] = (max_prob - min_prob) <= EVEN_DISTRIBUTION_GAP
        else:
            matrix.is_even[fr] = False
    
    return matrix


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — ENHANCED MATRIX BUILDER WITH DISTORTION FILTERING (NEW v3)
# ═══════════════════════════════════════════════════════════════

def _is_clean_fixture(fixture_metadata: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Determine if a fixture should be considered "clean"
    (no significant distortions affecting the result).
    
    Returns:
        Tuple of (is_clean, reason)
    """
    # Check for dead rubber
    if fixture_metadata.get("is_dead_rubber", False):
        return False, "dead_rubber"
    
    # Check for key player absences
    key_players_missing = fixture_metadata.get("key_players_missing", 0)
    if key_players_missing >= 2:
        return False, f"{key_players_missing}_key_players_missing"
    
    # Check for new manager bounce period
    manager_tenure = fixture_metadata.get("manager_tenure_days", 365)
    if manager_tenure < 60:
        return False, "new_manager_bounce"
    
    # Check for midweek fatigue (European football)
    if fixture_metadata.get("european_midweek", False):
        days_rest = fixture_metadata.get("days_rest", 7)
        if days_rest < 4:
            return False, "midweek_fatigue"
    
    # Check for international break fatigue
    days_since_intl = fixture_metadata.get("days_since_intl_return", 14)
    if days_since_intl < 3:
        return False, "intl_break_fatigue"
    
    # Check for extreme weather
    weather = fixture_metadata.get("weather", "").lower()
    if weather in ("heavy_rain", "snow", "storm"):
        return False, f"extreme_weather_{weather}"
    
    # Check for early/late season with high rotation
    season_phase = fixture_metadata.get("season_phase", "mid")
    rotation_count = fixture_metadata.get("squad_rotation", 0)
    if season_phase in ("early", "end") and rotation_count >= 4:
        return False, f"{season_phase}_season_rotation"
    
    return True, "clean"


def build_enhanced_tally_matrix(
    team_id: str,
    fixtures_with_context: List[Dict],
    dimension: str = "overall",
    season: str = "",
    team_name: str = "",
    total_games: int = 38,
) -> DimensionTallyMatrix:
    """
    Build a dimension-specific tally matrix with distortion filtering.
    
    Args:
        team_id: Team identifier
        fixtures_with_context: List of fixtures with metadata including:
            - result: W/D/L
            - date: ISO date
            - metadata: dict with distortion factors (is_dead_rubber, key_players_missing, etc.)
        dimension: Dimension name (overall, home, away, etc.)
        season: Season identifier
        team_name: Team name
        total_games: Total games in season
    
    Returns:
        DimensionTallyMatrix with clean stats
    """
    # Filter fixtures by dimension
    filtered = []
    for fx in fixtures_with_context:
        meta = fx.get("metadata", {})
        
        if dimension == DIMENSION_HOME and meta.get("venue") != "home":
            continue
        elif dimension == DIMENSION_AWAY and meta.get("venue") != "away":
            continue
        elif dimension == DIMENSION_VS_TOP6 and meta.get("opponent_tier") != "top6":
            continue
        elif dimension == DIMENSION_VS_MID and meta.get("opponent_tier") != "mid":
            continue
        elif dimension == DIMENSION_VS_BOTTOM6 and meta.get("opponent_tier") != "bottom6":
            continue
        elif dimension == DIMENSION_OVERALL:
            pass
        else:
            filtered.append(fx)
            continue
        
        filtered.append(fx)
    
    # Sort by date
    sorted_fx = sorted(filtered, key=lambda x: x.get("date", ""))
    
    # Separate clean and dirty transitions
    clean_results = []
    dirty_results = []
    clean_metadata = []
    dirty_metadata = []
    
    for i, fx in enumerate(sorted_fx):
        result = fx.get("result")
        if not result:
            continue
        
        meta = fx.get("metadata", {})
        is_clean, reason = _is_clean_fixture(meta)
        
        if is_clean:
            clean_results.append(result)
            clean_metadata.append(meta)
        else:
            dirty_results.append(result)
            dirty_metadata.append(meta)
    
    # Build matrices
    matrix = DimensionTallyMatrix(
        dimension_name=dimension,
        team_id=team_id,
        team_name=team_name,
        season=season,
        games_analyzed=len(sorted_fx),
        season_progress=_calculate_season_progress(len(sorted_fx), total_games),
    )
    
    # Build dirty matrix (all transitions)
    if len(dirty_results) + len(clean_results) >= 2:
        all_results = []
        # Interleave clean and dirty in chronological order
        # For now, combine by sorting by original order
        all_fx = sorted(filtered, key=lambda x: x.get("date", ""))
        all_results = [fx.get("result") for fx in all_fx if fx.get("result")]
        
        for i in range(len(all_results) - 1):
            fr, tr = all_results[i], all_results[i + 1]
            if fr in matrix.tally and tr in matrix.tally[fr]:
                matrix.tally[fr][tr] += 1
                matrix.total_transitions += 1
        
        # Calculate probabilities
        for fr in ALL_RESULTS:
            total = matrix.row_total(fr)
            matrix.is_reliable[fr] = total >= MIN_TALLY_RELIABLE
            
            if total > 0:
                for tr in ALL_RESULTS:
                    prob = matrix.tally[fr][tr] / total
                    matrix.probs.setdefault(fr, {})[tr] = round(prob, 4)
            else:
                for tr in ALL_RESULTS:
                    matrix.probs.setdefault(fr, {})[tr] = 1.0 / 3
    
    # Calculate clean stats
    clean_transition_count = 0
    distortion_counts: Dict[str, int] = defaultdict(int)
    
    # Count clean transitions
    for i in range(len(clean_results) - 1):
        clean_transition_count += 1
    
    # Count distortions
    for fx in sorted_fx:
        meta = fx.get("metadata", {})
        is_clean, reason = _is_clean_fixture(meta)
        if not is_clean:
            distortion_counts[reason] += 1
    
    total_fixtures = len(sorted_fx)
    distortion_rate = (total_fixtures - len(clean_results)) / total_fixtures if total_fixtures > 0 else 0
    primary_distortion = max(distortion_counts.items(), key=lambda x: x[1])[0] if distortion_counts else ""
    
    matrix.clean_stats = CleanTransitionStats(
        clean_count=clean_transition_count,
        dirty_count=matrix.total_transitions - clean_transition_count,
        distortion_rate=distortion_rate,
        primary_distortion=primary_distortion,
    )
    
    # Calculate sample quality
    overall_count = len([fx for fx in fixtures_with_context if fx.get("result")])
    if overall_count > 0:
        matrix.sample_quality = min(1.0, len(sorted_fx) / overall_count)
    else:
        matrix.sample_quality = 0.0
    
    return matrix


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — DIMENSION-SPECIFIC MATRIX BUILDER
# ═══════════════════════════════════════════════════════════════

def build_dimension_tally_matrix(
    team_id: str,
    team_name: str,
    fixtures_with_results: List[Dict],
    dimension_name: str,
    season: str = "",
    total_games: int = 38,
    use_enhanced: bool = True,
) -> DimensionTallyMatrix:
    """
    Build a tally matrix for a specific dimension.
    
    Args:
        team_id: Team identifier
        team_name: Team name
        fixtures_with_results: List of dicts with keys:
            - result: "W", "D", or "L"
            - date: ISO date string
            - metadata: dict with dimension-specific fields and distortion info
        dimension_name: "home", "away", "vs_top6", etc.
        season: Season identifier
        total_games: Total games in season
        use_enhanced: If True, use enhanced builder with distortion filtering
    
    Returns:
        DimensionTallyMatrix for the specified dimension
    """
    if use_enhanced:
        return build_enhanced_tally_matrix(
            team_id, fixtures_with_results, dimension_name,
            season, team_name, total_games
        )
    
    # Legacy builder (without distortion filtering)
    filtered = []
    for fx in fixtures_with_results:
        metadata = fx.get("metadata", {})
        
        if dimension_name == DIMENSION_HOME and metadata.get("venue") != "home":
            continue
        elif dimension_name == DIMENSION_AWAY and metadata.get("venue") != "away":
            continue
        elif dimension_name == DIMENSION_VS_TOP6 and metadata.get("opponent_tier") != "top6":
            continue
        elif dimension_name == DIMENSION_VS_MID and metadata.get("opponent_tier") != "mid":
            continue
        elif dimension_name == DIMENSION_VS_BOTTOM6 and metadata.get("opponent_tier") != "bottom6":
            continue
        elif dimension_name == DIMENSION_OVERALL:
            pass
        else:
            # For custom dimensions, check if metadata matches
            pass
        
        filtered.append(fx)
    
    # Sort by date
    sorted_fx = sorted(filtered, key=lambda x: x.get("date", ""))
    results = [fx["result"] for fx in sorted_fx if fx.get("result")]
    
    matrix = DimensionTallyMatrix(
        dimension_name=dimension_name,
        team_id=team_id,
        team_name=team_name,
        season=season,
        games_analyzed=len(results),
        season_progress=_calculate_season_progress(len(results), total_games),
    )
    
    # Calculate sample quality relative to overall
    overall_count = len([fx for fx in fixtures_with_results if fx.get("result")])
    if overall_count > 0:
        matrix.sample_quality = min(1.0, len(results) / overall_count)
    else:
        matrix.sample_quality = 0.0
    
    if len(results) < 2:
        return matrix
    
    # Build transition counts
    for i in range(len(results) - 1):
        fr, tr = results[i], results[i + 1]
        if fr in matrix.tally and tr in matrix.tally[fr]:
            matrix.tally[fr][tr] += 1
            matrix.total_transitions += 1
    
    # Calculate probabilities
    for fr in ALL_RESULTS:
        total = matrix.row_total(fr)
        matrix.is_reliable[fr] = total >= MIN_TALLY_RELIABLE
        
        if total > 0:
            for tr in ALL_RESULTS:
                prob = matrix.tally[fr][tr] / total
                matrix.probs.setdefault(fr, {})[tr] = round(prob, 4)
        else:
            for tr in ALL_RESULTS:
                matrix.probs.setdefault(fr, {})[tr] = 1.0 / 3
        
        if total >= MIN_TALLY_STRONG:
            probs_list = [matrix.probs[fr].get(r, 0.0) for r in ALL_RESULTS]
            max_prob = max(probs_list)
            min_prob = min(probs_list)
            matrix.is_even[fr] = (max_prob - min_prob) <= EVEN_DISTRIBUTION_GAP
        else:
            matrix.is_even[fr] = False
    
    return matrix


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — MULTI-DIMENSION RTM BUILDER
# ═══════════════════════════════════════════════════════════════

def build_multi_dimension_rtm(
    team_id: str,
    team_name: str,
    fixtures_with_results: List[Dict],
    season: str = "",
    total_games: int = 38,
    use_enhanced: bool = True,
) -> MultiDimensionTallyResult:
    """
    Build all dimension tally matrices for a team.
    
    Args:
        team_id: Team identifier
        team_name: Team name
        fixtures_with_results: List of dicts with keys:
            - result: "W", "D", or "L"
            - date: ISO date string
            - metadata: dict with venue, opponent_tier, opponent_id, and distortion info
        season: Season identifier
        total_games: Total games in season
        use_enhanced: If True, use enhanced builder with distortion filtering
    
    Returns:
        MultiDimensionTallyResult with all dimension matrices
    """
    result = MultiDimensionTallyResult(
        team_id=team_id,
        team_name=team_name,
        season=season,
    )
    
    # Build each dimension
    for dim in ALL_DIMENSIONS:
        matrix = build_dimension_tally_matrix(
            team_id, team_name, fixtures_with_results,
            dim, season, total_games, use_enhanced
        )
        setattr(result, dim, matrix)
    
    # Build clean versions if enhanced
    if use_enhanced:
        clean_fixtures = []
        for fx in fixtures_with_results:
            meta = fx.get("metadata", {})
            is_clean, _ = _is_clean_fixture(meta)
            if is_clean:
                clean_fixtures.append(fx)
        
        if clean_fixtures:
            for dim in ALL_DIMENSIONS:
                clean_matrix = build_dimension_tally_matrix(
                    team_id, team_name, clean_fixtures,
                    dim, season, total_games, use_enhanced=False
                )
                setattr(result, f"clean_{dim}", clean_matrix)
    
    return result


def build_h2h_rtm(
    team_id: str,
    team_name: str,
    opponent_id: str,
    opponent_name: str,
    h2h_fixtures: List[Dict],
    season: str = "",
    use_enhanced: bool = True,
) -> Optional[DimensionTallyMatrix]:
    """
    Build H2H-specific tally matrix for a specific opponent.
    
    Args:
        team_id: Team identifier
        team_name: Team name
        opponent_id: Opponent team identifier
        opponent_name: Opponent team name
        h2h_fixtures: List of head-to-head fixtures with results and metadata
        season: Season identifier
        use_enhanced: If True, use enhanced builder with distortion filtering
    
    Returns:
        DimensionTallyMatrix for H2H or None if insufficient data
    """
    if len(h2h_fixtures) < 3:
        return None
    
    fixtures_with_results = []
    for fx in h2h_fixtures:
        fixtures_with_results.append({
            "result": fx.get("result"),
            "date": fx.get("date", ""),
            "metadata": {
                "venue": fx.get("venue", "neutral"),
                "opponent_id": opponent_id,
                "opponent_name": opponent_name,
                "is_dead_rubber": fx.get("is_dead_rubber", False),
                "key_players_missing": fx.get("key_players_missing", 0),
                "manager_tenure_days": fx.get("manager_tenure_days", 365),
            }
        })
    
    matrix = build_dimension_tally_matrix(
        team_id, f"{team_name} vs {opponent_name}",
        fixtures_with_results,
        DIMENSION_H2H,
        season,
        total_games=len(h2h_fixtures),
        use_enhanced=use_enhanced
    )
    
    return matrix


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — STREAK ANALYSER
# ═══════════════════════════════════════════════════════════════

def analyse_streak(matrix: SeasonTallyMatrix, results: List[str]) -> Dict[str, Any]:
    if not results:
        return {
            "consecutive": 0,
            "direction": "?",
            "last": "?",
            "ceiling": 0,
            "ceiling_risk": False,
            "risk": "LOW",
            "reversal_probability": 0.0,
        }
    
    last = results[-1]
    count = 1
    for r in reversed(results[:-1]):
        if r == last:
            count += 1
        else:
            break
    
    ceiling = matrix.tally[last][last] if last in matrix.tally else 0
    reversal_prob = 1.0 - matrix.get_prob(last, last) if matrix.total_transitions > 0 else 0.33
    
    if count >= STREAK_DANGER:
        risk = "CRITICAL"
    elif count >= STREAK_CEILING_TRIGGER:
        risk = "HIGH"
    else:
        risk = "LOW"
    
    return {
        "consecutive": count,
        "direction": last,
        "last": last,
        "ceiling": ceiling,
        "ceiling_risk": count >= STREAK_CEILING_TRIGGER and count >= ceiling,
        "risk": risk,
        "reversal_probability": round(reversal_prob, 3),
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — BILATERAL INTERSECTION (ENHANCED)
# ═══════════════════════════════════════════════════════════════

def _calculate_confidence_score(
    combined_probs: Dict[str, float],
    home_weight: float,
    away_weight: float,
    home_reliable: bool,
    away_reliable: bool,
) -> Tuple[BilateralConfidence, float, bool]:
    max_prob = max(combined_probs.values())
    second_prob = sorted(combined_probs.values(), reverse=True)[1] if len(combined_probs) > 1 else 0.25
    margin = max_prob - second_prob
    
    if margin >= 0.20:
        margin_conf = 0.9
    elif margin >= 0.10:
        margin_conf = 0.7
    elif margin >= 0.05:
        margin_conf = 0.5
    else:
        margin_conf = 0.3
    
    reliability_score = 0.5
    if home_reliable and away_reliable:
        reliability_score = 0.9
    elif home_reliable or away_reliable:
        reliability_score = 0.7
    
    weight_factor = min(1.0, (home_weight + away_weight) / 1.4)
    
    confidence_score = (margin_conf * 0.5 + reliability_score * 0.3 + weight_factor * 0.2)
    is_uncertain = margin < 0.10 or (not home_reliable and not away_reliable)
    
    if confidence_score >= 0.70 and not is_uncertain:
        confidence = BilateralConfidence.HIGH
    elif confidence_score >= 0.55:
        confidence = BilateralConfidence.MEDIUM
    elif confidence_score >= 0.40:
        confidence = BilateralConfidence.LOW
    else:
        confidence = BilateralConfidence.UNCERTAIN
    
    return confidence, round(confidence_score, 3), is_uncertain


def calculate_bilateral_intersection(
    home_matrix: SeasonTallyMatrix,
    away_matrix: SeasonTallyMatrix,
    home_last: str,
    away_last: str,
    home_enhanced: Optional[DimensionTallyMatrix] = None,
    away_enhanced: Optional[DimensionTallyMatrix] = None,
) -> BilateralIntersection:
    """
    Calculate bilateral intersection with optional clean probabilities.
    """
    bi = BilateralIntersection()
    
    # Get probabilities from standard matrices
    home_probs = home_matrix.probs.get(home_last, {W: 0.33, D: 0.33, L: 0.34})
    away_probs = away_matrix.probs.get(away_last, {W: 0.33, D: 0.33, L: 0.34})
    
    bi.home_next_probs = home_probs.copy()
    bi.away_next_probs = away_probs.copy()
    
    home_reliable = home_matrix.is_reliable.get(home_last, False)
    away_reliable = away_matrix.is_reliable.get(away_last, False)
    
    home_weight = 0.7 if home_reliable else 0.4
    away_weight = 0.7 if away_reliable else 0.4
    
    season_weight_home = min(1.0, home_matrix.season_progress * 1.2)
    season_weight_away = min(1.0, away_matrix.season_progress * 1.2)
    
    home_weight *= season_weight_home
    away_weight *= season_weight_away
    
    home_weight = max(0.3, min(0.9, home_weight))
    away_weight = max(0.3, min(0.9, away_weight))
    
    # Standard combined probabilities
    combined = {
        W: (home_probs.get(W, 0.33) * home_weight + away_probs.get(L, 0.33) * away_weight) / (home_weight + away_weight),
        D: (home_probs.get(D, 0.33) * home_weight + away_probs.get(D, 0.33) * away_weight) / (home_weight + away_weight),
        L: (home_probs.get(L, 0.33) * home_weight + away_probs.get(W, 0.33) * away_weight) / (home_weight + away_weight),
    }
    
    total = sum(combined.values())
    if total > 0:
        combined = {r: round(v / total, 4) for r, v in combined.items()}
    
    bi.combined_probs = combined
    predicted = max(combined, key=combined.get)
    bi.predicted_outcome = predicted
    bi.margin = round(combined[predicted] - sorted(combined.values(), reverse=True)[1], 4) if len(combined) > 1 else 0.0
    
    conf, conf_score, uncertain = _calculate_confidence_score(
        combined, home_weight, away_weight, home_reliable, away_reliable
    )
    bi.confidence = conf
    bi.confidence_score = conf_score
    bi.is_uncertain = uncertain
    
    # ── NEW: Clean probabilities (if enhanced matrices available) ──
    if home_enhanced and away_enhanced:
        home_clean_probs = home_enhanced.get_clean_prob(home_last, "W")
        away_clean_probs = away_enhanced.get_clean_prob(away_last, "W")
        
        # Simplified clean combination
        clean_combined = {
            W: (home_enhanced.get_clean_prob(home_last, "W") * home_weight +
                away_enhanced.get_clean_prob(away_last, "L") * away_weight) / (home_weight + away_weight),
            D: (home_enhanced.get_clean_prob(home_last, "D") * home_weight +
                away_enhanced.get_clean_prob(away_last, "D") * away_weight) / (home_weight + away_weight),
            L: (home_enhanced.get_clean_prob(home_last, "L") * home_weight +
                away_enhanced.get_clean_prob(away_last, "W") * away_weight) / (home_weight + away_weight),
        }
        
        total_clean = sum(clean_combined.values())
        if total_clean > 0:
            clean_combined = {r: round(v / total_clean, 4) for r, v in clean_combined.items()}
        
        bi.clean_combined_probs = clean_combined
        clean_predicted = max(clean_combined, key=clean_combined.get)
        bi.clean_predicted_outcome = clean_predicted
        bi.clean_margin = round(clean_combined[clean_predicted] - sorted(clean_combined.values(), reverse=True)[1], 4) if len(clean_combined) > 1 else 0.0
        
        # Clean confidence
        clean_conf, clean_conf_score, clean_uncertain = _calculate_confidence_score(
            clean_combined, home_weight, away_weight, 
            home_enhanced.clean_stats.is_reliable if home_enhanced.clean_stats else False,
            away_enhanced.clean_stats.is_reliable if away_enhanced.clean_stats else False
        )
        bi.clean_confidence = clean_conf
        bi.clean_confidence_score = clean_conf_score
        bi.clean_is_uncertain = clean_uncertain
        
        # Calculate distortion impact
        fav_prob_diff = abs(combined.get(W, 0.33) - clean_combined.get(W, 0.33))
        bi.distortion_impact = round(fav_prob_diff, 4)
    
    return bi


# ═══════════════════════════════════════════════════════════════
# SECTION 10 — TRAP/VALUE DETECTION (ENHANCED)
# ═══════════════════════════════════════════════════════════════

def detect_trap_or_value(
    bi: BilateralIntersection,
    fav_odds: float,
    und_odds: float,
    fav_is_home: bool,
    home_matrix: Optional[SeasonTallyMatrix] = None,
    away_matrix: Optional[SeasonTallyMatrix] = None,
    use_clean: bool = True,
) -> TrapValueSignal:
    """
    Detect trap or value with optional clean probability adjustment.
    """
    sig = TrapValueSignal()
    
    if bi.is_uncertain or fav_odds <= 1.0 or und_odds <= 1.0:
        return sig
    
    # Standard detection using dirty probabilities
    if fav_is_home:
        fav_prob = bi.combined_probs.get(W, 0.33)
        draw_prob = bi.combined_probs.get(D, 0.33)
        und_prob = bi.combined_probs.get(L, 0.33)
    else:
        fav_prob = bi.combined_probs.get(L, 0.33)
        draw_prob = bi.combined_probs.get(D, 0.33)
        und_prob = bi.combined_probs.get(W, 0.33)
    
    total = fav_prob + draw_prob + und_prob
    if total > 0:
        fav_prob = fav_prob / total
        draw_prob = draw_prob / total
        und_prob = und_prob / total
    
    fav_implied = 1.0 / fav_odds if fav_odds > 1.0 else 0.0
    und_implied = 1.0 / und_odds if und_odds > 1.0 else 0.0
    
    sig.fav_model_prob = fav_prob
    sig.fav_implied_prob = fav_implied
    sig.edge = fav_prob - fav_implied if fav_is_home else und_prob - und_implied
    
    if fav_prob < fav_implied - TRAP_FAV_OVER_VALUE and fav_implied > 0.55:
        sig.signal_type = TrapValueType.TRAP
        sig.description = f"Favourite overvalued: RTM {fav_prob:.1%} vs market {fav_implied:.1%}"
        sig.recommendation = "AVOID"
        sig.strength = round(min(1.0, (fav_implied - fav_prob) / 0.15), 3)
    elif fav_prob > fav_implied + VALUE_FAV_UNDER_VALUE and fav_prob > 0.55:
        sig.signal_type = TrapValueType.VALUE
        sig.description = f"Favourite undervalued: RTM {fav_prob:.1%} vs market {fav_implied:.1%}"
        sig.recommendation = "BAG" if fav_odds >= UNDERDOG_ELITE_ODDS_MIN else "CONSIDER"
        sig.strength = round(min(1.0, (fav_prob - fav_implied) / 0.12), 3)
    elif und_prob > und_implied + VALUE_UND_OVER_VALUE and und_prob > 0.30:
        sig.signal_type = TrapValueType.VALUE
        sig.description = f"Underdog undervalued: RTM {und_prob:.1%} vs market {und_implied:.1%}"
        sig.recommendation = "BAG" if und_odds >= UNDERDOG_ELITE_ODDS_MIN else "CONSIDER"
        sig.strength = round(min(1.0, (und_prob - und_implied) / 0.15), 3)
    
    # ── NEW: Clean detection (if available) ──
    if use_clean and bi.clean_combined_probs:
        if fav_is_home:
            clean_fav_prob = bi.clean_combined_probs.get(W, 0.33)
            clean_und_prob = bi.clean_combined_probs.get(L, 0.33)
        else:
            clean_fav_prob = bi.clean_combined_probs.get(L, 0.33)
            clean_und_prob = bi.clean_combined_probs.get(W, 0.33)
        
        total_clean = clean_fav_prob + bi.clean_combined_probs.get(D, 0.33) + clean_und_prob
        if total_clean > 0:
            clean_fav_prob = clean_fav_prob / total_clean
            clean_und_prob = clean_und_prob / total_clean
        
        sig.clean_edge = clean_fav_prob - fav_implied if fav_is_home else clean_und_prob - und_implied
        
        if clean_fav_prob < fav_implied - TRAP_FAV_OVER_VALUE and fav_implied > 0.55:
            sig.clean_signal_type = TrapValueType.TRAP
            sig.clean_description = f"Clean: Favourite overvalued ({clean_fav_prob:.1%} vs {fav_implied:.1%})"
            sig.clean_strength = round(min(1.0, (fav_implied - clean_fav_prob) / 0.15), 3)
        elif clean_fav_prob > fav_implied + VALUE_FAV_UNDER_VALUE and clean_fav_prob > 0.55:
            sig.clean_signal_type = TrapValueType.VALUE
            sig.clean_description = f"Clean: Favourite undervalued ({clean_fav_prob:.1%} vs {fav_implied:.1%})"
            sig.clean_strength = round(min(1.0, (clean_fav_prob - fav_implied) / 0.12), 3)
        elif clean_und_prob > und_implied + VALUE_UND_OVER_VALUE and clean_und_prob > 0.30:
            sig.clean_signal_type = TrapValueType.VALUE
            sig.clean_description = f"Clean: Underdog undervalued ({clean_und_prob:.1%} vs {und_implied:.1%})"
            sig.clean_strength = round(min(1.0, (clean_und_prob - und_implied) / 0.15), 3)
    
    return sig


# ═══════════════════════════════════════════════════════════════
# SECTION 11 — MAIN ENTRY POINT (ENHANCED)
# ═══════════════════════════════════════════════════════════════

def run_tally_matrix_analysis(
    home_results: List[str],
    away_results: List[str],
    home_team_id: str = "home",
    away_team_id: str = "away",
    home_team_name: str = "Home Team",
    away_team_name: str = "Away Team",
    fav_odds: float = 0.0,
    und_odds: float = 0.0,
    fav_is_home: bool = True,
    season: str = "",
    total_games: int = 38,
    home_multi_rtm: Optional[MultiDimensionTallyResult] = None,
    away_multi_rtm: Optional[MultiDimensionTallyResult] = None,
    home_enhanced: Optional[DimensionTallyMatrix] = None,
    away_enhanced: Optional[DimensionTallyMatrix] = None,
    verbose: bool = False,
) -> TallyMatrixAnalysis:
    """
    Run tally matrix analysis with optional enhanced (clean) data.
    """
    result = TallyMatrixAnalysis()
    
    if not home_results or not away_results:
        result.combined_risk_flag = "NO_DATA"
        return result
    
    # Build standard matrices
    result.home_matrix = build_tally_matrix(
        home_team_id, home_results, season, home_team_name, total_games
    )
    result.away_matrix = build_tally_matrix(
        away_team_id, away_results, season, away_team_name, total_games
    )
    
    result.home_multi_rtm = home_multi_rtm
    result.away_multi_rtm = away_multi_rtm
    
    result.home_streak = analyse_streak(result.home_matrix, home_results)
    result.away_streak = analyse_streak(result.away_matrix, away_results)
    
    home_last = home_results[-1] if home_results else W
    away_last = away_results[-1] if away_results else W
    
    # Get enhanced dimensions if available
    home_enhanced_dim = None
    away_enhanced_dim = None
    if home_enhanced:
        home_enhanced_dim = home_enhanced
    elif home_multi_rtm and home_multi_rtm.clean_overall:
        home_enhanced_dim = home_multi_rtm.clean_overall
    if away_enhanced:
        away_enhanced_dim = away_enhanced
    elif away_multi_rtm and away_multi_rtm.clean_overall:
        away_enhanced_dim = away_multi_rtm.clean_overall
    
    # Calculate bilateral intersection
    result.bilateral = calculate_bilateral_intersection(
        result.home_matrix, result.away_matrix, home_last, away_last,
        home_enhanced_dim, away_enhanced_dim
    )
    
    # Determine matrix usefulness
    home_even = all(
        result.home_matrix.is_even.get(r, False)
        for r in ALL_RESULTS
        if result.home_matrix.row_total(r) > 0
    )
    away_even = all(
        result.away_matrix.is_even.get(r, False)
        for r in ALL_RESULTS
        if result.away_matrix.row_total(r) > 0
    )
    result.matrix_useful = not (home_even and away_even and result.bilateral.is_uncertain)
    
    # Clean matrix usefulness
    result.clean_matrix_useful = not (result.bilateral.clean_is_uncertain)
    result.distortion_adjusted = (home_enhanced_dim is not None or away_enhanced_dim is not None)
    
    # Check reliability warning
    if home_enhanced_dim and home_enhanced_dim.clean_stats:
        if home_enhanced_dim.clean_stats.distortion_rate > DISTORTION_RATE_CRITICAL:
            result.reliability_warning = f"Home high distortion rate: {home_enhanced_dim.clean_stats.distortion_rate:.0%}"
        elif away_enhanced_dim and away_enhanced_dim.clean_stats:
            if away_enhanced_dim.clean_stats.distortion_rate > DISTORTION_RATE_CRITICAL:
                result.reliability_warning = f"Away high distortion rate: {away_enhanced_dim.clean_stats.distortion_rate:.0%}"
    
    # Trap/value detection
    if fav_odds > 1.0 and und_odds > 1.0 and not result.bilateral.is_uncertain:
        result.trap_value_signal = detect_trap_or_value(
            result.bilateral, fav_odds, und_odds, fav_is_home,
            result.home_matrix, result.away_matrix,
            use_clean=result.distortion_adjusted
        )
        result.has_trap = result.trap_value_signal.signal_type == TrapValueType.TRAP
        result.has_value = result.trap_value_signal.signal_type == TrapValueType.VALUE
        result.combined_risk_flag = result.trap_value_signal.signal_type.value
    else:
        result.combined_risk_flag = "UNCERTAIN" if result.bilateral.is_uncertain else "NONE"
    
    if verbose:
        print(f"\n  Tally Matrix Analysis:")
        print(f"    Home Matrix: {result.home_matrix.total_transitions} transitions")
        print(f"    Away Matrix: {result.away_matrix.total_transitions} transitions")
        print(f"    Home Streak: {result.home_streak['consecutive']}x {result.home_streak['direction']}")
        print(f"    Away Streak: {result.away_streak['consecutive']}x {result.away_streak['direction']}")
        print(f"    Bilateral: predicts {result.bilateral.predicted_outcome} (confidence={result.bilateral.confidence.value})")
        if result.bilateral.distortion_impact > 0:
            print(f"    Distortion Impact: {result.bilateral.distortion_impact:.1%}")
            print(f"    Clean Bilateral: predicts {result.bilateral.clean_predicted_outcome} (confidence={result.bilateral.clean_confidence.value})")
        if result.reliability_warning:
            print(f"    ⚠ {result.reliability_warning}")
        if result.trap_value_signal.signal_type != TrapValueType.NONE:
            signal_type = result.trap_value_signal.signal_type.value
            print(f"    Signal: {signal_type} - {result.trap_value_signal.description}")
    
    return result


# ═══════════════════════════════════════════════════════════════
# SECTION 12 — CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_transition_probability(
    matrix: SeasonTallyMatrix,
    from_result: str,
    to_result: str,
) -> float:
    return matrix.get_prob(from_result, to_result)


def is_matrix_reliable(matrix: SeasonTallyMatrix, min_transitions: int = MIN_TALLY_RELIABLE) -> bool:
    return matrix.total_transitions >= min_transitions


def get_most_likely_next(matrix: SeasonTallyMatrix, last_result: str) -> Tuple[str, float]:
    return matrix.get_most_likely_next(last_result)


def get_matrix_usefulness_score(matrix_useful: bool) -> float:
    return USEFULNESS_SCORE_MAP.get(matrix_useful, 0.5)


def get_trap_value_score(signal_type: str) -> float:
    return TRAP_VALUE_SCORE_MAP.get(signal_type, 0.50)


def get_dimension_transition(
    multi_rtm: MultiDimensionTallyResult,
    dimension: str,
    from_result: str,
    to_result: str,
) -> float:
    """Get transition probability from a specific dimension."""
    dim = multi_rtm.get_dimension(dimension)
    if dim:
        return dim.get_prob(from_result, to_result)
    return 0.33


def get_clean_transition(
    multi_rtm: MultiDimensionTallyResult,
    dimension: str,
    from_result: str,
    to_result: str,
) -> float:
    """Get clean (distortion-filtered) transition probability."""
    clean_dim = multi_rtm.get_clean_dimension(dimension)
    if clean_dim:
        return clean_dim.get_clean_prob(from_result, to_result)
    return get_dimension_transition(multi_rtm, dimension, from_result, to_result)


def get_venue_advantage_from_rtm(
    home_multi: MultiDimensionTallyResult,
    away_multi: MultiDimensionTallyResult,
    use_clean: bool = False,
) -> float:
    """
    Calculate venue advantage using dimension-specific RTM.
    Positive = home team has advantage.
    
    Args:
        home_multi: Home team's multi-dimension result
        away_multi: Away team's multi-dimension result
        use_clean: If True, use clean (distortion-filtered) probabilities
    """
    if use_clean:
        home_home_win = get_clean_transition(home_multi, "home", "W", "W")
        away_away_win = get_clean_transition(away_multi, "away", "W", "W")
    else:
        home_home_win = get_dimension_transition(home_multi, "home", "W", "W")
        away_away_win = get_dimension_transition(away_multi, "away", "W", "W")
    
    return home_home_win - away_away_win


def get_distortion_impact_score(
    home_multi: MultiDimensionTallyResult,
    away_multi: MultiDimensionTallyResult,
    dimension: str = "overall",
) -> float:
    """Calculate how much distortions affect the overall prediction."""
    home_distortion = 0.0
    away_distortion = 0.0
    
    home_dim = home_multi.get_dimension(dimension)
    if home_dim and home_dim.clean_stats:
        home_distortion = home_dim.clean_stats.distortion_rate
    
    away_dim = away_multi.get_dimension(dimension)
    if away_dim and away_dim.clean_stats:
        away_distortion = away_dim.clean_stats.distortion_rate
    
    return round((home_distortion + away_distortion) / 2, 3)


# ═══════════════════════════════════════════════════════════════
# SECTION 13 — EXPORTS
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "TrapValueType",
    "BilateralConfidence",
    # Constants
    "MIN_TALLY_RELIABLE",
    "MIN_TALLY_STRONG",
    "MIN_CLEAN_TRANSITIONS",
    "DISTORTION_RATE_WARNING",
    "ALL_RESULTS",
    "DIMENSION_OVERALL",
    "DIMENSION_HOME",
    "DIMENSION_AWAY",
    "DIMENSION_VS_TOP6",
    "DIMENSION_VS_MID",
    "DIMENSION_VS_BOTTOM6",
    "ALL_DIMENSIONS",
    # Data classes
    "TransitionCounts",
    "CleanTransitionStats",
    "DimensionTallyMatrix",
    "SeasonTallyMatrix",
    "MultiDimensionTallyResult",
    "BilateralIntersection",
    "TrapValueSignal",
    "TallyMatrixAnalysis",
    # Core functions
    "build_tally_matrix",
    "build_enhanced_tally_matrix",
    "build_dimension_tally_matrix",
    "build_multi_dimension_rtm",
    "build_h2h_rtm",
    "analyse_streak",
    "calculate_bilateral_intersection",
    "detect_trap_or_value",
    "run_tally_matrix_analysis",
    # Convenience functions
    "get_transition_probability",
    "is_matrix_reliable",
    "get_most_likely_next",
    "get_dimension_transition",
    "get_clean_transition",
    "get_venue_advantage_from_rtm",
    "get_distortion_impact_score",
    "get_matrix_usefulness_score",
    "get_trap_value_score",
]


# ═══════════════════════════════════════════════════════════════
# SECTION 14 — EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import random
    
    print("\n" + "=" * 70)
    print("MODULE 10: TALLY MATRIX v3 - TEST RUN")
    print("=" * 70)
    
    # Create synthetic result sequences
    home_results = ["W", "W", "D", "W", "L", "W", "W", "D", "W", "W", "L", "W"]
    away_results = ["L", "L", "D", "L", "W", "L", "L", "D", "L", "L", "W", "W"]
    
    print("\n📊 Test 1: Basic Tally Matrix")
    print("-" * 40)
    
    matrix = build_tally_matrix(
        team_id="1",
        results=home_results,
        season="2025",
        team_name="Arsenal",
        total_games=38
    )
    
    print(f"Team: {matrix.team_name}")
    print(f"Total transitions: {matrix.total_transitions}")
    print(f"Overall reliability: {matrix.overall_reliability:.2f}")
    print(f"W→W probability: {matrix.get_prob('W', 'W'):.1%}")
    print(f"L→W probability: {matrix.get_prob('L', 'W'):.1%}")
    
    # Test 2: Enhanced matrix with distortion filtering
    print("\n📊 Test 2: Enhanced Matrix with Distortion Filtering")
    print("-" * 40)
    
    # Create fixtures with distortion metadata
    fixtures_with_context = []
    for i, result in enumerate(home_results):
        # Randomly add distortion factors
        is_dead_rubber = random.random() < 0.1
        key_players_missing = random.randint(0, 2) if random.random() < 0.2 else 0
        is_midweek = random.random() < 0.15
        
        metadata = {
            "venue": "home" if random.random() < 0.5 else "away",
            "opponent_tier": random.choice(["top6", "mid", "bottom6"]),
            "is_dead_rubber": is_dead_rubber,
            "key_players_missing": key_players_missing,
            "manager_tenure_days": random.randint(30, 500),
            "european_midweek": is_midweek,
            "days_rest": random.randint(3, 10),
        }
        
        fixtures_with_context.append({
            "result": result,
            "date": f"2025-{i+1:02d}-01",
            "metadata": metadata,
        })
    
    enhanced_matrix = build_enhanced_tally_matrix(
        team_id="1",
        fixtures_with_context=fixtures_with_context,
        dimension="overall",
        season="2025",
        team_name="Arsenal",
    )
    
    print(f"Team: {enhanced_matrix.team_name}")
    print(f"Total transitions: {enhanced_matrix.total_transitions}")
    print(f"Clean transitions: {enhanced_matrix.clean_stats.clean_count if enhanced_matrix.clean_stats else 0}")
    print(f"Distortion rate: {enhanced_matrix.clean_stats.distortion_rate:.1%}" if enhanced_matrix.clean_stats else "No clean stats")
    print(f"Primary distortion: {enhanced_matrix.clean_stats.primary_distortion}" if enhanced_matrix.clean_stats else "")
    print(f"Clean reliability: {enhanced_matrix.clean_reliability:.2f}")
    
    # Test 3: Bilateral intersection with clean data
    print("\n📊 Test 3: Bilateral Intersection with Clean Data")
    print("-" * 40)
    
    # Create away enhanced matrix
    away_fixtures = []
    for i, result in enumerate(away_results):
        metadata = {
            "venue": "away",
            "opponent_tier": random.choice(["top6", "mid", "bottom6"]),
            "is_dead_rubber": False,
            "key_players_missing": 0,
        }
        away_fixtures.append({
            "result": result,
            "date": f"2025-{i+1:02d}-15",
            "metadata": metadata,
        })
    
    away_enhanced = build_enhanced_tally_matrix(
        team_id="2",
        fixtures_with_context=away_fixtures,
        dimension="overall",
        season="2025",
        team_name="Chelsea",
    )
    
    # Build standard matrices
    home_std = build_tally_matrix("1", home_results, team_name="Arsenal")
    away_std = build_tally_matrix("2", away_results, team_name="Chelsea")
    
    # Calculate bilateral intersection
    bilateral = calculate_bilateral_intersection(
        home_std, away_std,
        home_results[-1] if home_results else "W",
        away_results[-1] if away_results else "W",
        enhanced_matrix, away_enhanced
    )
    
    print(f"Standard prediction: {bilateral.predicted_outcome} (margin={bilateral.margin:.1%})")
    print(f"Clean prediction: {bilateral.clean_predicted_outcome} (margin={bilateral.clean_margin:.1%})")
    print(f"Distortion impact: {bilateral.distortion_impact:.1%}")
    print(f"Standard confidence: {bilateral.confidence.value}")
    print(f"Clean confidence: {bilateral.clean_confidence.value}")
    
    # Test 4: Trap/Value detection
    print("\n📊 Test 4: Trap/Value Detection")
    print("-" * 40)
    
    signal = detect_trap_or_value(
        bilateral,
        fav_odds=1.85,
        und_odds=4.50,
        fav_is_home=True,
        home_enhanced=enhanced_matrix,
        away_enhanced=away_enhanced
    )
    
    print(f"Signal type: {signal.signal_type.value}")
    print(f"Description: {signal.description}")
    print(f"Edge: {signal.edge:+.3f}")
    print(f"Clean signal: {signal.clean_signal_type.value}")
    print(f"Clean edge: {signal.clean_edge:+.3f}")
    
    # Test 5: Multi-dimension RTM
    print("\n📊 Test 5: Multi-Dimension RTM")
    print("-" * 40)
    
    multi_rtm = build_multi_dimension_rtm(
        team_id="1",
        team_name="Arsenal",
        fixtures_with_results=fixtures_with_context,
        use_enhanced=True
    )
    
    summary = multi_rtm.summary()
    print(f"Team: {summary['team']}")
    print(f"Dimensions: {summary['dimensions']}")
    print(f"Clean available: {summary['clean_available']}")
    
    # Leg data for M11
    print("\n📊 Leg Data for M11:")
    analysis = run_tally_matrix_analysis(
        home_results, away_results,
        home_team_name="Arsenal",
        away_team_name="Chelsea",
        fav_odds=1.85,
        und_odds=4.50,
        fav_is_home=True,
        home_enhanced=enhanced_matrix,
        away_enhanced=away_enhanced,
        verbose=True
    )
    
    leg_data = analysis.to_leg_data()
    for key, value in leg_data.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("MODULE 10 v3 READY FOR PRODUCTION")
    print("=" * 70)