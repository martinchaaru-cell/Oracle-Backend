"""
The Match Oracle – Module 27: Head-to-Head Deep Analyzer (ENHANCED v4)
================================================================================
Module 4 only checks whether the favourite has won >50% of H2H games.
This module provides full H2H intelligence across 6 dimensions:

  1. Overall H2H win/draw/loss record
  2. Venue-specific record (at today's ground)
  3. Recent H2H form (last 3 meetings)
  4. Scoring pattern (high-scoring / BTTS / clean-sheet)
  5. Psychological dominance (streak analysis)
  6. Weighted composite H2H score (0–100)

ENHANCEMENTS IN VERSION 4 (May 2026):
-----------------------------------
1. ADDED: H2H Draw Rate Boost — increases draw probability when recent H2H has high draw rate
2. ADDED: Draw confidence scoring with sample size weighting
3. ADDED: Integration with M11 weighted decision system
4. ADDED: Draw boost factor to leg_data for probability adjustment
5. ADDED: Historical draw pattern tracking (last 6, last 3 games)
6. ADDED: Draw rate classification (NORMAL/ELEVATED/HIGH/EXTREME)

ENHANCEMENTS FOR DIMENSION RTM:
------------------------------
1. ADDED: H2H RTM (Results Transition Matrix) for head-to-head sequences
2. ADDED: Venue-specific H2H RTM (home/away in H2H)
3. ADDED: H2H bounce-back rate (what happens after a loss to this opponent)
4. ADDED: H2H streak analysis with psychological block detection
5. ADDED: H2H scoring pattern with per-game averages
6. ADDED: H2H confidence scoring based on sample size
7. ADDED: Integration with M2's DimensionRTM

WEIGHTED DECISION SUPPORT:
-------------------------
- dominance_score: 0-1 score from h2h_score (0-100 → 0-1)
- fav_edge_score: How strongly favourite dominates H2H
- draw_risk_score: High draw rate increases risk score
- normalized_score: Combined score for M11
- confidence_factor: For M13 Kelly scaling
- h2h_rtm_score: RTM-based bounce-back prediction
- draw_boost_factor: Multiplier for draw probability (NEW v4)
- draw_confidence: Confidence level in draw boost (NEW v4)
- to_leg_data(): Direct output for M11 aggregation

Feeds into: Module 5 (forensic flags), Module 11 (verdict enrichment),
            Module 17 (explainability context), Module 8 (dual pattern)

Usage:
    from module27 import run_h2h_deep_analyzer, H2HDeepAnalysis, calculate_draw_boost
    
    analysis = run_h2h_deep_analyzer(fixtures, fav_is_home=True)
    print(f"H2H Score: {analysis.h2h_score:.1f}/100")
    print(f"H2H RTM Bounce-back: {analysis.h2h_rtm.bounce_back_rate:.1%}")
    print(f"Draw Boost: {analysis.draw_boost_factor:.2f}x")
    
    # For M11
    leg_data = analysis.to_leg_data()
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — ENUMS
# ═══════════════════════════════════════════════════════════════

class H2HRTMConfidence(Enum):
    """Confidence level for H2H RTM analysis."""
    HIGH = "HIGH"           # 10+ games
    MEDIUM = "MEDIUM"       # 7-9 games
    LOW = "LOW"             # 5-6 games
    INSUFFICIENT = "INSUFFICIENT"  # <5 games


class DrawRateClass(Enum):
    """Classification of draw rate in H2H matches."""
    NORMAL = "NORMAL"           # <25% draws
    ELEVATED = "ELEVATED"       # 25-33% draws
    HIGH = "HIGH"               # 33-50% draws
    EXTREME = "EXTREME"         # >50% draws
    UNKNOWN = "UNKNOWN"


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — CONSTANTS
# ═══════════════════════════════════════════════════════════════

MIN_H2H_GAMES        = 5      # Minimum games for reliable analysis
DOMINANCE_THRESHOLD  = 6      # Consecutive wins = psychological block
RECENCY_WINDOW       = 3      # Number of recent H2H games to analyze
HIGH_SCORING_AVG     = 2.7    # Goals per game > this = high-scoring
LOW_SCORING_AVG      = 1.8    # Goals per game < this = low-scoring
BTTS_THRESHOLD       = 0.60   # Both teams scored in >60% = BTTS trend

# RTM-specific constants
MIN_H2H_FOR_RTM      = 5      # Minimum games for RTM reliability
MIN_H2H_FOR_BOUNCE   = 3      # Minimum for bounce-back detection
STREAK_BLOCK_THRESHOLD = 4    # 4+ consecutive wins = psychological block

# Weighted decision scores
H2H_LABEL_SCORE_MAP = {
    "FAV_DOMINANT": 0.90,
    "FAV_EDGE": 0.70,
    "NEUTRAL": 0.50,
    "UND_EDGE": 0.30,
    "UND_DOMINANT": 0.10,
    "INSUFFICIENT_DATA": 0.30,
    "UNKNOWN": 0.50,
}

DRAW_RISK_THRESHOLD = 0.35    # Draw rate above this increases risk
DRAW_RISK_PENALTY   = 0.15    # Penalty to score for high draw rate

# Draw boost constants (NEW v4)
DRAW_BOOST_MULTIPLIERS = {
    DrawRateClass.NORMAL: 1.00,
    DrawRateClass.ELEVATED: 1.08,
    DrawRateClass.HIGH: 1.15,
    DrawRateClass.EXTREME: 1.25,
}
DRAW_BOOST_MAX_FACTOR = 1.30   # Maximum draw boost multiplier
DRAW_BOOST_MIN_FACTOR = 0.95   # Minimum draw boost multiplier

# Draw rate thresholds (percentage of H2H games ending in draw)
DRAW_RATE_NORMAL_MAX = 0.25
DRAW_RATE_ELEVATED_MAX = 0.33
DRAW_RATE_HIGH_MAX = 0.50

# Recent draw weighting (last 3 games have higher impact)
RECENT_DRAW_WEIGHT = 0.60      # 60% weight on recent 3 games
OVERALL_DRAW_WEIGHT = 0.40     # 40% weight on overall history

# Sample size confidence for draw boost
MIN_DRAW_BOOST_SAMPLES = 5
HIGH_CONFIDENCE_SAMPLES = 10

# Confidence factor mapping
CONFIDENCE_MAP = {
    "HIGH": 1.0,      # 10+ games
    "MEDIUM": 0.7,    # 7-9 games
    "LOW": 0.4,       # 5-6 games
    "INSUFFICIENT": 0.2,
}

# RTM confidence mapping
RTM_CONFIDENCE_MAP = {
    H2HRTMConfidence.HIGH: 1.0,
    H2HRTMConfidence.MEDIUM: 0.7,
    H2HRTMConfidence.LOW: 0.4,
    H2HRTMConfidence.INSUFFICIENT: 0.2,
}


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class H2HFixtureDetail:
    """Detailed information for a single H2H fixture."""
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    venue: str           # "home" | "away" | "neutral"
    date: str            # YYYY-MM-DD
    competition: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "score": f"{self.home_goals}-{self.away_goals}",
            "venue": self.venue,
            "date": self.date,
            "competition": self.competition,
        }
    
    @property
    def outcome_from_fav_perspective(self, fav_is_home: bool) -> str:
        """Return 'W', 'D', or 'L' from favourite's perspective."""
        if self.home_goals == self.away_goals:
            return "D"
        elif self.home_goals > self.away_goals:
            return "W" if fav_is_home else "L"
        else:
            return "L" if fav_is_home else "W"


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
    def fav_edge(self) -> float:
        """Favourite's edge over underdog (0-1)."""
        return max(0.0, self.fav_win_rate - self.und_win_rate)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "games": self.games,
            "fav_wins": self.fav_wins,
            "draws": self.draws,
            "und_wins": self.und_wins,
            "fav_win_rate": round(self.fav_win_rate, 3),
            "draw_rate": round(self.draw_rate, 3),
            "is_one_sided": self.is_one_sided,
            "fav_edge": round(self.fav_edge, 3),
        }


@dataclass
class H2HRTM:
    """
    H2H-specific Results Transition Matrix.
    
    Tracks transitions in head-to-head matchups:
    - WIN→WIN: Team wins consecutive H2H matches
    - LOSS→WIN: Bounce-back after H2H loss
    - WIN→LOSS: Reversal pattern
    """
    # Transition counts
    transitions: Dict[Tuple[str, str], int] = field(default_factory=dict)
    
    # Transition probabilities
    probabilities: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    # Series metrics
    current_streak: int = 0
    streak_direction: str = "NONE"  # "W", "D", "L"
    max_streak: int = 0
    max_streak_direction: str = "NONE"
    
    # Bounce-back metrics
    bounce_back_rate: float = 0.0      # P(WIN | LOSS in H2H)
    bounce_back_count: int = 0
    bounce_back_total: int = 0
    
    # Pattern metrics
    win_ceiling: int = 0               # Max consecutive H2H wins
    loss_floor: int = 0                # Max consecutive H2H losses
    alternation_rate: float = 0.0      # How often results alternate
    
    # Confidence
    total_games: int = 0
    confidence: H2HRTMConfidence = H2HRTMConfidence.INSUFFICIENT
    
    @property
    def is_reliable(self) -> bool:
        return self.total_games >= MIN_H2H_FOR_RTM
    
    @property
    def has_psychological_block(self) -> bool:
        """True if underdog has 4+ consecutive H2H wins (psychological block)."""
        return self.max_streak >= STREAK_BLOCK_THRESHOLD and self.max_streak_direction == "L"
    
    def get_prob(self, from_result: str, to_result: str) -> float:
        """Get transition probability."""
        return self.probabilities.get((from_result, to_result), 0.33)
    
    def to_dimension_rtm(self, team_id: str, team_name: str, opponent_id: str, opponent_name: str) -> Optional[Any]:
        """Convert to M2's DimensionRTM object."""
        try:
            from module2 import DimensionRTM
        except ImportError:
            return None
        
        dim = DimensionRTM(dimension_name=f"h2h_vs_{opponent_id}")
        dim.sample_size = self.total_games
        dim.total_fixtures = self.total_games
        dim.is_reliable = self.is_reliable
        dim.bounce_back_rate = self.bounce_back_rate
        dim.win_ceiling = self.win_ceiling
        dim.resilience_score = self.bounce_back_rate * 100
        dim.current_win_streak = self.current_streak if self.streak_direction == "W" else 0
        dim.current_loss_streak = self.current_streak if self.streak_direction == "L" else 0
        
        # Map transition probabilities
        try:
            from module15 import OutcomeState
            for (fr, to), prob in self.probabilities.items():
                try:
                    prev = OutcomeState[fr]
                    curr = OutcomeState[to]
                    dim.probabilities[(prev, curr)] = prob
                except (KeyError, ValueError):
                    pass
        except ImportError:
            pass
        
        return dim
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_games": self.total_games,
            "confidence": self.confidence.value,
            "is_reliable": self.is_reliable,
            "current_streak": f"{self.current_streak}x{self.streak_direction}" if self.current_streak > 0 else "none",
            "max_streak": f"{self.max_streak}x{self.max_streak_direction}" if self.max_streak > 0 else "none",
            "bounce_back_rate": round(self.bounce_back_rate, 3),
            "win_ceiling": self.win_ceiling,
            "alternation_rate": round(self.alternation_rate, 3),
            "has_psychological_block": self.has_psychological_block,
            "probabilities": {
                f"{fr}→{to}": round(prob, 3)
                for (fr, to), prob in self.probabilities.items()
            },
        }


@dataclass
class H2HScoringPattern:
    """Scoring pattern analysis for H2H fixtures."""
    avg_goals: float = 0.0
    both_score_rate: float = 0.0
    clean_sheet_rate: float = 0.0
    label: str = "MODERATE"  # HIGH_SCORING, LOW_SCORING, BTTS, MODERATE
    confidence: str = "LOW"
    
    def summary(self) -> str:
        return (f"Avg {self.avg_goals:.1f} goals/game, "
                f"BTTS {self.both_score_rate:.0%}, "
                f"CS {self.clean_sheet_rate:.0%} - {self.label}")
    
    @property
    def score_modifier(self) -> float:
        """Return score modifier based on pattern (0.8-1.2)."""
        if self.label == "HIGH_SCORING":
            return 1.05
        elif self.label == "LOW_SCORING":
            return 0.95
        elif self.label == "BTTS":
            return 1.02
        return 1.00
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_goals": round(self.avg_goals, 2),
            "both_score_rate": round(self.both_score_rate, 3),
            "clean_sheet_rate": round(self.clean_sheet_rate, 3),
            "label": self.label,
            "confidence": self.confidence,
            "score_modifier": self.score_modifier,
        }


@dataclass
class H2HDominance:
    """Psychological dominance analysis."""
    streak_holder: str = "NONE"      # FAV | UND | NONE
    streak_length: int = 0
    psychological_block: bool = False
    recent_winner: str = "NONE"      # FAV | UND | DRAW
    trend: str = "STABLE"            # IMPROVING, DECLINING, STABLE
    
    def summary(self) -> str:
        if self.psychological_block:
            return f"⚠️ Psychological block: {self.streak_holder} on {self.streak_length}-match run"
        elif self.streak_length >= 3:
            return f"📈 Momentum: {self.streak_holder} on {self.streak_length}-match streak"
        return "No significant dominance"
    
    @property
    def dominance_modifier(self) -> float:
        """Return modifier for psychological block."""
        if self.psychological_block:
            return 0.70  # 30% penalty
        elif self.streak_length >= 4 and self.streak_holder == "FAV":
            return 1.10  # 10% boost for strong favourite streak
        elif self.streak_length >= 4 and self.streak_holder == "UND":
            return 0.85  # 15% penalty for underdog streak
        return 1.00
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "streak_holder": self.streak_holder,
            "streak_length": self.streak_length,
            "psychological_block": self.psychological_block,
            "recent_winner": self.recent_winner,
            "trend": self.trend,
            "dominance_modifier": self.dominance_modifier,
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 3.5 — DRAW BOOST ANALYSIS (NEW v4)
# ═══════════════════════════════════════════════════════════════

@dataclass
class DrawBoostAnalysis:
    """
    Analysis of H2H draw patterns for probability adjustment.
    
    This class calculates a draw boost factor that can be applied
    to the model's draw probability when recent H2H history shows
    elevated draw rates.
    
    Example: Machida vs Tokyo Verdy with 2 draws in last 6 H2H (33%)
    → draw boost factor of 1.08x, increasing draw probability from
    22% to ~24%.
    """
    overall_draw_rate: float = 0.0
    recent_draw_rate: float = 0.0      # Last 6 games
    very_recent_draw_rate: float = 0.0  # Last 3 games
    total_games: int = 0
    recent_games: int = 0
    very_recent_games: int = 0
    draw_rate_class: DrawRateClass = DrawRateClass.NORMAL
    draw_boost_factor: float = 1.00
    confidence: str = "LOW"
    confidence_score: float = 0.3
    
    def summary(self) -> str:
        return (f"Draw boost: {self.draw_boost_factor:.2f}x (overall={self.overall_draw_rate:.0%}, "
                f"recent={self.recent_draw_rate:.0%}, class={self.draw_rate_class.value})")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_draw_rate": round(self.overall_draw_rate, 3),
            "recent_draw_rate": round(self.recent_draw_rate, 3),
            "very_recent_draw_rate": round(self.very_recent_draw_rate, 3),
            "total_games": self.total_games,
            "draw_rate_class": self.draw_rate_class.value,
            "draw_boost_factor": round(self.draw_boost_factor, 3),
            "confidence": self.confidence,
            "confidence_score": round(self.confidence_score, 3),
        }


def calculate_draw_boost(
    fixtures: List[H2HFixtureDetail],
    fav_is_home: bool,
    recent_window: int = 6,
    very_recent_window: int = 3,
) -> DrawBoostAnalysis:
    """
    Calculate draw boost factor based on H2H draw rates.
    
    This function analyzes the historical draw rate between two teams
    and returns a multiplier that can be applied to the model's draw
    probability.
    
    The logic:
    - Weighted average of overall (40%) and recent (60%) draw rates
    - Higher weight on recent matches captures current pattern
    - Sample size affects confidence level
    
    Args:
        fixtures: List of H2H fixtures in chronological order
        fav_is_home: Whether favourite is home team in current fixture
        recent_window: Number of recent games to analyze (default 6)
        very_recent_window: Number of very recent games (default 3)
    
    Returns:
        DrawBoostAnalysis with boost factor and confidence
    """
    if len(fixtures) < MIN_H2H_GAMES:
        return DrawBoostAnalysis(
            total_games=len(fixtures),
            draw_boost_factor=1.00,
            confidence="INSUFFICIENT",
            confidence_score=0.1,
        )
    
    # Sort by date (oldest first for chronological analysis)
    sorted_fx = sorted(fixtures, key=lambda x: x.date)
    
    # Calculate draw rates from favourite's perspective
    draw_count = 0
    for fx in sorted_fx:
        if fx.outcome_from_fav_perspective(fav_is_home) == "D":
            draw_count += 1
    
    overall_draw_rate = draw_count / len(sorted_fx) if sorted_fx else 0.0
    
    # Calculate recent draw rates (last N games)
    recent_fx = sorted_fx[-recent_window:] if len(sorted_fx) >= recent_window else sorted_fx
    recent_draw_count = sum(1 for fx in recent_fx if fx.outcome_from_fav_perspective(fav_is_home) == "D")
    recent_draw_rate = recent_draw_count / len(recent_fx) if recent_fx else 0.0
    
    # Calculate very recent draw rates (last 3 games)
    very_recent_fx = sorted_fx[-very_recent_window:] if len(sorted_fx) >= very_recent_window else sorted_fx
    very_recent_draw_count = sum(1 for fx in very_recent_fx if fx.outcome_from_fav_perspective(fav_is_home) == "D")
    very_recent_draw_rate = very_recent_draw_count / len(very_recent_fx) if very_recent_fx else 0.0
    
    # Classify draw rate
    if overall_draw_rate >= DRAW_RATE_HIGH_MAX:
        draw_class = DrawRateClass.EXTREME
    elif overall_draw_rate >= DRAW_RATE_ELEVATED_MAX:
        draw_class = DrawRateClass.HIGH
    elif overall_draw_rate >= DRAW_RATE_NORMAL_MAX:
        draw_class = DrawRateClass.ELEVATED
    else:
        draw_class = DrawRateClass.NORMAL
    
    # Calculate weighted draw rate (recent games have higher weight)
    # Weight recent (last 6) at 60%, overall at 40%
    weighted_draw_rate = (recent_draw_rate * RECENT_DRAW_WEIGHT) + (overall_draw_rate * OVERALL_DRAW_WEIGHT)
    
    # Determine boost factor based on weighted draw rate
    if weighted_draw_rate >= 0.50:
        boost = DRAW_BOOST_MULTIPLIERS[DrawRateClass.EXTREME]
    elif weighted_draw_rate >= 0.33:
        boost = DRAW_BOOST_MULTIPLIERS[DrawRateClass.HIGH]
    elif weighted_draw_rate >= 0.25:
        boost = DRAW_BOOST_MULTIPLIERS[DrawRateClass.ELEVATED]
    else:
        boost = DRAW_BOOST_MULTIPLIERS[DrawRateClass.NORMAL]
    
    # Cap boost factor
    boost = min(DRAW_BOOST_MAX_FACTOR, max(DRAW_BOOST_MIN_FACTOR, boost))
    
    # Calculate confidence based on sample size
    total_games = len(sorted_fx)
    if total_games >= HIGH_CONFIDENCE_SAMPLES:
        confidence = "HIGH"
        confidence_score = 0.9
    elif total_games >= MIN_DRAW_BOOST_SAMPLES:
        confidence = "MEDIUM"
        confidence_score = 0.7
    else:
        confidence = "LOW"
        confidence_score = 0.4
    
    return DrawBoostAnalysis(
        overall_draw_rate=overall_draw_rate,
        recent_draw_rate=recent_draw_rate,
        very_recent_draw_rate=very_recent_draw_rate,
        total_games=total_games,
        recent_games=len(recent_fx),
        very_recent_games=len(very_recent_fx),
        draw_rate_class=draw_class,
        draw_boost_factor=round(boost, 3),
        confidence=confidence,
        confidence_score=confidence_score,
    )


@dataclass
class H2HDeepAnalysis:
    """Complete H2H deep analysis output."""
    total_games: int = 0
    is_reliable: bool = False
    confidence: str = "LOW"  # LOW / MEDIUM / HIGH / INSUFFICIENT

    # Individual components
    overall: H2HRecord = field(default_factory=H2HRecord)
    at_venue: H2HRecord = field(default_factory=H2HRecord)
    recent: H2HRecord = field(default_factory=H2HRecord)
    scoring: H2HScoringPattern = field(default_factory=H2HScoringPattern)
    dominance: H2HDominance = field(default_factory=H2HDominance)
    
    # NEW: H2H RTM
    h2h_rtm: H2HRTM = field(default_factory=H2HRTM)
    
    # Venue-specific RTM (available if enough data)
    h2h_rtm_home: Optional[H2HRTM] = None
    h2h_rtm_away: Optional[H2HRTM] = None

    # NEW v4: Draw boost analysis
    draw_boost: DrawBoostAnalysis = field(default_factory=DrawBoostAnalysis)

    # Composite score (0-100)
    h2h_score: float = 50.0
    h2h_label: str = "NEUTRAL"
    
    # RTM-enhanced score (includes bounce-back prediction)
    rtm_enhanced_score: float = 50.0
    
    # Flags for downstream modules
    flags: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # ═══════════════════════════════════════════════════════════
    # WEIGHTED DECISION PROPERTIES
    # ═══════════════════════════════════════════════════════════
    
    @property
    def dominance_score(self) -> float:
        """Convert h2h_score (0-100) to 0-1 score."""
        return self.h2h_score / 100.0
    
    @property
    def fav_edge_score(self) -> float:
        """How strongly favourite dominates H2H (0-1)."""
        if self.overall.fav_win_rate > self.overall.und_win_rate:
            return self.overall.fav_win_rate
        return 0.0
    
    @property
    def draw_risk_score(self) -> float:
        """Draw rate as risk score (0-1)."""
        return min(1.0, self.overall.draw_rate * 2)
    
    @property
    def rtm_score(self) -> float:
        """RTM-based score (0-1) incorporating bounce-back prediction."""
        if not self.h2h_rtm.is_reliable:
            return 0.5
        
        # Base from bounce-back rate
        base = self.h2h_rtm.bounce_back_rate
        
        # Adjust for psychological block
        if self.h2h_rtm.has_psychological_block:
            base = base * 0.7
        
        return round(min(1.0, max(0.0, base)), 3)
    
    @property
    def draw_boost_factor(self) -> float:
        """Draw boost factor for probability adjustment (NEW v4)."""
        return self.draw_boost.draw_boost_factor
    
    @property
    def draw_boost_confidence(self) -> float:
        """Confidence in draw boost (0-1)."""
        return self.draw_boost.confidence_score
    
    @property
    def normalized_score(self) -> float:
        """
        Combined score for weighted decision (0-1).
        Base = dominance_score, then apply adjustments from RTM and draw boost.
        """
        base = self.dominance_score
        
        # Apply scoring pattern modifier
        base *= self.scoring.score_modifier
        
        # Apply dominance modifier
        base *= self.dominance.dominance_modifier
        
        # Penalty for high draw rate
        if self.overall.draw_rate >= DRAW_RISK_THRESHOLD:
            base -= DRAW_RISK_PENALTY
        
        # Penalty for insufficient data
        if not self.is_reliable:
            base *= 0.7
        
        # RTM adjustment
        if self.h2h_rtm.is_reliable:
            # Blend with RTM score (70% traditional, 30% RTM)
            base = base * 0.7 + self.rtm_score * 0.3
        
        return round(max(0.0, min(1.0, base)), 3)
    
    @property
    def confidence_factor(self) -> float:
        """Confidence factor for M13 Kelly scaling."""
        base = CONFIDENCE_MAP.get(self.confidence, 0.5)
        
        # Boost for one-sided H2H
        if self.overall.is_one_sided:
            base = min(1.0, base * 1.2)
        
        # Penalty for high draw rate
        if self.overall.draw_rate >= DRAW_RISK_THRESHOLD:
            base *= 0.8
        
        # RTM confidence boost
        if self.h2h_rtm.is_reliable:
            base = min(1.0, base * 1.1)
        
        return round(base, 2)
    
    def to_leg_data(self) -> Dict[str, Any]:
        """Convert to leg_data format for M11 weighted decision."""
        return {
            "h2h_score": self.h2h_score,
            "h2h_normalized": self.normalized_score,
            "h2h_label": self.h2h_label,
            "h2h_dominance_score": self.dominance_score,
            "h2h_fav_edge": self.fav_edge_score,
            "h2h_draw_risk": self.draw_risk_score,
            "h2h_confidence_factor": self.confidence_factor,
            "h2h_reliable": self.is_reliable,
            "h2h_games": self.total_games,
            "h2h_rtm_score": self.rtm_score,
            "h2h_psychological_block": self.h2h_rtm.has_psychological_block,
            "h2h_bounce_back": self.h2h_rtm.bounce_back_rate,
            # NEW v4: Draw boost fields
            "h2h_draw_boost_factor": self.draw_boost_factor,
            "h2h_draw_boost_confidence": self.draw_boost_confidence,
            "h2h_draw_rate": self.overall.draw_rate,
            "h2h_recent_draw_rate": self.draw_boost.recent_draw_rate,
            "h2h_flags": self.flags,
        }
    
    def summary(self) -> str:
        """Full summary string."""
        lines = [
            f"H2H Analysis: {self.total_games} games (reliable={self.is_reliable}, confidence={self.confidence})",
            f"  Overall: FAV {self.overall.fav_wins}-{self.overall.draws}-{self.overall.und_wins} UND (fav {self.overall.fav_win_rate:.0%})",
            f"  Draw Rate: {self.overall.draw_rate:.0%} (boost: {self.draw_boost_factor:.2f}x)",
        ]
        if self.at_venue.games >= 3:
            lines.append(f"  At venue: FAV {self.at_venue.fav_wins}-{self.at_venue.draws}-{self.at_venue.und_wins} UND")
        lines.append(f"  Recent (last {RECENCY_WINDOW}): {self.recent.fav_wins}-{self.recent.draws}-{self.recent.und_wins}")
        lines.append(f"  Scoring: {self.scoring.summary()}")
        lines.append(f"  Dominance: {self.dominance.summary()}")
        lines.append(f"  Draw Boost: {self.draw_boost.summary()}")
        lines.append(f"  RTM: bounce-back={self.h2h_rtm.bounce_back_rate:.0%}, streak={self.h2h_rtm.current_streak}x{self.h2h_rtm.streak_direction}")
        lines.append(f"  Score: {self.h2h_score:.1f}/100 ({self.h2h_label})")
        lines.append(f"  RTM-Enhanced: {self.rtm_enhanced_score:.1f}/100")
        lines.append(f"  Normalized: {self.normalized_score:.3f}")
        if self.flags:
            lines.append(f"  Flags: {', '.join(self.flags)}")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_games": self.total_games,
            "is_reliable": self.is_reliable,
            "confidence": self.confidence,
            "overall": self.overall.to_dict(),
            "at_venue": self.at_venue.to_dict(),
            "recent": self.recent.to_dict(),
            "scoring": self.scoring.to_dict(),
            "dominance": self.dominance.to_dict(),
            "h2h_rtm": self.h2h_rtm.to_dict(),
            "draw_boost": self.draw_boost.to_dict(),
            "h2h_score": round(self.h2h_score, 1),
            "h2h_label": self.h2h_label,
            "rtm_enhanced_score": round(self.rtm_enhanced_score, 1),
            "normalized_score": self.normalized_score,
            "confidence_factor": self.confidence_factor,
            "flags": self.flags,
            "timestamp": self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — H2H RTM BUILDER
# ═══════════════════════════════════════════════════════════════

def _build_h2h_rtm(
    fixtures: List[H2HFixtureDetail],
    fav_is_home: bool,
    venue_filter: Optional[str] = None,
) -> H2HRTM:
    """
    Build H2H-specific transition matrix.
    
    Args:
        fixtures: List of H2H fixtures in chronological order
        fav_is_home: Whether favourite is home team in current fixture
        venue_filter: "home", "away", or None for all
    """
    rtm = H2HRTM()
    
    # Filter by venue if specified
    if venue_filter:
        filtered = [f for f in fixtures if f.venue == venue_filter]
    else:
        filtered = fixtures
    
    if len(filtered) < MIN_H2H_FOR_RTM:
        rtm.confidence = H2HRTMConfidence.INSUFFICIENT
        return rtm
    
    # Sort by date
    sorted_fx = sorted(filtered, key=lambda x: x.date)
    rtm.total_games = len(sorted_fx)
    
    # Extract outcomes from favourite's perspective
    outcomes = [fx.outcome_from_fav_perspective(fav_is_home) for fx in sorted_fx]
    
    # Set confidence based on sample size
    if len(outcomes) >= 10:
        rtm.confidence = H2HRTMConfidence.HIGH
    elif len(outcomes) >= 7:
        rtm.confidence = H2HRTMConfidence.MEDIUM
    elif len(outcomes) >= 5:
        rtm.confidence = H2HRTMConfidence.LOW
    
    # Calculate transitions
    transitions = {}
    prev_counts = {}
    
    for i in range(len(outcomes) - 1):
        fr, to = outcomes[i], outcomes[i+1]
        key = (fr, to)
        transitions[key] = transitions.get(key, 0) + 1
        prev_counts[fr] = prev_counts.get(fr, 0) + 1
    
    rtm.transitions = transitions
    
    # Calculate probabilities
    for (fr, to), count in transitions.items():
        total = prev_counts.get(fr, 0)
        rtm.probabilities[(fr, to)] = count / total if total > 0 else 0.33
    
    # Calculate bounce-back rate (LOSS → WIN)
    loss_to_win = transitions.get(("L", "W"), 0)
    total_losses = prev_counts.get("L", 0)
    rtm.bounce_back_rate = loss_to_win / total_losses if total_losses > 0 else 0.0
    rtm.bounce_back_count = loss_to_win
    rtm.bounce_back_total = total_losses
    
    # Calculate streaks
    current_streak = 1
    current_dir = outcomes[-1] if outcomes else "NONE"
    for i in range(len(outcomes) - 2, -1, -1):
        if outcomes[i] == current_dir:
            current_streak += 1
        else:
            break
    
    rtm.current_streak = current_streak
    rtm.streak_direction = current_dir
    
    # Calculate max streak
    max_streak = 1
    max_dir = outcomes[0] if outcomes else "NONE"
    running = 1
    for i in range(1, len(outcomes)):
        if outcomes[i] == outcomes[i-1]:
            running += 1
            if running > max_streak:
                max_streak = running
                max_dir = outcomes[i]
        else:
            running = 1
    
    rtm.max_streak = max_streak
    rtm.max_streak_direction = max_dir
    rtm.win_ceiling = max_streak if max_dir == "W" else 0
    rtm.loss_floor = max_streak if max_dir == "L" else 0
    
    # Calculate alternation rate
    if len(outcomes) > 1:
        alternations = sum(1 for i in range(1, len(outcomes)) if outcomes[i] != outcomes[i-1])
        rtm.alternation_rate = alternations / (len(outcomes) - 1)
    
    return rtm


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════

def _build_record(
    fixtures: List[H2HFixtureDetail],
    fav_is_home: bool,
) -> H2HRecord:
    """Build H2H record from a list of fixtures."""
    r = H2HRecord()
    for fx in fixtures:
        r.games += 1
        o = fx.outcome_from_fav_perspective(fav_is_home)
        if o == "W":
            r.fav_wins += 1
        elif o == "L":
            r.und_wins += 1
        else:
            r.draws += 1
    return r


def _streak_analysis(
    fixtures: List[H2HFixtureDetail],
    fav_is_home: bool,
) -> H2HDominance:
    """Analyze streak and psychological dominance with trend detection."""
    dom = H2HDominance()
    
    if not fixtures:
        return dom
    
    sorted_fx = sorted(fixtures, key=lambda x: x.date, reverse=True)
    
    # Calculate current streak
    streak, holder = 0, None
    for fx in sorted_fx:
        o = fx.outcome_from_fav_perspective(fav_is_home)
        if holder is None:
            holder, streak = o, 1
        elif o == holder:
            streak += 1
        else:
            break
    
    dom.streak_holder = holder or "NONE"
    dom.streak_length = streak
    dom.psychological_block = (streak >= DOMINANCE_THRESHOLD and holder == "L")
    
    # Recent winner (last 3 games)
    recent = sorted_fx[:RECENCY_WINDOW]
    fav_r = sum(1 for fx in recent if fx.outcome_from_fav_perspective(fav_is_home) == "W")
    und_r = sum(1 for fx in recent if fx.outcome_from_fav_perspective(fav_is_home) == "L")
    dom.recent_winner = "FAV" if fav_r > und_r else "UND" if und_r > fav_r else "DRAW"
    
    # Trend detection
    mid = len(sorted_fx) // 2
    if mid >= 2:
        first_half = sorted_fx[mid:]
        second_half = sorted_fx[:mid]
        
        fav_wins_first = sum(1 for fx in first_half if fx.outcome_from_fav_perspective(fav_is_home) == "W")
        fav_wins_second = sum(1 for fx in second_half if fx.outcome_from_fav_perspective(fav_is_home) == "W")
        
        if fav_wins_second > fav_wins_first + 1:
            dom.trend = "IMPROVING"
        elif fav_wins_first > fav_wins_second + 1:
            dom.trend = "DECLINING"
        else:
            dom.trend = "STABLE"
    
    return dom


def _scoring_pattern(fixtures: List[H2HFixtureDetail]) -> H2HScoringPattern:
    """Analyze scoring patterns in H2H fixtures."""
    if not fixtures:
        return H2HScoringPattern(confidence="INSUFFICIENT")
    
    total_goals = sum(fx.home_goals + fx.away_goals for fx in fixtures)
    btts_count = sum(1 for fx in fixtures if fx.home_goals > 0 and fx.away_goals > 0)
    clean_sheets = sum(1 for fx in fixtures if fx.home_goals == 0 or fx.away_goals == 0)
    
    avg = total_goals / len(fixtures)
    btts_rate = btts_count / len(fixtures)
    cs_rate = clean_sheets / len(fixtures)
    
    # Determine confidence based on sample size
    if len(fixtures) >= 10:
        confidence = "HIGH"
    elif len(fixtures) >= 7:
        confidence = "MEDIUM"
    elif len(fixtures) >= 4:
        confidence = "LOW"
    else:
        confidence = "INSUFFICIENT"
    
    if avg >= HIGH_SCORING_AVG:
        label = "HIGH_SCORING"
    elif avg <= LOW_SCORING_AVG:
        label = "LOW_SCORING"
    elif btts_rate >= BTTS_THRESHOLD:
        label = "BTTS"
    else:
        label = "MODERATE"
    
    return H2HScoringPattern(
        avg_goals=round(avg, 2),
        both_score_rate=round(btts_rate, 3),
        clean_sheet_rate=round(cs_rate, 3),
        label=label,
        confidence=confidence,
    )


def _calculate_h2h_score(
    overall: H2HRecord,
    at_venue: H2HRecord,
    recent: H2HRecord,
    dominance: H2HDominance,
    rtm: H2HRTM,
) -> Tuple[float, float]:
    """
    Calculate composite H2H score (0-100) and RTM-enhanced score.
    
    Weights:
    - Overall record: 20%
    - Venue-specific: 15%
    - Recent form: 20%
    - Psychological dominance: 25%
    - RTM bounce-back: 20%
    """
    # Traditional score
    score = 50.0
    
    # Overall record contribution (0-30 points)
    overall_score = (overall.fav_win_rate - 0.5) * 30
    score += overall_score
    
    # Venue-specific contribution (0-20 points)
    if at_venue.games >= 3:
        venue_score = (at_venue.fav_win_rate - 0.5) * 20
        score += venue_score
    
    # Recent form contribution (0-25 points)
    if recent.games > 0:
        recent_score = (recent.fav_wins / recent.games - 0.5) * 25
        score += recent_score
    
    # Psychological dominance adjustment (-15 to +10)
    if dominance.psychological_block:
        score -= 15
    elif dominance.streak_length >= 4 and dominance.streak_holder == "FAV":
        score += 10
    elif dominance.streak_length >= 4 and dominance.streak_holder == "UND":
        score -= 10
    
    # Recent winner adjustment (-5 to +5)
    if dominance.recent_winner == "FAV":
        score += 5
    elif dominance.recent_winner == "UND":
        score -= 5
    
    # Trend adjustment (-5 to +5)
    if dominance.trend == "IMPROVING":
        score += 5
    elif dominance.trend == "DECLINING":
        score -= 5
    
    traditional_score = max(0.0, min(100.0, score))
    
    # RTM-enhanced score
    rtm_score = traditional_score
    
    # RTM bounce-back adjustment
    if rtm.is_reliable:
        if rtm.bounce_back_rate > 0.55:
            # Underdog likely to bounce back - reduce favourite score
            rtm_score -= 15 * rtm.bounce_back_rate
        elif rtm.bounce_back_rate < 0.30:
            # Favourite likely to continue dominance - increase score
            rtm_score += 10
    
    # Psychological block from RTM
    if rtm.has_psychological_block:
        rtm_score -= 15
    
    # Alternation rate penalty (high alternation = unpredictable)
    if rtm.alternation_rate > 0.60:
        rtm_score -= 10
    
    rtm_score = max(0.0, min(100.0, rtm_score))
    
    return round(traditional_score, 1), round(rtm_score, 1)


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — MAIN ENGINE (UPDATED v4)
# ═══════════════════════════════════════════════════════════════

def run_h2h_deep_analyzer(
    fixtures: List[H2HFixtureDetail],
    fav_is_home: bool = True,
    include_rtm: bool = True,
    include_draw_boost: bool = True,
) -> H2HDeepAnalysis:
    """
    Full H2H deep analysis with optional RTM and draw boost.

    Args:
        fixtures: List of H2HFixtureDetail (any order, will be sorted)
        fav_is_home: True if today's home team is the favourite
        include_rtm: Whether to build H2H RTM matrix
        include_draw_boost: Whether to calculate draw boost (NEW v4)
    
    Returns:
        H2HDeepAnalysis with all metrics and weighted decision scores
    """
    result = H2HDeepAnalysis(total_games=len(fixtures))
    
    if len(fixtures) < MIN_H2H_GAMES:
        result.flags.append(
            f"Only {len(fixtures)} H2H games (min {MIN_H2H_GAMES}). "
            "Treated as NEUTRAL with reduced confidence."
        )
        result.h2h_label = "INSUFFICIENT_DATA"
        result.is_reliable = False
        result.confidence = "INSUFFICIENT"
        result.h2h_score = 30.0
        result.rtm_enhanced_score = 30.0
        return result

    result.is_reliable = True
    
    if len(fixtures) >= 10:
        result.confidence = "HIGH"
    elif len(fixtures) >= 7:
        result.confidence = "MEDIUM"
    else:
        result.confidence = "LOW"
    
    # Sort by date (oldest first for record building)
    sorted_fx = sorted(fixtures, key=lambda x: x.date)
    
    # ── Overall record ──
    result.overall = _build_record(sorted_fx, fav_is_home)
    
    # ── Venue-specific record ──
    venue_fx = [fx for fx in sorted_fx if fx.venue == "home"]
    if venue_fx:
        result.at_venue = _build_record(venue_fx, fav_is_home)
    
    # ── Recent record (last RECENCY_WINDOW games) ──
    recent_fx = sorted_fx[-RECENCY_WINDOW:] if len(sorted_fx) >= RECENCY_WINDOW else sorted_fx
    result.recent = _build_record(recent_fx, fav_is_home)
    
    # ── Scoring pattern ──
    result.scoring = _scoring_pattern(sorted_fx)
    
    # ── Dominance analysis ──
    result.dominance = _streak_analysis(sorted_fx, fav_is_home)
    
    # ── H2H RTM ─────────────────────────────────────────────
    if include_rtm:
        # Overall RTM
        result.h2h_rtm = _build_h2h_rtm(sorted_fx, fav_is_home)
        
        # Venue-specific RTM (if enough data)
        home_fx = [fx for fx in sorted_fx if fx.venue == "home"]
        away_fx = [fx for fx in sorted_fx if fx.venue == "away"]
        
        if len(home_fx) >= MIN_H2H_FOR_RTM:
            result.h2h_rtm_home = _build_h2h_rtm(home_fx, fav_is_home, venue_filter="home")
        if len(away_fx) >= MIN_H2H_FOR_RTM:
            result.h2h_rtm_away = _build_h2h_rtm(away_fx, fav_is_home, venue_filter="away")
    
    # ── NEW v4: Draw Boost Analysis ───────────────────────────
    if include_draw_boost:
        result.draw_boost = calculate_draw_boost(sorted_fx, fav_is_home)
        if result.draw_boost.draw_boost_factor > 1.02:
            result.flags.append(
                f"Elevated H2H draw rate: {result.overall.draw_rate:.0%} overall, "
                f"{result.draw_boost.recent_draw_rate:.0%} last 6 → draw boost {result.draw_boost.draw_boost_factor:.2f}x"
            )
    
    # ── Composite H2H Scores ──
    traditional_score, rtm_enhanced = _calculate_h2h_score(
        result.overall, result.at_venue, result.recent, result.dominance, result.h2h_rtm
    )
    result.h2h_score = traditional_score
    result.rtm_enhanced_score = rtm_enhanced
    
    # ── Label based on score ──
    if result.h2h_score >= 70:
        result.h2h_label = "FAV_DOMINANT"
    elif result.h2h_score >= 57:
        result.h2h_label = "FAV_EDGE"
    elif result.h2h_score >= 43:
        result.h2h_label = "NEUTRAL"
    elif result.h2h_score >= 30:
        result.h2h_label = "UND_EDGE"
    else:
        result.h2h_label = "UND_DOMINANT"
    
    # ── Generate flags for downstream ──
    if result.overall.draw_rate >= 0.40:
        result.flags.append(f"High H2H draw rate: {result.overall.draw_rate:.0%}")
    
    if result.draw_boost.draw_boost_factor >= 1.10:
        result.flags.append(f"Draw boost active: {result.draw_boost.draw_boost_factor:.2f}x ({result.draw_boost.draw_rate_class.value} draw rate)")
    
    if result.scoring.label == "HIGH_SCORING":
        result.flags.append(f"High-scoring H2H: avg {result.scoring.avg_goals:.1f} goals/game")
    elif result.scoring.label == "LOW_SCORING":
        result.flags.append(f"Low-scoring H2H: avg {result.scoring.avg_goals:.1f} goals/game")
    
    if result.scoring.both_score_rate >= BTTS_THRESHOLD:
        result.flags.append(f"BTTS in {result.scoring.both_score_rate:.0%} of H2H games")
    
    if result.overall.is_one_sided:
        result.flags.append("One-sided H2H record (>70% wins by one team)")
    
    if result.dominance.psychological_block:
        result.flags.append(
            f"Psychological block: underdog on "
            f"{result.dominance.streak_length}-match H2H run"
        )
    
    if result.dominance.trend == "IMPROVING":
        result.flags.append("H2H trend improving for favourite")
    elif result.dominance.trend == "DECLINING":
        result.flags.append("H2H trend declining for favourite")
    
    # RTM-specific flags
    if result.h2h_rtm.is_reliable:
        if result.h2h_rtm.bounce_back_rate > 0.55:
            result.flags.append(f"Underdog bounce-back threat ({result.h2h_rtm.bounce_back_rate:.0%} in H2H)")
        if result.h2h_rtm.has_psychological_block:
            result.flags.append(f"Psychological block: {result.h2h_rtm.max_streak}x{result.h2h_rtm.max_streak_direction} H2H streak")
        if result.h2h_rtm.alternation_rate > 0.65:
            result.flags.append("Unpredictable H2H pattern (high alternation rate)")
    
    return result


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — API INTEGRATION
# ═══════════════════════════════════════════════════════════════

def h2h_from_api_fixtures(
    api_fixtures: List[Dict],
    home_team_id: int,
    fav_is_home: bool = True,
    include_rtm: bool = True,
    include_draw_boost: bool = True,
) -> H2HDeepAnalysis:
    """
    Build H2HDeepAnalysis from raw API-Football /headtohead response.
    """
    details = []
    
    for fx in api_fixtures:
        status = fx.get("fixture", {}).get("status", {}).get("short", "")
        if status not in {"FT", "AET", "PEN"}:
            continue
        
        hg = fx["goals"].get("home")
        ag = fx["goals"].get("away")
        if hg is None or ag is None:
            continue
        
        is_home = fx["teams"]["home"]["id"] == home_team_id
        date_str = fx["fixture"].get("date", "")[:10]
        
        details.append(H2HFixtureDetail(
            home_team=fx["teams"]["home"]["name"],
            away_team=fx["teams"]["away"]["name"],
            home_goals=int(hg),
            away_goals=int(ag),
            venue="home" if is_home else "away",
            date=date_str,
            competition=fx.get("league", {}).get("name", ""),
        ))
    
    return run_h2h_deep_analyzer(details, fav_is_home, include_rtm, include_draw_boost)


def h2h_from_manual_input(
    h2h_data: List[Tuple[str, str, int, int, str]],
    fav_is_home: bool = True,
    include_rtm: bool = True,
    include_draw_boost: bool = True,
) -> H2HDeepAnalysis:
    """
    Build H2HDeepAnalysis from manual input.
    
    Args:
        h2h_data: List of (home_team, away_team, home_goals, away_goals, venue)
        fav_is_home: Whether favourite is home team
        include_rtm: Include RTM analysis
        include_draw_boost: Include draw boost analysis
    """
    details = []
    for i, (home, away, hg, ag, venue) in enumerate(h2h_data):
        details.append(H2HFixtureDetail(
            home_team=home,
            away_team=away,
            home_goals=hg,
            away_goals=ag,
            venue=venue,
            date=f"2000-{(i+1):02d}-01",
        ))
    
    return run_h2h_deep_analyzer(details, fav_is_home, include_rtm, include_draw_boost)


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════

@dataclass
class H2HAnalysisRequest:
    """Request for batch H2H analysis."""
    match_id: str
    fixtures: List[H2HFixtureDetail]
    fav_is_home: bool = True
    include_rtm: bool = True
    include_draw_boost: bool = True


def batch_h2h_analysis(
    requests: List[H2HAnalysisRequest],
    verbose: bool = False,
) -> Dict[str, H2HDeepAnalysis]:
    """
    Run H2H analysis on multiple fixtures.
    
    Args:
        requests: List of H2HAnalysisRequest objects
        verbose: Print progress
    
    Returns:
        Dictionary mapping match_id to H2HDeepAnalysis
    """
    results = {}
    
    for i, req in enumerate(requests):
        if verbose:
            print(f"  Analyzing H2H {i+1}/{len(requests)}: {req.match_id}")
        
        analysis = run_h2h_deep_analyzer(
            req.fixtures, 
            req.fav_is_home, 
            req.include_rtm,
            req.include_draw_boost
        )
        results[req.match_id] = analysis
    
    if verbose:
        reliable = sum(1 for a in results.values() if a.is_reliable)
        rtm_reliable = sum(1 for a in results.values() if a.h2h_rtm.is_reliable)
        draw_boost_active = sum(1 for a in results.values() if a.draw_boost.draw_boost_factor > 1.05)
        avg_score = sum(a.h2h_score for a in results.values()) / len(results) if results else 0
        print(f"\n  Batch summary: {len(results)} analyzed, {reliable} reliable, "
              f"{rtm_reliable} with RTM, {draw_boost_active} with draw boost, avg score={avg_score:.1f}")
    
    return results


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_h2h_score(analysis: H2HDeepAnalysis) -> float:
    """Extract H2H score (0-100)."""
    return analysis.h2h_score


def get_h2h_label(analysis: H2HDeepAnalysis) -> str:
    """Extract H2H label."""
    return analysis.h2h_label


def get_h2h_normalized_score(analysis: H2HDeepAnalysis) -> float:
    """Extract normalized score (0-1) for weighted decision."""
    return analysis.normalized_score


def is_h2h_reliable(analysis: H2HDeepAnalysis) -> bool:
    """Check if H2H analysis is reliable."""
    return analysis.is_reliable


def get_h2h_flags(analysis: H2HDeepAnalysis) -> List[str]:
    """Extract H2H flags for downstream modules."""
    return analysis.flags


def get_h2h_bounce_back(analysis: H2HDeepAnalysis) -> float:
    """Extract H2H bounce-back rate."""
    return analysis.h2h_rtm.bounce_back_rate


def has_psychological_block(analysis: H2HDeepAnalysis) -> bool:
    """Check if underdog has psychological block in H2H."""
    return analysis.h2h_rtm.has_psychological_block


def get_draw_boost_factor(analysis: H2HDeepAnalysis) -> float:
    """Extract draw boost factor for probability adjustment (NEW v4)."""
    return analysis.draw_boost.draw_boost_factor


def get_draw_boost_confidence(analysis: H2HDeepAnalysis) -> float:
    """Extract draw boost confidence (NEW v4)."""
    return analysis.draw_boost.confidence_score


# ═══════════════════════════════════════════════════════════════
# SECTION 10 — EXPORTS
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "H2HRTMConfidence",
    "DrawRateClass",  # NEW v4
    # Data classes
    "H2HFixtureDetail",
    "H2HRecord",
    "H2HRTM",
    "H2HScoringPattern",
    "H2HDominance",
    "DrawBoostAnalysis",  # NEW v4
    "H2HDeepAnalysis",
    "H2HAnalysisRequest",
    # Main functions
    "run_h2h_deep_analyzer",
    "h2h_from_api_fixtures",
    "h2h_from_manual_input",
    "batch_h2h_analysis",
    # Draw boost function (NEW v4)
    "calculate_draw_boost",
    # Convenience functions
    "get_h2h_score",
    "get_h2h_label",
    "get_h2h_normalized_score",
    "is_h2h_reliable",
    "get_h2h_flags",
    "get_h2h_bounce_back",
    "has_psychological_block",
    "get_draw_boost_factor",  # NEW v4
    "get_draw_boost_confidence",  # NEW v4
    # Constants
    "MIN_H2H_GAMES",
    "DOMINANCE_THRESHOLD",
    "RECENCY_WINDOW",
    "HIGH_SCORING_AVG",
    "LOW_SCORING_AVG",
    "BTTS_THRESHOLD",
    "MIN_H2H_FOR_RTM",
    "STREAK_BLOCK_THRESHOLD",
    # Draw boost constants (NEW v4)
    "DRAW_BOOST_MULTIPLIERS",
    "DRAW_BOOST_MAX_FACTOR",
    "DRAW_BOOST_MIN_FACTOR",
    "DRAW_RATE_NORMAL_MAX",
    "DRAW_RATE_ELEVATED_MAX",
    "DRAW_RATE_HIGH_MAX",
    "RECENT_DRAW_WEIGHT",
    "OVERALL_DRAW_WEIGHT",
    "MIN_DRAW_BOOST_SAMPLES",
    "HIGH_CONFIDENCE_SAMPLES",
]


# ═══════════════════════════════════════════════════════════════
# SECTION 11 — EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MODULE 27: HEAD-TO-HEAD DEEP ANALYZER v4 - TEST RUN")
    print("=" * 70)
    
    # Example 1: Machida vs Tokyo Verdy (Machida case with high draw rate)
    print("\n📊 EXAMPLE 1: Machida vs Tokyo Verdy (Elevated Draw Rate)")
    print("-" * 40)
    
    machida_fixtures = [
        H2HFixtureDetail(home_team="Machida", away_team="Tokyo Verdy", home_goals=1, away_goals=1, venue="home", date="2025-04-15"),
        H2HFixtureDetail(home_team="Tokyo Verdy", away_team="Machida", home_goals=2, away_goals=2, venue="away", date="2024-12-10"),
        H2HFixtureDetail(home_team="Machida", away_team="Tokyo Verdy", home_goals=0, away_goals=1, venue="home", date="2024-08-20"),
        H2HFixtureDetail(home_team="Tokyo Verdy", away_team="Machida", home_goals=1, away_goals=0, venue="away", date="2024-03-05"),
        H2HFixtureDetail(home_team="Machida", away_team="Tokyo Verdy", home_goals=1, away_goals=1, venue="home", date="2023-11-12"),
        H2HFixtureDetail(home_team="Tokyo Verdy", away_team="Machida", home_goals=0, away_goals=2, venue="away", date="2023-07-18"),
    ]
    
    analysis = run_h2h_deep_analyzer(machida_fixtures, fav_is_home=True, include_rtm=True, include_draw_boost=True)
    print(analysis.summary())
    
    print(f"\n🔢 DRAW BOOST DETAILS:")
    print(f"  Overall draw rate: {analysis.overall.draw_rate:.0%}")
    print(f"  Recent draw rate (last 6): {analysis.draw_boost.recent_draw_rate:.0%}")
    print(f"  Very recent draw rate (last 3): {analysis.draw_boost.very_recent_draw_rate:.0%}")
    print(f"  Draw rate class: {analysis.draw_boost.draw_rate_class.value}")
    print(f"  Draw boost factor: {analysis.draw_boost_factor:.2f}x")
    print(f"  Draw boost confidence: {analysis.draw_boost_confidence:.1%}")
    
    print(f"\n🔢 WEIGHTED DECISION SCORES:")
    print(f"  Dominance Score: {analysis.dominance_score:.3f}")
    print(f"  Fav Edge Score: {analysis.fav_edge_score:.3f}")
    print(f"  Draw Risk Score: {analysis.draw_risk_score:.3f}")
    print(f"  RTM Score: {analysis.rtm_score:.3f}")
    print(f"  Normalized Score: {analysis.normalized_score:.3f}")
    print(f"  Confidence Factor: {analysis.confidence_factor:.2f}")
    
    # Example 2: Barcelona vs Alavés (normal draw rate)
    print("\n" + "=" * 70)
    print("📊 EXAMPLE 2: Barcelona vs Alavés (Normal Draw Rate)")
    print("-" * 40)
    
    barca_fixtures = [
        H2HFixtureDetail(home_team="Barcelona", away_team="Alavés", home_goals=2, away_goals=0, venue="home", date="2025-02-15"),
        H2HFixtureDetail(home_team="Alavés", away_team="Barcelona", home_goals=1, away_goals=2, venue="away", date="2024-10-20"),
        H2HFixtureDetail(home_team="Barcelona", away_team="Alavés", home_goals=3, away_goals=1, venue="home", date="2024-04-10"),
        H2HFixtureDetail(home_team="Alavés", away_team="Barcelona", home_goals=0, away_goals=2, venue="away", date="2023-12-01"),
        H2HFixtureDetail(home_team="Barcelona", away_team="Alavés", home_goals=4, away_goals=0, venue="home", date="2023-08-25"),
    ]
    
    analysis2 = run_h2h_deep_analyzer(barca_fixtures, fav_is_home=True, include_draw_boost=True)
    print(analysis2.summary())
    print(f"\nDraw boost factor: {analysis2.draw_boost_factor:.2f}x (expected ~1.00x)")
    
    # Example 3: High draw rate case (Extreme)
    print("\n" + "=" * 70)
    print("📊 EXAMPLE 3: Draw-Heavy Rivalry (Extreme Draw Rate)")
    print("-" * 40)
    
    draw_heavy_fixtures = [
        H2HFixtureDetail(home_team="Team A", away_team="Team B", home_goals=1, away_goals=1, venue="home", date="2025-04-01"),
        H2HFixtureDetail(home_team="Team B", away_team="Team A", home_goals=0, away_goals=0, venue="away", date="2025-03-01"),
        H2HFixtureDetail(home_team="Team A", away_team="Team B", home_goals=2, away_goals=2, venue="home", date="2025-02-01"),
        H2HFixtureDetail(home_team="Team B", away_team="Team A", home_goals=1, away_goals=1, venue="away", date="2025-01-01"),
        H2HFixtureDetail(home_team="Team A", away_team="Team B", home_goals=0, away_goals=0, venue="home", date="2024-12-01"),
        H2HFixtureDetail(home_team="Team B", away_team="Team A", home_goals=2, away_goals=2, venue="away", date="2024-11-01"),
    ]
    
    analysis3 = run_h2h_deep_analyzer(draw_heavy_fixtures, fav_is_home=True, include_draw_boost=True)
    print(analysis3.summary())
    print(f"\nDraw boost factor: {analysis3.draw_boost_factor:.2f}x (expected ~1.25x)")
    
    # Leg data for M11
    print("\n" + "=" * 70)
    print("LEG DATA FOR M11:")
    print("=" * 70)
    leg_data = analysis.to_leg_data()
    for key, value in leg_data.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("MODULE 27 v4 READY FOR PRODUCTION")
    print("=" * 70)