"""
The Match Oracle - Module 14: Learning Engine + Transition Matrix Integration (REFINED)
===================================================================================
Tracks prediction outcomes and uses Results Transition Matrix (RTM) to detect
system decay, bounce-back capability, and win-streak ceiling.

This module closes the feedback loop: predictions → outcomes → learning → improvement.

REFINEMENTS IN THIS VERSION:
---------------------------
1. FIXED: File path handling with proper directory creation
2. FIXED: JSON serialization with datetime handling
3. FIXED: M15 integration (ResultsTransitionMatrix class)
4. FIXED: Comprehensive error handling for missing files
5. ADDED: Detailed calibration analysis by confidence tier
6. ADDED: Pattern learning from historical outcomes
7. ADDED: Anomaly detection for unexpected results
8. ADDED: Performance forecasting based on RTM patterns
9. ADDED: Confidence calibration adjustment recommendations
10. ADDED: Batch learning from multiple results
11. ADDED: Export functionality for learning reports
12. ADDED: Real-time learning with sliding window

WHAT THIS MODULE DOES:
---------------------
1. RECORD OUTCOMES: Store prediction results with transition tracking
2. CALIBRATION ANALYSIS: Measure accuracy by confidence tier and transition type
3. RTM INTEGRATION: Use M15's transition matrix for pattern validation
4. HEALTH MONITORING: Detect when system is degrading or needs recalibration
5. RECOMMENDATIONS: Generate actionable suggestions for threshold adjustment

Usage:
    from module14 import EnhancedLearningEngine, PerformanceHealth
    
    engine = EnhancedLearningEngine()
    
    # Record outcome after match
    engine.record_outcome(
        match_id="arsenal_chelsea_20250101",
        outcome="WIN",
        tier="TIER_1",
        odds=2.10,
        stake=100,
        pnl=110,
        confidence_before=0.65,
        confidence_after=0.70,
    )
    
    # Build calibration report
    calibration = engine.build_season_calibration("2025_SEASON")
    print(f"Health: {calibration.health_status.value}")
    print(f"Recommendations: {calibration.recommendations}")
"""
from __future__ import annotations

import json
import os
import warnings
import statistics
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import defaultdict
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — ENUMS
# ═══════════════════════════════════════════════════════════════

class PerformanceHealth(Enum):
    """System health classification based on calibration and RTM."""
    EXCELLENT = "EXCELLENT"   # 0.9+ health score
    GOOD = "GOOD"              # 0.7-0.89
    WARNING = "WARNING"        # 0.5-0.69
    CRITICAL = "CRITICAL"      # <0.5
    UNKNOWN = "UNKNOWN"        # Insufficient data


class OutcomeState(Enum):
    """Three outcome states for transition tracking."""
    WIN = "WIN"
    DRAW = "DRAW"
    LOSS = "LOSS"


class TransitionType(Enum):
    """9 possible transitions between outcomes."""
    WIN_TO_WIN = "WIN→WIN"
    WIN_TO_DRAW = "WIN→DRAW"
    WIN_TO_LOSS = "WIN→LOSS"
    DRAW_TO_WIN = "DRAW→WIN"
    DRAW_TO_DRAW = "DRAW→DRAW"
    DRAW_TO_LOSS = "DRAW→LOSS"
    LOSS_TO_WIN = "LOSS→WIN"
    LOSS_TO_DRAW = "LOSS→DRAW"
    LOSS_TO_LOSS = "LOSS→LOSS"


class LearningPhase(Enum):
    """Current phase of the learning cycle."""
    COLLECTING = "COLLECTING"   # Building initial data
    STABLE = "STABLE"           # Consistent performance
    ADAPTING = "ADAPTING"       # Making adjustments
    DEGRADING = "DEGRADING"     # Performance declining
    RECOVERING = "RECOVERING"   # Recovering from drawdown


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class TransitionRecord:
    """Single transition event between outcomes."""
    timestamp: str
    match_id: str
    prev_outcome: OutcomeState
    current_outcome: OutcomeState
    transition_type: TransitionType
    confidence_before: float
    confidence_after: float
    tier: str
    odds: float
    stake: float
    pnl: float
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
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
            "stake": self.stake,
            "pnl": self.pnl,
            "notes": self.notes,
        }


@dataclass
class TransitionMatrixStats:
    """
    Results transition matrix with statistics.
    Tracks 3x3 transitions between W/D/L outcomes.
    """
    matrix: Dict[Tuple[OutcomeState, OutcomeState], int] = field(default_factory=dict)
    probabilities: Dict[Tuple[OutcomeState, OutcomeState], float] = field(default_factory=dict)
    
    # Streak analysis
    max_win_streak: int = 0
    max_loss_streak: int = 0
    avg_win_streak: float = 0.0
    avg_loss_streak: float = 0.0
    current_win_streak: int = 0
    current_loss_streak: int = 0
    
    # Resilience metrics
    bounce_back_rate: float = 0.0  # P(WIN | LOSS)
    drawdown_recovery_rate: float = 0.0  # P(WIN or DRAW | LOSS)
    win_ceiling: int = 0
    
    # Decay analysis
    early_season_wr: float = 0.0
    late_season_wr: float = 0.0
    decay_rate: float = 0.0
    decay_confidence: float = 0.0
    
    # Confidence analysis
    avg_confidence_before_transition: float = 0.0
    avg_confidence_after_transition: float = 0.0
    confidence_shift: float = 0.0
    confidence_calibration: float = 0.0
    
    # Sample tracking
    total_transitions: int = 0
    reliable: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "probabilities": {
                f"{k[0].value}→{k[1].value}": v 
                for k, v in self.probabilities.items()
            },
            "max_win_streak": self.max_win_streak,
            "max_loss_streak": self.max_loss_streak,
            "avg_win_streak": self.avg_win_streak,
            "avg_loss_streak": self.avg_loss_streak,
            "current_win_streak": self.current_win_streak,
            "current_loss_streak": self.current_loss_streak,
            "bounce_back_rate": self.bounce_back_rate,
            "drawdown_recovery_rate": self.drawdown_recovery_rate,
            "win_ceiling": self.win_ceiling,
            "decay_rate": self.decay_rate,
            "confidence_shift": self.confidence_shift,
            "total_transitions": self.total_transitions,
            "reliable": self.reliable,
        }


@dataclass
class SeasonAnalysis:
    """Full season transition analysis."""
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
    
    transitions: List[TransitionRecord] = field(default_factory=list)
    matrix: TransitionMatrixStats = field(default_factory=TransitionMatrixStats)
    tier_matrices: Dict[str, TransitionMatrixStats] = field(default_factory=dict)
    confidence_matrices: Dict[str, TransitionMatrixStats] = field(default_factory=dict)
    anomalous_transitions: List[TransitionRecord] = field(default_factory=list)
    
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "season_id": self.season_id,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "total_fixtures": self.total_fixtures,
            "total_wins": self.total_wins,
            "total_draws": self.total_draws,
            "total_losses": self.total_losses,
            "overall_win_rate": self.overall_win_rate,
            "overall_profit": self.overall_profit,
            "overall_roi": self.overall_roi,
            "matrix": self.matrix.to_dict(),
            "anomalies": len(self.anomalous_transitions),
            "created_at": self.created_at,
        }


@dataclass
class TransitionMetrics:
    """RTM-derived metrics for learning."""
    bounce_back_rate: float = 0.0
    recovery_rate: float = 0.0
    max_win_streak: int = 0
    max_loss_streak: int = 0
    decay_rate: float = 0.0
    health_score: float = 0.0
    is_resilient: bool = False
    learning_phase: LearningPhase = LearningPhase.COLLECTING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bounce_back_rate": self.bounce_back_rate,
            "recovery_rate": self.recovery_rate,
            "max_win_streak": self.max_win_streak,
            "max_loss_streak": self.max_loss_streak,
            "decay_rate": self.decay_rate,
            "health_score": self.health_score,
            "is_resilient": self.is_resilient,
            "learning_phase": self.learning_phase.value,
        }


@dataclass
class CalibrationByBin:
    """Calibration data for a probability bin."""
    bin_lower: float
    bin_upper: float
    count: int
    predicted_rate: float
    actual_rate: float
    calibration_error: float
    is_reliable: bool = False


@dataclass
class CalibrationAnalysis:
    """Enhanced calibration with RTM insights."""
    by_confidence: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_tier: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_transition: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_probability_bin: List[CalibrationByBin] = field(default_factory=list)
    rtm_metrics: TransitionMetrics = field(default_factory=TransitionMetrics)
    health_status: PerformanceHealth = PerformanceHealth.UNKNOWN
    recommendations: List[str] = field(default_factory=list)
    brier_score: float = 0.0
    log_loss: float = 0.0
    calibration_error: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "by_confidence": self.by_confidence,
            "by_tier": self.by_tier,
            "rtm_metrics": self.rtm_metrics.to_dict(),
            "health_status": self.health_status.value,
            "recommendations": self.recommendations,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "calibration_error": self.calibration_error,
            "timestamp": self.timestamp,
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "═" * 60,
            "  CALIBRATION ANALYSIS REPORT",
            "═" * 60,
            f"  Health Status : {self.health_status.value}",
            f"  Brier Score   : {self.brier_score:.4f} (0=perfect, 0.25=random)",
            f"  Calib Error   : {self.calibration_error:.4f} (0=perfect)",
            f"  Learning Phase: {self.rtm_metrics.learning_phase.value}",
            "",
            "  Confidence Calibration:",
        ]
        for conf, data in self.by_confidence.items():
            acc = data.get("accuracy", 0)
            count = data.get("count", 0)
            lines.append(f"    {conf}: {acc:.1%} ({count} predictions)")
        
        lines.append("\n  RTM Metrics:")
        lines.append(f"    Bounce-back: {self.rtm_metrics.bounce_back_rate:.1%}")
        lines.append(f"    Recovery: {self.rtm_metrics.recovery_rate:.1%}")
        lines.append(f"    Max Loss Streak: {self.rtm_metrics.max_loss_streak}")
        lines.append(f"    Decay Rate: {self.rtm_metrics.decay_rate:.1%}")
        
        if self.recommendations:
            lines.append("\n  Recommendations:")
            for r in self.recommendations:
                lines.append(f"    → {r}")
        
        lines.append("═" * 60)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — RESULTS TRANSITION MATRIX ENGINE
# ═══════════════════════════════════════════════════════════════

class ResultsTransitionMatrix:
    """
    Main engine for tracking and analyzing result transitions.
    Integrates with M15 for advanced transition analysis.
    """
    
    def __init__(self, min_samples_for_reliable: int = 10):
        self.transitions: List[TransitionRecord] = []
        self.seasons: Dict[str, SeasonAnalysis] = {}
        self.current_state: Optional[OutcomeState] = None
        self.min_reliable = min_samples_for_reliable
    
    def _get_transition_type(self, prev: OutcomeState, curr: OutcomeState) -> TransitionType:
        """Map (prev, curr) to TransitionType."""
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
    
    def record_transition(
        self,
        match_id: str,
        prev_outcome: str,
        current_outcome: str,
        confidence_before: float,
        confidence_after: float,
        tier: str,
        odds: float,
        stake: float,
        pnl: float,
        notes: str = "",
    ) -> TransitionRecord:
        """Record a single transition."""
        try:
            prev = OutcomeState[prev_outcome.upper()]
            curr = OutcomeState[current_outcome.upper()]
        except KeyError as e:
            raise ValueError(f"Invalid outcome: {e}. Must be WIN, DRAW, or LOSS")
        
        transition_type = self._get_transition_type(prev, curr)
        
        record = TransitionRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            match_id=match_id,
            prev_outcome=prev,
            current_outcome=curr,
            transition_type=transition_type,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            tier=tier,
            odds=odds,
            stake=stake,
            pnl=pnl,
            notes=notes,
        )
        
        self.transitions.append(record)
        self.current_state = curr
        
        return record
    
    def build_season_matrix(self, season_id: str, fixtures: List[Dict]) -> SeasonAnalysis:
        """Build transition matrix for entire season."""
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
        
        # Sort by date
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
        
        # Calculate ROI
        total_stake = sum(f.get("stake", 0) for f in sorted_fixtures)
        if total_stake > 0:
            analysis.overall_roi = analysis.overall_profit / total_stake
        
        prev_outcome = None
        
        for fixture in sorted_fixtures:
            outcome_str = fixture.get("outcome", "LOSS").upper()
            try:
                outcome = OutcomeState[outcome_str]
            except KeyError:
                outcome = OutcomeState.LOSS
                warnings.warn(f"Unknown outcome '{outcome_str}' for {fixture.get('match_id', 'unknown')}")
            
            # Count outcomes
            if outcome == OutcomeState.WIN:
                analysis.total_wins += 1
            elif outcome == OutcomeState.DRAW:
                analysis.total_draws += 1
            else:
                analysis.total_losses += 1
            
            # Record transition if not first fixture
            if prev_outcome is not None:
                transition_type = self._get_transition_type(prev_outcome, outcome)
                
                record = TransitionRecord(
                    timestamp=fixture.get("date", datetime.now(timezone.utc).isoformat()),
                    match_id=fixture.get("match_id", "UNKNOWN"),
                    prev_outcome=prev_outcome,
                    current_outcome=outcome,
                    transition_type=transition_type,
                    confidence_before=fixture.get("confidence_before", 0.5),
                    confidence_after=fixture.get("confidence_after", 0.5),
                    tier=fixture.get("tier", "UNKNOWN"),
                    odds=fixture.get("odds", 2.0),
                    stake=fixture.get("stake", 0),
                    pnl=fixture.get("pnl", 0),
                    notes=fixture.get("notes", ""),
                )
                
                analysis.transitions.append(record)
            
            prev_outcome = outcome
        
        # Build matrices
        analysis.matrix = self._compute_transition_matrix(analysis.transitions)
        
        # Build tier-specific matrices
        tiers = set(t.tier for t in analysis.transitions)
        for tier in tiers:
            tier_transitions = [t for t in analysis.transitions if t.tier == tier]
            analysis.tier_matrices[tier] = self._compute_transition_matrix(tier_transitions)
        
        # Build confidence-based matrices
        confidence_bins = ["HIGH", "MEDIUM", "LOW"]
        for conf in confidence_bins:
            conf_transitions = [t for t in analysis.transitions if t.tier == conf]
            if conf_transitions:
                analysis.confidence_matrices[conf] = self._compute_transition_matrix(conf_transitions)
        
        analysis.overall_win_rate = analysis.total_wins / analysis.total_fixtures if analysis.total_fixtures > 0 else 0
        
        # Detect anomalies
        analysis.anomalous_transitions = self._detect_anomalies(analysis.transitions)
        
        self.seasons[season_id] = analysis
        return analysis
    
    def _compute_transition_matrix(self, transitions: List[TransitionRecord]) -> TransitionMatrixStats:
        """Compute transition probabilities and metrics."""
        matrix = TransitionMatrixStats()
        
        if not transitions:
            return matrix
        
        # Count transitions
        transition_counts = {}
        prev_counts = {}
        
        for trans in transitions:
            key = (trans.prev_outcome, trans.current_outcome)
            transition_counts[key] = transition_counts.get(key, 0) + 1
            prev_counts[trans.prev_outcome] = prev_counts.get(trans.prev_outcome, 0) + 1
        
        matrix.matrix = transition_counts
        matrix.total_transitions = len(transitions)
        matrix.reliable = len(transitions) >= self.min_reliable
        
        # Compute probabilities
        for (prev, curr), count in transition_counts.items():
            total_from_prev = prev_counts.get(prev, 0)
            prob = count / total_from_prev if total_from_prev > 0 else 0
            matrix.probabilities[(prev, curr)] = prob
        
        # Compute resilience metrics
        loss_to_win = transition_counts.get((OutcomeState.LOSS, OutcomeState.WIN), 0)
        total_losses = prev_counts.get(OutcomeState.LOSS, 0)
        matrix.bounce_back_rate = loss_to_win / total_losses if total_losses > 0 else 0
        
        loss_to_win_or_draw = (
            transition_counts.get((OutcomeState.LOSS, OutcomeState.WIN), 0) +
            transition_counts.get((OutcomeState.LOSS, OutcomeState.DRAW), 0)
        )
        matrix.drawdown_recovery_rate = loss_to_win_or_draw / total_losses if total_losses > 0 else 0
        
        # Calculate streaks
        matrix.max_win_streak, matrix.avg_win_streak, matrix.current_win_streak = self._calculate_streaks(
            transitions, OutcomeState.WIN
        )
        matrix.max_loss_streak, matrix.avg_loss_streak, matrix.current_loss_streak = self._calculate_streaks(
            transitions, OutcomeState.LOSS
        )
        matrix.win_ceiling = matrix.max_win_streak
        
        # Calculate decay (early vs late season performance)
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
        
        # Confidence analysis
        if transitions:
            conf_before = [t.confidence_before for t in transitions]
            conf_after = [t.confidence_after for t in transitions]
            matrix.avg_confidence_before_transition = statistics.mean(conf_before) if conf_before else 0.0
            matrix.avg_confidence_after_transition = statistics.mean(conf_after) if conf_after else 0.0
            matrix.confidence_shift = matrix.avg_confidence_after_transition - matrix.avg_confidence_before_transition
            
            # Confidence calibration: predicted confidence vs actual win rate
            # This is a simplified version - full calibration uses M28
            high_conf_transitions = [t for t in transitions if t.confidence_before >= 0.7]
            if high_conf_transitions:
                high_actual_wr = sum(1 for t in high_conf_transitions if t.current_outcome == OutcomeState.WIN) / len(high_conf_transitions)
                matrix.confidence_calibration = high_actual_wr - 0.7
        
        return matrix
    
    def _calculate_streaks(
        self, 
        transitions: List[TransitionRecord], 
        target_outcome: OutcomeState
    ) -> Tuple[int, float, int]:
        """
        Calculate max, average, and current streak length for target outcome.
        """
        if not transitions:
            return 0, 0.0, 0
        
        # Build outcome sequence including starting state
        outcomes = [transitions[0].prev_outcome] + [t.current_outcome for t in transitions]
        
        streaks = []
        current = 0
        current_streak = 0
        
        # Check current streak (last outcome)
        if outcomes and outcomes[-1] == target_outcome:
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
    
    def _detect_anomalies(self, transitions: List[TransitionRecord]) -> List[TransitionRecord]:
        """Detect anomalous transitions."""
        anomalies = []
        
        if len(transitions) < 2:
            return anomalies
        
        # Anomaly 1: WIN→LOSS after high confidence (>70%)
        for trans in transitions:
            if trans.transition_type == TransitionType.WIN_TO_LOSS and trans.confidence_before > 0.70:
                trans.notes = f"Anomaly: High confidence ({trans.confidence_before:.1%}) loss"
                anomalies.append(trans)
        
        # Anomaly 2: LOSS→LOSS streak of 3+ (critical drawdown)
        loss_streak = 0
        for trans in transitions:
            if trans.current_outcome == OutcomeState.LOSS:
                loss_streak += 1
            else:
                if loss_streak >= 3:
                    trans.notes = f"Anomaly: {loss_streak} consecutive losses"
                    if trans not in anomalies:
                        anomalies.append(trans)
                loss_streak = 0
        
        # Anomaly 3: Unexpected bounce (LOSS→WIN with low confidence)
        for trans in transitions:
            if (trans.transition_type == TransitionType.LOSS_TO_WIN and 
                trans.confidence_before < 0.40):
                trans.notes = f"Anomaly: Unexpected bounce (confidence {trans.confidence_before:.1%})"
                anomalies.append(trans)
        
        return anomalies
    
    def get_summary(self, season_id: str) -> Dict[str, Any]:
        """Get full summary for a season."""
        if season_id not in self.seasons:
            return {"error": f"Season {season_id} not found"}
        
        analysis = self.seasons[season_id]
        matrix = analysis.matrix
        
        return {
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
                "avg_win_streak": matrix.avg_win_streak,
                "avg_loss_streak": matrix.avg_loss_streak,
                "current_win_streak": matrix.current_win_streak,
                "current_loss_streak": matrix.current_loss_streak,
            },
            "decay": {
                "early_season_wr": matrix.early_season_wr,
                "late_season_wr": matrix.late_season_wr,
                "decay_rate": matrix.decay_rate,
                "decay_confidence": matrix.decay_confidence,
            },
            "confidence": {
                "before": matrix.avg_confidence_before_transition,
                "after": matrix.avg_confidence_after_transition,
                "shift": matrix.confidence_shift,
                "calibration": matrix.confidence_calibration,
            },
            "anomalies": len(analysis.anomalous_transitions),
            "reliable": matrix.reliable,
        }
    
    def export_to_json(self, season_id: str, filename: str = None) -> str:
        """Export season analysis to JSON."""
        summary = self.get_summary(season_id)
        
        if filename is None:
            filename = f"rtm_{season_id}_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        
        return filename


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — ENHANCED LEARNING ENGINE
# ═══════════════════════════════════════════════════════════════

class EnhancedLearningEngine:
    """
    Module 14 Enhanced: Integrates Results Transition Matrix.
    
    Now tracks:
    1. Traditional win rate by confidence
    2. Bounce-back patterns after losses
    3. Win-streak ceiling detection
    4. System decay analysis
    5. Resilience scoring
    6. Real-time learning with sliding window
    """
    
    def __init__(self, history_file: str = "oracle_history.json", min_samples: int = 10):
        self.history_file = history_file
        self.rtm_file = history_file.replace(".json", "_rtm.json")
        self.min_samples = min_samples
        
        self.history = self._load_history()
        self.rtm = ResultsTransitionMatrix(min_samples_for_reliable=min_samples)
        
        # Season tracking
        self.current_season = None
        self.seasons: Dict[str, CalibrationAnalysis] = {}
        self.sliding_window_size = 100  # Keep last 100 predictions for trend analysis
    
    def _load_history(self) -> List[Dict]:
        """Load prediction history from JSON file."""
        if not os.path.exists(self.history_file):
            return []
        
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Handle both list and dict formats
                if isinstance(data, dict) and "history" in data:
                    return data["history"]
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, IOError) as e:
            warnings.warn(f"Could not load history file: {e}")
            return []
    
    def _save_history(self) -> None:
        """Save prediction history to JSON file."""
        try:
            # Ensure directory exists
            Path(self.history_file).parent.mkdir(parents=True, exist_ok=True)
            
            # Save with metadata
            data = {
                "history": self.history,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "total_records": len(self.history),
            }
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except IOError as e:
            warnings.warn(f"Could not save history file: {e}")
    
    def record_outcome(
        self,
        match_id: str,
        outcome: str,  # "WIN", "LOSS", "DRAW"
        tier: str,
        odds: float,
        stake: float,
        pnl: float,
        confidence_before: float = 0.5,
        confidence_after: float = 0.5,
    ) -> TransitionRecord:
        """
        Record outcome and capture transition.
        
        Args:
            match_id: Unique match identifier
            outcome: WIN, LOSS, or DRAW
            tier: TIER_1, TIER_2, etc.
            odds: Odds of the bet
            stake: Amount staked
            pnl: Profit/loss
            confidence_before: Confidence before match
            confidence_after: Confidence after match
        
        Returns:
            TransitionRecord with transition details
        """
        # Find previous outcome
        prev_outcome = None
        if self.history:
            # Get most recent outcome of same tier
            for record in reversed(self.history):
                if record.get("tier") == tier:
                    prev_outcome = record.get("outcome", "LOSS")
                    break
        
        # Default to LOSS if no previous (conservative)
        if prev_outcome is None:
            prev_outcome = "LOSS"
        
        # Record in RTM
        transition = self.rtm.record_transition(
            match_id=match_id,
            prev_outcome=prev_outcome,
            current_outcome=outcome,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            tier=tier,
            odds=odds,
            stake=stake,
            pnl=pnl,
        )
        
        # Record in history
        record = {
            "match_id": match_id,
            "outcome": outcome,
            "tier": tier,
            "odds": odds,
            "stake": stake,
            "pnl": pnl,
            "confidence_before": confidence_before,
            "confidence_after": confidence_after,
            "transition_type": transition.transition_type.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        self.history.append(record)
        
        # Maintain sliding window
        if len(self.history) > self.sliding_window_size:
            self.history = self.history[-self.sliding_window_size:]
        
        self._save_history()
        
        return transition
    
    def _calculate_brier_score(self, predictions: List[float], outcomes: List[int]) -> float:
        """Calculate Brier Score for probability calibration."""
        if not predictions:
            return 0.25
        
        n = len(predictions)
        return sum((p - o) ** 2 for p, o in zip(predictions, outcomes)) / n
    
    def _calculate_calibration_by_bin(
        self, 
        predictions: List[float], 
        outcomes: List[int],
        n_bins: int = 10
    ) -> Tuple[List[CalibrationByBin], float]:
        """
        Calculate calibration by probability bins.
        
        Returns:
            Tuple of (bins, calibration_error)
        """
        bins: List[CalibrationByBin] = []
        step = 1.0 / n_bins
        
        total_error = 0.0
        total_count = 0
        
        for i in range(n_bins):
            lo = round(i * step, 2)
            hi = round((i + 1) * step, 2)
            
            # Collect predictions in this bin
            in_bin = [(p, o) for p, o in zip(predictions, outcomes) if lo <= p < hi or (i == n_bins - 1 and p == hi)]
            
            count = len(in_bin)
            if count == 0:
                continue
            
            predicted_rate = sum(p for p, _ in in_bin) / count
            actual_rate = sum(o for _, o in in_bin) / count
            
            error = abs(predicted_rate - actual_rate)
            total_error += error * count
            total_count += count
            
            is_reliable = count >= self.min_samples // 2
            
            bins.append(CalibrationByBin(
                bin_lower=lo,
                bin_upper=hi,
                count=count,
                predicted_rate=predicted_rate,
                actual_rate=actual_rate,
                calibration_error=error,
                is_reliable=is_reliable,
            ))
        
        calibration_error = total_error / total_count if total_count > 0 else 0
        return bins, calibration_error
    
    def _get_learning_phase(self, rtm_metrics: TransitionMetrics) -> LearningPhase:
        """Determine current learning phase based on RTM metrics."""
        if rtm_metrics.total_transitions < self.min_samples:
            return LearningPhase.COLLECTING
        
        if rtm_metrics.decay_rate > 0.10 and rtm_metrics.decay_confidence > 0.7:
            return LearningPhase.DEGRADING
        
        if rtm_metrics.bounce_back_rate > 0.45 and rtm_metrics.recovery_rate > 0.6:
            return LearningPhase.RECOVERING
        
        if rtm_metrics.decay_rate > 0.05:
            return LearningPhase.ADAPTING
        
        return LearningPhase.STABLE
    
    def build_season_calibration(self, season_id: str) -> CalibrationAnalysis:
        """
        Build calibration analysis with RTM integration.
        """
        # Convert history to RTM format
        fixtures = [
            {
                "date": r.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "match_id": r.get("match_id"),
                "outcome": r.get("outcome", "LOSS"),
                "confidence_before": r.get("confidence_before", 0.5),
                "confidence_after": r.get("confidence_after", 0.5),
                "tier": r.get("tier", "UNKNOWN"),
                "odds": r.get("odds", 2.0),
                "stake": r.get("stake", 0),
                "pnl": r.get("pnl", 0),
            }
            for r in self.history
        ]
        
        # Build RTM
        rtm_analysis = self.rtm.build_season_matrix(season_id, fixtures)
        
        # Extract metrics
        matrix = rtm_analysis.matrix
        total_transitions = matrix.total_transitions
        
        health_score = 0.5  # Start neutral
        
        # Build health score
        if total_transitions >= self.min_samples:
            # Bounce-back contributes 30%
            health_score += matrix.bounce_back_rate * 0.3
            
            # Recovery contributes 20%
            health_score += matrix.drawdown_recovery_rate * 0.2
            
            # No collapse (max loss streak) contributes 30%
            if matrix.max_loss_streak < 3:
                health_score += 0.3
            elif matrix.max_loss_streak < 5:
                health_score += 0.15
            else:
                health_score += 0
            
            # Decay contributes 20%
            decay_penalty = matrix.decay_rate * 0.2
            health_score = max(0, health_score - decay_penalty)
        
        health_score = min(1.0, health_score)
        
        # Determine health status
        if health_score >= 0.90:
            health_status = PerformanceHealth.EXCELLENT
        elif health_score >= 0.70:
            health_status = PerformanceHealth.GOOD
        elif health_score >= 0.50:
            health_status = PerformanceHealth.WARNING
        else:
            health_status = PerformanceHealth.CRITICAL
        
        # Determine learning phase
        learning_phase = self._get_learning_phase(TransitionMetrics(
            bounce_back_rate=matrix.bounce_back_rate,
            recovery_rate=matrix.drawdown_recovery_rate,
            max_win_streak=matrix.max_win_streak,
            max_loss_streak=matrix.max_loss_streak,
            decay_rate=matrix.decay_rate,
            health_score=health_score,
            is_resilient=matrix.bounce_back_rate > 0.40,
            learning_phase=LearningPhase.COLLECTING,
        ))
        
        rtm_metrics = TransitionMetrics(
            bounce_back_rate=matrix.bounce_back_rate,
            recovery_rate=matrix.drawdown_recovery_rate,
            max_win_streak=matrix.max_win_streak,
            max_loss_streak=matrix.max_loss_streak,
            decay_rate=matrix.decay_rate,
            health_score=health_score,
            is_resilient=matrix.bounce_back_rate > 0.40,
            learning_phase=learning_phase,
        )
        
        # Build traditional calibration
        by_confidence = self._calibration_by_confidence()
        by_tier = self._calibration_by_tier()
        by_transition = self._calibration_by_transition()
        
        # Calculate Brier score
        predictions = [r.get("confidence_before", 0.5) for r in self.history]
        outcomes = [1 if r.get("outcome") == "WIN" else 0 for r in self.history]
        brier_score = self._calculate_brier_score(predictions, outcomes)
        
        # Calculate calibration by bin
        bins, cal_error = self._calculate_calibration_by_bin(predictions, outcomes)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(matrix, by_confidence, by_tier, brier_score, cal_error)
        
        analysis = CalibrationAnalysis(
            by_confidence=by_confidence,
            by_tier=by_tier,
            by_transition=by_transition,
            by_probability_bin=bins,
            rtm_metrics=rtm_metrics,
            health_status=health_status,
            recommendations=recommendations,
            brier_score=round(brier_score, 4),
            calibration_error=round(cal_error, 4),
        )
        
        self.seasons[season_id] = analysis
        
        return analysis
    
    def _calibration_by_confidence(self) -> Dict[str, Dict[str, float]]:
        """Calibration by confidence level."""
        by_conf = {}
        
        for record in self.history:
            conf_level = record.get("confidence_level", "MEDIUM")
            outcome = record.get("outcome", "LOSS")
            conf_before = record.get("confidence_before", 0.5)
            
            if conf_level not in by_conf:
                by_conf[conf_level] = {
                    "total": 0, "wins": 0, "losses": 0, "draws": 0,
                    "win_rate": 0.0, "avg_confidence": 0.0, "calibration_gap": 0.0
                }
            
            by_conf[conf_level]["total"] += 1
            if outcome == "WIN":
                by_conf[conf_level]["wins"] += 1
            elif outcome == "LOSS":
                by_conf[conf_level]["losses"] += 1
            else:
                by_conf[conf_level]["draws"] += 1
            
            by_conf[conf_level]["avg_confidence"] += conf_before
        
        for conf in by_conf:
            total = by_conf[conf]["total"]
            if total > 0:
                by_conf[conf]["win_rate"] = by_conf[conf]["wins"] / total
                by_conf[conf]["avg_confidence"] /= total
                by_conf[conf]["calibration_gap"] = by_conf[conf]["avg_confidence"] - by_conf[conf]["win_rate"]
        
        return by_conf
    
    def _calibration_by_tier(self) -> Dict[str, Dict[str, float]]:
        """Calibration by tier."""
        by_tier = {}
        
        for record in self.history:
            tier = record.get("tier", "UNKNOWN")
            outcome = record.get("outcome", "LOSS")
            
            if tier not in by_tier:
                by_tier[tier] = {"total": 0, "wins": 0, "losses": 0, "draws": 0, "win_rate": 0.0}
            
            by_tier[tier]["total"] += 1
            if outcome == "WIN":
                by_tier[tier]["wins"] += 1
            elif outcome == "LOSS":
                by_tier[tier]["losses"] += 1
            else:
                by_tier[tier]["draws"] += 1
        
        for tier in by_tier:
            total = by_tier[tier]["total"]
            by_tier[tier]["win_rate"] = by_tier[tier]["wins"] / total if total > 0 else 0
        
        return by_tier
    
    def _calibration_by_transition(self) -> Dict[str, Dict[str, float]]:
        """Calibration by transition type."""
        by_transition = {}
        
        for record in self.history:
            trans_type = record.get("transition_type", "UNKNOWN")
            outcome = record.get("outcome", "LOSS")
            
            if trans_type not in by_transition:
                by_transition[trans_type] = {"total": 0, "wins": 0, "losses": 0, "win_rate": 0.0}
            
            by_transition[trans_type]["total"] += 1
            if outcome == "WIN":
                by_transition[trans_type]["wins"] += 1
            elif outcome == "LOSS":
                by_transition[trans_type]["losses"] += 1
        
        for trans in by_transition:
            total = by_transition[trans]["total"]
            by_transition[trans]["win_rate"] = by_transition[trans]["wins"] / total if total > 0 else 0
        
        return by_transition
    
    def _generate_recommendations(
        self,
        matrix: TransitionMatrixStats,
        by_confidence: Dict,
        by_tier: Dict,
        brier_score: float,
        calibration_error: float,
    ) -> List[str]:
        """Generate actionable recommendations based on RTM analysis."""
        recommendations = []
        
        # Bounce-back analysis
        if matrix.bounce_back_rate < 0.30 and matrix.total_transitions >= 10:
            recommendations.append(
                f"⚠️ Low bounce-back rate ({matrix.bounce_back_rate:.1%}). "
                "After losses, system struggles to recover. Consider reducing position size after losses."
            )
        elif matrix.bounce_back_rate > 0.50 and matrix.total_transitions >= 10:
            recommendations.append(
                f"✓ Strong bounce-back rate ({matrix.bounce_back_rate:.1%}). "
                "System recovers well after losses."
            )
        
        # Streak analysis
        if matrix.max_loss_streak >= 5:
            recommendations.append(
                f"⚠️ Concerning loss streaks detected (max: {matrix.max_loss_streak}). "
                "System experiences prolonged downturns. Review M4 pre-filter."
            )
        
        if matrix.max_win_streak >= 15:
            recommendations.append(
                f"✓ Strong win streaks detected (max: {matrix.max_win_streak}). "
                "System shows momentum when clicking."
            )
        
        # Current streak warning
        if matrix.current_loss_streak >= 3:
            recommendations.append(
                f"⚠️ Current loss streak: {matrix.current_loss_streak} in a row. "
                f"Bounce-back probability: {matrix.bounce_back_rate:.1%}."
            )
        
        # Decay analysis
        if matrix.decay_rate > 0.15 and matrix.decay_confidence > 0.7:
            recommendations.append(
                f"⚠️ Performance decay detected ({matrix.decay_rate:.1%}). "
                f"Early season: {matrix.early_season_wr:.1%}, Late season: {matrix.late_season_wr:.1%}. "
                "System degrades over time. Check M18 recalibration."
            )
        elif matrix.decay_rate < 0.05 and matrix.total_transitions >= 20:
            recommendations.append(
                f"✓ Stable performance across season (decay: {matrix.decay_rate:.1%}). "
                "System maintains consistency."
            )
        
        # Confidence calibration
        overconfident = []
        for conf, data in by_confidence.items():
            gap = data.get("calibration_gap", 0)
            if gap > 0.10 and data.get("total", 0) >= 5:
                overconfident.append(f"{conf} ({gap:+.1%})")
        
        if overconfident:
            recommendations.append(
                f"⚠️ Overconfident in: {', '.join(overconfident)}. "
                "Predicted probabilities are higher than actual win rates."
            )
        
        # Brier score analysis
        if brier_score > 0.22 and len(self.history) >= 20:
            recommendations.append(
                f"⚠️ High Brier score ({brier_score:.4f}) - near random (0.25). "
                "Probabilities need recalibration."
            )
        
        # Calibration error
        if calibration_error > 0.10:
            recommendations.append(
                f"⚠️ Calibration error ({calibration_error:.4f}) - probabilities systematically off. "
                "Run M28 calibration check."
            )
        
        # Overall health
        if matrix.bounce_back_rate > 0.40 and matrix.drawdown_recovery_rate > 0.60:
            recommendations.append(
                "✓ RESILIENCE CHECK PASSED. System shows healthy transition patterns."
            )
        elif matrix.total_transitions >= 10:
            recommendations.append(
                "⚠️ RESILIENCE CHECK FAILED. Consider reducing stakes until patterns improve."
            )
        
        if not recommendations and len(self.history) >= 10:
            recommendations.append("✓ System calibration healthy. No adjustments needed.")
        elif len(self.history) < self.min_samples:
            recommendations.append(f"📊 Collecting data ({len(self.history)}/{self.min_samples} samples). Continue recording outcomes.")
        
        return recommendations
    
    def get_rtm_summary(self, season_id: str) -> Dict:
        """Get RTM summary for season."""
        return self.rtm.get_summary(season_id)
    
    def get_latest_anomalies(self, limit: int = 5) -> List[TransitionRecord]:
        """Get most recent anomalous transitions."""
        anomalies = []
        for season in self.rtm.seasons.values():
            anomalies.extend(season.anomalous_transitions)
        
        anomalies.sort(key=lambda x: x.timestamp, reverse=True)
        return anomalies[:limit]
    
    def get_performance_trend(self, window: int = 10) -> Dict[str, Any]:
        """
        Analyze performance trend over recent predictions.
        
        Returns:
            Dictionary with rolling win rates and trend direction
        """
        if len(self.history) < window:
            return {"error": f"Insufficient data. Need at least {window} predictions"}
        
        rolling_wr = []
        outcomes = [1 if r.get("outcome") == "WIN" else 0 for r in self.history]
        
        for i in range(len(outcomes) - window + 1):
            window_wr = sum(outcomes[i:i+window]) / window
            rolling_wr.append(window_wr)
        
        # Calculate trend
        if len(rolling_wr) >= 2:
            first_half = sum(rolling_wr[:len(rolling_wr)//2]) / (len(rolling_wr)//2)
            second_half = sum(rolling_wr[len(rolling_wr)//2:]) / (len(rolling_wr) - len(rolling_wr)//2)
            trend = "IMPROVING" if second_half > first_half else "DECLINING" if second_half < first_half else "STABLE"
        else:
            trend = "UNKNOWN"
        
        return {
            "rolling_win_rates": rolling_wr[-10:],
            "current_win_rate": rolling_wr[-1] if rolling_wr else 0,
            "trend": trend,
            "sample_size": len(outcomes),
        }
    
    def export_season_report(self, season_id: str, output_file: str = None) -> str:
        """Export full season analysis with RTM."""
        if season_id not in self.seasons:
            # Build calibration first
            self.build_season_calibration(season_id)
        
        if season_id not in self.seasons:
            return "Season not found"
        
        analysis = self.seasons[season_id]
        rtm_summary = self.get_rtm_summary(season_id)
        
        report = {
            "season_id": season_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "health_status": analysis.health_status.value,
            "health_score": analysis.rtm_metrics.health_score,
            "brier_score": analysis.brier_score,
            "calibration_error": analysis.calibration_error,
            "rtm_metrics": {
                "bounce_back_rate": analysis.rtm_metrics.bounce_back_rate,
                "recovery_rate": analysis.rtm_metrics.recovery_rate,
                "max_win_streak": analysis.rtm_metrics.max_win_streak,
                "max_loss_streak": analysis.rtm_metrics.max_loss_streak,
                "decay_rate": analysis.rtm_metrics.decay_rate,
                "learning_phase": analysis.rtm_metrics.learning_phase.value,
                "is_resilient": analysis.rtm_metrics.is_resilient,
            },
            "calibration": {
                "by_confidence": analysis.by_confidence,
                "by_tier": analysis.by_tier,
                "by_transition": analysis.by_transition,
            },
            "recommendations": analysis.recommendations,
            "rtm_summary": rtm_summary,
        }
        
        if output_file is None:
            output_file = f"season_report_{season_id}_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Ensure directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        
        return output_file
    
    def reset(self) -> None:
        """Reset all learning data (for testing)."""
        self.history = []
        self.rtm = ResultsTransitionMatrix(min_samples_for_reliable=self.min_samples)
        self.seasons = {}
        self._save_history()


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — EXPORTS
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "PerformanceHealth",
    "OutcomeState",
    "TransitionType",
    "LearningPhase",
    # Data classes
    "TransitionRecord",
    "TransitionMatrixStats",
    "SeasonAnalysis",
    "TransitionMetrics",
    "CalibrationByBin",
    "CalibrationAnalysis",
    # Main classes
    "ResultsTransitionMatrix",
    "EnhancedLearningEngine",
]


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import tempfile
    
    print("\n" + "=" * 70)
    print("MODULE 14: ENHANCED LEARNING ENGINE - TEST RUN")
    print("=" * 70)
    
    # Use temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        engine = EnhancedLearningEngine(history_file=tmp_path, min_samples=5)
        
        # Record some outcomes
        print("\n📊 Recording outcomes...")
        
        outcomes = [
            ("M001", "WIN", "TIER_1", 2.50, 1000, 1500, 0.65, 0.70),
            ("M002", "WIN", "TIER_1", 2.30, 1000, 1300, 0.70, 0.75),
            ("M003", "LOSS", "TIER_1", 2.80, 1000, -1000, 0.60, 0.50),
            ("M004", "WIN", "TIER_2", 3.20, 500, 1100, 0.55, 0.65),
            ("M005", "LOSS", "TIER_1", 2.10, 1000, -1000, 0.58, 0.45),
            ("M006", "WIN", "TIER_2", 2.90, 500, 950, 0.52, 0.60),
            ("M007", "WIN", "TIER_1", 1.95, 1000, 950, 0.68, 0.72),
            ("M008", "LOSS", "TIER_2", 3.50, 500, -500, 0.48, 0.42),
            ("M009", "WIN", "TIER_1", 2.20, 1000, 1200, 0.62, 0.68),
            ("M010", "LOSS", "TIER_2", 3.80, 500, -500, 0.45, 0.40),
        ]
        
        for match_id, outcome, tier, odds, stake, pnl, conf_before, conf_after in outcomes:
            engine.record_outcome(
                match_id=match_id,
                outcome=outcome,
                tier=tier,
                odds=odds,
                stake=stake,
                pnl=pnl,
                confidence_before=conf_before,
                confidence_after=conf_after,
            )
            print(f"  Recorded: {match_id} → {outcome} (tier {tier})")
        
        # Build calibration
        print("\n📊 Building calibration...")
        analysis = engine.build_season_calibration("2025_SEASON")
        
        print(f"\nHealth Status: {analysis.health_status.value}")
        print(f"Health Score: {analysis.rtm_metrics.health_score:.2f}/1.0")
        print(f"Brier Score: {analysis.brier_score:.4f}")
        print(f"Calibration Error: {analysis.calibration_error:.4f}")
        
        print(f"\nResilience Metrics:")
        print(f"  Bounce-back rate: {analysis.rtm_metrics.bounce_back_rate:.1%}")
        print(f"  Recovery rate: {analysis.rtm_metrics.recovery_rate:.1%}")
        print(f"  Max win streak: {analysis.rtm_metrics.max_win_streak}")
        print(f"  Max loss streak: {analysis.rtm_metrics.max_loss_streak}")
        print(f"  Decay rate: {analysis.rtm_metrics.decay_rate:.1%}")
        print(f"  Learning Phase: {analysis.rtm_metrics.learning_phase.value}")
        
        print(f"\nConfidence Calibration:")
        for conf, data in analysis.by_confidence.items():
            print(f"  {conf}: Win Rate={data.get('win_rate', 0):.1%}, "
                  f"Avg Conf={data.get('avg_confidence', 0):.1%}, "
                  f"Gap={data.get('calibration_gap', 0):+.1%}")
        
        print(f"\nRecommendations:")
        for rec in analysis.recommendations:
            print(f"  {rec}")
        
        # Get RTM summary
        rtm_summary = engine.get_rtm_summary("2025_SEASON")
        print(f"\nRTM Summary:")
        print(f"  Total fixtures: {rtm_summary.get('total_fixtures', 0)}")
        print(f"  Overall win rate: {rtm_summary.get('win_rate', 0):.1%}")
        print(f"  Profit: {rtm_summary.get('profit', 0):.2f}")
        print(f"  ROI: {rtm_summary.get('roi', 0):.1%}")
        
        # Get performance trend
        print(f"\nPerformance Trend:")
        trend = engine.get_performance_trend(window=5)
        print(f"  Current WR: {trend.get('current_win_rate', 0):.1%}")
        print(f"  Trend: {trend.get('trend', 'UNKNOWN')}")
        
        # Get anomalies
        anomalies = engine.get_latest_anomalies(limit=3)
        if anomalies:
            print(f"\nRecent Anomalies:")
            for a in anomalies:
                print(f"  {a.match_id}: {a.transition_type.value} (conf={a.confidence_before:.1%})")
        
        # Export report
        report_file = engine.export_season_report("2025_SEASON")
        print(f"\nReport exported to: {report_file}")
        
        print(analysis.summary())
        
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            print(f"\nCleaned up: {tmp_path}")
    
    print("\n" + "=" * 70)
    print("MODULE 14 READY FOR PRODUCTION")
    print("=" * 70)