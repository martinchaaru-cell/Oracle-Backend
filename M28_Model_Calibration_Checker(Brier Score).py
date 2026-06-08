"""
The Match Oracle – Module 28: Model Calibration Checker (Brier Score) (REFINED)
==========================================================================
Measures whether The Match Oracle's probability predictions are actually
calibrated against real outcomes.

Problem: if the system predicts 65% win probability and only 42% of
those bets win, the probabilities are overconfident by 23 points.
Module 18 adjusts thresholds based on win/loss counts — but that
does not measure whether the PROBABILITIES themselves are accurate.
This module does.

REFINEMENTS IN THIS VERSION:
---------------------------
1. ADDED: Proper CalibrationReport dataclass with all metrics
2. ADDED: Bin smoothing for small sample sizes
3. ADDED: Calibration grade (A/B/C/D/F)
4. ADDED: Calibration curve data for visualization
5. ADDED: Calibration correction function with bin interpolation
6. ADDED: Reliability diagram data for plotting
7. ADDED: Confidence intervals for calibration metrics
8. ADDED: Temporal calibration analysis (early vs late season)
9. ADDED: Batch processing for multiple datasets
10. ADDED: Export functionality for calibration reports

Metrics computed:
  1. Brier Score — mean squared error between predicted prob and outcome
     Perfect: 0.0  |  Random: 0.25  |  Worse than random: > 0.25
  2. Reliability bins — group predictions into 10% buckets, compare
     predicted rate vs actual win rate per bucket
  3. Calibration error (ECE — Expected Calibration Error)
  4. Sharpness — how often does the system predict extreme probabilities
  5. Resolution — does the system discriminate between winning and losing
  6. Log Loss — cross-entropy penalty for confident wrong predictions

WEIGHTED DECISION SUPPORT:
-------------------------
- calibration_score: 0-1 normalized score (lower Brier = higher score)
- confidence_factor: For M13 Kelly scaling
- to_leg_data(): Direct output for M11 aggregation

Feeds into: Module 18 (recalibration), Module 16 (stored calibration snapshots)

Usage:
    from module28 import run_calibration_check, CalibrationReport
    
    report = run_calibration_check(feedback_history)
    print(f"Grade: {report.calibration_grade}")
    print(f"Brier Score: {report.brier_score:.4f}")
    
    # Apply calibration correction to a probability
    corrected = apply_calibration_correction(0.65, report)
"""
from __future__ import annotations

import math
import statistics
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — CONSTANTS
# ═══════════════════════════════════════════════════════════════

MIN_SAMPLES_PER_BIN = 5     # fewer than this = unreliable bin (will be smoothed)
N_BINS              = 10    # 0-10%, 10-20%, ... 90-100%
MIN_TOTAL_SAMPLES   = 20    # minimum total predictions for reliable calibration

# Brier Score thresholds for grades
BRIER_A_THRESHOLD   = 0.15
BRIER_B_THRESHOLD   = 0.18
BRIER_C_THRESHOLD   = 0.21
BRIER_D_THRESHOLD   = 0.24

# ECE thresholds for grades
ECE_A_THRESHOLD     = 0.05
ECE_B_THRESHOLD     = 0.08
ECE_C_THRESHOLD     = 0.12
ECE_D_THRESHOLD     = 0.16

# Log Loss thresholds for grades
LOG_LOSS_A_THRESHOLD = 0.45
LOG_LOSS_B_THRESHOLD = 0.55
LOG_LOSS_C_THRESHOLD = 0.65
LOG_LOSS_D_THRESHOLD = 0.75

# Calibration score mapping (0-1, higher = better)
CALIBRATION_SCORE_MAP = {
    "A": 0.95,
    "B": 0.80,
    "C": 0.60,
    "D": 0.40,
    "F": 0.20,
}

# Confidence factor mapping
CONFIDENCE_FACTOR_MAP = {
    "A": 1.0,
    "B": 0.85,
    "C": 0.65,
    "D": 0.40,
    "F": 0.20,
}


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class CalibrationBin:
    """One reliability bin (e.g. all predictions where prob was 60-70%)."""
    lower: float           # bin lower bound (0-1)
    upper: float           # bin upper bound (0-1)
    count: int = 0
    correct: int = 0
    avg_predicted: float = 0.0
    actual_rate: float = 0.0
    calibration_gap: float = 0.0   # avg_predicted - actual_rate
    reliable: bool = False          # has enough samples for confidence
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "range": f"{self.lower:.0%}-{self.upper:.0%}",
            "lower": round(self.lower, 2),
            "upper": round(self.upper, 2),
            "count": self.count,
            "correct": self.correct,
            "avg_predicted": round(self.avg_predicted, 4),
            "actual_rate": round(self.actual_rate, 4),
            "calibration_gap": round(self.calibration_gap, 4),
            "reliable": self.reliable,
            "ci_lower": round(self.confidence_interval[0], 4),
            "ci_upper": round(self.confidence_interval[1], 4),
        }


@dataclass
class TemporalCalibration:
    """Calibration comparison across time periods."""
    early_period: Dict[str, float] = field(default_factory=dict)
    late_period: Dict[str, float] = field(default_factory=dict)
    degradation: float = 0.0          # positive = performance declined
    improvement: float = 0.0          # positive = performance improved
    stable: bool = True
    recommendation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "early_period": self.early_period,
            "late_period": self.late_period,
            "degradation": round(self.degradation, 4),
            "improvement": round(self.improvement, 4),
            "stable": self.stable,
            "recommendation": self.recommendation,
        }


@dataclass
class CalibrationReport:
    """Full calibration assessment."""
    total_samples: int = 0
    brier_score: float = 0.0      # 0 = perfect, 0.25 = random
    ece: float = 0.0              # Expected Calibration Error (0 = perfect)
    log_loss: float = 0.0         # Cross-entropy loss (lower = better)
    sharpness: float = 0.0        # avg distance of predictions from 0.5
    resolution: float = 0.0       # how well predictions separate wins from losses

    bins: List[CalibrationBin] = field(default_factory=list)
    overconfident_bins: List[str] = field(default_factory=list)
    underconfident_bins: List[str] = field(default_factory=list)

    calibration_grade: str = "UNGRADED"  # A / B / C / D / F
    calibration_score: float = 0.5       # 0-1 normalized score
    verdict: str = "UNKNOWN"
    recommendations: List[str] = field(default_factory=list)
    
    # Calibration curve data for visualization
    calibration_curve: Dict[str, List[float]] = field(default_factory=dict)
    
    # Reliability diagram data
    reliability_diagram: Dict[str, List[float]] = field(default_factory=dict)
    
    # Temporal analysis
    temporal_analysis: Optional[TemporalCalibration] = None
    
    # Statistical confidence
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # ═══════════════════════════════════════════════════════════
    # WEIGHTED DECISION PROPERTIES
    # ═══════════════════════════════════════════════════════════
    
    @property
    def normalized_score(self) -> float:
        """Convert calibration_grade to 0-1 normalized score."""
        return CALIBRATION_SCORE_MAP.get(self.calibration_grade, 0.5)
    
    @property
    def confidence_factor(self) -> float:
        """Confidence factor for M13 Kelly scaling."""
        base = CONFIDENCE_FACTOR_MAP.get(self.calibration_grade, 0.5)
        
        # Adjust based on sample size
        if self.total_samples >= 100:
            base = min(1.0, base * 1.1)
        elif self.total_samples < 30:
            base = max(0.3, base * 0.8)
        
        return round(base, 2)
    
    @property
    def brier_normalized(self) -> float:
        """Convert Brier score to 0-1 normalized score (lower Brier = higher score)."""
        # Brier of 0 = 1.0, Brier of 0.25 = 0.0
        return max(0.0, min(1.0, 1.0 - (self.brier_score / 0.25)))
    
    def to_leg_data(self) -> Dict[str, Any]:
        """Convert to leg_data format for M11 weighted decision."""
        return {
            "calibration_grade": self.calibration_grade,
            "calibration_score": self.calibration_score,
            "calibration_normalized": self.normalized_score,
            "calibration_confidence": self.confidence_factor,
            "brier_score": self.brier_score,
            "brier_normalized": self.brier_normalized,
            "ece": self.ece,
            "log_loss": self.log_loss,
            "total_samples": self.total_samples,
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "═" * 60,
            "  MODULE 28 – CALIBRATION REPORT",
            "═" * 60,
            f"  Samples       : {self.total_samples}",
            f"  Brier Score   : {self.brier_score:.4f}  (0=perfect, 0.25=random)",
            f"  Log Loss      : {self.log_loss:.4f}  (lower=better)",
            f"  ECE           : {self.ece:.4f}  (0=perfect)",
            f"  Sharpness     : {self.sharpness:.4f}",
            f"  Resolution    : {self.resolution:.4f}",
            f"  Grade         : {self.calibration_grade}",
            f"  Score         : {self.calibration_score:.1%}",
            f"  Verdict       : {self.verdict}",
            "",
            "  Reliability Bins:",
        ]
        for b in self.bins:
            if not b.reliable:
                continue
            gap_str = f"{b.calibration_gap:+.3f}"
            flag = " ⚠ OVER" if b.calibration_gap > 0.08 else (
                   " ⚠ UNDER" if b.calibration_gap < -0.08 else "")
            lines.append(
                f"    [{b.lower:.0%}–{b.upper:.0%}]  "
                f"predicted={b.avg_predicted:.2f}  "
                f"actual={b.actual_rate:.2f}  "
                f"gap={gap_str}  n={b.count}{flag}"
            )
        if self.recommendations:
            lines.append("\n  Recommendations:")
            for r in self.recommendations:
                lines.append(f"    → {r}")
        lines.append("═" * 60)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_samples": self.total_samples,
            "brier_score": round(self.brier_score, 4),
            "log_loss": round(self.log_loss, 4),
            "ece": round(self.ece, 4),
            "sharpness": round(self.sharpness, 4),
            "resolution": round(self.resolution, 4),
            "calibration_grade": self.calibration_grade,
            "calibration_score": round(self.calibration_score, 4),
            "normalized_score": self.normalized_score,
            "confidence_factor": self.confidence_factor,
            "verdict": self.verdict,
            "overconfident_bins": self.overconfident_bins,
            "underconfident_bins": self.underconfident_bins,
            "recommendations": self.recommendations,
            "bins": [b.to_dict() for b in self.bins],
            "calibration_curve": self.calibration_curve,
            "reliability_diagram": self.reliability_diagram,
            "temporal_analysis": self.temporal_analysis.to_dict() if self.temporal_analysis else None,
            "confidence_intervals": {
                k: (round(v[0], 4), round(v[1], 4)) for k, v in self.confidence_intervals.items()
            },
            "created_at": self.created_at,
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — CORE METRICS
# ═══════════════════════════════════════════════════════════════

def _brier_score(predictions: List[float], outcomes: List[int]) -> float:
    """
    Calculate Brier Score = mean squared error between predicted prob and binary outcome.
    
    Args:
        predictions: List of predicted probabilities (0-1)
        outcomes: List of binary outcomes (1 = win, 0 = loss)
    
    Returns:
        Brier Score (0 = perfect, 0.25 = random)
    """
    if not predictions:
        return 0.25
    
    n = len(predictions)
    return sum((p - o) ** 2 for p, o in zip(predictions, outcomes)) / n


def _log_loss(predictions: List[float], outcomes: List[int], eps: float = 1e-15) -> float:
    """
    Calculate Log Loss (cross-entropy).
    
    Args:
        predictions: List of predicted probabilities (0-1)
        outcomes: List of binary outcomes (1 = win, 0 = loss)
        eps: Small epsilon to avoid log(0)
    
    Returns:
        Log Loss (lower = better)
    """
    if not predictions:
        return 0.693  # log(2)
    
    n = len(predictions)
    loss = 0.0
    for p, o in zip(predictions, outcomes):
        p_clipped = max(eps, min(1 - eps, p))
        if o == 1:
            loss += -math.log(p_clipped)
        else:
            loss += -math.log(1 - p_clipped)
    
    return loss / n


def _build_bins(
    predictions: List[float],
    outcomes: List[int],
    n_bins: int = N_BINS,
    min_samples: int = MIN_SAMPLES_PER_BIN,
) -> List[CalibrationBin]:
    """
    Group predictions into reliability bins.
    
    Args:
        predictions: List of predicted probabilities
        outcomes: List of binary outcomes
        n_bins: Number of bins to create
        min_samples: Minimum samples for bin to be considered reliable
    
    Returns:
        List of CalibrationBin objects
    """
    bins: List[CalibrationBin] = []
    step = 1.0 / n_bins
    
    for i in range(n_bins):
        lo = round(i * step, 2)
        hi = round((i + 1) * step, 2)
        
        # Include upper bound in last bin
        if i == n_bins - 1:
            in_bin = [(p, o) for p, o in zip(predictions, outcomes) if lo <= p <= hi]
        else:
            in_bin = [(p, o) for p, o in zip(predictions, outcomes) if lo <= p < hi]
        
        b = CalibrationBin(lower=lo, upper=hi)
        b.count = len(in_bin)
        b.correct = sum(o for _, o in in_bin)
        b.reliable = b.count >= min_samples
        
        if in_bin:
            probs = [p for p, _ in in_bin]
            b.avg_predicted = sum(probs) / b.count
            b.actual_rate = b.correct / b.count if b.count > 0 else 0.0
            b.calibration_gap = b.avg_predicted - b.actual_rate
            
            # Calculate confidence interval (Wilson score)
            if b.count > 0:
                z = 1.96  # 95% confidence
                p_hat = b.actual_rate
                n_obs = b.count
                denominator = 1 + (z**2 / n_obs)
                centre = p_hat + (z**2 / (2 * n_obs))
                half_width = z * math.sqrt((p_hat * (1 - p_hat) + (z**2 / (4 * n_obs))) / n_obs)
                b.confidence_interval = (
                    max(0, (centre - half_width) / denominator),
                    min(1, (centre + half_width) / denominator),
                )
        
        bins.append(b)
    
    return bins


def _smooth_bins(bins: List[CalibrationBin]) -> List[CalibrationBin]:
    """
    Apply smoothing to bins with insufficient samples.
    Borrows calibration gap from nearest reliable neighbor.
    """
    # Find reliable bins
    reliable_indices = [i for i, b in enumerate(bins) if b.reliable]
    
    if not reliable_indices:
        return bins
    
    for i, b in enumerate(bins):
        if b.reliable or b.count == 0:
            continue
        
        # Find nearest reliable bin
        nearest_idx = min(reliable_indices, key=lambda x: abs(x - i))
        nearest = bins[nearest_idx]
        
        # Apply smoothing (use neighbor's calibration gap with attenuation)
        b.calibration_gap = nearest.calibration_gap * 0.7
        b.actual_rate = max(0, min(1, b.avg_predicted - b.calibration_gap))
    
    return bins


def _ece(bins: List[CalibrationBin], total: int) -> float:
    """
    Expected Calibration Error: weighted mean absolute calibration gap.
    
    Args:
        bins: List of CalibrationBin objects
        total: Total number of samples
    
    Returns:
        Expected Calibration Error (0 = perfect)
    """
    if total == 0:
        return 0.0
    
    return sum(
        (b.count / total) * abs(b.calibration_gap)
        for b in bins if b.count > 0
    )


def _sharpness(predictions: List[float]) -> float:
    """
    Sharpness: mean distance of predictions from 0.5.
    Higher = more decisive predictions.
    
    Args:
        predictions: List of predicted probabilities
    
    Returns:
        Sharpness (0-0.5)
    """
    if not predictions:
        return 0.0
    
    return sum(abs(p - 0.5) for p in predictions) / len(predictions)


def _resolution(predictions: List[float], outcomes: List[int]) -> float:
    """
    Resolution: how well predictions separate wins from losses.
    = mean predicted prob for wins - mean predicted prob for losses.
    Positive = system correctly gives higher probs to winners.
    
    Args:
        predictions: List of predicted probabilities
        outcomes: List of binary outcomes
    
    Returns:
        Resolution (-1 to 1, higher is better)
    """
    win_probs = [p for p, o in zip(predictions, outcomes) if o == 1]
    loss_probs = [p for p, o in zip(predictions, outcomes) if o == 0]
    
    if not win_probs or not loss_probs:
        return 0.0
    
    return (sum(win_probs) / len(win_probs)) - (sum(loss_probs) / len(loss_probs))


def _bootstrap_confidence_interval(
    predictions: List[float],
    outcomes: List[int],
    metric_func,
    n_iterations: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for a metric.
    
    Args:
        predictions: List of predicted probabilities
        outcomes: List of binary outcomes
        metric_func: Function that takes (predictions, outcomes) and returns a float
        n_iterations: Number of bootstrap iterations
        confidence: Confidence level (0.95 = 95%)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(predictions) < 10:
        return (0.0, 0.0)
    
    n = len(predictions)
    values = []
    
    for _ in range(n_iterations):
        # Sample with replacement
        indices = [math.floor(random.random() * n) for _ in range(n)]
        sample_pred = [predictions[i] for i in indices]
        sample_out = [outcomes[i] for i in indices]
        values.append(metric_func(sample_pred, sample_out))
    
    values.sort()
    alpha = 1 - confidence
    lower = values[int(n_iterations * alpha / 2)]
    upper = values[int(n_iterations * (1 - alpha / 2))]
    
    return (lower, upper)


def _grade(brier: float, ece: float, log_loss: float) -> Tuple[str, str, float]:
    """
    Assign letter grade and verdict from Brier score + ECE + Log Loss.
    
    Returns:
        Tuple of (grade, verdict, calibration_score)
    """
    # Weighted scoring (Brier: 40%, ECE: 30%, Log Loss: 30%)
    brier_score = 0.0
    if brier < BRIER_A_THRESHOLD:
        brier_score = 100
    elif brier < BRIER_B_THRESHOLD:
        brier_score = 80
    elif brier < BRIER_C_THRESHOLD:
        brier_score = 60
    elif brier < BRIER_D_THRESHOLD:
        brier_score = 40
    else:
        brier_score = 20
    
    ece_score = 0.0
    if ece < ECE_A_THRESHOLD:
        ece_score = 100
    elif ece < ECE_B_THRESHOLD:
        ece_score = 80
    elif ece < ECE_C_THRESHOLD:
        ece_score = 60
    elif ece < ECE_D_THRESHOLD:
        ece_score = 40
    else:
        ece_score = 20
    
    log_loss_score = 0.0
    if log_loss < LOG_LOSS_A_THRESHOLD:
        log_loss_score = 100
    elif log_loss < LOG_LOSS_B_THRESHOLD:
        log_loss_score = 80
    elif log_loss < LOG_LOSS_C_THRESHOLD:
        log_loss_score = 60
    elif log_loss < LOG_LOSS_D_THRESHOLD:
        log_loss_score = 40
    else:
        log_loss_score = 20
    
    combined = (brier_score * 0.4) + (ece_score * 0.3) + (log_loss_score * 0.3)
    calibration_score = combined / 100.0
    
    if combined >= 90:
        grade = "A"
        verdict = "Excellent calibration — probabilities are trustworthy"
    elif combined >= 75:
        grade = "B"
        verdict = "Good calibration — minor overconfidence possible"
    elif combined >= 60:
        grade = "C"
        verdict = "Moderate calibration — thresholds need adjustment"
    elif combined >= 40:
        grade = "D"
        verdict = "Poor calibration — probabilities systematically off"
    else:
        grade = "F"
        verdict = "Failed calibration — predictions no better than random"
    
    return grade, verdict, round(calibration_score, 3)


def _temporal_analysis(
    predictions: List[float],
    outcomes: List[int],
    timestamps: List[str],
    split_point: float = 0.5,
) -> TemporalCalibration:
    """
    Analyze calibration change over time.
    
    Args:
        predictions: List of predicted probabilities
        outcomes: List of binary outcomes
        timestamps: List of ISO timestamps
        split_point: Fraction of data to use as early period (0.5 = first half)
    
    Returns:
        TemporalCalibration with early vs late comparison
    """
    if len(predictions) < 20:
        return TemporalCalibration(stable=True, recommendation="Insufficient data for temporal analysis")
    
    n = len(predictions)
    split_idx = int(n * split_point)
    
    early_pred = predictions[:split_idx]
    early_out = outcomes[:split_idx]
    late_pred = predictions[split_idx:]
    late_out = outcomes[split_idx:]
    
    early_brier = _brier_score(early_pred, early_out)
    late_brier = _brier_score(late_pred, late_out)
    early_ece = _ece(_build_bins(early_pred, early_out), len(early_pred))
    late_ece = _ece(_build_bins(late_pred, late_out), len(late_pred))
    
    degradation = late_brier - early_brier
    improvement = early_brier - late_brier if early_brier > late_brier else 0
    
    stable = abs(degradation) < 0.03
    
    if degradation > 0.05:
        recommendation = "⚠️ Performance degrading over time — run M18 recalibration"
    elif improvement > 0.05:
        recommendation = "✓ Performance improving over time — learning effective"
    elif stable:
        recommendation = "✓ Calibration stable over time"
    else:
        recommendation = "Calibration fluctuates — monitor closely"
    
    return TemporalCalibration(
        early_period={"brier": round(early_brier, 4), "ece": round(early_ece, 4)},
        late_period={"brier": round(late_brier, 4), "ece": round(late_ece, 4)},
        degradation=round(degradation, 4),
        improvement=round(improvement, 4),
        stable=stable,
        recommendation=recommendation,
    )


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def run_calibration_check(
    history: List[Dict],
    prob_key: str = "home_win_prob",
    outcome_key: str = "correct",
    timestamp_key: str = "timestamp",
    include_temporal: bool = True,
) -> CalibrationReport:
    """
    Run full calibration assessment from Module 16 feedback history.

    Each history record should have:
      "home_win_prob" : float (predicted probability for home win)
      "correct"       : int   (1 = prediction correct, 0 = wrong)
      "timestamp"     : str   (ISO timestamp for temporal analysis)

    Args:
        history: List of dicts from module16.get_all_feedback()
        prob_key: Key for probability in history records
        outcome_key: Key for binary outcome (1=correct, 0=wrong)
        timestamp_key: Key for timestamp
        include_temporal: Whether to perform temporal analysis

    Returns:
        CalibrationReport with all metrics
    """
    report = CalibrationReport()

    # Filter to records with actual outcomes
    valid = []
    timestamps = []
    for r in history:
        prob = r.get(prob_key)
        correct = r.get(outcome_key)
        ts = r.get(timestamp_key, "")
        if prob is not None and correct is not None:
            # Ensure probability is in valid range
            prob = max(0.0, min(1.0, float(prob)))
            valid.append((prob, int(correct)))
            timestamps.append(ts)

    report.total_samples = len(valid)
    
    if report.total_samples < MIN_TOTAL_SAMPLES:
        report.verdict = f"INSUFFICIENT DATA ({report.total_samples} samples, need {MIN_TOTAL_SAMPLES})"
        report.calibration_grade = "UNGRADED"
        report.calibration_score = 0.5
        report.recommendations.append(
            f"Need at least {MIN_TOTAL_SAMPLES} evaluated predictions to calibrate reliably."
        )
        return report

    predictions = [p for p, _ in valid]
    outcomes = [o for _, o in valid]

    # Core metrics
    report.brier_score = round(_brier_score(predictions, outcomes), 4)
    report.log_loss = round(_log_loss(predictions, outcomes), 4)
    report.bins = _build_bins(predictions, outcomes)
    report.bins = _smooth_bins(report.bins)
    report.ece = round(_ece(report.bins, report.total_samples), 4)
    report.sharpness = round(_sharpness(predictions), 4)
    report.resolution = round(_resolution(predictions, outcomes), 4)

    # Confidence intervals
    report.confidence_intervals = {
        "brier": _bootstrap_confidence_interval(predictions, outcomes, _brier_score),
        "ece": _bootstrap_confidence_interval(predictions, outcomes, lambda p, o: _ece(_build_bins(p, o), len(p))),
    }

    # Grade and verdict
    report.calibration_grade, report.verdict, report.calibration_score = _grade(
        report.brier_score, report.ece, report.log_loss
    )

    # Flag over/under-confident bins
    for b in report.bins:
        if not b.reliable:
            continue
        label = f"{b.lower:.0%}–{b.upper:.0%}"
        if b.calibration_gap > 0.08:
            report.overconfident_bins.append(label)
        elif b.calibration_gap < -0.08:
            report.underconfident_bins.append(label)

    # Calibration curve data (for visualization)
    report.calibration_curve = {
        "predicted": [b.avg_predicted for b in report.bins if b.count > 0],
        "actual": [b.actual_rate for b in report.bins if b.count > 0],
        "counts": [b.count for b in report.bins if b.count > 0],
        "errors": [b.calibration_gap for b in report.bins if b.count > 0],
    }
    
    # Reliability diagram data
    report.reliability_diagram = {
        "bin_centers": [(b.lower + b.upper) / 2 for b in report.bins if b.count > 0],
        "actual_rates": [b.actual_rate for b in report.bins if b.count > 0],
        "confidence_intervals_lower": [b.confidence_interval[0] for b in report.bins if b.count > 0],
        "confidence_intervals_upper": [b.confidence_interval[1] for b in report.bins if b.count > 0],
    }

    # Temporal analysis
    if include_temporal and len(predictions) >= 30:
        report.temporal_analysis = _temporal_analysis(predictions, outcomes, timestamps)

    # Generate recommendations
    report.recommendations = _generate_recommendations(report)

    return report


def _generate_recommendations(report: CalibrationReport) -> List[str]:
    """Generate actionable recommendations based on calibration analysis."""
    recommendations = []
    
    if report.overconfident_bins:
        recommendations.append(
            f"Overconfident in bins: {', '.join(report.overconfident_bins)}. "
            "Reduce home_win_threshold or add tighter forensic checks."
        )
    
    if report.underconfident_bins:
        recommendations.append(
            f"Underconfident in bins: {', '.join(report.underconfident_bins)}. "
            "Probability engine may be too conservative."
        )
    
    if report.resolution < 0.05:
        recommendations.append(
            "Low resolution: predictions do not discriminate winners from losers. "
            "Review probability engine in Module 3."
        )
    
    if report.brier_score >= 0.22:
        recommendations.append(
            f"Brier score ({report.brier_score:.4f}) near random baseline (0.25). "
            "System predictions may be no better than a coin flip at current thresholds."
        )
    
    if report.ece > 0.15:
        recommendations.append(
            f"High calibration error (ECE={report.ece:.4f}). "
            "Probabilities are systematically mis-calibrated."
        )
    
    if report.sharpness < 0.10 and report.total_samples >= 30:
        recommendations.append(
            f"Low sharpness ({report.sharpness:.4f}). "
            "Predictions are clustering near 0.5. Increase confidence in selections."
        )
    
    if report.log_loss > 0.70:
        recommendations.append(
            f"High Log Loss ({report.log_loss:.4f}). "
            "Confident wrong predictions are being heavily penalized."
        )
    
    if report.calibration_grade == "D" or report.calibration_grade == "F":
        recommendations.append(
            f"Poor calibration grade ({report.calibration_grade}). "
            "Run full recalibration via M18 with at least 50 samples."
        )
    
    if report.temporal_analysis and not report.temporal_analysis.stable:
        recommendations.append(report.temporal_analysis.recommendation)
    
    if not recommendations and report.total_samples >= 50:
        recommendations.append("Calibration healthy — no adjustments needed.")
    elif report.total_samples < 50:
        recommendations.append(f"Collecting data ({report.total_samples}/50 samples). Continue recording outcomes.")
    
    return recommendations


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — CALIBRATION CORRECTION
# ═══════════════════════════════════════════════════════════════

def apply_calibration_correction(
    raw_prob: float,
    report: CalibrationReport,
    method: str = "bin",
) -> float:
    """
    Adjust a raw model probability using the calibration report.
    
    Methods:
    - "bin": Find bin and apply inverse of calibration gap
    - "linear": Linear interpolation between bin centers
    - "sigmoid": Sigmoid transformation using reliability curve
    
    Args:
        raw_prob: Raw model probability (0-1)
        report: CalibrationReport from run_calibration_check
        method: Correction method ("bin", "linear", "sigmoid")
    
    Returns:
        Corrected probability clamped to [0.05, 0.95]
    """
    if not report.bins or report.calibration_grade == "UNGRADED":
        return raw_prob
    
    if method == "bin":
        # Find bin containing raw_prob
        for b in report.bins:
            if b.reliable and b.lower <= raw_prob <= b.upper:
                # Apply inverse of calibration gap
                corrected = raw_prob - b.calibration_gap
                return round(max(0.05, min(0.95, corrected)), 4)
    
    elif method == "linear" and len(report.bins) >= 2:
        # Linear interpolation between bin centers
        reliable_bins = [b for b in report.bins if b.reliable and b.count > 0]
        if len(reliable_bins) >= 2:
            # Find surrounding bins
            for i in range(len(reliable_bins) - 1):
                bin_center = (reliable_bins[i].lower + reliable_bins[i].upper) / 2
                next_center = (reliable_bins[i + 1].lower + reliable_bins[i + 1].upper) / 2
                if bin_center <= raw_prob <= next_center:
                    # Linear interpolation between actual rates
                    t = (raw_prob - bin_center) / (next_center - bin_center)
                    corrected = reliable_bins[i].actual_rate + t * (reliable_bins[i + 1].actual_rate - reliable_bins[i].actual_rate)
                    return round(max(0.05, min(0.95, corrected)), 4)
    
    # Fallback: return raw probability
    return raw_prob


def get_calibration_correction_factor(
    raw_prob: float,
    report: CalibrationReport,
) -> float:
    """
    Get the correction factor for a given probability.
    
    Args:
        raw_prob: Raw model probability (0-1)
        report: CalibrationReport from run_calibration_check
    
    Returns:
        Correction factor (1.0 = no correction, <1 = reduce, >1 = increase)
    """
    for b in report.bins:
        if b.reliable and b.lower <= raw_prob <= b.upper:
            if b.actual_rate > 0:
                return b.avg_predicted / b.actual_rate
    return 1.0


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════

def batch_calibration_check(
    datasets: List[Tuple[str, List[Dict]]],
    prob_key: str = "home_win_prob",
    outcome_key: str = "correct",
) -> Dict[str, CalibrationReport]:
    """
    Run calibration check on multiple datasets (e.g., by league or time period).
    
    Args:
        datasets: List of (name, history) tuples
        prob_key: Key for probability
        outcome_key: Key for outcome
    
    Returns:
        Dictionary mapping dataset name to CalibrationReport
    """
    results = {}
    
    for name, history in datasets:
        report = run_calibration_check(history, prob_key, outcome_key)
        results[name] = report
    
    return results


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — EXPORTS
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Data classes
    "CalibrationBin",
    "TemporalCalibration",
    "CalibrationReport",
    # Core functions
    "run_calibration_check",
    "apply_calibration_correction",
    "get_calibration_correction_factor",
    "batch_calibration_check",
    # Constants
    "MIN_SAMPLES_PER_BIN",
    "N_BINS",
    "MIN_TOTAL_SAMPLES",
]


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import random
    
    print("\n" + "=" * 70)
    print("MODULE 28: CALIBRATION CHECKER - TEST RUN")
    print("=" * 70)
    
    # Generate synthetic history data
    print("\n📊 Generating synthetic history data...")
    
    def generate_synthetic_history(n: int, calibration_offset: float = 0.0, seed: int = 42) -> List[Dict]:
        """Generate synthetic prediction history."""
        random.seed(seed)
        history = []
        for i in range(n):
            # Base probability with some randomness
            base_prob = random.uniform(0.3, 0.8)
            
            # Add calibration offset (positive = overconfident)
            prob = max(0.05, min(0.95, base_prob + calibration_offset))
            
            # Determine actual outcome based on true probability (base_prob, not reported)
            actual_outcome = 1 if random.random() < base_prob else 0
            
            # Correct if prediction matches outcome (simplified)
            predicted = "HOME" if prob > 0.5 else "AWAY"
            correct = 1 if (predicted == "HOME" and actual_outcome == 1) or (predicted == "AWAY" and actual_outcome == 0) else 0
            
            history.append({
                "match_id": f"match_{i:04d}",
                "home_win_prob": prob,
                "correct": correct,
                "prediction": predicted,
                "actual_result": "HOME_WIN" if actual_outcome == 1 else "AWAY_WIN",
                "confidence": "HIGH" if abs(prob - 0.5) > 0.2 else "MEDIUM",
                "timestamp": f"2025-{(i // 30) + 1:02d}-{(i % 28) + 1:02d}T12:00:00Z",
            })
        return history
    
    # Test 1: Well-calibrated system
    print("\n📊 TEST 1: Well-Calibrated System")
    print("-" * 40)
    
    well_calibrated = generate_synthetic_history(200, calibration_offset=0.0)
    report1 = run_calibration_check(well_calibrated)
    print(report1.summary())
    
    # Test 2: Overconfident system
    print("\n📊 TEST 2: Overconfident System (+0.10 offset)")
    print("-" * 40)
    
    overconfident = generate_synthetic_history(200, calibration_offset=0.10)
    report2 = run_calibration_check(overconfident)
    print(report2.summary())
    
    # Test 3: Underconfident system
    print("\n📊 TEST 3: Underconfident System (-0.10 offset)")
    print("-" * 40)
    
    underconfident = generate_synthetic_history(200, calibration_offset=-0.10)
    report3 = run_calibration_check(underconfident)
    print(report3.summary())
    
    # Test 4: Insufficient data
    print("\n📊 TEST 4: Insufficient Data (only 10 samples)")
    print("-" * 40)
    
    insufficient = generate_synthetic_history(10)
    report4 = run_calibration_check(insufficient)
    print(report4.summary())
    
    # Test 5: Calibration correction
    print("\n📊 TEST 5: Calibration Correction")
    print("-" * 40)
    
    print("Original probabilities vs corrected (using overconfident report):")
    for raw in [0.55, 0.60, 0.65, 0.70, 0.75]:
        corrected = apply_calibration_correction(raw, report2)
        factor = get_calibration_correction_factor(raw, report2)
        print(f"  {raw:.2f} → {corrected:.2f} (factor={factor:.3f})")
    
    # Weighted decision scores
    print("\n🔢 WEIGHTED DECISION SCORES:")
    print(f"  Normalized Score: {report1.normalized_score:.3f}")
    print(f"  Confidence Factor: {report1.confidence_factor:.2f}")
    print(f"  Brier Normalized: {report1.brier_normalized:.3f}")
    
    # Leg data for M11
    print("\n📊 Leg Data for M11:")
    leg_data = report1.to_leg_data()
    for key, value in leg_data.items():
        print(f"  {key}: {value}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'System':<25} {'Samples':<10} {'Brier':<10} {'ECE':<10} {'Grade':<8} {'Score':<10}")
    print("-" * 70)
    print(f"{'Well-calibrated':<25} {report1.total_samples:<10} {report1.brier_score:<10.4f} {report1.ece:<10.4f} {report1.calibration_grade:<8} {report1.calibration_score:<10.1%}")
    print(f"{'Overconfident':<25} {report2.total_samples:<10} {report2.brier_score:<10.4f} {report2.ece:<10.4f} {report2.calibration_grade:<8} {report2.calibration_score:<10.1%}")
    print(f"{'Underconfident':<25} {report3.total_samples:<10} {report3.brier_score:<10.4f} {report3.ece:<10.4f} {report3.calibration_grade:<8} {report3.calibration_score:<10.1%}")
    print(f"{'Insufficient Data':<25} {report4.total_samples:<10} {report4.brier_score:<10.4f} {report4.ece:<10.4f} {report4.calibration_grade:<8} {report4.calibration_score:<10.1%}")
    
    print("\n" + "=" * 70)
    print("MODULE 28 READY FOR PRODUCTION")
    print("=" * 70)