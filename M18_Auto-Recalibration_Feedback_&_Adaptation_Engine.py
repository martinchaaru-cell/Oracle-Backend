"""
The Match Oracle - Module 18: Auto-Recalibration, Feedback & Adaptation Engine (REFINED)
===================================================================================
Merged from:
  - Original Module 23 (auto-recalibration: thresholds, weights, league sensitivity)
  - Module 20 (live feedback loop: evaluate predictions vs results, performance report)
  - Module 21 (autonomous adaptation: persist/load config via Module 16 DB)

REFINEMENTS IN THIS VERSION:
---------------------------
1. ADDED: DecisionWeights dataclass (centralized decision weights for M11)
2. ADDED: Weighted decision calculation with configurable weights
3. ADDED: Weight adjustment based on module performance (learning)
4. ADDED: Weight normalization (auto sum to 1.0)
5. ADDED: Comprehensive weight persistence and versioning
6. ADDED: Performance-based weight optimization
7. ADDED: League-specific calibration with volatility adjustments
8. ADDED: Confidence calibration correction based on historical accuracy
9. ADDED: Export functionality for config and weights

WEIGHTED DECISION SYSTEM:
-------------------------
DecisionWeights control how much each module influences the final
APPROVE/CAUTION/REJECT decision in M11. Weights are:
- Auto-adjusted based on historical module performance
- Normalized to sum to 1.0
- Persisted across sessions via Module 16

Weight adjustment logic:
- Modules with higher historical accuracy get higher weights
- Learning rate controls how quickly weights adapt
- Maximum weight per module is capped at 0.35
- Minimum weight per module is 0.03

Usage:
    from module18 import SystemConfig, DecisionWeights, run_full_learning_cycle
    
    # Get current config
    config = load_config_from_db(db)
    
    # Run learning cycle
    result = run_full_learning_cycle(verdicts, results, db)
    
    # Access updated weights
    new_weights = result["new_config"].decision_weights
"""
from __future__ import annotations

import logging
import statistics
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

# Set up logging
logger = logging.getLogger("oracle_beast.module18")


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — DECISION WEIGHTS (NEW for M11)
# ═══════════════════════════════════════════════════════════════

@dataclass
class DecisionWeights:
    """
    Weights for each module in the final decision.
    Sum should equal 1.0 (auto-normalized on update).
    
    These weights represent how much each module influences
    the final APPROVE/CAUTION/REJECT decision in M11.
    
    Default weights based on empirical testing:
    - Oracle modules (M4+M5): 35% (core fundamentals)
    - AI consensus (M7): 15% (external validation)
    - Pattern modules (M8+M9+M10): 25% (pattern detection)
    - Supporting modules (M6+M27+M26): 25% (context)
    """
    # Core Oracle modules (M4 + M5)
    oracle_prefilter: float = 0.20   # M4 Asymmetric pre-filter
    oracle_forensics: float = 0.15   # M5 Failure score
    
    # Intelligence modules
    ai_consensus: float = 0.15       # M7 Multi-AI verdict
    dual_pattern: float = 0.10       # M8 Pattern clash
    underdog: float = 0.10           # M9 Underdog value
    matrix: float = 0.05             # M10 Transition matrix
    
    # Supporting modules
    personnel: float = 0.10          # M6 Injuries/suspensions
    h2h: float = 0.08                # M27 H2H deep analysis
    context: float = 0.07            # M26 Match context
    
    # Version tracking
    version: int = 1
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def __post_init__(self):
        """Normalize weights to sum to 1.0 after initialization."""
        self.normalize()
    
    def normalize(self) -> None:
        """Ensure weights sum to 1.0."""
        total = sum([
            self.oracle_prefilter,
            self.oracle_forensics,
            self.ai_consensus,
            self.dual_pattern,
            self.underdog,
            self.matrix,
            self.personnel,
            self.h2h,
            self.context,
        ])
        
        if total > 0 and abs(total - 1.0) > 0.01:
            factor = 1.0 / total
            self.oracle_prefilter *= factor
            self.oracle_forensics *= factor
            self.ai_consensus *= factor
            self.dual_pattern *= factor
            self.underdog *= factor
            self.matrix *= factor
            self.personnel *= factor
            self.h2h *= factor
            self.context *= factor
            
            logger.debug(f"Weights normalized: total was {total:.3f}, now 1.0")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "oracle_prefilter": self.oracle_prefilter,
            "oracle_forensics": self.oracle_forensics,
            "ai_consensus": self.ai_consensus,
            "dual_pattern": self.dual_pattern,
            "underdog": self.underdog,
            "matrix": self.matrix,
            "personnel": self.personnel,
            "h2h": self.h2h,
            "context": self.context,
            "version": self.version,
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionWeights":
        """Create from dictionary."""
        return cls(
            oracle_prefilter=data.get("oracle_prefilter", 0.20),
            oracle_forensics=data.get("oracle_forensics", 0.15),
            ai_consensus=data.get("ai_consensus", 0.15),
            dual_pattern=data.get("dual_pattern", 0.10),
            underdog=data.get("underdog", 0.10),
            matrix=data.get("matrix", 0.05),
            personnel=data.get("personnel", 0.10),
            h2h=data.get("h2h", 0.08),
            context=data.get("context", 0.07),
            version=data.get("version", 1),
            last_updated=data.get("last_updated", datetime.now(timezone.utc).isoformat()),
        )
    
    def summary(self) -> str:
        """Human-readable summary."""
        oracle_total = self.oracle_prefilter + self.oracle_forensics
        pattern_total = self.dual_pattern + self.underdog + self.matrix
        support_total = self.personnel + self.h2h + self.context
        
        return (
            f"Decision Weights (sum=1.0):\n"
            f"  Oracle (M4+M5): {oracle_total:.0%}\n"
            f"    - Prefilter (M4): {self.oracle_prefilter:.0%}\n"
            f"    - Forensics (M5): {self.oracle_forensics:.0%}\n"
            f"  AI (M7): {self.ai_consensus:.0%}\n"
            f"  Pattern (M8+M9+M10): {pattern_total:.0%}\n"
            f"    - Dual Pattern (M8): {self.dual_pattern:.0%}\n"
            f"    - Underdog (M9): {self.underdog:.0%}\n"
            f"    - Matrix (M10): {self.matrix:.0%}\n"
            f"  Support (M6+M27+M26): {support_total:.0%}\n"
            f"    - Personnel (M6): {self.personnel:.0%}\n"
            f"    - H2H (M27): {self.h2h:.0%}\n"
            f"    - Context (M26): {self.context:.0%}"
        )
    
    def get_weight(self, module_name: str) -> float:
        """Get weight by module name."""
        weights = {
            "oracle_prefilter": self.oracle_prefilter,
            "oracle_forensics": self.oracle_forensics,
            "ai_consensus": self.ai_consensus,
            "dual_pattern": self.dual_pattern,
            "underdog": self.underdog,
            "matrix": self.matrix,
            "personnel": self.personnel,
            "h2h": self.h2h,
            "context": self.context,
        }
        return weights.get(module_name, 0.0)


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — SYSTEM CONFIG (with Decision Weights)
# ═══════════════════════════════════════════════════════════════

@dataclass
class SystemConfig:
    """
    Single source of truth for all adaptive thresholds and weights.
    Includes DecisionWeights for M11 weighted scoring.
    """
    # Core prediction gates
    home_win_threshold: float = 0.57
    opponent_win_cap: float = 0.25
    min_edge: float = 0.04

    # Risk tuning
    high_risk_league_multiplier: float = 1.2
    low_risk_league_multiplier: float = 0.9

    # Module weights (legacy - kept for backward compatibility)
    oracle_weight: float = 1.0
    dual_weight: float = 1.0
    underdog_weight: float = 1.0
    matrix_weight: float = 1.0

    # Micro-adjustments
    edge_threshold: float = 0.05
    confidence_threshold: float = 0.55
    risk_tolerance: float = 1.0

    # NEW: Decision weights for M11 weighted scoring
    decision_weights: DecisionWeights = field(default_factory=DecisionWeights)

    # Tracking
    recalibration_count: int = 0
    last_updated: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    
    def clamp(self) -> "SystemConfig":
        """Clamp all values to safe operational bounds."""
        # Core gates
        self.home_win_threshold = min(max(self.home_win_threshold, 0.52), 0.70)
        self.opponent_win_cap = min(max(self.opponent_win_cap, 0.15), 0.35)
        self.min_edge = min(max(self.min_edge, 0.02), 0.15)
        
        # Risk tuning
        self.high_risk_league_multiplier = min(max(self.high_risk_league_multiplier, 1.0), 1.5)
        self.low_risk_league_multiplier = min(max(self.low_risk_league_multiplier, 0.7), 1.0)
        
        # Legacy weights
        self.oracle_weight = min(max(self.oracle_weight, 0.5), 2.0)
        self.dual_weight = min(max(self.dual_weight, 0.5), 2.0)
        self.underdog_weight = min(max(self.underdog_weight, 0.5), 2.0)
        self.matrix_weight = min(max(self.matrix_weight, 0.5), 2.0)
        
        # Micro-adjustments
        self.edge_threshold = min(max(self.edge_threshold, 0.03), 0.15)
        self.confidence_threshold = min(max(self.confidence_threshold, 0.50), 0.75)
        self.risk_tolerance = min(max(self.risk_tolerance, 0.5), 1.5)
        
        # Decision weights normalization
        self.decision_weights.normalize()
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "home_win_threshold": self.home_win_threshold,
            "opponent_win_cap": self.opponent_win_cap,
            "min_edge": self.min_edge,
            "high_risk_league_multiplier": self.high_risk_league_multiplier,
            "low_risk_league_multiplier": self.low_risk_league_multiplier,
            "oracle_weight": self.oracle_weight,
            "dual_weight": self.dual_weight,
            "underdog_weight": self.underdog_weight,
            "matrix_weight": self.matrix_weight,
            "edge_threshold": self.edge_threshold,
            "confidence_threshold": self.confidence_threshold,
            "risk_tolerance": self.risk_tolerance,
            "decision_weights": self.decision_weights.to_dict(),
            "recalibration_count": self.recalibration_count,
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemConfig":
        """Create SystemConfig from dictionary."""
        weights_data = data.get("decision_weights", {})
        return cls(
            home_win_threshold=data.get("home_win_threshold", 0.57),
            opponent_win_cap=data.get("opponent_win_cap", 0.25),
            min_edge=data.get("min_edge", 0.04),
            high_risk_league_multiplier=data.get("high_risk_league_multiplier", 1.2),
            low_risk_league_multiplier=data.get("low_risk_league_multiplier", 0.9),
            oracle_weight=data.get("oracle_weight", 1.0),
            dual_weight=data.get("dual_weight", 1.0),
            underdog_weight=data.get("underdog_weight", 1.0),
            matrix_weight=data.get("matrix_weight", 1.0),
            edge_threshold=data.get("edge_threshold", 0.05),
            confidence_threshold=data.get("confidence_threshold", 0.55),
            risk_tolerance=data.get("risk_tolerance", 1.0),
            decision_weights=DecisionWeights.from_dict(weights_data),
            recalibration_count=data.get("recalibration_count", 0),
            last_updated=data.get("last_updated", datetime.now(timezone.utc).isoformat()),
        )


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — WEIGHT ADJUSTMENT BASED ON PERFORMANCE
# ═══════════════════════════════════════════════════════════════

def adjust_weights_by_performance(
    weights: DecisionWeights,
    module_performance: Dict[str, float],
    learning_rate: float = 0.05,
    min_weight: float = 0.03,
    max_weight: float = 0.35,
) -> DecisionWeights:
    """
    Adjust decision weights based on historical module performance.
    
    Args:
        weights: Current DecisionWeights
        module_performance: Dict of module_name -> accuracy (0-1)
        learning_rate: How much to adjust per iteration (0-0.1)
        min_weight: Minimum weight per module
        max_weight: Maximum weight per module
    
    Returns:
        Adjusted DecisionWeights
    """
    # Map module names to weight attributes
    module_map = {
        "oracle_prefilter": "oracle_prefilter",
        "oracle_forensics": "oracle_forensics",
        "ai_consensus": "ai_consensus",
        "dual_pattern": "dual_pattern",
        "underdog": "underdog",
        "matrix": "matrix",
        "personnel": "personnel",
        "h2h": "h2h",
        "context": "context",
    }
    
    # Target weights based on performance
    # Better performing modules get higher target weights
    adjustments = {}
    
    # Calculate baseline performance
    perf_values = list(module_performance.values())
    avg_perf = statistics.mean(perf_values) if perf_values else 0.5
    
    for module, accuracy in module_performance.items():
        if module in module_map:
            attr = module_map[module]
            current = getattr(weights, attr)
            
            # Target weight based on relative performance
            # Higher than average = increase weight
            performance_ratio = accuracy / max(avg_perf, 0.01)
            
            # Target range: 0.05-0.30 based on performance
            if performance_ratio >= 1.2:
                target = min(0.30, current * 1.2)
            elif performance_ratio >= 1.0:
                target = min(0.25, current * 1.05)
            elif performance_ratio >= 0.8:
                target = max(0.08, current * 0.95)
            else:
                target = max(0.05, current * 0.85)
            
            # Clamp to bounds
            target = min(max_weight, max(min_weight, target))
            
            # Gradual adjustment
            adjustment = (target - current) * learning_rate
            new_value = current + adjustment
            new_value = min(max_weight, max(min_weight, new_value))
            adjustments[attr] = new_value
            
            logger.debug(f"Weight adjustment for {module}: {current:.3f} → {new_value:.3f} (target={target:.3f}, perf={accuracy:.1%})")
    
    # Apply adjustments
    for attr, new_value in adjustments.items():
        setattr(weights, attr, new_value)
    
    # Normalize to sum to 1.0
    weights.normalize()
    
    logger.info(f"Weights adjusted: {weights.to_dict()}")
    
    return weights


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — WEIGHTED DECISION CALCULATOR (for M11)
# ═══════════════════════════════════════════════════════════════

def calculate_weighted_decision(
    leg_data: Dict[str, Any],
    weights: DecisionWeights = None,
) -> Tuple[float, str, str, Dict[str, float]]:
    """
    Calculate weighted decision score using decision weights.
    
    This function is designed to be called by M11 to aggregate
    all module outputs into a single weighted score.
    
    Args:
        leg_data: Dictionary containing scores from all modules:
            - pre_filter_passed: bool (M4)
            - pre_filter_score: float (M4 weighted, 0-1)
            - failure_score: float (M5, 0-10 scale)
            - ai_verdict: str (M7: APPROVE/CAUTION/REJECT)
            - ai_score: float (M7 normalized, 0-1)
            - dual_risk_level: str (M8: LOW/MEDIUM/HIGH/CRITICAL)
            - underdog_edge: float (M9, -0.5 to 0.5)
            - matrix_useful: bool (M10)
            - personnel_advantage: float (M6, 0-100)
            - h2h_score: float (M27, 0-100)
            - match_importance: float (M26, 0-1)
        weights: DecisionWeights (uses default if None)
    
    Returns:
        Tuple of (total_score, final_status, confidence, contributions)
    """
    if weights is None:
        weights = DecisionWeights()
    
    total_score = 0.0
    contributions = {}
    
    # M4 Pre-filter (use weighted score if available)
    m4_score = leg_data.get("pre_filter_score", 0.0)
    if m4_score == 0.0:
        m4_score = 1.0 if leg_data.get("pre_filter_passed", False) else 0.0
    contributions["oracle_prefilter"] = m4_score * weights.oracle_prefilter
    total_score += contributions["oracle_prefilter"]
    
    # M5 Forensics (normalize failure score: 0-10 -> 1.0 to 0.0)
    failure_score = leg_data.get("failure_score", 5.0)
    m5_score = max(0.0, min(1.0, 1.0 - (failure_score / 10.0)))
    contributions["oracle_forensics"] = m5_score * weights.oracle_forensics
    total_score += contributions["oracle_forensics"]
    
    # M7 AI (use pre-computed score if available)
    ai_score = leg_data.get("ai_score", 0.5)
    if ai_score == 0.5:
        # Fallback to verdict mapping
        ai_verdict = leg_data.get("ai_verdict", "CAUTION")
        ai_score = {"APPROVE": 1.0, "CAUTION": 0.5, "REJECT": 0.0}.get(ai_verdict, 0.5)
    contributions["ai_consensus"] = ai_score * weights.ai_consensus
    total_score += contributions["ai_consensus"]
    
    # M8 Dual Pattern
    dual_risk = leg_data.get("dual_risk_level", "MEDIUM")
    dual_score = leg_data.get("dual_risk_score", 0.5)
    if dual_score == 0.5:
        dual_score = {"LOW": 0.8, "MEDIUM": 0.5, "HIGH": 0.2, "CRITICAL": 0.1}.get(dual_risk, 0.5)
    contributions["dual_pattern"] = dual_score * weights.dual_pattern
    total_score += contributions["dual_pattern"]
    
    # M9 Underdog
    underdog_edge = leg_data.get("underdog_edge", 0.0)
    underdog_score = leg_data.get("underdog_score", 0.5)
    if underdog_score == 0.5:
        underdog_score = min(1.0, max(0.0, underdog_edge / 0.15))
    contributions["underdog"] = underdog_score * weights.underdog
    total_score += contributions["underdog"]
    
    # M10 Matrix
    matrix_useful = leg_data.get("matrix_useful", False)
    matrix_score = leg_data.get("matrix_score", 0.5)
    if matrix_score == 0.5:
        matrix_score = 0.7 if matrix_useful else 0.3
    contributions["matrix"] = matrix_score * weights.matrix
    total_score += contributions["matrix"]
    
    # M6 Personnel
    personnel_advantage = leg_data.get("personnel_advantage", 50.0)
    personnel_score = leg_data.get("personnel_score", 0.5)
    if personnel_score == 0.5:
        personnel_score = min(1.0, max(0.0, personnel_advantage / 50.0))
    contributions["personnel"] = personnel_score * weights.personnel
    total_score += contributions["personnel"]
    
    # M27 H2H
    h2h_score_raw = leg_data.get("h2h_score", 50.0)
    h2h_score = leg_data.get("h2h_normalized", 0.5)
    if h2h_score == 0.5:
        h2h_score = h2h_score_raw / 100.0
    contributions["h2h"] = h2h_score * weights.h2h
    total_score += contributions["h2h"]
    
    # M26 Context
    context_importance = leg_data.get("match_importance", 0.5)
    contributions["context"] = context_importance * weights.context
    total_score += contributions["context"]
    
    # Thresholds for decision
    if total_score >= 0.70:
        final_status = "APPROVED"
        confidence = "HIGH"
    elif total_score >= 0.50:
        final_status = "CAUTION"
        confidence = "MEDIUM"
    else:
        final_status = "REJECTED"
        confidence = "LOW"
    
    return round(total_score, 3), final_status, confidence, contributions


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — LIVE FEEDBACK LOOP (from M20)
# ═══════════════════════════════════════════════════════════════

@dataclass
class MatchResult:
    """
    Actual outcome for one match.
    
    Attributes:
        match: Match identifier (match_id)
        actual_outcome: "HOME_WIN", "AWAY_WIN", or "DRAW"
        home_goals: Number of goals scored by home team
        away_goals: Number of goals scored by away team
    """
    match: str
    actual_outcome: str   # HOME_WIN / AWAY_WIN / DRAW
    home_goals: int = 0
    away_goals: int = 0


@dataclass
class EvaluationRecord:
    """Record of one prediction vs actual outcome."""
    match: str
    predicted: str
    actual: str
    correct: bool
    confidence: str
    odds: float
    edge: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class PerformanceReport:
    """Performance statistics from evaluated predictions."""
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    high_conf_accuracy: float = 0.0
    medium_conf_accuracy: float = 0.0
    low_conf_accuracy: float = 0.0
    profit_estimate: float = 0.0
    roi: float = 0.0
    brier_score: float = 0.0
    calibration_error: float = 0.0
    notes: List[str] = field(default_factory=list)


def evaluate_predictions(
    verdicts: List[Any],
    results: List[MatchResult],
) -> List[EvaluationRecord]:
    """
    Compare MasterVerdict predictions against actual match results.
    
    Args:
        verdicts: List of MasterVerdict objects (from Module 11)
        results: List of MatchResult objects (actual outcomes)
    
    Returns:
        List of EvaluationRecord for each evaluated prediction
    """
    records = []
    result_map = {r.match: r for r in results}
    
    for v in verdicts:
        # Safely extract leg and match_id
        leg = getattr(v, 'oracle', None)
        if leg is None:
            continue
        leg_obj = getattr(leg, 'leg', None)
        if leg_obj is None:
            continue
            
        match_name = getattr(leg_obj, 'match_id', None)
        if match_name is None or match_name not in result_map:
            continue
        
        r = result_map[match_name]
        
        # Get prediction (selection from leg)
        predicted = getattr(leg_obj, 'selection', '?')
        
        # Map predicted selection to outcome format
        home_team = getattr(leg_obj, 'home_profile', None)
        home_name = getattr(home_team, 'team_name', '') if home_team else ''
        
        # Determine if prediction was correct
        if predicted == home_name and r.actual_outcome == "HOME_WIN":
            correct = True
        elif predicted != home_name and predicted != "Draw" and r.actual_outcome == "AWAY_WIN":
            correct = True
        elif predicted == "Draw" and r.actual_outcome == "DRAW":
            correct = True
        else:
            correct = False
        
        records.append(EvaluationRecord(
            match=match_name,
            predicted=predicted,
            actual=r.actual_outcome,
            correct=correct,
            confidence=getattr(v, 'final_confidence', 'LOW'),
            odds=getattr(leg_obj, 'odds', 0.0),
            edge=getattr(leg, 'edge', 0.0) if hasattr(leg, 'edge') else 0.0,
        ))
    
    return records


def build_performance_report(records: List[EvaluationRecord]) -> PerformanceReport:
    """
    Build performance report from evaluation records.
    
    Args:
        records: List of EvaluationRecord
    
    Returns:
        PerformanceReport with accuracy, ROI, and recommendations
    """
    report = PerformanceReport()
    
    if not records:
        report.notes.append("No records to evaluate")
        return report
    
    report.total_predictions = len(records)
    report.correct_predictions = sum(1 for r in records if r.correct)
    report.accuracy = report.correct_predictions / report.total_predictions if report.total_predictions > 0 else 0.0
    
    # Accuracy by confidence level
    high = [r for r in records if r.confidence == "HIGH"]
    medium = [r for r in records if r.confidence == "MEDIUM"]
    low = [r for r in records if r.confidence == "LOW"]
    
    if high:
        report.high_conf_accuracy = sum(1 for r in high if r.correct) / len(high)
    if medium:
        report.medium_conf_accuracy = sum(1 for r in medium if r.correct) / len(medium)
    if low:
        report.low_conf_accuracy = sum(1 for r in low if r.correct) / len(low)
    
    # Profit estimate (assuming 1 unit stake per bet)
    profit = sum((r.odds - 1) if r.correct else -1 for r in records)
    report.profit_estimate = round(profit, 2)
    report.roi = round(profit / report.total_predictions, 4) if report.total_predictions > 0 else 0.0
    
    # Brier score (simplified - uses confidence as probability)
    predictions = []
    outcomes = []
    for r in records:
        # Map confidence to probability
        conf_probs = {"HIGH": 0.7, "MEDIUM": 0.55, "LOW": 0.5}
        pred_prob = conf_probs.get(r.confidence, 0.5)
        predictions.append(pred_prob)
        outcomes.append(1 if r.correct else 0)
    
    if predictions:
        report.brier_score = round(sum((p - o) ** 2 for p, o in zip(predictions, outcomes)) / len(predictions), 4)
    
    # Generate notes
    if report.accuracy < 0.5:
        report.notes.append("Low overall accuracy — system needs tightening")
    if report.high_conf_accuracy < 0.6 and high:
        report.notes.append("High-confidence predictions underperforming")
    if report.medium_conf_accuracy < 0.5 and medium:
        report.notes.append("Medium-confidence predictions below random")
    if report.roi < 0:
        report.notes.append(f"Negative ROI: {report.roi:.1%}")
    if report.brier_score > 0.22:
        report.notes.append(f"High Brier score ({report.brier_score:.4f}) - confidence calibration needed")
    
    return report


def generate_adjustments(report: PerformanceReport) -> Dict[str, Any]:
    """
    Translate performance gaps into concrete adjustment flags.
    
    Args:
        report: PerformanceReport from build_performance_report
    
    Returns:
        Dictionary of adjustment flags
    """
    adj: Dict[str, Any] = {
        "increase_thresholds": False,
        "reduce_risk": False,
        "confidence_bias": "neutral",
        "raise_edge_requirement": False,
        "calibration_needed": False,
    }
    
    if report.accuracy < 0.5:
        adj["increase_thresholds"] = True
        adj["reduce_risk"] = True
    
    if report.high_conf_accuracy < 0.6 and report.high_conf_accuracy > 0:
        adj["confidence_bias"] = "overconfident"
    
    if report.roi < 0:
        adj["reduce_risk"] = True
    
    if report.accuracy < 0.45:
        adj["raise_edge_requirement"] = True
    
    if report.brier_score > 0.22:
        adj["calibration_needed"] = True
    
    return adj


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — AUTO-RECALIBRATION (from M23)
# ═══════════════════════════════════════════════════════════════

@dataclass
class PerformanceSnapshot:
    """Performance snapshot from historical data."""
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    home_failures: int = 0
    underdog_hits: int = 0
    draw_failures: int = 0
    avg_edge: float = 0.0
    avg_confidence: float = 0.0
    brier_score: float = 0.0


def analyze_performance(history: List[Dict[str, Any]]) -> PerformanceSnapshot:
    """
    Build a PerformanceSnapshot from Module 16 DB history rows.
    
    Args:
        history: List of dicts from module16.get_all_feedback()
    
    Returns:
        PerformanceSnapshot with aggregated statistics
    """
    snapshot = PerformanceSnapshot()
    
    if not history:
        return snapshot
    
    edges = []
    confidences = []
    predictions = []
    outcomes = []
    
    for h in history:
        snapshot.total_bets += 1
        
        actual = h.get("actual_result")
        predicted = h.get("prediction")
        correct = h.get("correct", 0)
        
        if correct:
            snapshot.wins += 1
        else:
            snapshot.losses += 1
        
        # Track specific failure patterns
        if actual == "DRAW" and predicted != "Draw":
            snapshot.draw_failures += 1
        if predicted == "HOME" and actual != "HOME_WIN":
            snapshot.home_failures += 1
        if predicted == "AWAY" and actual == "AWAY_WIN":
            snapshot.underdog_hits += 1
        
        # Metrics
        if "edge" in h and h["edge"]:
            try:
                edges.append(float(h["edge"]))
            except (ValueError, TypeError):
                pass
        
        # Confidence score mapping
        conf = h.get("confidence", "LOW")
        conf_score = {"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.5}.get(conf, 0.5)
        confidences.append(conf_score)
        
        # For Brier score
        predictions.append(conf_score)
        outcomes.append(1 if correct else 0)
    
    if edges:
        snapshot.avg_edge = statistics.mean(edges)
    if confidences:
        snapshot.avg_confidence = statistics.mean(confidences)
    if predictions and outcomes:
        snapshot.brier_score = sum((p - o) ** 2 for p, o in zip(predictions, outcomes)) / len(predictions)
    
    return snapshot


def recalibrate_system(config: SystemConfig, snapshot: PerformanceSnapshot) -> SystemConfig:
    """
    Adjust system thresholds based on performance snapshot.
    
    Args:
        config: Current SystemConfig
        snapshot: PerformanceSnapshot from historical data
    
    Returns:
        Updated SystemConfig
    """
    total = max(snapshot.total_bets, 1)
    
    # 1. Home failure: tighten home gate
    if snapshot.home_failures > total * 0.4:
        config.home_win_threshold += 0.02
        config.opponent_win_cap -= 0.02
        logger.info(f"M18: Home failure rate high ({snapshot.home_failures/total:.1%}) - raising threshold")

    # 2. Draw failure: require more edge
    if snapshot.draw_failures > total * 0.25:
        config.min_edge += 0.01
        config.edge_threshold += 0.01
        logger.info(f"M18: Draw failure rate high ({snapshot.draw_failures/total:.1%}) - raising edge requirement")

    # 3. Underdog hits: trust underdog signals more
    if snapshot.underdog_hits > total * 0.3:
        config.underdog_weight += 0.2
        # Also adjust decision weights
        config.decision_weights.underdog = min(0.20, config.decision_weights.underdog + 0.02)
        logger.info(f"M18: Underdog success rate high ({snapshot.underdog_hits/total:.1%}) - increasing underdog weight")

    # 4. Low edge quality: lean harder on oracle
    if snapshot.avg_edge < 0.03 and snapshot.avg_edge > 0:
        config.oracle_weight += 0.2
        config.decision_weights.oracle_prefilter = min(0.25, config.decision_weights.oracle_prefilter + 0.02)
        config.decision_weights.oracle_forensics = min(0.20, config.decision_weights.oracle_forensics + 0.02)
        logger.info(f"M18: Low average edge ({snapshot.avg_edge:.3f}) - increasing oracle weight")

    # 5. Overconfident system: raise confidence bar
    if snapshot.avg_confidence > 0.75 and snapshot.losses > snapshot.wins:
        config.home_win_threshold += 0.01
        config.confidence_threshold += 0.02
        logger.info(f"M18: Overconfidence detected - raising thresholds")

    # 6. Brier score calibration
    if snapshot.brier_score > 0.22:
        config.confidence_threshold += 0.03
        logger.info(f"M18: High Brier score ({snapshot.brier_score:.4f}) - raising confidence threshold")

    # 7. General risk reduction
    if snapshot.losses > snapshot.wins and snapshot.total_bets >= 20:
        config.risk_tolerance = max(config.risk_tolerance * 0.9, 0.5)
        logger.info(f"M18: losing record ({snapshot.wins}/{snapshot.losses}) - reducing risk tolerance")

    # Normalize decision weights
    config.decision_weights.normalize()
    
    # Increment recalibration counter
    config.recalibration_count += 1
    config.last_updated = datetime.now(timezone.utc).isoformat()

    # Clamp all values
    config.clamp()
    
    return config


def adjust_for_league(config: SystemConfig, league_volatility: float) -> SystemConfig:
    """
    Scale thresholds based on measured league volatility (0–1).
    """
    if league_volatility > 0.6:
        config.home_win_threshold = min(
            config.home_win_threshold * config.high_risk_league_multiplier, 0.70
        )
        config.min_edge *= 1.2
        logger.info(f"M18: High volatility league ({league_volatility:.2f}) - raised thresholds")
    elif league_volatility < 0.3:
        config.home_win_threshold = max(
            config.home_win_threshold * config.low_risk_league_multiplier, 0.52
        )
        config.min_edge *= 0.9
        logger.info(f"M18: Low volatility league ({league_volatility:.2f}) - lowered thresholds")
    
    config.clamp()
    return config


def run_auto_recalibration(
    config: SystemConfig,
    history: List[Dict[str, Any]],
    league_volatility: float = 0.5,
    min_samples: int = 10,
) -> SystemConfig:
    """
    Full recalibration pipeline: snapshot → adjust → league scale.
    """
    if len(history) < min_samples:
        logger.info(f"M18: Insufficient samples ({len(history)} < {min_samples}) - skipping recalibration")
        return config
    
    snapshot = analyze_performance(history)
    config = recalibrate_system(config, snapshot)
    config = adjust_for_league(config, league_volatility)
    
    logger.info(f"M18: Recalibration complete - home_threshold={config.home_win_threshold:.3f}, min_edge={config.min_edge:.3f}")
    
    return config


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — CONFIG PERSISTENCE (from M21)
# ═══════════════════════════════════════════════════════════════

def persist_config(config: SystemConfig, db=None) -> bool:
    """Save the current SystemConfig to the database."""
    if db is not None and hasattr(db, "save_config"):
        try:
            db.save_config(config.to_dict())
            logger.info("M18: Configuration persisted to database")
            return True
        except Exception as e:
            logger.error(f"M18: Failed to persist config: {e}")
            return False
    else:
        logger.warning("M18: No database module provided - config not persisted")
        return False


def load_config_from_db(db=None) -> SystemConfig:
    """Load a previously saved SystemConfig from database."""
    if db is not None and hasattr(db, "load_config"):
        try:
            data = db.load_config()
            if data:
                config = SystemConfig.from_dict(data)
                logger.info("M18: Configuration loaded from database")
                return config.clamp()
        except Exception as e:
            logger.error(f"M18: Failed to load config: {e}")
    
    logger.info("M18: Using default configuration")
    return SystemConfig().clamp()


def adjust_confidence_label(confidence: str, config: SystemConfig) -> str:
    """Dynamically adjust confidence labels based on system state."""
    if config.confidence_threshold > 0.65 and confidence == "HIGH":
        return "MEDIUM"
    return confidence


def adjust_probability_calibration(
    raw_prob: float,
    historical_accuracy: Dict[str, float],
    min_samples: int = 20,
) -> float:
    """
    Calibrate probabilities based on historical confidence accuracy.
    
    Args:
        raw_prob: Raw model probability (0-1)
        historical_accuracy: Dict mapping confidence level to historical accuracy
        min_samples: Minimum samples per confidence level for adjustment
    
    Returns:
        Calibrated probability
    """
    # Determine confidence level buckets
    if raw_prob >= 0.70:
        level = "HIGH"
    elif raw_prob >= 0.55:
        level = "MEDIUM"
    else:
        level = "LOW"
    
    # Get historical accuracy for this level
    hist_acc = historical_accuracy.get(level, raw_prob)
    samples = historical_accuracy.get(f"{level}_samples", 0)
    
    if samples >= min_samples:
        # Apply calibration: move probability toward historical accuracy
        calibrated = (raw_prob + hist_acc) / 2
    else:
        calibrated = raw_prob
    
    return max(0.05, min(0.95, calibrated))


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — MASTER ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def run_full_learning_cycle(
    verdicts: List[Any],
    results: List[MatchResult],
    db=None,
    league_volatility: float = 0.5,
    min_samples: int = 10,
    update_weights: bool = True,
) -> Dict[str, Any]:
    """
    One-call wrapper for the complete post-match learning cycle.
    
    Args:
        verdicts: List of MasterVerdict objects
        results: List of MatchResult objects (actual outcomes)
        db: Database module for persistence and history
        league_volatility: Volatility score for league (0-1)
        min_samples: Minimum samples for recalibration
        update_weights: Whether to update decision weights based on performance
    
    Returns:
        Dictionary with all intermediate outputs for inspection
    """
    # Step 1 & 2: Feedback
    records = evaluate_predictions(verdicts, results)
    report = build_performance_report(records)
    adjustments = generate_adjustments(report)
    
    logger.info(f"M18: Performance - Accuracy={report.accuracy:.1%}, ROI={report.roi:.1%}")
    
    # Step 3: Load config
    config = load_config_from_db(db)
    
    # Step 4: Apply adjustment flags
    if adjustments.get("increase_thresholds"):
        config.edge_threshold += 0.01
        config.confidence_threshold += 0.02
        logger.info("M18: Applied threshold increase")
    
    if adjustments.get("reduce_risk"):
        config.risk_tolerance = max(config.risk_tolerance * 0.9, 0.5)
        logger.info("M18: Applied risk reduction")
    
    if adjustments.get("confidence_bias") == "overconfident":
        config.confidence_threshold += 0.03
        logger.info("M18: Applied overconfidence correction")
    
    if adjustments.get("raise_edge_requirement"):
        config.min_edge += 0.01
        logger.info("M18: Raised edge requirement")
    
    # Step 5: Update decision weights based on module performance
    if update_weights and db is not None:
        try:
            # Calculate module performance from history
            history = db.get_all_feedback() if hasattr(db, "get_all_feedback") else []
            if len(history) >= min_samples:
                module_performance = {}
                
                # Oracle performance (M4/M5 combined)
                oracle_correct = sum(1 for h in history if h.get("status", "").startswith("APPROVED") and h.get("correct"))
                oracle_total = sum(1 for h in history if h.get("status", "").startswith("APPROVED"))
                module_performance["oracle_prefilter"] = oracle_correct / max(oracle_total, 1)
                module_performance["oracle_forensics"] = oracle_correct / max(oracle_total, 1)
                
                # AI performance
                ai_correct = sum(1 for h in history if "AI CONSENSUS" in h.get("status", "") and h.get("correct"))
                ai_total = sum(1 for h in history if "AI CONSENSUS" in h.get("status", ""))
                module_performance["ai_consensus"] = ai_correct / max(ai_total, 1)
                
                # Use overall accuracy as proxy for other modules
                overall_acc = report.accuracy if report.accuracy > 0 else 0.5
                for module in ["dual_pattern", "underdog", "matrix", "personnel", "h2h", "context"]:
                    module_performance[module] = overall_acc
                
                # Adjust weights
                config.decision_weights = adjust_weights_by_performance(
                    config.decision_weights, module_performance, learning_rate=0.03
                )
                logger.info("M18: Decision weights updated based on module performance")
        except Exception as e:
            logger.warning(f"M18: Weight update failed: {e}")
    
    # Step 6: Full recalibration
    if db is not None and hasattr(db, "get_all_feedback"):
        history = db.get_all_feedback()
        config = run_auto_recalibration(config, history, league_volatility, min_samples)
    
    # Step 7: Persist
    persist_config(config, db)
    
    return {
        "records": records,
        "performance": report,
        "adjustments": adjustments,
        "new_config": config,
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — DEBUG OUTPUT FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def print_recalibration(config: SystemConfig) -> None:
    """Pretty print SystemConfig to console."""
    print("\n" + "=" * 60)
    print("   AUTO-RECALIBRATION REPORT")
    print("=" * 60)
    print(f"Home Win Threshold   : {config.home_win_threshold:.3f}")
    print(f"Opponent Win Cap     : {config.opponent_win_cap:.3f}")
    print(f"Min Edge             : {config.min_edge:.3f}")
    print(f"Edge Threshold       : {config.edge_threshold:.3f}")
    print(f"Confidence Threshold : {config.confidence_threshold:.3f}")
    print(f"Risk Tolerance       : {config.risk_tolerance:.2f}")
    print(f"\nLegacy Weights:")
    print(f"  Oracle Weight   : {config.oracle_weight:.2f}")
    print(f"  Dual Weight     : {config.dual_weight:.2f}")
    print(f"  Underdog Weight : {config.underdog_weight:.2f}")
    print(f"  Matrix Weight   : {config.matrix_weight:.2f}")
    print("\n" + config.decision_weights.summary())
    print(f"\nRecalibrations Run : {config.recalibration_count}")
    print(f"Last Updated       : {config.last_updated}")
    print("=" * 60)


def print_performance_report(report: PerformanceReport) -> None:
    """Pretty print PerformanceReport to console."""
    print("\n" + "=" * 60)
    print("   PERFORMANCE REPORT")
    print("=" * 60)
    print(f"Total Predictions : {report.total_predictions}")
    print(f"Correct           : {report.correct_predictions}")
    print(f"Accuracy          : {report.accuracy * 100:.1f}%")
    print(f"High Conf Acc     : {report.high_conf_accuracy * 100:.1f}%")
    print(f"Medium Conf Acc   : {report.medium_conf_accuracy * 100:.1f}%")
    print(f"Low Conf Acc      : {report.low_conf_accuracy * 100:.1f}%")
    print(f"Profit Estimate   : {report.profit_estimate:+.2f} units")
    print(f"ROI               : {report.roi * 100:+.1f}%")
    print(f"Brier Score       : {report.brier_score:.4f}")
    for note in report.notes:
        print(f"⚠  {note}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════
# SECTION 10 — EXPORTS
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Weighted decision
    "DecisionWeights",
    "calculate_weighted_decision",
    "adjust_weights_by_performance",
    # System Config
    "SystemConfig",
    # Data classes
    "MatchResult",
    "EvaluationRecord",
    "PerformanceReport",
    "PerformanceSnapshot",
    # Core functions
    "evaluate_predictions",
    "build_performance_report",
    "generate_adjustments",
    "analyze_performance",
    "recalibrate_system",
    "adjust_for_league",
    "run_auto_recalibration",
    # Persistence
    "persist_config",
    "load_config_from_db",
    "adjust_confidence_label",
    "adjust_probability_calibration",
    # Master entry point
    "run_full_learning_cycle",
    # Debug output
    "print_recalibration",
    "print_performance_report",
]


# ═══════════════════════════════════════════════════════════════
# SECTION 11 — EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MODULE 18: AUTO-RECALIBRATION WITH WEIGHTED DECISIONS")
    print("=" * 70)
    
    # Test DecisionWeights
    print("\n📊 DECISION WEIGHTS TEST")
    print("-" * 40)
    weights = DecisionWeights()
    print(weights.summary())
    
    # Test weighted decision calculation
    print("\n📊 WEIGHTED DECISION TEST")
    print("-" * 40)
    
    test_leg_data = {
        "pre_filter_passed": True,
        "pre_filter_score": 0.85,
        "failure_score": 3.5,
        "ai_verdict": "APPROVE",
        "ai_score": 0.82,
        "dual_risk_level": "LOW",
        "dual_risk_score": 0.8,
        "underdog_edge": 0.08,
        "underdog_score": 0.53,
        "matrix_useful": True,
        "matrix_score": 0.7,
        "personnel_advantage": 65,
        "personnel_score": 0.65,
        "h2h_score": 72,
        "h2h_normalized": 0.72,
        "match_importance": 0.85,
    }
    
    score, status, confidence, contributions = calculate_weighted_decision(test_leg_data, weights)
    
    print(f"Total Score: {score}")
    print(f"Final Status: {status}")
    print(f"Confidence: {confidence}")
    print("\nContributions:")
    for module, contribution in sorted(contributions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {module}: {contribution:.3f}")
    
    # Test weight adjustment
    print("\n📊 WEIGHT ADJUSTMENT TEST")
    print("-" * 40)
    
    module_performance = {
        "oracle_prefilter": 0.65,
        "oracle_forensics": 0.60,
        "ai_consensus": 0.55,
        "dual_pattern": 0.45,
        "underdog": 0.70,
        "matrix": 0.50,
        "personnel": 0.40,
        "h2h": 0.55,
        "context": 0.60,
    }
    
    adjusted_weights = adjust_weights_by_performance(weights, module_performance, learning_rate=0.05)
    print(adjusted_weights.summary())
    
    # Test probability calibration
    print("\n📊 PROBABILITY CALIBRATION TEST")
    print("-" * 40)
    
    historical_accuracy = {
        "HIGH": 0.62,
        "HIGH_samples": 50,
        "MEDIUM": 0.55,
        "MEDIUM_samples": 30,
        "LOW": 0.51,
        "LOW_samples": 20,
    }
    
    for raw_prob in [0.75, 0.60, 0.45]:
        cal = adjust_probability_calibration(raw_prob, historical_accuracy)
        print(f"  {raw_prob:.2f} → {cal:.2f}")
    
    print("\n" + "=" * 70)
    print("MODULE 18 READY FOR PRODUCTION")
    print("=" * 70)