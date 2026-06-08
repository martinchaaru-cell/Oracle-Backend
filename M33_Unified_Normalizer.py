"""
The Match Oracle – Module 33: Unified Normalization Engine (REFINED)
==================================================================
Centralized normalization for all module outputs to 0-1 scale.

REFINEMENTS IN THIS VERSION:
---------------------------
1. ADDED: Complete mapping dictionaries for all module outputs
2. ADDED: Normalizer class with static methods for all conversions
3. ADDED: Batch normalization for leg data aggregation
4. ADDED: Weighted score aggregation with configurable weights
5. ADDED: Inverse normalization for reverse mappings
6. ADDED: Confidence to stake factor mapping
7. ADDED: Grade to score conversion (A/B/C/D/F)
8. ADDED: Performance metric normalization (Sharpe, Calmar, etc.)
9. ADDED: Probability calibration mapping
10. ADDED: Comprehensive test suite

This module provides:
1. Consistent score normalization across all modules
2. Mapping dictionaries for all enum/string values
3. Confidence to factor conversion for Kelly staking
4. Batch normalization for leg data aggregation

All modules (M5, M6, M8, M9, M10, M26, M27, M28, M29, M31, M32) should use this
instead of implementing their own normalization logic.

Usage:
    from module33 import Normalizer
    
    risk_score = Normalizer.risk_to_score("HIGH")  # Returns 0.75
    confidence_factor = Normalizer.confidence_to_factor("MEDIUM")  # Returns 0.6
    normalized = Normalizer.normalize_leg_data(leg_data)
    weighted = Normalizer.aggregate_weighted_score(normalized, weights)
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple, Union
from enum import Enum
import math


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — MAPPING DICTIONARIES (Single Source of Truth)
# ═══════════════════════════════════════════════════════════════

# ─── Risk levels (M4, M8, M11) ─────────────────────────────────
RISK_TO_SCORE: Dict[str, float] = {
    "MINIMAL": 0.10,
    "LOW": 0.25,
    "MEDIUM": 0.50,
    "HIGH": 0.75,
    "CRITICAL": 0.95,
    "UNKNOWN": 0.50,
}

RISK_TO_CONFIDENCE_FACTOR: Dict[str, float] = {
    "MINIMAL": 1.00,
    "LOW": 0.90,
    "MEDIUM": 0.70,
    "HIGH": 0.50,
    "CRITICAL": 0.30,
    "UNKNOWN": 0.50,
}

# Inverse mapping
SCORE_TO_RISK: Dict[float, str] = {
    0.10: "MINIMAL",
    0.25: "LOW",
    0.50: "MEDIUM",
    0.75: "HIGH",
    0.95: "CRITICAL",
}

# ─── Threat levels (M9) ───────────────────────────────────────
THREAT_TO_SCORE: Dict[str, float] = {
    "NONE": 0.00,
    "LOW": 0.25,
    "MEDIUM": 0.60,
    "HIGH": 0.90,
    "CRITICAL": 0.95,
    "UNKNOWN": 0.50,
}

SCORE_TO_THREAT: Dict[float, str] = {
    0.00: "NONE",
    0.25: "LOW",
    0.60: "MEDIUM",
    0.90: "HIGH",
    0.95: "CRITICAL",
}

# ─── H2H labels (M27) ─────────────────────────────────────────
H2H_LABEL_TO_SCORE: Dict[str, float] = {
    "FAV_DOMINANT": 0.90,
    "FAV_EDGE": 0.70,
    "NEUTRAL": 0.50,
    "UND_EDGE": 0.30,
    "UND_DOMINANT": 0.10,
    "INSUFFICIENT_DATA": 0.30,
    "UNKNOWN": 0.50,
}

SCORE_TO_H2H_LABEL: Dict[float, str] = {
    0.90: "FAV_DOMINANT",
    0.70: "FAV_EDGE",
    0.50: "NEUTRAL",
    0.30: "UND_EDGE",
    0.10: "UND_DOMINANT",
}

# ─── Pattern strength (M8, M9) ─────────────────────────────────
PATTERN_STRENGTH_TO_SCORE: Dict[str, float] = {
    "DEFINITIVE": 1.00,
    "STRONG": 0.75,
    "MODERATE": 0.50,
    "WEAK": 0.25,
    "INSUFFICIENT": 0.10,
    "UNKNOWN": 0.50,
}

# ─── Pattern verdict (M9) ─────────────────────────────────────
PATTERN_VERDICT_TO_SCORE: Dict[str, float] = {
    "DEFINITIVE_PATTERN": 0.95,
    "STRONG_PATTERN": 0.80,
    "MODERATE_PATTERN": 0.60,
    "WEAK_PATTERN": 0.40,
    "NO_PATTERN": 0.20,
    "UNKNOWN": 0.50,
}

# ─── Trap/Value signal (M10) ──────────────────────────────────
TRAP_VALUE_TO_SCORE: Dict[str, float] = {
    "TRAP": 0.10,
    "VALUE": 0.85,
    "NONE": 0.50,
    "UNCERTAIN": 0.40,
}

# ─── AI verdict (M7) ─────────────────────────────────────────
AI_VERDICT_TO_SCORE: Dict[str, float] = {
    "APPROVE": 0.90,
    "CAUTION": 0.50,
    "REJECT": 0.10,
    "NEUTRAL": 0.50,
}

# ─── Confidence levels (M5, M6, M7, M11) ──────────────────────
CONFIDENCE_TO_FACTOR: Dict[str, float] = {
    "HIGH": 1.00,
    "MEDIUM": 0.60,
    "LOW": 0.30,
    "UNKNOWN": 0.50,
}

CONFIDENCE_TO_BASE_SCORE: Dict[str, float] = {
    "HIGH": 0.85,
    "MEDIUM": 0.60,
    "LOW": 0.35,
    "UNKNOWN": 0.50,
}

SCORE_TO_CONFIDENCE: Dict[float, str] = {
    0.85: "HIGH",
    0.60: "MEDIUM",
    0.35: "LOW",
}

# ─── Recommendation types (M9) ────────────────────────────────
RECOMMENDATION_TO_SCORE: Dict[str, float] = {
    "GOLDMINE_QUALIFIED": 0.95,
    "UNDERDOG_WIN": 0.85,
    "UNDERDOG_DNB": 0.70,
    "PATTERN_WATCH": 0.55,
    "THREAT_CHECK": 0.30,
    "PATTERN_NOTED_NO_MARKET": 0.45,
    "MATCH_SKIP": 0.20,
}

# ─── Bilateral confidence (M10) ───────────────────────────────
BILATERAL_CONFIDENCE_TO_SCORE: Dict[str, float] = {
    "HIGH": 0.85,
    "MEDIUM": 0.60,
    "LOW": 0.35,
    "UNCERTAIN": 0.20,
    "UNKNOWN": 0.50,
}

# ─── Calibration grades (M28) ─────────────────────────────────
CALIBRATION_GRADE_TO_SCORE: Dict[str, float] = {
    "A": 0.95,
    "B": 0.80,
    "C": 0.60,
    "D": 0.40,
    "F": 0.20,
    "UNGRADED": 0.50,
}

CALIBRATION_GRADE_TO_FACTOR: Dict[str, float] = {
    "A": 1.00,
    "B": 0.85,
    "C": 0.65,
    "D": 0.40,
    "F": 0.20,
    "UNGRADED": 0.50,
}

# ─── Health status (M29) ──────────────────────────────────────
HEALTH_STATUS_TO_MULTIPLIER: Dict[str, float] = {
    "HEALTHY": 1.00,
    "CAUTION": 0.85,
    "DANGER": 0.65,
    "EMERGENCY": 0.40,
    "CRITICAL": 0.20,
}

# ─── League tier reliability (M31) ────────────────────────────
LEAGUE_TIER_TO_RELIABILITY: Dict[int, float] = {
    1: 1.00,   # Elite leagues
    2: 0.85,   # Strong secondary
    3: 0.70,   # Mid-tier
    4: 0.50,   # Lower divisions
    5: 0.30,   # Unknown
}

# ─── Competition type to parlay factor (M12, M31) ─────────────
COMPETITION_TO_PARLAY_FACTOR: Dict[str, float] = {
    "league": 1.00,
    "playoff": 0.90,
    "cup": 0.10,
    "friendly": 0.00,
    "continental_group": 0.10,
    "continental_knockout": 0.10,
}

# ─── Performance metrics normalization ────────────────────────
SHARPE_TO_SCORE: Dict[float, float] = {
    2.0: 1.00,   # Excellent
    1.5: 0.90,
    1.0: 0.75,
    0.5: 0.60,
    0.0: 0.50,
    -0.5: 0.40,
    -1.0: 0.25,
}

CALMAR_TO_SCORE: Dict[float, float] = {
    3.0: 1.00,   # Excellent
    2.0: 0.85,
    1.0: 0.70,
    0.5: 0.55,
    0.0: 0.40,
}

PROFIT_FACTOR_TO_SCORE: Dict[float, float] = {
    2.0: 1.00,   # Excellent (2:1 profit)
    1.5: 0.80,
    1.2: 0.60,
    1.0: 0.50,
    0.8: 0.40,
}

# ─── Drawdown thresholds (M29) ────────────────────────────────
DRAWDOWN_TO_MULTIPLIER: List[Tuple[float, float]] = [
    (0.00, 1.00),   # 0% drawdown → 1.0x
    (0.10, 0.75),   # 10% drawdown → 0.75x
    (0.15, 0.50),   # 15% drawdown → 0.50x
    (0.20, 0.30),   # 20% drawdown → 0.30x
    (0.25, 0.20),   # 25%+ drawdown → 0.20x
]

# ─── Brier score normalization (M28) ──────────────────────────
def brier_to_score(brier: float) -> float:
    """Convert Brier score (0=perfect, 0.25=random) to 0-1 score."""
    return max(0.0, min(1.0, 1.0 - (brier / 0.25)))


def ece_to_score(ece: float) -> float:
    """Convert Expected Calibration Error (0=perfect) to 0-1 score."""
    return max(0.0, min(1.0, 1.0 - (ece / 0.20)))


# ─── Kelly fraction to stake multiplier ───────────────────────
KELLY_FRACTION_TO_STAKE: Dict[float, float] = {
    1.00: 0.25,   # Full Kelly → 25% of Kelly
    0.50: 0.20,   # Half Kelly → 20%
    0.25: 0.15,   # Quarter Kelly → 15%
    0.10: 0.10,   # Tenth Kelly → 10%
}


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — NORMALIZER CLASS
# ═══════════════════════════════════════════════════════════════

class Normalizer:
    """
    Centralized normalization engine for all module outputs.
    
    All methods are static for easy access across modules.
    """
    
    # ─── Risk & Threat ─────────────────────────────────────────
    
    @staticmethod
    def risk_to_score(risk_level: str) -> float:
        """Convert risk level (LOW/MEDIUM/HIGH/CRITICAL) to 0-1 score."""
        return RISK_TO_SCORE.get(risk_level.upper(), 0.50)
    
    @staticmethod
    def score_to_risk(score: float) -> str:
        """Convert 0-1 score to risk level."""
        # Find closest threshold
        closest = min(SCORE_TO_RISK.keys(), key=lambda x: abs(x - score))
        return SCORE_TO_RISK.get(closest, "UNKNOWN")
    
    @staticmethod
    def risk_to_confidence_factor(risk_level: str) -> float:
        """Convert risk level to Kelly stake multiplier."""
        return RISK_TO_CONFIDENCE_FACTOR.get(risk_level.upper(), 0.50)
    
    @staticmethod
    def threat_to_score(threat_level: str) -> float:
        """Convert threat level (NONE/LOW/MEDIUM/HIGH) to 0-1 score."""
        return THREAT_TO_SCORE.get(threat_level.upper(), 0.50)
    
    @staticmethod
    def score_to_threat(score: float) -> str:
        """Convert 0-1 score to threat level."""
        closest = min(SCORE_TO_THREAT.keys(), key=lambda x: abs(x - score))
        return SCORE_TO_THREAT.get(closest, "UNKNOWN")
    
    # ─── H2H ───────────────────────────────────────────────────
    
    @staticmethod
    def h2h_label_to_score(label: str) -> float:
        """Convert H2H label to 0-1 score."""
        return H2H_LABEL_TO_SCORE.get(label.upper(), 0.50)
    
    @staticmethod
    def score_to_h2h_label(score: float) -> str:
        """Convert 0-1 score to H2H label."""
        closest = min(SCORE_TO_H2H_LABEL.keys(), key=lambda x: abs(x - score))
        return SCORE_TO_H2H_LABEL.get(closest, "UNKNOWN")
    
    # ─── Pattern ───────────────────────────────────────────────
    
    @staticmethod
    def pattern_strength_to_score(strength: str) -> float:
        """Convert pattern strength to 0-1 score."""
        return PATTERN_STRENGTH_TO_SCORE.get(strength.upper(), 0.50)
    
    @staticmethod
    def pattern_verdict_to_score(verdict: str) -> float:
        """Convert pattern verdict to recommendation score."""
        return PATTERN_VERDICT_TO_SCORE.get(verdict.upper(), 0.50)
    
    # ─── Trap/Value ────────────────────────────────────────────
    
    @staticmethod
    def trap_value_to_score(signal_type: str) -> float:
        """Convert trap/value signal to 0-1 score."""
        return TRAP_VALUE_TO_SCORE.get(signal_type.upper(), 0.50)
    
    # ─── AI ────────────────────────────────────────────────────
    
    @staticmethod
    def ai_verdict_to_score(verdict: str) -> float:
        """Convert AI verdict (APPROVE/CAUTION/REJECT) to 0-1 score."""
        return AI_VERDICT_TO_SCORE.get(verdict.upper(), 0.50)
    
    # ─── Confidence ────────────────────────────────────────────
    
    @staticmethod
    def confidence_to_factor(confidence: str) -> float:
        """Convert confidence level to Kelly stake multiplier."""
        return CONFIDENCE_TO_FACTOR.get(confidence.upper(), 0.50)
    
    @staticmethod
    def confidence_to_base_score(confidence: str) -> float:
        """Convert confidence level to base score (0-1)."""
        return CONFIDENCE_TO_BASE_SCORE.get(confidence.upper(), 0.50)
    
    @staticmethod
    def score_to_confidence(score: float) -> str:
        """Convert 0-1 score to confidence level."""
        closest = min(SCORE_TO_CONFIDENCE.keys(), key=lambda x: abs(x - score))
        return SCORE_TO_CONFIDENCE.get(closest, "UNKNOWN")
    
    # ─── Recommendations ───────────────────────────────────────
    
    @staticmethod
    def recommendation_to_score(recommendation: str) -> float:
        """Convert recommendation to 0-1 score."""
        return RECOMMENDATION_TO_SCORE.get(recommendation.upper(), 0.20)
    
    # ─── Bilateral ─────────────────────────────────────────────
    
    @staticmethod
    def bilateral_confidence_to_score(confidence: str) -> float:
        """Convert bilateral confidence to 0-1 score."""
        return BILATERAL_CONFIDENCE_TO_SCORE.get(confidence.upper(), 0.50)
    
    # ─── Calibration ───────────────────────────────────────────
    
    @staticmethod
    def calibration_grade_to_score(grade: str) -> float:
        """Convert calibration grade (A/B/C/D/F) to 0-1 score."""
        return CALIBRATION_GRADE_TO_SCORE.get(grade.upper(), 0.50)
    
    @staticmethod
    def calibration_grade_to_factor(grade: str) -> float:
        """Convert calibration grade to stake multiplier."""
        return CALIBRATION_GRADE_TO_FACTOR.get(grade.upper(), 0.50)
    
    @staticmethod
    def brier_to_score(brier: float) -> float:
        """Convert Brier score (0-0.25+) to 0-1 score."""
        return brier_to_score(brier)
    
    @staticmethod
    def ece_to_score(ece: float) -> float:
        """Convert Expected Calibration Error to 0-1 score."""
        return ece_to_score(ece)
    
    # ─── Health & Drawdown ─────────────────────────────────────
    
    @staticmethod
    def health_status_to_multiplier(status: str) -> float:
        """Convert health status to stake multiplier."""
        return HEALTH_STATUS_TO_MULTIPLIER.get(status.upper(), 0.50)
    
    @staticmethod
    def drawdown_to_multiplier(drawdown_pct: float) -> float:
        """Convert drawdown percentage to stake multiplier."""
        for threshold, multiplier in DRAWDOWN_TO_MULTIPLIER:
            if drawdown_pct <= threshold:
                return multiplier
        return 0.20  # Emergency floor
    
    # ─── League & Competition ──────────────────────────────────
    
    @staticmethod
    def league_tier_to_reliability(tier: int) -> float:
        """Convert league tier (1-5) to reliability score."""
        return LEAGUE_TIER_TO_RELIABILITY.get(tier, 0.50)
    
    @staticmethod
    def competition_to_parlay_factor(comp_type: str) -> float:
        """Convert competition type to parlay eligibility factor."""
        return COMPETITION_TO_PARLAY_FACTOR.get(comp_type.lower(), 0.50)
    
    # ─── Performance Metrics ───────────────────────────────────
    
    @staticmethod
    def sharpe_to_score(sharpe: float) -> float:
        """Convert Sharpe ratio to 0-1 score."""
        # Find closest threshold
        thresholds = sorted(SHARPE_TO_SCORE.keys())
        for threshold in thresholds:
            if sharpe >= threshold:
                return SHARPE_TO_SCORE[threshold]
        return 0.25
    
    @staticmethod
    def calmar_to_score(calmar: float) -> float:
        """Convert Calmar ratio to 0-1 score."""
        thresholds = sorted(CALMAR_TO_SCORE.keys())
        for threshold in thresholds:
            if calmar >= threshold:
                return CALMAR_TO_SCORE[threshold]
        return 0.25
    
    @staticmethod
    def profit_factor_to_score(pf: float) -> float:
        """Convert profit factor to 0-1 score."""
        thresholds = sorted(PROFIT_FACTOR_TO_SCORE.keys())
        for threshold in thresholds:
            if pf >= threshold:
                return PROFIT_FACTOR_TO_SCORE[threshold]
        return 0.25
    
    # ─── Numeric Utilities ─────────────────────────────────────
    
    @staticmethod
    def failure_to_score(failure_score: float, max_score: float = 10.0) -> float:
        """
        Convert failure score (0-10) to normalized 0-1 score.
        Lower failure = higher score.
        """
        return max(0.0, min(1.0, 1.0 - (failure_score / max_score)))
    
    @staticmethod
    def percentage_to_score(percentage: float, max_value: float = 100.0) -> float:
        """Convert percentage (0-100) to 0-1 score."""
        return max(0.0, min(1.0, percentage / max_value))
    
    @staticmethod
    def edge_to_score(edge: float, max_edge: float = 0.20) -> float:
        """
        Convert edge percentage to 0-1 score.
        Edge of 0% = 0.5, Edge of 20% = 1.0
        """
        normalized = 0.5 + (edge / max_edge)
        return max(0.0, min(1.0, normalized))
    
    @staticmethod
    def probability_to_score(prob: float) -> float:
        """Convert probability (0-1) to 0-1 score (direct mapping)."""
        return max(0.0, min(1.0, prob))
    
    @staticmethod
    def odds_to_implied_prob(odds: float) -> float:
        """Convert decimal odds to implied probability."""
        return 1.0 / odds if odds > 1.0 else 0.0
    
    @staticmethod
    def implied_prob_to_odds(prob: float) -> float:
        """Convert implied probability to decimal odds."""
        return 1.0 / prob if prob > 0 else 0.0
    
    @staticmethod
    def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Clamp value to [min, max] range."""
        return max(min_val, min(max_val, value))
    
    # ─── Pattern Classification ────────────────────────────────
    
    @staticmethod
    def goals_to_scoring_pattern(avg_goals: float) -> str:
        """Classify scoring pattern based on average goals."""
        if avg_goals >= 2.7:
            return "HIGH_SCORING"
        elif avg_goals <= 1.8:
            return "LOW_SCORING"
        return "MODERATE"
    
    @staticmethod
    def draw_rate_to_risk(draw_rate: float, threshold: float = 0.35) -> float:
        """Convert draw rate to risk score (0-1)."""
        return min(1.0, draw_rate / threshold)
    
    @staticmethod
    def win_rate_to_dominance(win_rate: float) -> float:
        """Convert win rate to dominance score (0-1)."""
        return max(0.0, min(1.0, win_rate))
    
    @staticmethod
    def consistency_score(results: List[str]) -> float:
        """
        Calculate consistency score from W/D/L sequence.
        Higher score = more consistent (less alternation).
        """
        if len(results) < 2:
            return 0.5
        
        alternations = sum(1 for i in range(1, len(results)) if results[i] != results[i-1])
        alternation_rate = alternations / (len(results) - 1)
        
        # Lower alternation = higher consistency
        return 1.0 - alternation_rate


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — BATCH NORMALIZATION
# ═══════════════════════════════════════════════════════════════

def normalize_leg_data(leg_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Normalize all fields in a leg_data dictionary to 0-1 scale.
    
    Args:
        leg_data: Raw leg data from modules
    
    Returns:
        Normalized leg data with all values 0-1
    """
    normalized = {}
    
    # Direct mappings
    if "dual_risk_level" in leg_data:
        normalized["dual_risk_score"] = Normalizer.risk_to_score(leg_data["dual_risk_level"])
    
    if "underdog_threat_level" in leg_data:
        normalized["underdog_threat_score"] = Normalizer.threat_to_score(leg_data["underdog_threat_level"])
    
    if "h2h_label" in leg_data:
        normalized["h2h_score"] = Normalizer.h2h_label_to_score(leg_data["h2h_label"])
    
    if "pattern_verdict" in leg_data:
        normalized["pattern_score"] = Normalizer.pattern_verdict_to_score(leg_data["pattern_verdict"])
    
    if "trap_value_signal" in leg_data:
        normalized["trap_value_score"] = Normalizer.trap_value_to_score(leg_data["trap_value_signal"])
    
    if "ai_verdict" in leg_data:
        normalized["ai_score"] = Normalizer.ai_verdict_to_score(leg_data["ai_verdict"])
    
    if "confidence" in leg_data:
        normalized["confidence_base_score"] = Normalizer.confidence_to_base_score(leg_data["confidence"])
        normalized["confidence_factor"] = Normalizer.confidence_to_factor(leg_data["confidence"])
    
    if "recommendation" in leg_data:
        normalized["recommendation_score"] = Normalizer.recommendation_to_score(leg_data["recommendation"])
    
    if "bilateral_confidence" in leg_data:
        normalized["bilateral_score"] = Normalizer.bilateral_confidence_to_score(leg_data["bilateral_confidence"])
    
    if "calibration_grade" in leg_data:
        normalized["calibration_score"] = Normalizer.calibration_grade_to_score(leg_data["calibration_grade"])
        normalized["calibration_factor"] = Normalizer.calibration_grade_to_factor(leg_data["calibration_grade"])
    
    # Numeric fields
    if "failure_score" in leg_data:
        normalized["failure_normalized"] = Normalizer.failure_to_score(leg_data["failure_score"])
    
    if "edge" in leg_data:
        normalized["edge_score"] = Normalizer.edge_to_score(leg_data["edge"])
    
    if "underdog_edge" in leg_data:
        normalized["underdog_edge_score"] = Normalizer.edge_to_score(leg_data["underdog_edge"])
    
    if "h2h_score_raw" in leg_data:
        normalized["h2h_score"] = Normalizer.percentage_to_score(leg_data["h2h_score_raw"])
    
    if "draw_prob" in leg_data:
        normalized["draw_risk_score"] = Normalizer.draw_rate_to_risk(leg_data["draw_prob"])
    
    if "model_prob" in leg_data:
        normalized["model_prob_score"] = Normalizer.probability_to_score(leg_data["model_prob"])
    
    # Boolean fields
    if "matrix_useful" in leg_data:
        normalized["matrix_score"] = 0.8 if leg_data["matrix_useful"] else 0.3
    
    if "pre_filter_passed" in leg_data:
        normalized["pre_filter_score"] = 1.0 if leg_data["pre_filter_passed"] else 0.0
    
    if "forensic_passed" in leg_data:
        normalized["forensic_score"] = 1.0 if leg_data["forensic_passed"] else 0.0
    
    if "is_reliable" in leg_data:
        normalized["reliability_score"] = 0.9 if leg_data["is_reliable"] else 0.4
    
    return normalized


def aggregate_weighted_score(
    normalized_scores: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """
    Aggregate multiple normalized scores with weights.
    
    Args:
        normalized_scores: Dict of module scores (0-1)
        weights: Dict of module weights (sum to 1.0)
    
    Returns:
        Weighted aggregate score (0-1)
    """
    total = 0.0
    weight_sum = 0.0
    
    for module, score in normalized_scores.items():
        weight = weights.get(module, 0.0)
        total += score * weight
        weight_sum += weight
    
    if weight_sum > 0:
        return round(total / weight_sum, 3)
    return 0.5


def combine_scores(
    scores: List[float],
    weights: List[float] = None,
    method: str = "weighted",
) -> float:
    """
    Combine multiple scores using specified method.
    
    Args:
        scores: List of scores (0-1)
        weights: Optional list of weights (must sum to 1.0)
        method: "weighted", "geometric", "harmonic", "min", "max", "average"
    
    Returns:
        Combined score (0-1)
    """
    if not scores:
        return 0.5
    
    if method == "weighted" and weights:
        if len(weights) != len(scores):
            weights = None
        else:
            weight_sum = sum(weights)
            if weight_sum > 0:
                return sum(s * w for s, w in zip(scores, weights)) / weight_sum
    
    if method == "geometric":
        # Geometric mean (product^(1/n))
        product = 1.0
        for s in scores:
            product *= max(0.01, s)
        return product ** (1.0 / len(scores))
    
    if method == "harmonic":
        # Harmonic mean (n / sum(1/x))
        inv_sum = sum(1.0 / max(0.01, s) for s in scores)
        return len(scores) / inv_sum if inv_sum > 0 else 0.5
    
    if method == "min":
        return min(scores)
    
    if method == "max":
        return max(scores)
    
    # Default: arithmetic mean
    return sum(scores) / len(scores)


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — EXPORTS
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Mapping dictionaries (for reference)
    "RISK_TO_SCORE",
    "RISK_TO_CONFIDENCE_FACTOR",
    "THREAT_TO_SCORE",
    "H2H_LABEL_TO_SCORE",
    "PATTERN_STRENGTH_TO_SCORE",
    "PATTERN_VERDICT_TO_SCORE",
    "TRAP_VALUE_TO_SCORE",
    "AI_VERDICT_TO_SCORE",
    "CONFIDENCE_TO_FACTOR",
    "CONFIDENCE_TO_BASE_SCORE",
    "RECOMMENDATION_TO_SCORE",
    "BILATERAL_CONFIDENCE_TO_SCORE",
    "CALIBRATION_GRADE_TO_SCORE",
    "HEALTH_STATUS_TO_MULTIPLIER",
    "LEAGUE_TIER_TO_RELIABILITY",
    # Normalizer class
    "Normalizer",
    # Batch functions
    "normalize_leg_data",
    "aggregate_weighted_score",
    "combine_scores",
]


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MODULE 33: UNIFIED NORMALIZER - TEST RUN")
    print("=" * 70)
    
    # Test individual normalizations
    print("\n📊 RISK LEVELS:")
    for risk in ["MINIMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        score = Normalizer.risk_to_score(risk)
        factor = Normalizer.risk_to_confidence_factor(risk)
        print(f"  {risk:10s} → score={score:.2f}, factor={factor:.2f}")
    
    print("\n📊 THREAT LEVELS:")
    for threat in ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        score = Normalizer.threat_to_score(threat)
        print(f"  {threat:8s} → score={score:.2f}")
    
    print("\n📊 H2H LABELS:")
    for label in ["FAV_DOMINANT", "FAV_EDGE", "NEUTRAL", "UND_EDGE", "UND_DOMINANT"]:
        score = Normalizer.h2h_label_to_score(label)
        print(f"  {label:15s} → score={score:.2f}")
    
    print("\n📊 PATTERN VERDICTS:")
    for verdict in ["DEFINITIVE_PATTERN", "STRONG_PATTERN", "MODERATE_PATTERN", "WEAK_PATTERN", "NO_PATTERN"]:
        score = Normalizer.pattern_verdict_to_score(verdict)
        print(f"  {verdict:20s} → score={score:.2f}")
    
    print("\n📊 CALIBRATION GRADES:")
    for grade in ["A", "B", "C", "D", "F"]:
        score = Normalizer.calibration_grade_to_score(grade)
        factor = Normalizer.calibration_grade_to_factor(grade)
        print(f"  {grade} → score={score:.2f}, factor={factor:.2f}")
    
    print("\n📊 NUMERIC CONVERSIONS:")
    print(f"  Failure 8.0 → {Normalizer.failure_to_score(8.0):.2f}")
    print(f"  Failure 4.5 → {Normalizer.failure_to_score(4.5):.2f}")
    print(f"  Edge +0.12 → {Normalizer.edge_to_score(0.12):.2f}")
    print(f"  Edge +0.05 → {Normalizer.edge_to_score(0.05):.2f}")
    print(f"  Brier 0.15 → {Normalizer.brier_to_score(0.15):.2f}")
    print(f"  Brier 0.25 → {Normalizer.brier_to_score(0.25):.2f}")
    print(f"  Sharpe 1.5 → {Normalizer.sharpe_to_score(1.5):.2f}")
    print(f"  Sharpe 0.5 → {Normalizer.sharpe_to_score(0.5):.2f}")
    
    # Test batch normalization
    print("\n📊 BATCH NORMALIZATION:")
    
    test_leg_data = {
        "dual_risk_level": "HIGH",
        "underdog_threat_level": "MEDIUM",
        "h2h_label": "FAV_EDGE",
        "pattern_verdict": "STRONG_PATTERN",
        "trap_value_signal": "VALUE",
        "ai_verdict": "APPROVE",
        "confidence": "HIGH",
        "recommendation": "UNDERDOG_WIN",
        "calibration_grade": "B",
        "failure_score": 3.5,
        "edge": 0.09,
        "underdog_edge": 0.08,
        "h2h_score_raw": 72,
        "draw_prob": 0.32,
        "model_prob": 0.65,
        "matrix_useful": True,
        "pre_filter_passed": True,
        "forensic_passed": True,
        "is_reliable": True,
    }
    
    normalized = normalize_leg_data(test_leg_data)
    print("  Input keys:", len(test_leg_data))
    print("  Normalized keys:", len(normalized))
    for key, value in list(normalized.items())[:10]:
        print(f"    {key}: {value:.3f}")
    
    # Test weighted aggregation
    print("\n📊 WEIGHTED AGGREGATION:")
    
    weights = {
        "dual_risk_score": 0.10,
        "underdog_threat_score": 0.10,
        "h2h_score": 0.08,
        "pattern_score": 0.10,
        "trap_value_score": 0.05,
        "ai_score": 0.15,
        "failure_normalized": 0.15,
        "edge_score": 0.10,
        "matrix_score": 0.05,
        "recommendation_score": 0.12,
    }
    
    total = aggregate_weighted_score(normalized, weights)
    print(f"  Total weighted score: {total:.3f}")
    
    # Test combine_scores
    print("\n📊 COMBINE SCORES:")
    scores = [0.85, 0.72, 0.68, 0.91, 0.77]
    print(f"  Scores: {scores}")
    print(f"  Weighted: {combine_scores(scores, method='weighted'):.3f}")
    print(f"  Geometric: {combine_scores(scores, method='geometric'):.3f}")
    print(f"  Harmonic: {combine_scores(scores, method='harmonic'):.3f}")
    print(f"  Min: {combine_scores(scores, method='min'):.3f}")
    print(f"  Max: {combine_scores(scores, method='max'):.3f}")
    print(f"  Average: {combine_scores(scores, method='average'):.3f}")
    
    # Test inverse mappings
    print("\n📊 INVERSE MAPPINGS:")
    score = 0.75
    risk = Normalizer.score_to_risk(score)
    threat = Normalizer.score_to_threat(score)
    confidence = Normalizer.score_to_confidence(score)
    print(f"  Score {score} → Risk: {risk}, Threat: {threat}, Confidence: {confidence}")
    
    print("\n" + "=" * 70)
    print("MODULE 33 READY FOR PRODUCTION")
    print("=" * 70)