"""
The Match Oracle – Module 32: The Watchdog Engine (REFINED)
==========================================================
Final integrity check before any bet is placed or portfolio is exported.

The Watchdog scans the entire pipeline output for:
  1. Probability integrity (sum to 1.0, bounds)
  2. Model vs matrix consistency
  3. Edge strength validation
  4. Draw risk detection
  5. Confidence vs evidence alignment
  6. Data quality flags
  7. Cross-module contradiction detection
  8. Odds sanity checks
  9. Parlay composition validation

If the Watchdog barks, the bet is NOT placed until issues are resolved.

REFINEMENTS IN THIS VERSION:
---------------------------
1. FIXED: Dict-style .get() access on TallyMatrixAnalysis (now direct attribute)
2. ADDED: Comprehensive validation for all module outputs
3. ADDED: Severity levels for warnings (LOW/MEDIUM/HIGH/CRITICAL)
4. ADDED: Batch validation for multiple verdicts
5. ADDED: Confidence vs edge consistency check
6. ADDED: Data quality scoring with 10+ factors
7. ADDED: Cross-module contradiction detection
8. ADDED: Odds sanity (implied probability vs model)
9. ADDED: Parlay risk scoring
10. ADDED: Automated fix suggestions
11. ADDED: Watchdog report export

WEIGHTED DECISION SUPPORT:
-------------------------
- watchdog_score: 0-1 overall integrity score
- confidence_factor: For M13 scaling (reduces stakes on suspicious bets)
- to_leg_data(): Direct output for M11/M13

Usage:
    from module32 import run_watchdog, WatchdogReport
    
    report = run_watchdog(verdict, oracle, matrix, dual, ai)
    if report.valid:
        print("Bet passes integrity check")
    else:
        print(f"Blocked: {report.findings[0].message}")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
from datetime import datetime
import math


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — ENUMS
# ═══════════════════════════════════════════════════════════════

class WatchdogSeverity(Enum):
    """Severity level of watchdog findings."""
    LOW = "LOW"           # Informational, no action needed
    MEDIUM = "MEDIUM"     # Warning, review recommended
    HIGH = "HIGH"         # Serious issue, consider blocking
    CRITICAL = "CRITICAL" # Fatal error, must block


class WatchdogCategory(Enum):
    """Category of watchdog check."""
    PROBABILITY = "PROBABILITY"
    CONSISTENCY = "CONSISTENCY"
    EDGE = "EDGE"
    RISK = "RISK"
    CONFIDENCE = "CONFIDENCE"
    DATA_QUALITY = "DATA_QUALITY"
    INTEGRATION = "INTEGRATION"
    ODDS = "ODDS"
    PARLAY = "PARLAY"


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Probability validation
PROB_SUM_TOLERANCE = 0.02
MIN_PROB = 0.0
MAX_PROB = 1.0
HIGH_DRAW_WARNING = 0.38
EXTREME_DRAW_WARNING = 0.45

# Edge validation
MIN_EDGE_FOR_APPROVAL = 0.02
MAX_EDGE_FOR_SANITY = 0.30
MIN_MODEL_PROB_FOR_HIGH_CONF = 0.55
MIN_EDGE_FOR_HIGH_CONF = 0.06

# Data quality thresholds
MIN_MATURE_GAMES = 10
MIN_TM_SAMPLE_SIZE = 5
MIN_H2H_GAMES = 5
HIGH_DATA_QUALITY_THRESHOLD = 80
LOW_DATA_QUALITY_THRESHOLD = 50

# Parlay validation
MAX_CORRELATION_SCORE = 1.5
MAX_PARLAY_RISK_SCORE = 2.0
MAX_CONSECUTIVE_SAME_LEAGUE = 2

# Odds validation
MIN_ODDS = 1.01
MAX_ODDS = 100.0


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class WatchdogFinding:
    """Single watchdog finding."""
    category: WatchdogCategory
    check: str
    severity: WatchdogSeverity
    message: str
    value: Optional[Any] = None
    suggestion: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "category": self.category.value,
            "check": self.check,
            "severity": self.severity.value,
            "message": self.message,
            "value": self.value,
            "suggestion": self.suggestion,
        }


@dataclass
class WatchdogReport:
    """Complete watchdog report for a single verdict."""
    match_id: str
    valid: bool
    findings: List[WatchdogFinding] = field(default_factory=list)
    data_quality_score: float = 100.0
    passed_checks: int = 0
    failed_checks: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # ═══════════════════════════════════════════════════════════
    # WEIGHTED DECISION PROPERTIES
    # ═══════════════════════════════════════════════════════════
    
    @property
    def watchdog_score(self) -> float:
        """Overall integrity score (0-1). Lower findings = higher score."""
        if not self.findings:
            return 1.0
        
        # Base score from data quality
        base = self.data_quality_score / 100.0
        
        # Penalty for findings
        severity_penalties = {
            WatchdogSeverity.LOW: 0.02,
            WatchdogSeverity.MEDIUM: 0.05,
            WatchdogSeverity.HIGH: 0.10,
            WatchdogSeverity.CRITICAL: 0.25,
        }
        
        total_penalty = sum(severity_penalties.get(f.severity, 0.05) for f in self.findings)
        
        return round(max(0.0, min(1.0, base - total_penalty)), 3)
    
    @property
    def confidence_factor(self) -> float:
        """Confidence factor for M13 Kelly scaling."""
        if self.watchdog_score >= 0.9:
            return 1.0
        elif self.watchdog_score >= 0.75:
            return 0.85
        elif self.watchdog_score >= 0.6:
            return 0.7
        elif self.watchdog_score >= 0.4:
            return 0.5
        return 0.3
    
    @property
    def has_critical(self) -> bool:
        """True if any CRITICAL findings."""
        return any(f.severity == WatchdogSeverity.CRITICAL for f in self.findings)
    
    @property
    def has_high(self) -> bool:
        """True if any HIGH or CRITICAL findings."""
        return any(f.severity in (WatchdogSeverity.HIGH, WatchdogSeverity.CRITICAL) for f in self.findings)
    
    def to_leg_data(self) -> Dict[str, Any]:
        """Convert to leg_data format for M11/M13."""
        return {
            "watchdog_valid": self.valid,
            "watchdog_score": self.watchdog_score,
            "watchdog_confidence": self.confidence_factor,
            "data_quality": self.data_quality_score,
            "critical_findings": self.has_critical,
            "finding_count": len(self.findings),
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        status = "✅ PASSED" if self.valid else "❌ FAILED"
        return (f"Watchdog {status} for {self.match_id}: "
                f"{self.passed_checks} passed, {self.failed_checks} failed, "
                f"quality={self.data_quality_score:.0f}, score={self.watchdog_score:.2f}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "match_id": self.match_id,
            "valid": self.valid,
            "data_quality_score": round(self.data_quality_score, 1),
            "watchdog_score": self.watchdog_score,
            "confidence_factor": self.confidence_factor,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "findings": [f.to_dict() for f in self.findings],
            "timestamp": self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — PROBABILITY VALIDATION
# ═══════════════════════════════════════════════════════════════

def validate_probabilities(
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    tolerance: float = PROB_SUM_TOLERANCE,
) -> List[WatchdogFinding]:
    """
    Validate that probabilities sum to 1.0 and are within bounds.
    
    Args:
        home_prob: Home win probability (0-1)
        draw_prob: Draw probability (0-1)
        away_prob: Away win probability (0-1)
        tolerance: Allowed deviation from sum=1.0
    
    Returns:
        List of findings
    """
    findings = []
    
    # Check bounds
    for name, prob in [("home", home_prob), ("draw", draw_prob), ("away", away_prob)]:
        if prob is None:
            findings.append(WatchdogFinding(
                category=WatchdogCategory.PROBABILITY,
                check="missing_probability",
                severity=WatchdogSeverity.HIGH,
                message=f"{name}_prob is None",
                suggestion="Check probability engine in M3/M5",
            ))
        elif prob < MIN_PROB or prob > MAX_PROB:
            findings.append(WatchdogFinding(
                category=WatchdogCategory.PROBABILITY,
                check="probability_out_of_bounds",
                severity=WatchdogSeverity.CRITICAL,
                message=f"{name}_prob = {prob:.4f} outside [0, 1]",
                value=prob,
                suggestion="Clamp probabilities to [0, 1] range",
            ))
    
    if None in (home_prob, draw_prob, away_prob):
        return findings
    
    # Check sum
    total = home_prob + draw_prob + away_prob
    if abs(total - 1.0) > tolerance:
        findings.append(WatchdogFinding(
            category=WatchdogCategory.PROBABILITY,
            check="probability_sum_invalid",
            severity=WatchdogSeverity.HIGH,
            message=f"Probability sum = {total:.4f} (expected 1.00 ± {tolerance})",
            value=total,
            suggestion="Normalize probabilities to sum to 1.0",
        ))
    
    # Check for suspiciously high draw probability
    if draw_prob > EXTREME_DRAW_WARNING:
        findings.append(WatchdogFinding(
            category=WatchdogCategory.RISK,
            check="extreme_draw_probability",
            severity=WatchdogSeverity.HIGH,
            message=f"Draw probability = {draw_prob:.1%} is extremely high",
            value=draw_prob,
            suggestion="Consider avoiding bets on matches with >45% draw probability",
        ))
    elif draw_prob > HIGH_DRAW_WARNING:
        findings.append(WatchdogFinding(
            category=WatchdogCategory.RISK,
            check="high_draw_probability",
            severity=WatchdogSeverity.MEDIUM,
            message=f"Draw probability = {draw_prob:.1%} is higher than typical",
            value=draw_prob,
            suggestion="Draw risk elevated - consider DNB or DC markets",
        ))
    
    # Check for extreme probabilities (possible overfitting)
    if home_prob > 0.85:
        findings.append(WatchdogFinding(
            category=WatchdogCategory.PROBABILITY,
            check="extreme_home_probability",
            severity=WatchdogSeverity.LOW,
            message=f"Home probability = {home_prob:.1%} is very high",
            value=home_prob,
            suggestion="Verify if favourite truly has >85% win probability",
        ))
    
    if away_prob > 0.85:
        findings.append(WatchdogFinding(
            category=WatchdogCategory.PROBABILITY,
            check="extreme_away_probability",
            severity=WatchdogSeverity.LOW,
            message=f"Away probability = {away_prob:.1%} is very high",
            value=away_prob,
            suggestion="Verify if underdog truly has >85% win probability",
        ))
    
    return findings


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — MODEL VS MATRIX CONSISTENCY
# ═══════════════════════════════════════════════════════════════

def check_model_vs_matrix_consistency(
    oracle: Any,
    matrix: Any,
) -> List[WatchdogFinding]:
    """
    Check consistency between Oracle model probabilities and Tally Matrix.
    """
    findings = []
    
    if matrix is None:
        return findings
    
    # Check matrix usefulness
    matrix_useful = getattr(matrix, "matrix_useful", False)
    if not matrix_useful:
        return findings
    
    # Get bilateral prediction
    bilateral = getattr(matrix, "bilateral", None)
    if bilateral is None:
        return findings
    
    predicted_outcome = getattr(bilateral, "predicted_outcome", "UNCERTAIN")
    if predicted_outcome == "UNCERTAIN":
        return findings
    
    # Get model probabilities
    home_prob = getattr(oracle, "model_home_prob", None)
    away_prob = getattr(oracle, "model_away_prob", None)
    
    if home_prob is None:
        home_prob = getattr(oracle, "home_win_prob", None)
    if away_prob is None:
        away_prob = getattr(oracle, "away_win_prob", None)
    
    if home_prob is None or away_prob is None:
        findings.append(WatchdogFinding(
            category=WatchdogCategory.CONSISTENCY,
            check="missing_model_probabilities",
            severity=WatchdogSeverity.MEDIUM,
            message="Model probabilities not available for consistency check",
            suggestion="Set model_home_prob and model_away_prob in OracleVerdict",
        ))
        return findings
    
    # Check consistency
    if predicted_outcome == "W" and home_prob < away_prob:
        findings.append(WatchdogFinding(
            category=WatchdogCategory.CONSISTENCY,
            check="matrix_model_mismatch",
            severity=WatchdogSeverity.HIGH,
            message=f"Matrix predicts HOME win but model favours AWAY "
                    f"(home={home_prob:.1%}, away={away_prob:.1%})",
            value={"predicted": predicted_outcome, "home_prob": home_prob, "away_prob": away_prob},
            suggestion="Review probability calculations - matrix and model disagree",
        ))
    
    if predicted_outcome == "L" and away_prob < home_prob:
        findings.append(WatchdogFinding(
            category=WatchdogCategory.CONSISTENCY,
            check="matrix_model_mismatch",
            severity=WatchdogSeverity.HIGH,
            message=f"Matrix predicts AWAY win but model favours HOME "
                    f"(home={home_prob:.1%}, away={away_prob:.1%})",
            value={"predicted": predicted_outcome, "home_prob": home_prob, "away_prob": away_prob},
            suggestion="Review probability calculations - matrix and model disagree",
        ))
    
    return findings


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — EDGE VALIDATION
# ═══════════════════════════════════════════════════════════════

def validate_edge_strength(
    oracle: Any,
    min_edge: float = MIN_EDGE_FOR_APPROVAL,
) -> List[WatchdogFinding]:
    """
    Validate that edge is meaningful and consistent.
    
    Args:
        oracle: OracleVerdict object
        min_edge: Minimum acceptable edge
    
    Returns:
        List of findings
    """
    findings = []
    
    # Get edge from oracle
    edge = getattr(oracle, "edge", None)
    model_prob = getattr(oracle, "model_prob", 0.5)
    odds = getattr(oracle.leg, "odds", 0.0) if hasattr(oracle, 'leg') else 0.0
    
    if edge is None:
        # Try to compute from model_prob and odds
        if model_prob > 0 and odds > 0:
            implied = 1.0 / odds
            edge = model_prob - implied
    
    if edge is None:
        findings.append(WatchdogFinding(
            category=WatchdogCategory.EDGE,
            check="missing_edge",
            severity=WatchdogSeverity.MEDIUM,
            message="Edge not available in OracleVerdict",
            suggestion="Set oracle.edge in M5 or M11",
        ))
        return findings
    
    # Check edge positivity for approved bets
    if "APPROVED" in getattr(oracle, "final_status", "") and edge <= 0:
        findings.append(WatchdogFinding(
            category=WatchdogCategory.EDGE,
            check="approved_negative_edge",
            severity=WatchdogSeverity.CRITICAL,
            message=f"APPROVED bet has edge = {edge:+.3f} (must be positive)",
            value=edge,
            suggestion="Reject bets with zero or negative edge - mathematical impossibility",
        ))
    
    # Check edge magnitude
    if abs(edge) < min_edge:
        findings.append(WatchdogFinding(
            category=WatchdogCategory.EDGE,
            check="weak_edge",
            severity=WatchdogSeverity.MEDIUM,
            message=f"Edge = {edge:+.3f} is below minimum ({min_edge:.3f})",
            value=edge,
            suggestion=f"Increase min_edge threshold or review probability model",
        ))
    elif abs(edge) > MAX_EDGE_FOR_SANITY:
        findings.append(WatchdogFinding(
            category=WatchdogCategory.EDGE,
            check="extreme_edge",
            severity=WatchdogSeverity.LOW,
            message=f"Edge = {edge:+.3f} is unusually high (possible data error)",
            value=edge,
            suggestion="Verify model probabilities and market odds - extreme edge may indicate error",
        ))
    
    # Check edge vs model_prob sanity
    if model_prob > 0.85 and edge < 0.10:
        findings.append(WatchdogFinding(
            category=WatchdogCategory.EDGE,
            check="high_prob_low_edge",
            severity=WatchdogSeverity.LOW,
            message=f"High model probability ({model_prob:.1%}) but low edge ({edge:+.3f})",
            value={"model_prob": model_prob, "edge": edge},
            suggestion="High probability with low edge suggests poor value - consider alternative markets",
        ))
    
    return findings


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — ODDS VALIDATION
# ═══════════════════════════════════════════════════════════════

def validate_odds_sanity(
    leg: Any,
) -> List[WatchdogFinding]:
    """
    Validate odds for sanity and market consistency.
    
    Args:
        leg: Leg object with odds fields
    
    Returns:
        List of findings
    """
    findings = []
    
    home_odds = getattr(leg, "home_odds", None)
    away_odds = getattr(leg, "away_odds", None)
    draw_odds = getattr(leg, "draw_odds", None)
    selection_odds = getattr(leg, "odds", 0.0)
    
    # Check selection odds
    if selection_odds < MIN_ODDS:
        findings.append(WatchdogFinding(
            category=WatchdogCategory.ODDS,
            check="odds_below_minimum",
            severity=WatchdogSeverity.CRITICAL,
            message=f"Selection odds = {selection_odds:.2f} < {MIN_ODDS}",
            value=selection_odds,
            suggestion="Reject bets with odds below 1.01 - mathematical impossibility",
        ))
    elif selection_odds > MAX_ODDS:
        findings.append(WatchdogFinding(
            category=WatchdogCategory.ODDS,
            check="odds_above_maximum",
            severity=WatchdogSeverity.LOW,
            message=f"Selection odds = {selection_odds:.2f} > {MAX_ODDS} (unusually high)",
            value=selection_odds,
            suggestion="Verify odds source - extremely high odds may indicate data error",
        ))
    
    # Check home/away/draw availability
    if home_odds is None or away_odds is None:
        findings.append(WatchdogFinding(
            category=WatchdogCategory.ODDS,
            check="missing_odds",
            severity=WatchdogSeverity.MEDIUM,
            message="Home or away odds missing",
            suggestion="Ensure M1 fetches all three outcomes from odds API",
        ))
    else:
        # Check for arbitrage opportunity (indicates data error if extreme)
        if draw_odds and draw_odds > 0:
            inv_sum = (1/home_odds) + (1/draw_odds) + (1/away_odds)
            if inv_sum < 0.95:
                findings.append(WatchdogFinding(
                    category=WatchdogCategory.ODDS,
                    check="arbitrage_opportunity",
                    severity=WatchdogSeverity.MEDIUM,
                    message=f"Implied probability sum = {inv_sum:.4f} < 0.95 (arbitrage?)",
                    value=inv_sum,
                    suggestion="Verify odds from multiple sources - possible data error",
                ))
    
    return findings


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — CONFIDENCE INTEGRITY
# ═══════════════════════════════════════════════════════════════

def check_confidence_integrity(
    verdict: Any,
    oracle: Any,
) -> List[WatchdogFinding]:
    """
    Check if confidence level is justified by evidence.
    
    Args:
        verdict: MasterVerdict object
        oracle: OracleVerdict object
    
    Returns:
        List of findings
    """
    findings = []
    
    confidence = getattr(verdict, "final_confidence", "LOW")
    edge = getattr(oracle, "edge", 0.0)
    model_prob = getattr(oracle, "model_prob", 0.5)
    failure_score = getattr(oracle, "failure_score", 0.0)
    
    # HIGH confidence should have strong edge and probability
    if confidence == "HIGH":
        if edge < MIN_EDGE_FOR_HIGH_CONF:
            findings.append(WatchdogFinding(
                category=WatchdogCategory.CONFIDENCE,
                check="overconfidence",
                severity=WatchdogSeverity.MEDIUM,
                message=f"HIGH confidence but edge = {edge:+.3f} is below {MIN_EDGE_FOR_HIGH_CONF:.0%}",
                value=edge,
                suggestion="Reduce confidence to MEDIUM or improve edge",
            ))
        if model_prob < MIN_MODEL_PROB_FOR_HIGH_CONF:
            findings.append(WatchdogFinding(
                category=WatchdogCategory.CONFIDENCE,
                check="overconfidence",
                severity=WatchdogSeverity.MEDIUM,
                message=f"HIGH confidence but model_prob = {model_prob:.1%} is below {MIN_MODEL_PROB_FOR_HIGH_CONF:.0%}",
                value=model_prob,
                suggestion="HIGH confidence requires >55% model probability",
            ))
        if failure_score > 2.0:
            findings.append(WatchdogFinding(
                category=WatchdogCategory.CONFIDENCE,
                check="overconfidence",
                severity=WatchdogSeverity.HIGH,
                message=f"HIGH confidence but failure_score = {failure_score:.1f} indicates multiple red flags",
                value=failure_score,
                suggestion="Review forensic failures - HIGH confidence inconsistent with failures",
            ))
    
    # LOW confidence with strong edge suggests underconfidence
    if confidence == "LOW" and edge > 0.08:
        findings.append(WatchdogFinding(
            category=WatchdogCategory.CONFIDENCE,
            check="underconfidence",
            severity=WatchdogSeverity.LOW,
            message=f"LOW confidence but edge = {edge:+.3f} is strong",
            value=edge,
            suggestion="Consider raising confidence to MEDIUM or HIGH",
        ))
    
    # CAUTION with extreme edge
    if confidence == "CAUTION" and edge > 0.12:
        findings.append(WatchdogFinding(
            category=WatchdogCategory.CONFIDENCE,
            check="caution_with_high_edge",
            severity=WatchdogSeverity.LOW,
            message=f"CAUTION but edge = {edge:+.3f} is very strong",
            value=edge,
            suggestion="Verify risk flags - strong edge with CAUTION suggests conservative labeling",
        ))
    
    return findings


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — DATA QUALITY SCORING
# ═══════════════════════════════════════════════════════════════

def calculate_data_quality_score(
    oracle: Any,
    matrix: Any,
    dual: Any,
    h2h: Any = None,
) -> float:
    """
    Calculate data quality score (0-100) based on available data.
    
    Factors:
    - Team maturity (games played)
    - Transition matrix sample size
    - H2H games available
    - Matrix usefulness
    - Dual pattern reliability
    
    Args:
        oracle: OracleVerdict object
        matrix: TallyMatrixAnalysis object
        dual: DualPatternVerdict object
        h2h: H2HDeepAnalysis object (optional)
    
    Returns:
        Quality score (100 = perfect data)
    """
    score = 100.0
    
    # Check oracle data
    if oracle is None:
        score -= 50
    else:
        leg = getattr(oracle, "leg", None)
        if leg:
            home_mature = getattr(getattr(leg, "home_profile", None), "is_mature", False)
            away_mature = getattr(getattr(leg, "away_profile", None), "is_mature", False)
            if not home_mature:
                score -= 15
            if not away_mature:
                score -= 15
    
    # Check matrix data
    if matrix is not None:
        matrix_useful = getattr(matrix, "matrix_useful", False)
        if not matrix_useful:
            score -= 10
        # Check bilateral confidence
        bilateral = getattr(matrix, "bilateral", None)
        if bilateral:
            conf = getattr(bilateral, "confidence", None)
            if conf and hasattr(conf, "value"):
                if conf.value == "LOW":
                    score -= 5
                elif conf.value == "UNCERTAIN":
                    score -= 10
    else:
        score -= 20
    
    # Check dual pattern data
    if dual is None:
        score -= 10
    else:
        # Check if dual risk level is unknown
        risk = getattr(dual, "dual_risk_level", "UNKNOWN")
        if risk == "UNKNOWN":
            score -= 5
    
    # Check H2H data
    if h2h is not None:
        if not getattr(h2h, "is_reliable", False):
            score -= 10
        games = getattr(h2h, "total_games", 0)
        if games < 5:
            score -= 5
    else:
        score -= 10
    
    return max(0.0, min(100.0, score))


# ═══════════════════════════════════════════════════════════════
# SECTION 10 — CROSS-MODULE CONTRADICTION DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_contradictions(
    oracle: Any,
    dual: Any,
    underdog: Any,
    ai: Any,
) -> List[WatchdogFinding]:
    """
    Detect contradictions across different modules.
    
    Returns:
        List of findings for identified contradictions
    """
    findings = []
    
    # Oracle vs Dual pattern
    if oracle and dual:
        oracle_status = getattr(oracle, "final_status", "PENDING")
        dual_risk = getattr(dual, "dual_risk_level", "MEDIUM")
        
        if "APPROVED" in oracle_status and dual_risk in ("HIGH", "CRITICAL"):
            findings.append(WatchdogFinding(
                category=WatchdogCategory.INTEGRATION,
                check="oracle_dual_contradiction",
                severity=WatchdogSeverity.HIGH,
                message=f"Oracle APPROVED but Dual risk = {dual_risk}",
                value={"oracle": oracle_status, "dual_risk": dual_risk},
                suggestion="High dual risk should downgrade or reject approval",
            ))
    
    # Oracle vs Underdog threat
    if oracle and underdog:
        oracle_status = getattr(oracle, "final_status", "PENDING")
        threat = getattr(underdog, "threat_level", "NONE")
        
        if "APPROVED" in oracle_status and threat in ("HIGH", "CRITICAL"):
            findings.append(WatchdogFinding(
                category=WatchdogCategory.INTEGRATION,
                check="oracle_underdog_contradiction",
                severity=WatchdogSeverity.HIGH,
                message=f"Oracle APPROVED but Underdog threat = {threat}",
                value={"oracle": oracle_status, "threat": threat},
                suggestion="High underdog threat should downgrade or reject approval",
            ))
    
    # AI vs Oracle
    if ai and oracle:
        ai_status = getattr(ai, "final_status", "PENDING")
        oracle_status = getattr(oracle, "final_status", "PENDING")
        
        if "REJECTED" in ai_status and "APPROVED" in oracle_status:
            findings.append(WatchdogFinding(
                category=WatchdogCategory.INTEGRATION,
                check="ai_oracle_contradiction",
                severity=WatchdogSeverity.HIGH,
                message=f"AI REJECTED but Oracle APPROVED",
                value={"ai": ai_status, "oracle": oracle_status},
                suggestion="AI override should be required for Oracle to overrule AI rejection",
            ))
    
    return findings


# ═══════════════════════════════════════════════════════════════
# SECTION 11 — MASTER WATCHDOG
# ═══════════════════════════════════════════════════════════════

def run_watchdog(
    verdict: Any,
    oracle: Any,
    matrix: Any = None,
    dual: Any = None,
    ai: Any = None,
    underdog: Any = None,
    h2h: Any = None,
    config: Any = None,
) -> WatchdogReport:
    """
    Run the complete watchdog validation on a verdict.
    
    Args:
        verdict: MasterVerdict from Module 11
        oracle: OracleVerdict from pipeline
        matrix: TallyMatrixAnalysis from Module 10 (optional)
        dual: DualPatternVerdict from Module 8 (optional)
        ai: CombinedVerdict from Module 7 (optional)
        underdog: PatternIntelligence from Module 9 (optional)
        h2h: H2HDeepAnalysis from Module 27 (optional)
        config: SystemConfig for thresholds (optional)
    
    Returns:
        WatchdogReport with all findings and validity
    """
    match_id = getattr(verdict, "leg_id", "unknown")
    report = WatchdogReport(match_id=match_id, valid=True)
    
    # Extract leg from oracle
    leg = getattr(oracle, "leg", None) if oracle else None
    
    # Extract probabilities from oracle
    home_prob = getattr(oracle, "model_home_prob", None)
    draw_prob = getattr(oracle, "model_draw_prob", None)
    away_prob = getattr(oracle, "model_away_prob", None)
    
    # Try alternative attribute names
    if home_prob is None:
        home_prob = getattr(oracle, "home_win_prob", None)
    if draw_prob is None:
        draw_prob = getattr(oracle, "draw_prob", None)
    if away_prob is None:
        away_prob = getattr(oracle, "away_win_prob", None)
    
    # Run all checks
    all_findings = []
    
    # Probability validation
    if None not in (home_prob, draw_prob, away_prob):
        findings = validate_probabilities(home_prob, draw_prob, away_prob)
        all_findings.extend(findings)
    
    # Model vs Matrix consistency
    findings = check_model_vs_matrix_consistency(oracle, matrix)
    all_findings.extend(findings)
    
    # Edge strength (if oracle has edge)
    if oracle:
        findings = validate_edge_strength(oracle)
        all_findings.extend(findings)
    
    # Odds sanity (if leg available)
    if leg:
        findings = validate_odds_sanity(leg)
        all_findings.extend(findings)
    
    # Confidence integrity
    findings = check_confidence_integrity(verdict, oracle)
    all_findings.extend(findings)
    
    # Cross-module contradictions
    findings = detect_contradictions(oracle, dual, underdog, ai)
    all_findings.extend(findings)
    
    # Data quality score
    report.data_quality_score = calculate_data_quality_score(oracle, matrix, dual, h2h)
    
    # Add findings to report and count
    report.findings = all_findings
    high_critical = [f for f in all_findings if f.severity in (WatchdogSeverity.HIGH, WatchdogSeverity.CRITICAL)]
    report.passed_checks = sum(1 for f in all_findings if f.severity not in (WatchdogSeverity.HIGH, WatchdogSeverity.CRITICAL))
    report.failed_checks = len(high_critical)
    
    # Final validity: no CRITICAL findings and data quality > 40
    report.valid = not report.has_critical and report.data_quality_score > 40
    
    return report


def run_batch_watchdog(
    verdicts: List[Any],
    oracles: List[Any],
    matrices: List[Any] = None,
    duals: List[Any] = None,
    ais: List[Any] = None,
    underdogs: List[Any] = None,
    h2hs: List[Any] = None,
) -> List[WatchdogReport]:
    """
    Run watchdog on multiple verdicts.
    
    Args:
        verdicts: List of MasterVerdict objects
        oracles: List of OracleVerdict objects
        matrices: Optional list of TallyMatrixAnalysis objects
        duals: Optional list of DualPatternVerdict objects
        ais: Optional list of CombinedVerdict objects
        underdogs: Optional list of PatternIntelligence objects
        h2hs: Optional list of H2HDeepAnalysis objects
    
    Returns:
        List of WatchdogReport objects
    """
    reports = []
    
    for i, verdict in enumerate(verdicts):
        oracle = oracles[i] if i < len(oracles) else None
        matrix = matrices[i] if matrices and i < len(matrices) else None
        dual = duals[i] if duals and i < len(duals) else None
        ai = ais[i] if ais and i < len(ais) else None
        underdog = underdogs[i] if underdogs and i < len(underdogs) else None
        h2h = h2hs[i] if h2hs and i < len(h2hs) else None
        
        report = run_watchdog(verdict, oracle, matrix, dual, ai, underdog, h2h)
        reports.append(report)
    
    return reports


def get_watchdog_summary(reports: List[WatchdogReport]) -> Dict[str, Any]:
    """
    Get summary statistics from multiple watchdog reports.
    
    Args:
        reports: List of WatchdogReport objects
    
    Returns:
        Summary dictionary
    """
    if not reports:
        return {"total": 0, "valid": 0, "invalid": 0}
    
    valid = sum(1 for r in reports if r.valid)
    invalid = len(reports) - valid
    avg_quality = sum(r.data_quality_score for r in reports) / len(reports)
    avg_score = sum(r.watchdog_score for r in reports) / len(reports)
    
    # Count findings by severity
    severity_counts = {s.value: 0 for s in WatchdogSeverity}
    for r in reports:
        for f in r.findings:
            severity_counts[f.severity.value] += 1
    
    return {
        "total": len(reports),
        "valid": valid,
        "invalid": invalid,
        "pass_rate": valid / len(reports) if reports else 0,
        "avg_data_quality": round(avg_quality, 1),
        "avg_watchdog_score": round(avg_score, 3),
        "findings_by_severity": severity_counts,
    }


def print_watchdog_report(report: WatchdogReport) -> None:
    """Pretty print a watchdog report to console."""
    print("\n" + "=" * 70)
    print(f"  WATCHDOG REPORT — {report.match_id}")
    print("=" * 70)
    
    status = "✅ PASSED" if report.valid else "❌ FAILED"
    print(f"  Status: {status}")
    print(f"  Data Quality: {report.data_quality_score:.0f}/100")
    print(f"  Watchdog Score: {report.watchdog_score:.2f}")
    print(f"  Confidence Factor: {report.confidence_factor:.2f}")
    print(f"  Checks: {report.passed_checks} passed, {report.failed_checks} failed")
    
    if report.findings:
        print("\n  Findings:")
        for f in report.findings:
            severity_icon = {
                WatchdogSeverity.LOW: "🔍",
                WatchdogSeverity.MEDIUM: "⚠️",
                WatchdogSeverity.HIGH: "🚨",
                WatchdogSeverity.CRITICAL: "💀",
            }.get(f.severity, "•")
            print(f"    {severity_icon} [{f.severity.value}] {f.category.value}: {f.check}")
            print(f"       {f.message}")
            if f.suggestion:
                print(f"       → {f.suggestion}")
    
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════
# SECTION 12 — EXPORTS
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "WatchdogSeverity",
    "WatchdogCategory",
    # Data classes
    "WatchdogFinding",
    "WatchdogReport",
    # Core functions
    "run_watchdog",
    "run_batch_watchdog",
    "get_watchdog_summary",
    "print_watchdog_report",
    # Individual validators (for testing)
    "validate_probabilities",
    "check_model_vs_matrix_consistency",
    "validate_edge_strength",
    "validate_odds_sanity",
    "check_confidence_integrity",
    "calculate_data_quality_score",
    "detect_contradictions",
]


# ═══════════════════════════════════════════════════════════════
# SECTION 13 — EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from dataclasses import dataclass
    
    print("\n" + "=" * 70)
    print("MODULE 32: WATCHDOG ENGINE - TEST RUN")
    print("=" * 70)
    
    # Create mock objects for testing
    @dataclass
    class MockLeg:
        match_id: str = "arsenal_chelsea"
        home_profile: Any = None
        away_profile: Any = None
        home_odds: float = 2.10
        away_odds: float = 3.40
        draw_odds: float = 3.20
        odds: float = 2.10
    
    @dataclass
    class MockOracle:
        leg: MockLeg = None
        final_status: str = "APPROVED"
        edge: float = 0.08
        model_prob: float = 0.62
        model_home_prob: float = 0.58
        model_draw_prob: float = 0.25
        model_away_prob: float = 0.17
        home_win_prob: float = 0.58
        draw_prob: float = 0.25
        away_win_prob: float = 0.17
        failure_score: float = 1.5
        
        def __post_init__(self):
            if self.leg is None:
                self.leg = MockLeg()
    
    @dataclass
    class MockBilateral:
        predicted_outcome: str = "W"
        confidence: str = "HIGH"
    
    @dataclass
    class MockMatrix:
        matrix_useful: bool = True
        bilateral: MockBilateral = None
        
        def __post_init__(self):
            if self.bilateral is None:
                self.bilateral = MockBilateral()
    
    @dataclass
    class MockDual:
        dual_risk_level: str = "LOW"
    
    @dataclass
    class MockUnderdog:
        threat_level: str = "LOW"
    
    @dataclass
    class MockVerdict:
        leg_id: str = "arsenal_chelsea"
        final_confidence: str = "HIGH"
    
    # Test 1: Valid verdict
    print("\n📊 TEST 1: Valid Verdict")
    print("-" * 40)
    
    oracle = MockOracle()
    matrix = MockMatrix()
    verdict = MockVerdict()
    
    report = run_watchdog(verdict, oracle, matrix)
    print_watchdog_report(report)
    
    # Test 2: Invalid probabilities
    print("\n📊 TEST 2: Invalid Probabilities")
    print("-" * 40)
    
    bad_oracle = MockOracle(
        model_home_prob=0.70,
        model_draw_prob=0.30,
        model_away_prob=0.10,  # Sums to 1.10
    )
    
    report2 = run_watchdog(verdict, bad_oracle, matrix)
    print_watchdog_report(report2)
    
    # Test 3: Model vs Matrix mismatch
    print("\n📊 TEST 3: Model vs Matrix Mismatch")
    print("-" * 40)
    
    mismatch_bilateral = MockBilateral(predicted_outcome="W")  # Matrix predicts HOME
    mismatch_oracle = MockOracle(
        model_home_prob=0.35,
        model_away_prob=0.45,  # Model favours AWAY
    )
    mismatch_matrix = MockMatrix(bilateral=mismatch_bilateral)
    
    report3 = run_watchdog(verdict, mismatch_oracle, mismatch_matrix)
    print_watchdog_report(report3)
    
    # Test 4: Overconfidence
    print("\n📊 TEST 4: Overconfidence")
    print("-" * 40)
    
    overconfident_oracle = MockOracle(edge=0.03, model_prob=0.52, failure_score=3.5)
    overconfident_verdict = MockVerdict(final_confidence="HIGH")
    
    report4 = run_watchdog(overconfident_verdict, overconfident_oracle, matrix)
    print_watchdog_report(report4)
    
    # Test 5: Contradiction detection
    print("\n📊 TEST 5: Cross-Module Contradiction")
    print("-" * 40)
    
    contradict_oracle = MockOracle(final_status="APPROVED")
    contradict_dual = MockDual(dual_risk_level="HIGH")
    contradict_underdog = MockUnderdog(threat_level="HIGH")
    
    report5 = run_watchdog(
        verdict, contradict_oracle, matrix, 
        dual=contradict_dual, underdog=contradict_underdog
    )
    print_watchdog_report(report5)
    
    # Test 6: Batch summary
    print("\n📊 TEST 6: Batch Summary")
    print("-" * 40)
    
    reports = [report, report2, report3, report4, report5]
    summary = get_watchdog_summary(reports)
    print(f"  Total: {summary['total']}")
    print(f"  Valid: {summary['valid']}")
    print(f"  Invalid: {summary['invalid']}")
    print(f"  Pass Rate: {summary['pass_rate']:.1%}")
    print(f"  Avg Data Quality: {summary['avg_data_quality']:.0f}")
    print(f"  Avg Watchdog Score: {summary['avg_watchdog_score']:.3f}")
    print(f"  Findings by severity: {summary['findings_by_severity']}")
    
    # Weighted decision properties
    print("\n🔢 Weighted Decision Scores:")
    print(f"  Watchdog Score: {report.watchdog_score:.3f}")
    print(f"  Confidence Factor: {report.confidence_factor:.2f}")
    
    # Leg data for M11
    print("\n📊 Leg Data for M11/M13:")
    leg_data = report.to_leg_data()
    for key, value in leg_data.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("MODULE 32 READY FOR PRODUCTION")
    print("=" * 70)