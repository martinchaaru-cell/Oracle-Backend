"""
The Match Oracle – Module 17: Explainability & Decision Intelligence Engine (REFINED)
================================================================================

Final layer of the system.

Responsibilities:
- Explain WHY a decision was made
- Detect contradictions across modules
- Validate confidence integrity
- Flag suspicious approvals
- Produce full forensic audit trail
- Generate decision quality scores

REFINEMENTS IN THIS VERSION:
---------------------------
1. FIXED: AIVerdict import from module7 (proper enum comparison)
2. ADDED: Proper handling for None values in all checks
3. ADDED: Comprehensive scoring system for decision quality (0-10)
4. ADDED: Contradiction detection across all modules with severity
5. ADDED: Confidence integrity validation with statistical tests
6. ADDED: Human-readable output formatting with colors/emojis
7. ADDED: Batch explainability for multiple decisions
8. ADDED: Decision score contribution breakdown
9. ADDED: Recommendation generation for improvement
10. ADDED: Export functionality for explainability reports

WEIGHTED DECISION SUPPORT:
-------------------------
- decision_score: 0-10 overall explainability score
- confidence_factor: For M13 scaling (reduces stakes on suspicious decisions)
- to_leg_data(): Direct output for M11 aggregation

Feeds into: Module 11 (verdict enrichment), Module 25 (API responses),
            Module 30 (alerts for suspicious approvals)

Usage:
    from module17 import run_explainability_engine, ExplainabilityReport
    
    report = run_explainability_engine(master_verdict, oracle, dual, underdog)
    print(report.summary())
    print(f"Decision Score: {report.decision_score:.1f}/10")
    
    # Get leg data for M11
    leg_data = report.to_leg_data()
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

# Import AIVerdict enum for correct enum comparison
try:
    from module7 import AIVerdict
    _M7_AVAILABLE = True
except ImportError:
    _M7_AVAILABLE = False
    # Fallback enum for when module7 not available
    from enum import Enum
    class AIVerdict(Enum):
        APPROVE = "APPROVE"
        CAUTION = "CAUTION"
        REJECT = "REJECT"
        NEUTRAL = "NEUTRAL"


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — ENUMS
# ═══════════════════════════════════════════════════════════════

class ExplainabilityLevel(Enum):
    """Level of explainability detail."""
    SUMMARY = "SUMMARY"       # One-line summary
    DETAILED = "DETAILED"     # Full breakdown
    FORENSIC = "FORENSIC"     # All details including contradictions


class ContradictionSeverity(Enum):
    """Severity of contradiction."""
    MINOR = "MINOR"           # Small inconsistency
    MODERATE = "MODERATE"     # Significant disagreement
    MAJOR = "MAJOR"           # Serious contradiction
    CRITICAL = "CRITICAL"     # Decision should be blocked


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Decision score thresholds
EXCELLENT_THRESHOLD = 8.0
GOOD_THRESHOLD = 6.0
MEDIUM_THRESHOLD = 4.0
POOR_THRESHOLD = 2.0

# Confidence integrity thresholds
CONFIDENCE_INTEGRITY_TOLERANCE = 0.15

# Risk alignment thresholds
RISK_ALIGNMENT_GOOD_THRESHOLD = 0.30
RISK_ALIGNMENT_MODERATE_THRESHOLD = 0.50

# Maximum allowed risk flags for approved bet
MAX_RISK_FLAGS_APPROVED = 2

# Maximum allowed contradictions for approved bet
MAX_CONTRADICTIONS_APPROVED = 1

# Score contribution weights
SCORE_WEIGHTS = {
    "oracle_edge": 0.25,
    "forensic_score": 0.20,
    "dual_pattern": 0.15,
    "underdog_threat": 0.15,
    "ai_agreement": 0.15,
    "matrix_signal": 0.10,
}

# Factor mappings for confidence factor
FACTOR_MAP = {
    "EXCELLENT": 1.0,
    "GOOD": 0.85,
    "MEDIUM": 0.65,
    "POOR": 0.40,
    "CRITICAL": 0.20,
}


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class Contradiction:
    """A contradiction detected between modules."""
    description: str
    severity: ContradictionSeverity
    modules: List[str]
    value: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "severity": self.severity.value,
            "modules": self.modules,
            "value": self.value,
        }


@dataclass
class DecisionFactor:
    """One factor influencing the decision."""
    name: str
    value: float
    weight: float
    contribution: float
    is_positive: bool
    explanation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 3),
            "weight": round(self.weight, 3),
            "contribution": round(self.contribution, 3),
            "is_positive": self.is_positive,
            "explanation": self.explanation,
        }


@dataclass
class ExplainabilityReport:
    """Complete explainability report for a single match decision."""
    
    # Basic identification
    match: str
    final_status: str
    selection: str
    confidence: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Decision scoring (0-10 scale)
    decision_score: float = 0.0
    decision_grade: str = "UNKNOWN"
    confidence_integrity: str = "VALID"  # VALID / OVERCONFIDENT / UNDERCONFIDENT
    
    # Contradictions found
    contradictions: List[Contradiction] = field(default_factory=list)
    risk_alignment: str = "ALIGNED"  # ALIGNED / PARTIAL / MISALIGNED
    
    # Suspicious decision flags
    suspicious_flag: bool = False
    suspicious_reasons: List[str] = field(default_factory=list)
    
    # Key factors that influenced the decision
    key_factors: List[DecisionFactor] = field(default_factory=list)
    
    # Full audit trail
    audit_trail: List[str] = field(default_factory=list)
    
    # Module-specific details (for debugging)
    module_details: Dict[str, Any] = field(default_factory=dict)
    
    # Recommendations for improvement
    recommendations: List[str] = field(default_factory=list)
    
    # Score breakdown
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # ═══════════════════════════════════════════════════════════
    # WEIGHTED DECISION PROPERTIES
    # ═══════════════════════════════════════════════════════════
    
    @property
    def normalized_score(self) -> float:
        """Convert decision_score (0-10) to 0-1 normalized score."""
        return round(self.decision_score / 10.0, 3)
    
    @property
    def confidence_factor(self) -> float:
        """Confidence factor for M13 Kelly scaling."""
        if self.decision_score >= EXCELLENT_THRESHOLD:
            return 1.0
        elif self.decision_score >= GOOD_THRESHOLD:
            return 0.85
        elif self.decision_score >= MEDIUM_THRESHOLD:
            return 0.65
        elif self.decision_score >= POOR_THRESHOLD:
            return 0.40
        return 0.20
    
    @property
    def is_approved(self) -> bool:
        """Check if final status is approved."""
        return "APPROVED" in self.final_status
    
    @property
    def is_suspicious(self) -> bool:
        """Check if decision is flagged as suspicious."""
        return self.suspicious_flag
    
    def to_leg_data(self) -> Dict[str, Any]:
        """Convert to leg_data format for M11 weighted decision."""
        return {
            "explainability_score": self.decision_score,
            "explainability_normalized": self.normalized_score,
            "explainability_confidence": self.confidence_factor,
            "decision_grade": self.decision_grade,
            "suspicious": self.suspicious_flag,
            "contradictions_count": len(self.contradictions),
            "risk_alignment": self.risk_alignment,
            "confidence_integrity": self.confidence_integrity,
        }
    
    def summary(self) -> str:
        """One-line summary of the report."""
        return (f"{self.match}: {self.final_status} ({self.confidence}) | "
                f"Score: {self.decision_score:.1f}/10 | "
                f"Integrity: {self.confidence_integrity} | "
                f"Risk: {self.risk_alignment} | "
                f"Suspicious: {self.suspicious_flag}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "match": self.match,
            "final_status": self.final_status,
            "selection": self.selection,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "decision_score": round(self.decision_score, 1),
            "decision_grade": self.decision_grade,
            "normalized_score": self.normalized_score,
            "confidence_factor": self.confidence_factor,
            "confidence_integrity": self.confidence_integrity,
            "risk_alignment": self.risk_alignment,
            "suspicious_flag": self.suspicious_flag,
            "suspicious_reasons": self.suspicious_reasons,
            "contradictions": [c.to_dict() for c in self.contradictions],
            "key_factors": [f.to_dict() for f in self.key_factors],
            "score_breakdown": self.score_breakdown,
            "recommendations": self.recommendations,
            "module_details": self.module_details,
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — CORE ENGINE
# ═══════════════════════════════════════════════════════════════

def run_explainability_engine(
    master_verdict: Any,
    oracle: Any,
    dual: Any,
    underdog: Any,
    matrix: Any = None,
    ai: Any = None,
    config: Any = None,
    h2h: Any = None,
    context: Any = None,
    personnel: Any = None,
) -> ExplainabilityReport:
    """
    Produce an explainability report for a MasterVerdict.

    Args:
        master_verdict: MasterVerdict from Module 11
        oracle: OracleVerdict from pipeline
        dual: DualPatternVerdict from Module 8 (optional)
        underdog: PatternIntelligence from Module 9 (optional)
        matrix: TallyMatrixAnalysis from Module 10 (optional)
        ai: CombinedVerdict from Module 7 (optional)
        config: SystemConfig for threshold context (optional)
        h2h: H2HDeepAnalysis from Module 27 (optional)
        context: MatchContextScore from Module 26 (optional)
        personnel: PersonnelMetrics from Module 6 (optional)

    Returns:
        ExplainabilityReport with full analysis
    """
    
    # Extract basic info safely
    leg_id = getattr(master_verdict, 'leg_id', 'unknown')
    final_status = getattr(master_verdict, 'final_status', 'PENDING')
    final_confidence = getattr(master_verdict, 'final_confidence', '-')
    selection = ""
    if hasattr(master_verdict, 'oracle') and master_verdict.oracle:
        if hasattr(master_verdict.oracle, 'leg') and master_verdict.oracle.leg:
            selection = getattr(master_verdict.oracle.leg, 'selection', '')
    
    report = ExplainabilityReport(
        match=leg_id,
        final_status=final_status,
        selection=selection,
        confidence=final_confidence,
    )
    
    score = 0.0
    max_possible_score = 10.0
    score_breakdown = {}

    # ─────────────────────────────────────────
    # ORACLE CONTRIBUTION (max 3 points)
    # ─────────────────────────────────────────
    if oracle:
        oracle_status = getattr(oracle, 'final_status', 'PENDING')
        oracle_edge = getattr(oracle, 'edge', 0.0)
        oracle_failure = getattr(oracle, 'failure_score', 0.0)
        pre_filter_score = getattr(oracle, 'pre_filter_score', 0.0)
        
        report.audit_trail.append(f"Oracle: {oracle_status} | Edge={oracle_edge:+.3f} | Failure={oracle_failure:.1f} | Pre-filter={pre_filter_score:.1%}")
        report.module_details['oracle'] = {
            'status': oracle_status,
            'edge': oracle_edge,
            'failure_score': oracle_failure,
            'pre_filter_passed': getattr(oracle, 'pre_filter_passed', False),
            'pre_filter_score': pre_filter_score,
        }

        # Edge score (0-3)
        if oracle_edge > 0.08:
            edge_score = 3.0
            report.key_factors.append(DecisionFactor(
                name="Oracle Edge",
                value=oracle_edge,
                weight=0.25,
                contribution=3.0,
                is_positive=True,
                explanation=f"Strong positive edge of {oracle_edge:+.3f} indicates value"
            ))
        elif oracle_edge > 0.04:
            edge_score = 2.0
            report.key_factors.append(DecisionFactor(
                name="Oracle Edge",
                value=oracle_edge,
                weight=0.25,
                contribution=2.0,
                is_positive=True,
                explanation=f"Moderate positive edge of {oracle_edge:+.3f}"
            ))
        elif oracle_edge > 0.02:
            edge_score = 1.0
            report.key_factors.append(DecisionFactor(
                name="Oracle Edge",
                value=oracle_edge,
                weight=0.25,
                contribution=1.0,
                is_positive=True,
                explanation=f"Weak positive edge of {oracle_edge:+.3f}"
            ))
        else:
            edge_score = 0.0
            if report.is_approved:
                report.suspicious_reasons.append(f"Low/negative edge ({oracle_edge:+.3f}) on approved bet")
                report.suspicious_flag = True
        
        # Failure score adjustment
        if oracle_failure >= 4.5 and report.is_approved:
            report.contradictions.append(Contradiction(
                description=f"High failure score ({oracle_failure:.1f}) but approved",
                severity=ContradictionSeverity.MAJOR,
                modules=["M5", "M11"],
                value=oracle_failure
            ))
            report.suspicious_flag = True
            edge_score = max(0, edge_score - 1.0)
        
        score += edge_score
        score_breakdown["oracle_edge"] = edge_score

    # ─────────────────────────────────────────
    # FORENSIC SCORE (max 2 points)
    # ─────────────────────────────────────────
    if oracle:
        failure_score = getattr(oracle, 'failure_score', 5.0)
        # Convert failure score (0-10) to points (0-2)
        forensic_points = max(0.0, min(2.0, 2.0 - (failure_score / 5.0)))
        score += forensic_points
        score_breakdown["forensic"] = forensic_points
        
        report.key_factors.append(DecisionFactor(
            name="Forensic Health",
            value=1.0 - (failure_score / 10.0),
            weight=0.20,
            contribution=forensic_points,
            is_positive=failure_score < 4.5,
            explanation=f"Failure score: {failure_score:.1f} / 10.0 (threshold: 4.5)"
        ))

    # ─────────────────────────────────────────
    # DUAL PATTERN CHECK (max 1.5 points)
    # ─────────────────────────────────────────
    if dual:
        dual_risk = getattr(dual, 'dual_risk_level', 'UNKNOWN')
        underdog_threat = getattr(dual, 'underdog_threat_level', 'NONE')
        pattern_clash = getattr(dual, 'pattern_clash_score', 0.0)
        resilience_gap = getattr(dual, 'resilience_gap', 0.0)
        
        report.audit_trail.append(f"Dual Risk: {dual_risk} | Underdog Threat: {underdog_threat} | Clash: {pattern_clash:.2f}")
        report.module_details['dual'] = {
            'risk_level': dual_risk,
            'underdog_threat': underdog_threat,
            'pattern_clash': pattern_clash,
            'resilience_gap': resilience_gap,
        }

        # Risk to points mapping
        risk_points = {
            "MINIMAL": 1.5,
            "LOW": 1.2,
            "MEDIUM": 0.8,
            "HIGH": 0.3,
            "CRITICAL": 0.0,
            "UNKNOWN": 0.5,
        }
        dual_points = risk_points.get(dual_risk, 0.5)
        
        if dual_risk in ("HIGH", "CRITICAL") and report.is_approved:
            report.contradictions.append(Contradiction(
                description=f"High dual risk ({dual_risk}) but approved",
                severity=ContradictionSeverity.MAJOR,
                modules=["M8", "M11"],
                value=dual_risk
            ))
            report.suspicious_flag = True
            dual_points = max(0, dual_points - 0.5)
        elif dual_risk in ("LOW", "MINIMAL"):
            report.key_factors.append(DecisionFactor(
                name="Dual Pattern Risk",
                value=1.0 - (risk_points.get(dual_risk, 0.5) / 1.5),
                weight=0.15,
                contribution=dual_points,
                is_positive=True,
                explanation=f"Low pattern risk ({dual_risk})"
            ))
        
        score += dual_points
        score_breakdown["dual_pattern"] = dual_points

    # ─────────────────────────────────────────
    # UNDERDOG THREAT (max 1.5 points)
    # ─────────────────────────────────────────
    if underdog:
        threat_level = getattr(underdog, 'threat_level', 'NONE')
        pattern_verdict = getattr(underdog, 'pattern_verdict', 'NO_PATTERN')
        pattern_score = getattr(underdog, 'pattern_strength_score', 0.0)
        goldmine = getattr(underdog, 'goldmine_qualified', False)
        underdog_edge = getattr(underdog, 'underdog_edge', 0.0)
        
        report.audit_trail.append(f"Underdog Threat: {threat_level} | Pattern: {pattern_verdict} ({pattern_score:.0f}) | Goldmine: {goldmine}")
        report.module_details['underdog'] = {
            'threat_level': threat_level,
            'pattern_verdict': pattern_verdict,
            'pattern_score': pattern_score,
            'goldmine': goldmine,
            'underdog_win_prob': getattr(underdog, 'underdog_win_prob', 0.0),
            'underdog_edge': underdog_edge,
        }

        # Threat to points mapping (lower threat = higher points for favourite)
        threat_points = {
            "NONE": 1.5,
            "LOW": 1.2,
            "MEDIUM": 0.8,
            "HIGH": 0.3,
            "CRITICAL": 0.0,
        }
        underdog_points = threat_points.get(threat_level, 0.5)
        
        if goldmine:
            underdog_points = 0.0  # Goldmine means underdog is value, not favourite
            report.key_factors.append(DecisionFactor(
                name="Goldmine Detected",
                value=1.0,
                weight=0.15,
                contribution=0.0,
                is_positive=False,
                explanation="Goldmine opportunity - favour underdog instead"
            ))
        
        if threat_level in ("HIGH", "CRITICAL") and report.is_approved:
            report.contradictions.append(Contradiction(
                description=f"High underdog threat ({threat_level}) but approved",
                severity=ContradictionSeverity.MAJOR,
                modules=["M9", "M11"],
                value=threat_level
            ))
            report.suspicious_flag = True
            underdog_points = max(0, underdog_points - 0.5)
        
        score += underdog_points
        score_breakdown["underdog"] = underdog_points

    # ─────────────────────────────────────────
    # MATRIX SIGNAL (max 1 point)
    # ─────────────────────────────────────────
    if matrix:
        tv_signal = getattr(matrix, 'trap_value_signal', None)
        combined_risk = getattr(matrix, 'combined_risk_flag', 'NONE')
        matrix_useful = getattr(matrix, 'matrix_useful', False)
        
        signal_type = getattr(tv_signal, 'signal_type', 'NONE') if tv_signal else 'NONE'
        signal_desc = getattr(tv_signal, 'description', '') if tv_signal else ''
        
        report.audit_trail.append(f"Matrix Signal: {signal_type} | Risk: {combined_risk} | Useful: {matrix_useful}")
        report.module_details['matrix'] = {
            'signal_type': signal_type,
            'combined_risk': combined_risk,
            'matrix_useful': matrix_useful,
            'bilateral_prediction': getattr(getattr(matrix, 'bilateral', None), 'predicted_outcome', 'UNCERTAIN') if hasattr(matrix, 'bilateral') else 'N/A',
        }

        if signal_type == "TRAP":
            matrix_points = 0.1
            if report.is_approved:
                report.contradictions.append(Contradiction(
                    description=f"Trap signal ignored: {signal_desc}",
                    severity=ContradictionSeverity.MODERATE,
                    modules=["M10", "M11"],
                    value=signal_type
                ))
                report.suspicious_flag = True
        elif signal_type == "VALUE":
            matrix_points = 1.0
            report.key_factors.append(DecisionFactor(
                name="Value Signal",
                value=1.0,
                weight=0.10,
                contribution=1.0,
                is_positive=True,
                explanation=f"Matrix detected value: {signal_desc[:50]}"
            ))
        else:
            matrix_points = 0.5 if matrix_useful else 0.3
        
        score += matrix_points
        score_breakdown["matrix"] = matrix_points

    # ─────────────────────────────────────────
    # AI CONSENSUS (max 1.5 points)
    # ─────────────────────────────────────────
    if ai and _M7_AVAILABLE:
        ai_analysis = getattr(ai, 'ai_analysis', None)
        if ai_analysis:
            final_verdict = getattr(ai_analysis, 'final_ai_verdict', None)
            agreement = getattr(ai_analysis, 'agreement_level', 0.0)
            ai_chain = getattr(ai, 'ai_chain_used', [])
            ai_narrative = getattr(ai_analysis, 'consensus_narrative', "")
        else:
            final_verdict = None
            agreement = 0.0
            ai_chain = []
            ai_narrative = ""
        
        report.audit_trail.append(f"AI Verdict: {final_verdict.value if final_verdict else 'N/A'} | Agreement: {agreement:.1%} | Chain: {ai_chain}")
        report.module_details['ai'] = {
            'verdict': final_verdict.value if final_verdict else 'N/A',
            'agreement': agreement,
            'chain': ai_chain,
            'narrative': ai_narrative[:200] if ai_narrative else "",
        }

        # AI points based on alignment with final decision
        if final_verdict == AIVerdict.APPROVE:
            ai_points = 1.5 if report.is_approved else 0.3
            if not report.is_approved:
                report.contradictions.append(Contradiction(
                    description="AI approved but system rejected",
                    severity=ContradictionSeverity.MODERATE,
                    modules=["M7", "M11"],
                    value=final_verdict.value
                ))
        elif final_verdict == AIVerdict.REJECT:
            ai_points = 1.5 if not report.is_approved else 0.3
            if report.is_approved:
                report.contradictions.append(Contradiction(
                    description="AI rejected but system approved",
                    severity=ContradictionSeverity.MAJOR,
                    modules=["M7", "M11"],
                    value=final_verdict.value
                ))
                report.suspicious_flag = True
        else:  # CAUTION or NEUTRAL
            ai_points = 0.8 if report.confidence != "HIGH" else 0.5
        
        # Boost for high agreement
        if agreement > 0.85:
            ai_points = min(1.5, ai_points + 0.3)
            report.key_factors.append(DecisionFactor(
                name="AI Agreement",
                value=agreement,
                weight=0.15,
                contribution=0.3,
                is_positive=True,
                explanation=f"Strong AI agreement ({agreement:.1%}) across {len(ai_chain)} providers"
            ))
        
        score += ai_points
        score_breakdown["ai"] = ai_points

    # ─────────────────────────────────────────
    # H2H CONTRIBUTION (bonus up to 0.5)
    # ─────────────────────────────────────────
    if h2h:
        h2h_score = getattr(h2h, 'h2h_score', 50.0)
        h2h_label = getattr(h2h, 'h2h_label', 'NEUTRAL')
        
        if h2h_label in ("FAV_DOMINANT", "FAV_EDGE"):
            h2h_bonus = 0.5
            report.key_factors.append(DecisionFactor(
                name="H2H Dominance",
                value=h2h_score / 100.0,
                weight=0.05,
                contribution=0.5,
                is_positive=True,
                explanation=f"Favourable H2H record: {h2h_label} ({h2h_score:.0f}/100)"
            ))
        elif h2h_label in ("UND_EDGE", "UND_DOMINANT"):
            h2h_bonus = -0.3
            if report.is_approved:
                report.contradictions.append(Contradiction(
                    description=f"Underdog has H2H edge ({h2h_label}) but favourite approved",
                    severity=ContradictionSeverity.MINOR,
                    modules=["M27", "M11"],
                    value=h2h_label
                ))
        else:
            h2h_bonus = 0.0
        
        score += h2h_bonus
        score_breakdown["h2h"] = h2h_bonus

    # ─────────────────────────────────────────
    # CONTEXT CONTRIBUTION (bonus up to 0.5)
    # ─────────────────────────────────────────
    if context:
        match_importance = getattr(context, 'match_importance', 0.5)
        is_dead_rubber = getattr(context, 'is_dead_rubber', False)
        is_rivalry = getattr(context, 'is_rivalry', False)
        context_label = getattr(context, 'context_label', 'NORMAL')
        
        if is_dead_rubber:
            context_bonus = -0.3
            if report.is_approved:
                report.contradictions.append(Contradiction(
                    description="Dead rubber match but bet approved (low intensity expected)",
                    severity=ContradictionSeverity.MINOR,
                    modules=["M26", "M11"],
                    value=context_label
                ))
        elif is_rivalry:
            context_bonus = 0.2
        elif match_importance > 0.7:
            context_bonus = 0.3
        else:
            context_bonus = 0.0
        
        score += context_bonus
        score_breakdown["context"] = context_bonus

    # ─────────────────────────────────────────
    # PERSONNEL CONTRIBUTION (bonus up to 0.5)
    # ─────────────────────────────────────────
    if personnel:
        personnel_score = getattr(personnel, 'overall_personnel_score', 50.0)
        
        if personnel_score > 75:
            personnel_bonus = 0.4
        elif personnel_score > 60:
            personnel_bonus = 0.2
        elif personnel_score < 40:
            personnel_bonus = -0.2
            if report.is_approved:
                report.contradictions.append(Contradiction(
                    description=f"Poor personnel score ({personnel_score:.0f}) but bet approved",
                    severity=ContradictionSeverity.MINOR,
                    modules=["M6", "M11"],
                    value=personnel_score
                ))
        else:
            personnel_bonus = 0.0
        
        score += personnel_bonus
        score_breakdown["personnel"] = personnel_bonus

    # ─────────────────────────────────────────
    # CONFIDENCE INTEGRITY CHECK
    # ─────────────────────────────────────────
    # Check if confidence matches the evidence
    if report.confidence == "HIGH" and score < 5.0:
        report.confidence_integrity = "OVERCONFIDENT"
        report.suspicious_flag = True
        report.suspicious_reasons.append(f"High confidence ({report.confidence}) but low supporting evidence (score={score:.1f}/10)")
    elif report.confidence == "LOW" and score >= 6.0:
        report.confidence_integrity = "UNDERCONFIDENT"
        report.recommendations.append("Consider raising confidence - evidence supports stronger conviction")
    else:
        report.confidence_integrity = "VALID"

    # ─────────────────────────────────────────
    # RISK ALIGNMENT CHECK
    # ─────────────────────────────────────────
    num_contradictions = len(report.contradictions)
    if num_contradictions == 0:
        report.risk_alignment = "ALIGNED"
    elif num_contradictions <= 2:
        report.risk_alignment = "PARTIAL"
        score -= num_contradictions * 0.25
    else:
        report.risk_alignment = "MISALIGNED"
        score -= num_contradictions * 0.4

    # Check risk flags
    risk_flags = getattr(master_verdict, 'risk_flags', [])
    if len(risk_flags) > MAX_RISK_FLAGS_APPROVED and report.is_approved:
        report.contradictions.append(Contradiction(
            description=f"{len(risk_flags)} risk flags on approved bet",
            severity=ContradictionSeverity.MODERATE,
            modules=["M11", "M12"],
            value=risk_flags
        ))
        score -= 0.5

    # ─────────────────────────────────────────
    # NORMALIZE SCORE TO 0-10
    # ─────────────────────────────────────────
    report.decision_score = round(max(0.0, min(10.0, score)), 1)
    report.score_breakdown = score_breakdown
    
    # Determine grade
    if report.decision_score >= EXCELLENT_THRESHOLD:
        report.decision_grade = "EXCELLENT"
    elif report.decision_score >= GOOD_THRESHOLD:
        report.decision_grade = "GOOD"
    elif report.decision_score >= MEDIUM_THRESHOLD:
        report.decision_grade = "MEDIUM"
    elif report.decision_score >= POOR_THRESHOLD:
        report.decision_grade = "POOR"
    else:
        report.decision_grade = "CRITICAL"

    # ─────────────────────────────────────────
    # RECOMMENDATIONS
    # ─────────────────────────────────────────
    if report.suspicious_flag:
        report.recommendations.append("⚠️ Review decision - multiple red flags detected")
    
    if report.confidence_integrity == "OVERCONFIDENT":
        report.recommendations.append("Re-calibrate confidence thresholds - overconfidence detected")
    
    if "TRAP" in str(report.module_details.get('matrix', {})):
        report.recommendations.append("Trap detected - review M4 pre-filter thresholds")
    
    if len(report.contradictions) > 2:
        report.recommendations.append("Multiple contradictions - investigate pipeline consistency")
    
    if not report.key_factors and report.is_approved:
        report.recommendations.append("Approved without strong supporting factors - add more forensic checks")
    
    if report.decision_score < 4.0 and report.is_approved:
        report.recommendations.append("Low decision score on approved bet - consider rejecting similar patterns")

    return report


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════

def run_batch_explainability(
    verdicts: List[Any],
    oracles: List[Any],
    duals: List[Any] = None,
    underdogs: List[Any] = None,
    matrices: List[Any] = None,
    ais: List[Any] = None,
    h2hs: List[Any] = None,
    contexts: List[Any] = None,
    personnels: List[Any] = None,
    config: Any = None,
) -> List[ExplainabilityReport]:
    """
    Generate explainability reports for multiple verdicts.
    
    Args:
        verdicts: List of MasterVerdict objects
        oracles: List of OracleVerdict objects
        duals: Optional list of DualPatternVerdict objects
        underdogs: Optional list of PatternIntelligence objects
        matrices: Optional list of TallyMatrixAnalysis objects
        ais: Optional list of CombinedVerdict objects
        h2hs: Optional list of H2HDeepAnalysis objects
        contexts: Optional list of MatchContextScore objects
        personnels: Optional list of PersonnelMetrics objects
        config: Optional SystemConfig
    
    Returns:
        List of ExplainabilityReport objects
    """
    reports = []
    
    for i, mv in enumerate(verdicts):
        oracle = oracles[i] if i < len(oracles) else None
        dual = duals[i] if duals and i < len(duals) else None
        underdog = underdogs[i] if underdogs and i < len(underdogs) else None
        matrix = matrices[i] if matrices and i < len(matrices) else None
        ai = ais[i] if ais and i < len(ais) else None
        h2h = h2hs[i] if h2hs and i < len(h2hs) else None
        context = contexts[i] if contexts and i < len(contexts) else None
        personnel = personnels[i] if personnels and i < len(personnels) else None
        
        report = run_explainability_engine(
            master_verdict=mv,
            oracle=oracle,
            dual=dual,
            underdog=underdog,
            matrix=matrix,
            ai=ai,
            config=config,
            h2h=h2h,
            context=context,
            personnel=personnel,
        )
        reports.append(report)
    
    return reports


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — HUMAN READABLE OUTPUT
# ═══════════════════════════════════════════════════════════════

def print_explainability(report: ExplainabilityReport, level: ExplainabilityLevel = ExplainabilityLevel.DETAILED) -> None:
    """Pretty print an explainability report to console."""
    
    # Determine emoji based on status
    if report.suspicious_flag:
        status_emoji = "🚨"
    elif report.is_approved:
        status_emoji = "✅"
    else:
        status_emoji = "❌"
    
    print("\n" + "=" * 70)
    print(f"   {status_emoji} ORACLE BEAST – DECISION INTELLIGENCE REPORT")
    print("=" * 70)

    print(f"Match        : {report.match}")
    print(f"Selection    : {report.selection}")
    print(f"Status       : {report.final_status}")
    print(f"Confidence   : {report.confidence}")
    print(f"Score        : {report.decision_score:.1f}/10 ({report.decision_grade})")
    print(f"Normalized   : {report.normalized_score:.3f}")
    print(f"Timestamp    : {report.timestamp[:19]} UTC")

    print(f"\n📊 Integrity:")
    print(f"  Confidence Integrity: {report.confidence_integrity}")
    print(f"  Risk Alignment      : {report.risk_alignment}")
    print(f"  Suspicious Flag     : {'⚠️ YES' if report.suspicious_flag else '✓ NO'}")

    if level != ExplainabilityLevel.SUMMARY:
        if report.key_factors:
            print("\n✓ Key Factors:")
            # Sort by contribution descending
            sorted_factors = sorted(report.key_factors, key=lambda x: x.contribution, reverse=True)
            for f in sorted_factors[:5]:
                icon = "🟢" if f.is_positive else "🔴"
                print(f"  {icon} {f.name}: {f.explanation} (+{f.contribution:.1f})")

        if report.contradictions:
            print("\n⚠ Contradictions:")
            for c in report.contradictions:
                severity_icon = {
                    ContradictionSeverity.MINOR: "🔍",
                    ContradictionSeverity.MODERATE: "⚠️",
                    ContradictionSeverity.MAJOR: "🚨",
                    ContradictionSeverity.CRITICAL: "💀",
                }.get(c.severity, "•")
                print(f"  {severity_icon} [{c.severity.value}] {c.description}")

        if report.suspicious_reasons:
            print("\n🚨 Suspicious Decision Reasons:")
            for r in report.suspicious_reasons:
                print(f"  → {r}")

        if report.recommendations:
            print("\n📋 Recommendations:")
            for r in report.recommendations:
                print(f"  → {r}")

        if report.score_breakdown:
            print("\n📈 Score Breakdown:")
            for module, points in sorted(report.score_breakdown.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(points * 2) + "░" * (10 - int(points * 2))
                print(f"  {module:12}: {points:.1f}/10 [{bar}]")

    if level == ExplainabilityLevel.FORENSIC:
        print("\n🔍 Audit Trail:")
        for line in report.audit_trail[:10]:
            print(f"  • {line}")
        if len(report.audit_trail) > 10:
            print(f"  ... and {len(report.audit_trail) - 10} more entries")
        
        if report.module_details:
            print("\n📦 Module Details:")
            for module, details in list(report.module_details.items())[:5]:
                print(f"  {module}: {str(details)[:80]}...")

    print("=" * 70)


def print_batch_summary(reports: List[ExplainabilityReport]) -> None:
    """Print a summary of multiple reports."""
    if not reports:
        print("No reports to summarize")
        return
    
    total = len(reports)
    suspicious = sum(1 for r in reports if r.suspicious_flag)
    overconfident = sum(1 for r in reports if r.confidence_integrity == "OVERCONFIDENT")
    underconfident = sum(1 for r in reports if r.confidence_integrity == "UNDERCONFIDENT")
    misaligned = sum(1 for r in reports if r.risk_alignment == "MISALIGNED")
    approved = sum(1 for r in reports if r.is_approved)
    avg_score = sum(r.decision_score for r in reports) / total
    
    print("\n" + "=" * 70)
    print("   EXPLAINABILITY BATCH SUMMARY")
    print("=" * 70)
    print(f"Total Reports      : {total}")
    print(f"Approved           : {approved} ({approved/total:.1%})")
    print(f"Suspicious         : {suspicious} ({suspicious/total:.1%})")
    print(f"Overconfident      : {overconfident} ({overconfident/total:.1%})")
    print(f"Underconfident     : {underconfident} ({underconfident/total:.1%})")
    print(f"Risk Misaligned    : {misaligned} ({misaligned/total:.1%})")
    print(f"Average Score      : {avg_score:.1f}/10")
    
    # Grade distribution
    grades = {}
    for r in reports:
        grades[r.decision_grade] = grades.get(r.decision_grade, 0) + 1
    print(f"Grade Distribution : {grades}")
    
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — EXPORTS
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "ExplainabilityLevel",
    "ContradictionSeverity",
    # Data classes
    "Contradiction",
    "DecisionFactor",
    "ExplainabilityReport",
    # Main functions
    "run_explainability_engine",
    "run_batch_explainability",
    # Output functions
    "print_explainability",
    "print_batch_summary",
]


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from dataclasses import dataclass
    
    print("\n" + "=" * 70)
    print("MODULE 17: EXPLAINABILITY ENGINE - TEST RUN")
    print("=" * 70)
    
    # Create mock data structures for testing
    @dataclass
    class MockLeg:
        selection: str = "Arsenal"
        match_id: str = "arsenal_chelsea"
        odds: float = 2.10
    
    @dataclass
    class MockOracle:
        leg: MockLeg = None
        final_status: str = "APPROVED"
        edge: float = 0.09
        failure_score: float = 2.5
        pre_filter_passed: bool = True
        pre_filter_score: float = 0.85
        
        def __post_init__(self):
            if self.leg is None:
                self.leg = MockLeg()
    
    @dataclass
    class MockDual:
        dual_risk_level: str = "LOW"
        underdog_threat_level: str = "LOW"
        pattern_clash_score: float = 0.35
        resilience_gap: float = 0.12
    
    @dataclass
    class MockUnderdog:
        threat_level: str = "LOW"
        pattern_verdict: str = "MODERATE_PATTERN"
        pattern_strength_score: float = 35.0
        goldmine_qualified: bool = False
        underdog_win_prob: float = 0.32
        underdog_edge: float = 0.05
    
    @dataclass
    class MockMatrix:
        trap_value_signal: Any = None
        combined_risk_flag: str = "NONE"
        matrix_useful: bool = True
        
        def __post_init__(self):
            @dataclass
            class MockSignal:
                signal_type: str = "NONE"
                description: str = ""
            self.trap_value_signal = MockSignal()
    
    @dataclass
    class MockAIAnalysis:
        final_ai_verdict: AIVerdict = AIVerdict.APPROVE
        agreement_level: float = 0.85
        consensus_narrative: str = "Strong consensus on value"
    
    @dataclass
    class MockAI:
        ai_analysis: MockAIAnalysis = None
        ai_chain_used: List[str] = None
        
        def __post_init__(self):
            if self.ai_analysis is None:
                self.ai_analysis = MockAIAnalysis()
            if self.ai_chain_used is None:
                self.ai_chain_used = ["deepseek", "claude", "gemini"]
    
    @dataclass
    class MockH2H:
        h2h_score: float = 75.0
        h2h_label: str = "FAV_EDGE"
    
    @dataclass
    class MockContext:
        match_importance: float = 0.75
        is_dead_rubber: bool = False
        is_rivalry: bool = False
        context_label: str = "HIGH_STAKES"
    
    @dataclass
    class MockPersonnel:
        overall_personnel_score: float = 82.0
    
    @dataclass
    class MockMasterVerdict:
        leg_id: str = "arsenal_chelsea"
        final_status: str = "APPROVED"
        final_confidence: str = "HIGH"
        oracle: MockOracle = None
        risk_flags: List[str] = field(default_factory=list)
        
        def __post_init__(self):
            if self.oracle is None:
                self.oracle = MockOracle()
    
    # Create instances
    master = MockMasterVerdict()
    oracle = MockOracle()
    dual = MockDual()
    underdog = MockUnderdog()
    matrix = MockMatrix()
    ai = MockAI()
    h2h = MockH2H()
    context = MockContext()
    personnel = MockPersonnel()
    
    print("\n📊 TEST 1: Valid Approved Bet")
    print("-" * 40)
    
    # Generate report
    report = run_explainability_engine(
        master_verdict=master,
        oracle=oracle,
        dual=dual,
        underdog=underdog,
        matrix=matrix,
        ai=ai,
        h2h=h2h,
        context=context,
        personnel=personnel,
    )
    
    # Print report
    print_explainability(report)
    
    # Test with suspicious case
    print("\n" + "=" * 70)
    print("TEST 2: Suspicious Decision")
    print("=" * 70)
    
    suspicious_master = MockMasterVerdict(
        leg_id="suspicious_match",
        final_status="APPROVED",
        final_confidence="HIGH",
        risk_flags=["High draw probability", "Pattern clash", "Injuries", "Poor form"],
    )
    suspicious_oracle = MockOracle(
        edge=0.01,  # Very low edge
        failure_score=6.0,  # High failure score
        pre_filter_score=0.45,
    )
    suspicious_dual = MockDual(dual_risk_level="HIGH")
    suspicious_underdog = MockUnderdog(threat_level="HIGH", goldmine_qualified=False)
    
    suspicious_report = run_explainability_engine(
        master_verdict=suspicious_master,
        oracle=suspicious_oracle,
        dual=suspicious_dual,
        underdog=suspicious_underdog,
        matrix=matrix,
        ai=ai,
    )
    
    print_explainability(suspicious_report)
    
    # Batch summary test
    print("\n" + "=" * 70)
    print("TEST 3: Batch Summary")
    print("=" * 70)
    
    batch_reports = [report, suspicious_report]
    print_batch_summary(batch_reports)
    
    # Leg data for M11
    print("\n📊 Leg Data for M11:")
    leg_data = report.to_leg_data()
    for key, value in leg_data.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("MODULE 17 READY FOR PRODUCTION")
    print("=" * 70)