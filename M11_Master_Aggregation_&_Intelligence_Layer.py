"""
The Match Oracle - Module 11: Master Aggregation & Intelligence Layer (ENHANCED v6)
======================================================================
Combines Modules 4-10 into a unified MasterVerdict per match.

This is the central nervous system of the pipeline. It orchestrates:
- Oracle pipeline (Modules 4-6) → forensic checks, pre-filter, probabilities
- Dual pattern engine (M8) → pattern clash and underdog threat
- Underdog scanner (M9) → RTM pattern intelligence
- Season tally matrix (M10) → transition probabilities and trap detection
- AI intelligence layer (M7) → multi-AI consensus (if enabled)
- Match context (M26) → motivation, dead rubber, rivalry detection
- H2H deep analysis (M27) → psychological dominance and draw boost
- Pattern reliability (M8/M9) → trustworthiness of detected patterns
- Distortion tracking (M1/M10/M15) → clean vs dirty probability adjustment

REFINEMENTS IN THIS VERSION (v6):
------------------------------
1. CHANGED: M8 conflict detection now triggers HARD REJECT (not CAUTION)
2. CHANGED: Removed conflict stake multiplier (conflict = zero stake)
3. CHANGED: Zero score for conflicts ensures rejection
4. ADDED: REJECTED (H2H CONFLICT) status for final verdict
5. ADDED: Early termination when conflict detected in weighted decision

PREVIOUS ENHANCEMENTS (v5):
------------------------------
- M8 conflict detection integration (H2H vs current season)
- M27 draw boost factor for probability adjustment
- Pattern reliability penalty for overconfident approvals
- Tier performance confidence scoring
- Conflict severity escalation (LOW/MEDIUM/HIGH)
- Stake multiplier adjustment based on conflict severity
- Final verdict override when multiple modules disagree

WEIGHTED DECISION FORMULA (v6):
-------------------------
Total Score = Σ(module_score × module_weight × reliability_weight)

CONFLICT RESOLUTION (v6):
-----------------------
- If ANY conflict detected → OVERRIDE to REJECTED (zero score)
- No stake calculation for conflicted legs

ADJUSTED WEIGHTS (v5):
---------------------
| Module | Weight |
|--------|--------|
| M4 Pre-filter | 18% |
| M5 Forensics | 15% |
| M7 AI | 12% |
| M8 Dual Pattern | 12% |
| M9 Underdog | 8% |
| M10 Matrix | 5% |
| M6 Personnel | 10% |
| M27 H2H | 12% |
| M26 Context | 12% |

Usage:
    from module11 import run_master_aggregation, MasterVerdict
    
    verdict = run_master_aggregation(leg, fav_is_home=True, run_ai=True)
    
    print(f"Final: {verdict.final_status} ({verdict.weighted_score:.3f})")
    print(f"Clean Score: {verdict.clean_weighted_score:.3f}")
    print(f"Conflict Detected: {verdict.conflict_detected}")
    print(f"Contributions: {verdict.weighted_contributions}")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import math
import warnings

from module2 import Leg, BetMarket


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — MODULE IMPORTS (with graceful fallbacks)
# ═══════════════════════════════════════════════════════════════

# Module 4: Asymmetric Pre-filter
try:
    from module4 import run_asymmetric_prefilter, PreFilterResult, detect_favourite
    _M4_AVAILABLE = True
except ImportError:
    _M4_AVAILABLE = False
    def run_asymmetric_prefilter(leg, verbose=False):
        from dataclasses import dataclass
        @dataclass
        class FakeResult:
            passed: bool = True
            weighted_pass_score: float = 0.5
            checks_passed: int = 4
            checks_evaluated: int = 8
        return FakeResult()
    def detect_favourite(leg):
        return "HOME"

# Module 5: Forensic Checks
try:
    from module5 import run_forensic_checks, REJECTION_THRESHOLD, get_forensic_summary
    _M5_AVAILABLE = True
except ImportError:
    _M5_AVAILABLE = False
    REJECTION_THRESHOLD = 4.5
    def run_forensic_checks(leg, league_size=20):
        return 0.0
    def get_forensic_summary(leg):
        return {"failure_score": 0.0, "passes": True}

# Module 6: Personnel Forensics
try:
    from module6 import analyze_personnel_for_leg, PersonnelMetrics
    _M6_AVAILABLE = True
except ImportError:
    _M6_AVAILABLE = False
    PersonnelMetrics = None
    def analyze_personnel_for_leg(leg, api_key=None, season=2025, verbose=False):
        from dataclasses import dataclass
        @dataclass
        class FakeMetrics:
            overall_personnel_score: float = 50.0
            normalized_score: float = 0.5
            confidence_factor: float = 0.5
            home_advantage_adjustment: float = 0.0
            summary: str = "Personnel data unavailable"
        return FakeMetrics(), FakeMetrics()

# Module 7: AI Intelligence
try:
    from module7 import run_ai_intelligence, CombinedVerdict, AIVerdict
    _M7_AVAILABLE = True
except ImportError:
    _M7_AVAILABLE = False
    run_ai_intelligence = None
    CombinedVerdict = None
    AIVerdict = None

# Module 8: Dual Pattern Engine (v6 with HARD REJECT)
try:
    from module8 import run_dual_pattern_engine, DualPatternVerdict
    _M8_AVAILABLE = True
except ImportError:
    _M8_AVAILABLE = False
    DualPatternVerdict = None
    def run_dual_pattern_engine(leg, fav_is_home=True, store_in_leg=True, use_clean=True):
        return None

# Module 9: Underdog Scanner
try:
    from module9 import run_underdog_scanner, PatternIntelligence
    _M9_AVAILABLE = True
except ImportError:
    _M9_AVAILABLE = False
    PatternIntelligence = None
    def run_underdog_scanner(leg, fav_is_home=True, mode="OPPORTUNITY", use_clean=True, **kwargs):
        return None

# Module 10: Season Tally Matrix
try:
    from module10 import run_tally_matrix_analysis, TallyMatrixAnalysis
    _M10_AVAILABLE = True
except ImportError:
    _M10_AVAILABLE = False
    TallyMatrixAnalysis = None
    def run_tally_matrix_analysis(home_results, away_results, **kwargs):
        return None

# Module 26: Match Context
try:
    from module26 import context_from_leg, MatchContextScore
    _M26_AVAILABLE = True
except ImportError:
    _M26_AVAILABLE = False
    MatchContextScore = None
    def context_from_leg(leg):
        return None

# Module 27: H2H Deep Analysis
try:
    from module27 import run_h2h_deep_analyzer, H2HDeepAnalysis, get_draw_boost_factor
    _M27_AVAILABLE = True
except ImportError:
    _M27_AVAILABLE = False
    H2HDeepAnalysis = None
    get_draw_boost_factor = None
    def run_h2h_deep_analyzer(h2h_data, fav_is_home=True):
        return None

# Module 28: Calibration Correction
try:
    from module28 import apply_calibration_correction, run_calibration_check
    _M28_AVAILABLE = True
except ImportError:
    _M28_AVAILABLE = False
    def apply_calibration_correction(prob, report):
        return prob

# Module 29: Drawdown Status
try:
    from module29 import DrawdownStatus
    _M29_AVAILABLE = True
except ImportError:
    _M29_AVAILABLE = False
    DrawdownStatus = None

# Module 31: League Intelligence
try:
    from module31 import build_league_profile, LeagueProfile
    _M31_AVAILABLE = True
except ImportError:
    _M31_AVAILABLE = False
    LeagueProfile = None
    def build_league_profile(league_id, league_name, **kwargs):
        return None


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — CONSTANTS (v6)
# ═══════════════════════════════════════════════════════════════

# Conflict detection - HARD REJECT (v6)
CONFLICT_HARD_REJECT = True  # If True, conflict = REJECTED, not CAUTION

# Verdict thresholds
APPROVED_THRESHOLD = 0.70
CAUTION_THRESHOLD = 0.50

# Minimum scores for approval without conflict
MIN_SCORE_FOR_APPROVAL = 0.70
MIN_SCORE_FOR_CAUTION = 0.50

# Tier performance critical threshold
TIER_CRITICAL_THRESHOLD = 0.30  # 30% win rate
TIER_MIN_GAMES = 5


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — DATA STRUCTURES (Enhanced v6)
# ═══════════════════════════════════════════════════════════════

@dataclass
class OracleVerdict:
    """
    Oracle pipeline verdict from Modules 4-6.
    
    Attributes:
        leg: The leg being analysed
        pre_filter_passed: Did M4 asymmetric pre-filter pass?
        pre_filter_score: Weighted score from M4 (0-1)
        failure_score: Cumulative failure score from M5
        forensic_passed: Did M5 forensic checks pass?
        final_status: APPROVED / CAUTION / REJECTED
        model_prob: Model's win probability for selection
        clean_model_prob: Clean probability (distortions removed)
        edge: Model probability minus implied probability
        clean_edge: Clean probability minus implied
        confidence_tier: HIGH / MEDIUM / LOW based on edge and checks
        dual_risk_level: Risk level from M8
        failure_details: Dict of failure reasons with weights
    """
    leg: Leg
    pre_filter_passed: bool = False
    pre_filter_score: float = 0.0
    failure_score: float = 0.0
    forensic_passed: bool = False
    final_status: str = "PENDING"
    model_prob: float = 0.0
    clean_model_prob: float = 0.0
    edge: float = 0.0
    clean_edge: float = 0.0
    confidence_tier: str = "LOW"
    dual_risk_level: str = "LOW"
    failure_details: Dict[str, float] = field(default_factory=dict)
    
    # Normalized scores for weighted decision
    normalized_scores: Dict[str, float] = field(default_factory=dict)
    weighted_contributions: Dict[str, float] = field(default_factory=dict)
    weighted_total: float = 0.0
    clean_weighted_total: float = 0.0
    
    @property
    def is_approved(self) -> bool:
        """Return True if verdict is APPROVED or APPROVED variant."""
        return "APPROVED" in self.final_status
    
    @property
    def is_rejected(self) -> bool:
        """Return True if verdict is REJECTED or REJECTED variant."""
        return "REJECTED" in self.final_status
    
    @property
    def reliability_penalty(self) -> float:
        """Penalty factor for unreliable patterns (0.7-1.0)."""
        if hasattr(self.leg, 'features') and self.leg.features.get("patterns_reliable", True):
            return 1.0
        return 0.7
    
    @property
    def summary(self) -> str:
        """Human-readable summary."""
        return (f"Oracle: {self.final_status} | "
                f"edge={self.edge:+.3f} | "
                f"clean_edge={self.clean_edge:+.3f} | "
                f"prob={self.model_prob:.1%} | "
                f"failure={self.failure_score:.1f} | "
                f"weighted={self.weighted_total:.3f}")


@dataclass
class MasterVerdict:
    """
    Final aggregated verdict from all modules.
    
    ENHANCED v6: Now includes HARD REJECT on conflict detection.
    """
    leg_id: str
    oracle: OracleVerdict
    ai: Optional[Any] = None
    dual_pattern: Optional[Any] = None
    underdog: Optional[Any] = None
    season_matrix: Optional[Any] = None
    match_context: Optional[Any] = None
    h2h_analysis: Optional[Any] = None
    
    final_status: str = "PENDING"
    final_confidence: str = "-"
    risk_flags: List[str] = field(default_factory=list)
    decision_notes: List[str] = field(default_factory=list)
    
    weighted_score: float = 0.0
    clean_weighted_score: float = 0.0
    weighted_contributions: Dict[str, float] = field(default_factory=dict)
    
    # Pattern reliability
    patterns_reliable: bool = True
    reliability_score: float = 1.0
    distortion_warning: Optional[str] = None
    clean_bounce_back: float = 0.0
    
    # Draw boost from H2H
    h2h_draw_boost_factor: float = 1.0
    h2h_draw_rate: float = 0.0
    
    # NEW v5/v6: Conflict detection
    conflict_detected: bool = False
    conflict_severity: str = "NONE"  # NONE / LOW / MEDIUM / HIGH
    conflict_reasons: List[str] = field(default_factory=list)
    override_applied: bool = False
    final_stake_multiplier: float = 0.0  # NEW v6: 0 for conflicts, 1.0 otherwise
    
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    @property
    def is_approved(self) -> bool:
        """Return True if final verdict is approved."""
        return "APPROVED" in self.final_status
    
    @property
    def is_caution(self) -> bool:
        """Return True if final verdict is caution."""
        return "CAUTION" in self.final_status
    
    @property
    def is_rejected(self) -> bool:
        """Return True if final verdict is rejected."""
        return "REJECTED" in self.final_status
    
    @property
    def is_h2h_conflict(self) -> bool:
        """Return True if rejected due to H2H conflict."""
        return "H2H CONFLICT" in self.final_status
    
    @property
    def reliability_adjusted_score(self) -> float:
        """Score adjusted for pattern reliability."""
        return self.weighted_score * self.reliability_score
    
    @property
    def confidence_tier_from_score(self) -> str:
        """Derive confidence tier from weighted score."""
        if self.weighted_score >= 0.75:
            return "HIGH"
        elif self.weighted_score >= 0.55:
            return "MEDIUM"
        return "LOW"
    
    @property
    def summary(self) -> str:
        """Human-readable summary."""
        reliable_str = " ✓reliable" if self.patterns_reliable else " ⚠unreliable"
        
        # NEW v6: Conflict shows REJECTED
        if self.conflict_detected:
            conflict_str = f" 🚨H2H CONFLICT REJECTED"
        else:
            conflict_str = ""
        
        draw_str = f" draw_boost={self.h2h_draw_boost_factor:.2f}x" if self.h2h_draw_boost_factor != 1.0 else ""
        clean_str = f" clean={self.clean_weighted_score:.3f}" if self.clean_weighted_score != self.weighted_score else ""
        
        return (f"{self.leg_id}: {self.final_status} "
                f"({self.final_confidence}){reliable_str}{conflict_str}{draw_str} | "
                f"Weighted: {self.weighted_score:.3f}{clean_str} | "
                f"Risk flags: {len(self.risk_flags)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        leg = self.oracle.leg
        return {
            "leg_id": self.leg_id,
            "match": getattr(leg, 'match_id', 'unknown'),
            "selection": leg.selection,
            "odds": leg.odds,
            "market": leg.market.value if leg.market else "straight_win",
            "final_status": self.final_status,
            "final_confidence": self.final_confidence,
            "edge": round(self.oracle.edge, 4),
            "clean_edge": round(self.oracle.clean_edge, 4),
            "model_prob": round(self.oracle.model_prob, 4),
            "weighted_score": round(self.weighted_score, 3),
            "clean_weighted_score": round(self.clean_weighted_score, 3),
            "h2h_draw_boost": round(self.h2h_draw_boost_factor, 3),
            "weighted_contributions": {k: round(v, 3) for k, v in self.weighted_contributions.items()},
            "risk_flags": self.risk_flags[:5],
            "decision_notes": self.decision_notes[:3],
            "patterns_reliable": self.patterns_reliable,
            "reliability_score": round(self.reliability_score, 3),
            "distortion_warning": self.distortion_warning,
            # NEW v5/v6 fields
            "conflict_detected": self.conflict_detected,
            "conflict_severity": self.conflict_severity,
            "conflict_reasons": self.conflict_reasons[:3],
            "override_applied": self.override_applied,
            "final_stake_multiplier": round(self.final_stake_multiplier, 2),
            "created_at": self.created_at,
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — CONFLICT DETECTION (UPDATED v6 - HARD REJECT)
# ═══════════════════════════════════════════════════════════════

def _detect_conflicts(
    m4_score: float,
    m5_score: float,
    m8_dual_score: float,
    m9_score: float,
    m10_score: float,
    m6_score: float,
    m26_score: float,
    m27_h2h_score: float,
    m27_draw_boost: float,
    dual_pattern: Any = None,
) -> Tuple[bool, str, List[str], float]:
    """
    Detect conflicts between modules, especially H2H vs current season.
    
    NEW v6: If conflict detected, returns stake_multiplier = 0.0 (HARD REJECT)
    
    Returns:
        Tuple of (conflict_detected, severity, reasons, stake_multiplier)
        0.0 stake_multiplier = HARD REJECT
    """
    conflicts = []
    severity_count = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    
    # Get M8 conflict info if available
    m8_conflict = False
    m8_conflict_severity = "NONE"
    if dual_pattern and hasattr(dual_pattern, 'h2h_conflict_detected'):
        m8_conflict = dual_pattern.h2h_conflict_detected
        m8_conflict_severity = getattr(dual_pattern, 'conflict_severity', "NONE")
        
        if m8_conflict:
            conflicts.append(f"M8: {getattr(dual_pattern, 'conflict_reasons', ['H2H vs current season conflict'])[0]}")
            if m8_conflict_severity == "HIGH":
                severity_count["HIGH"] += 1
            elif m8_conflict_severity == "MEDIUM":
                severity_count["MEDIUM"] += 1
            else:
                severity_count["LOW"] += 1
    
    # Check H2H vs M4 Pre-filter
    if m27_h2h_score > 0.70 and m4_score < 0.50:
        conflicts.append(f"H2H strong ({m27_h2h_score:.0%}) but M4 pre-filter weak ({m4_score:.0%})")
        severity_count["MEDIUM"] += 1
    
    # Check H2H vs M5 Forensics
    if m27_h2h_score > 0.70 and m5_score < 0.45:
        conflicts.append(f"H2H strong ({m27_h2h_score:.0%}) but M5 forensics weak ({m5_score:.0%})")
        severity_count["MEDIUM"] += 1
    
    # Check H2H vs M8 Dual Pattern
    if m27_h2h_score > 0.70 and m8_dual_score < 0.50:
        conflicts.append(f"H2H strong ({m27_h2h_score:.0%}) but M8 dual pattern weak ({m8_dual_score:.0%})")
        severity_count["HIGH"] += 1
    
    # Check H2H vs M26 Context
    if m27_h2h_score > 0.70 and m26_score < 0.50:
        conflicts.append(f"H2H strong ({m27_h2h_score:.0%}) but M26 context weak ({m26_score:.0%})")
        severity_count["MEDIUM"] += 1
    
    # Check draw boost vs M8 (high draw probability but dual pattern says low draw)
    if m27_draw_boost > 1.10 and m8_dual_score > 0.65:
        conflicts.append(f"High draw boost ({m27_draw_boost:.2f}x) but M8 expects favourite win")
        severity_count["LOW"] += 1
    
    # Determine overall severity
    total_conflicts = len(conflicts)
    
    if total_conflicts == 0:
        return False, "NONE", [], 1.0
    
    # NEW v6: ANY conflict = HARD REJECT (stake_multiplier = 0.0)
    # No more stake reduction tiers. Conflict = NO BET.
    
    if severity_count.get("HIGH", 0) >= 1 or total_conflicts >= 3:
        severity = "HIGH"
        stake_multiplier = 0.0  # HARD REJECT
    elif severity_count.get("MEDIUM", 0) >= 1 or total_conflicts >= 2:
        severity = "MEDIUM"
        stake_multiplier = 0.0  # HARD REJECT
    else:
        severity = "LOW"
        stake_multiplier = 0.0  # HARD REJECT
    
    return True, severity, conflicts, stake_multiplier


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — TIER PERFORMANCE CHECK
# ═══════════════════════════════════════════════════════════════

def _check_tier_performance(
    leg: Leg,
    fav_is_home: bool,
    standings: Dict[int, Dict] = None,
    league_size: int = 20,
) -> Tuple[bool, float, str]:
    """
    Check if favourite has CRITICAL tier performance.
    
    Returns:
        Tuple of (is_critical, penalty_multiplier, reason)
    """
    if not standings:
        return False, 1.0, ""
    
    # Determine favourite
    if fav_is_home:
        fav_profile = leg.home_profile
        venue = "home"
    else:
        fav_profile = leg.away_profile
        venue = "away"
    
    if not fav_profile:
        return False, 1.0, ""
    
    fav_id = int(fav_profile.team_id) if fav_profile.team_id else 0
    if fav_id == 0 or fav_id not in standings:
        return False, 1.0, ""
    
    fav_pos = standings[fav_id].get("position", 10)
    top_threshold = max(1, int(league_size * 0.30))
    bottom_threshold = league_size - top_threshold + 1
    
    # Determine opponent tier
    opp_profile = leg.away_profile if fav_is_home else leg.home_profile
    opp_id = int(opp_profile.team_id) if opp_profile and opp_profile.team_id else 0
    if opp_id == 0 or opp_id not in standings:
        return False, 1.0, ""
    
    opp_pos = standings[opp_id].get("position", 10)
    
    if opp_pos <= top_threshold:
        opp_tier = "top6"
    elif opp_pos >= bottom_threshold:
        opp_tier = "bottom6"
    else:
        opp_tier = "mid"
    
    # Get tier performance from leg features (set by M8/M1)
    tier_perf_key = f"{venue}_vs_{opp_tier}"
    tier_perf = None
    
    if hasattr(leg, 'features'):
        if venue == "home":
            tier_perf = leg.features.get("home_tier_performance", {}).get(opp_tier)
        else:
            tier_perf = leg.features.get("away_tier_performance", {}).get(opp_tier)
    
    if tier_perf and hasattr(tier_perf, 'games_played'):
        if tier_perf.games_played >= TIER_MIN_GAMES:
            if tier_perf.win_rate < TIER_CRITICAL_THRESHOLD:
                return True, 0.50, f"{fav_profile.team_name} {venue} vs {opp_tier}: {tier_perf.win_rate:.0%} win rate over {tier_perf.games_played} games (CRITICAL)"
    
    return False, 1.0, ""


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — COLLECT LEG DATA (Enhanced v6)
# ═══════════════════════════════════════════════════════════════

def _collect_leg_data(
    leg: Leg,
    oracle_verdict: OracleVerdict,
    dual_verdict: Any,
    underdog_res: Any,
    season_matrix: Any,
    ai_verdict: Any,
    h2h_analysis: Any,
    match_context: Any,
    personnel_home: Any = None,
    personnel_away: Any = None,
) -> Dict[str, Any]:
    """
    Collect all module outputs into a single dictionary for weighted decision.
    
    ENHANCED v6: Now includes M8 conflict flags for HARD REJECT.
    """
    leg_data = {}
    
    # M4: Pre-filter
    leg_data["pre_filter_passed"] = oracle_verdict.pre_filter_passed
    leg_data["pre_filter_score"] = oracle_verdict.pre_filter_score
    
    # M5: Failure score
    leg_data["failure_score"] = oracle_verdict.failure_score
    
    # M7: AI verdict
    if ai_verdict and hasattr(ai_verdict, 'ai_analysis'):
        ai_analysis = ai_verdict.ai_analysis
        ai_verdict_str = getattr(ai_analysis, 'final_ai_verdict', None)
        if ai_verdict_str:
            leg_data["ai_verdict"] = ai_verdict_str.value if hasattr(ai_verdict_str, 'value') else str(ai_verdict_str)
        else:
            leg_data["ai_verdict"] = "CAUTION"
        leg_data["ai_score"] = getattr(ai_analysis, 'normalized_score', 0.5)
        leg_data["ai_agreement"] = getattr(ai_analysis, 'agreement_level', 0.5)
    else:
        leg_data["ai_verdict"] = "CAUTION"
        leg_data["ai_score"] = 0.5
        leg_data["ai_agreement"] = 0.5
    
    # M8: Dual pattern (v6 with HARD REJECT flags)
    if dual_verdict:
        leg_data["dual_risk_level"] = getattr(dual_verdict, 'dual_risk_level', 'MEDIUM')
        leg_data["dual_risk_score"] = getattr(dual_verdict, 'risk_score', 0.5)
        leg_data["clean_dual_risk_score"] = getattr(dual_verdict, 'clean_risk_score', 0.5)
        leg_data["patterns_reliable"] = getattr(dual_verdict, 'patterns_reliable', True)
        leg_data["distortion_warning"] = getattr(dual_verdict, 'distortion_warning', None)
        # NEW v6: Conflict flags from M8 (now triggers HARD REJECT)
        leg_data["h2h_conflict_detected"] = getattr(dual_verdict, 'h2h_conflict_detected', False)
        leg_data["conflict_severity"] = getattr(dual_verdict, 'conflict_severity', 'NONE')
        leg_data["conflict_reasons"] = getattr(dual_verdict, 'conflict_reasons', [])
    else:
        leg_data["dual_risk_level"] = "MEDIUM"
        leg_data["dual_risk_score"] = 0.5
        leg_data["clean_dual_risk_score"] = 0.5
        leg_data["patterns_reliable"] = True
        leg_data["distortion_warning"] = None
        leg_data["h2h_conflict_detected"] = False
        leg_data["conflict_severity"] = "NONE"
        leg_data["conflict_reasons"] = []
    
    # M9: Underdog
    if underdog_res:
        leg_data["underdog_edge"] = getattr(underdog_res, 'underdog_edge', 0.0)
        leg_data["clean_underdog_edge"] = getattr(underdog_res, 'clean_underdog_edge', 0.0)
        leg_data["underdog_score"] = getattr(underdog_res, 'normalized_score', 0.5)
        leg_data["clean_underdog_score"] = getattr(underdog_res, 'clean_normalized_score', 0.5)
        leg_data["threat_level"] = getattr(underdog_res, 'threat_level', 'NONE')
        leg_data["clean_threat_level"] = getattr(underdog_res, 'clean_threat_level', 'NONE')
        leg_data["pattern_reliability_score"] = getattr(underdog_res, 'pattern_reliability_score', 0.5)
    else:
        leg_data["underdog_edge"] = 0.0
        leg_data["clean_underdog_edge"] = 0.0
        leg_data["underdog_score"] = 0.5
        leg_data["clean_underdog_score"] = 0.5
        leg_data["threat_level"] = "NONE"
        leg_data["clean_threat_level"] = "NONE"
        leg_data["pattern_reliability_score"] = 0.5
    
    # M10: Matrix
    if season_matrix:
        leg_data["matrix_useful"] = getattr(season_matrix, 'matrix_useful', False)
        leg_data["clean_matrix_useful"] = getattr(season_matrix, 'clean_matrix_useful', False)
        leg_data["matrix_score"] = getattr(season_matrix, 'normalized_score', 0.5)
        leg_data["clean_matrix_score"] = getattr(season_matrix, 'clean_normalized_score', 0.5)
        leg_data["distortion_impact"] = getattr(season_matrix, 'bilateral', None)
        if leg_data["distortion_impact"]:
            leg_data["distortion_impact"] = getattr(leg_data["distortion_impact"], 'distortion_impact', 0.0)
    else:
        leg_data["matrix_useful"] = False
        leg_data["clean_matrix_useful"] = False
        leg_data["matrix_score"] = 0.3
        leg_data["clean_matrix_score"] = 0.3
        leg_data["distortion_impact"] = 0.0
    
    # M6: Personnel
    if personnel_home and personnel_away:
        home_score = personnel_home.overall_personnel_score
        away_score = personnel_away.overall_personnel_score
        leg_data["personnel_advantage"] = home_score - away_score + 50
        leg_data["personnel_score_home"] = personnel_home.normalized_score
        leg_data["personnel_score_away"] = personnel_away.normalized_score
    else:
        leg_data["personnel_advantage"] = 50.0
        leg_data["personnel_score_home"] = 0.5
        leg_data["personnel_score_away"] = 0.5
    
    # M27: H2H (v5/v6 with draw boost)
    if h2h_analysis:
        leg_data["h2h_score"] = getattr(h2h_analysis, 'h2h_score', 50.0)
        leg_data["h2h_normalized"] = getattr(h2h_analysis, 'normalized_score', 0.5)
        leg_data["h2h_clean_normalized"] = getattr(h2h_analysis, 'normalized_score', 0.5)
        leg_data["h2h_bounce_back"] = getattr(h2h_analysis, 'h2h_rtm', None)
        if leg_data["h2h_bounce_back"]:
            leg_data["h2h_bounce_back"] = getattr(leg_data["h2h_bounce_back"], 'bounce_back_rate', 0.33)
        # Draw boost fields
        leg_data["h2h_draw_boost_factor"] = getattr(h2h_analysis, 'draw_boost_factor', 1.0)
        leg_data["h2h_draw_rate"] = getattr(h2h_analysis, 'overall', None)
        if leg_data["h2h_draw_rate"]:
            leg_data["h2h_draw_rate"] = getattr(leg_data["h2h_draw_rate"], 'draw_rate', 0.0)
        else:
            leg_data["h2h_draw_rate"] = 0.0
        leg_data["h2h_recent_draw_rate"] = getattr(h2h_analysis, 'draw_boost', None)
        if leg_data["h2h_recent_draw_rate"]:
            leg_data["h2h_recent_draw_rate"] = getattr(leg_data["h2h_recent_draw_rate"], 'recent_draw_rate', 0.0)
    else:
        leg_data["h2h_score"] = 50.0
        leg_data["h2h_normalized"] = 0.5
        leg_data["h2h_clean_normalized"] = 0.5
        leg_data["h2h_bounce_back"] = 0.33
        leg_data["h2h_draw_boost_factor"] = 1.0
        leg_data["h2h_draw_rate"] = 0.0
        leg_data["h2h_recent_draw_rate"] = 0.0
    
    # M26: Match context
    if match_context:
        leg_data["match_importance"] = getattr(match_context, 'match_importance', 0.5)
        leg_data["is_rivalry"] = getattr(match_context, 'is_rivalry', False)
        leg_data["is_dead_rubber"] = getattr(match_context, 'is_dead_rubber', False)
        leg_data["is_six_pointer"] = getattr(match_context, 'is_six_pointer', False)
        leg_data["context_home_adj"] = getattr(match_context, 'home_advantage_adjustment', 0.0)
    else:
        leg_data["match_importance"] = 0.5
        leg_data["is_rivalry"] = False
        leg_data["is_dead_rubber"] = False
        leg_data["is_six_pointer"] = False
        leg_data["context_home_adj"] = 0.0
    
    # M31: League adjustment
    leg_data["league_tier"] = getattr(leg, 'league_tier', 3)
    leg_data["league_country"] = getattr(leg, 'league_country', 'unknown')
    leg_data["league_id"] = getattr(leg, 'league_id', 0)
    
    return leg_data


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — WEIGHTED DECISION CALCULATION (Enhanced v6 - HARD REJECT)
# ═══════════════════════════════════════════════════════════════

class DecisionWeights:
    """Decision weights for M11 aggregation (v5)."""
    
    def __init__(self):
        self.oracle_prefilter = 0.18   # M4 (18%)
        self.oracle_forensics = 0.15   # M5 (15%)
        self.ai_consensus = 0.12       # M7 (12%)
        self.dual_pattern = 0.12       # M8 (12%)
        self.underdog = 0.08           # M9 (8%)
        self.matrix = 0.05             # M10 (5%)
        self.personnel = 0.10          # M6 (10%)
        self.h2h = 0.12                # M27 (12%)
        self.context = 0.12            # M26 (12%)
    
    def to_dict(self) -> Dict[str, float]:
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
        }


def _calculate_weighted_decision(
    leg_data: Dict[str, Any],
    weights: DecisionWeights = None,
) -> Tuple[float, float, str, str, Dict[str, float], bool, str, List[str], bool]:
    """
    Enhanced weighted decision calculation with HARD REJECT on conflict.
    
    NEW v6: If conflict detected, returns zero score and REJECTED status.
    
    Returns:
        Tuple of (weighted_score, clean_weighted_score, status, confidence, contributions,
                  conflict_detected, conflict_severity, conflict_reasons, override_applied)
    """
    if weights is None:
        weights = DecisionWeights()
    
    # Get conflict flags from M8
    conflict_detected = leg_data.get("h2h_conflict_detected", False)
    conflict_severity = leg_data.get("conflict_severity", "NONE")
    conflict_reasons = leg_data.get("conflict_reasons", [])
    draw_boost = leg_data.get("h2h_draw_boost_factor", 1.0)
    
    # ─── NEW v6: HARD REJECT on conflict ─────────────────────────────
    if conflict_detected:
        # Zero score, REJECTED status, no stake
        total_score = 0.0
        clean_total_score = 0.0
        contributions = {}
        
        final_status = "REJECTED (H2H CONFLICT)"
        final_confidence = "LOW"
        override_applied = True
        
        return (total_score, clean_total_score, final_status, final_confidence, 
                contributions, conflict_detected, conflict_severity, conflict_reasons, override_applied)
    
    # Calculate standard weighted score (only if NO conflict)
    total_score = 0.0
    clean_total_score = 0.0
    contributions = {}
    
    # M4 Pre-filter
    m4_score = leg_data.get("pre_filter_score", 0.0)
    if m4_score == 0.0:
        m4_score = 1.0 if leg_data.get("pre_filter_passed", False) else 0.0
    contributions["oracle_prefilter"] = m4_score * weights.oracle_prefilter
    total_score += contributions["oracle_prefilter"]
    clean_total_score += contributions["oracle_prefilter"]
    
    # M5 Forensics
    failure_score = leg_data.get("failure_score", 5.0)
    m5_score = max(0.0, min(1.0, 1.0 - (failure_score / 10.0)))
    contributions["oracle_forensics"] = m5_score * weights.oracle_forensics
    total_score += contributions["oracle_forensics"]
    clean_total_score += contributions["oracle_forensics"]
    
    # M7 AI
    ai_score = leg_data.get("ai_score", 0.5)
    contributions["ai_consensus"] = ai_score * weights.ai_consensus
    total_score += contributions["ai_consensus"]
    clean_total_score += contributions["ai_consensus"]
    
    # M8 Dual Pattern (only if no conflict - already handled above)
    dual_score = leg_data.get("dual_risk_score", 0.5)
    clean_dual_score = leg_data.get("clean_dual_risk_score", dual_score)
    
    # Invert risk score (lower risk = higher score)
    dual_normalized = 1.0 - dual_score
    clean_dual_normalized = 1.0 - clean_dual_score
    
    contributions["dual_pattern"] = dual_normalized * weights.dual_pattern
    total_score += contributions["dual_pattern"]
    contributions["clean_dual_pattern"] = clean_dual_normalized * weights.dual_pattern
    clean_total_score += contributions["clean_dual_pattern"]
    
    # M9 Underdog
    underdog_score = leg_data.get("underdog_score", 0.5)
    clean_underdog_score = leg_data.get("clean_underdog_score", underdog_score)
    
    contributions["underdog"] = underdog_score * weights.underdog
    total_score += contributions["underdog"]
    contributions["clean_underdog"] = clean_underdog_score * weights.underdog
    clean_total_score += contributions["clean_underdog"]
    
    # M10 Matrix
    matrix_score = leg_data.get("matrix_score", 0.3)
    clean_matrix_score = leg_data.get("clean_matrix_score", matrix_score)
    
    contributions["matrix"] = matrix_score * weights.matrix
    total_score += contributions["matrix"]
    contributions["clean_matrix"] = clean_matrix_score * weights.matrix
    clean_total_score += contributions["clean_matrix"]
    
    # M6 Personnel
    personnel_score = leg_data.get("personnel_advantage", 50.0) / 100.0
    personnel_score = min(1.0, max(0.0, personnel_score))
    contributions["personnel"] = personnel_score * weights.personnel
    total_score += contributions["personnel"]
    clean_total_score += contributions["personnel"]
    
    # M27 H2H (with draw boost)
    h2h_score = leg_data.get("h2h_normalized", 0.5)
    clean_h2h_score = leg_data.get("h2h_clean_normalized", h2h_score)
    
    # Apply draw boost to H2H score
    h2h_score_boosted = min(1.0, h2h_score * draw_boost)
    clean_h2h_score_boosted = min(1.0, clean_h2h_score * draw_boost)
    
    contributions["h2h"] = h2h_score_boosted * weights.h2h
    total_score += contributions["h2h"]
    contributions["clean_h2h"] = clean_h2h_score_boosted * weights.h2h
    clean_total_score += contributions["clean_h2h"]
    
    # M26 Context
    context_importance = leg_data.get("match_importance", 0.5)
    contributions["context"] = context_importance * weights.context
    total_score += contributions["context"]
    clean_total_score += contributions["context"]
    
    # Round scores
    total_score = round(total_score, 3)
    clean_total_score = round(clean_total_score, 3)
    
    # Apply pattern reliability penalty
    patterns_reliable = leg_data.get("patterns_reliable", True)
    if not patterns_reliable:
        reliability_penalty = 0.7
        total_score = round(total_score * reliability_penalty, 3)
    
    # Determine base verdict
    if total_score >= APPROVED_THRESHOLD:
        base_status = "APPROVED"
        base_confidence = "HIGH"
    elif total_score >= CAUTION_THRESHOLD:
        base_status = "CAUTION"
        base_confidence = "MEDIUM"
    else:
        base_status = "REJECTED"
        base_confidence = "LOW"
    
    # No conflict to override (already handled at top)
    final_status = base_status
    final_confidence = base_confidence
    override_applied = False
    
    return (total_score, clean_total_score, final_status, final_confidence, 
            contributions, conflict_detected, conflict_severity, conflict_reasons, override_applied)


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — ORACLE PIPELINE (Modules 4-6)
# ═══════════════════════════════════════════════════════════════

def run_oracle_pipeline(
    leg: Leg,
    fav_is_home: bool = True,
    league_size: int = 20,
    weights: Optional[Any] = None,
    verbose: bool = False,
) -> OracleVerdict:
    """
    Run the core Oracle pipeline: M4 pre-filter + M5 forensic checks.
    
    Args:
        leg: Leg object with home_profile and away_profile
        fav_is_home: Whether favourite is home team
        league_size: Number of teams in league (for tier calculations)
        weights: DecisionWeights for weighted scoring (optional)
        verbose: Print detailed output
    
    Returns:
        OracleVerdict with pipeline results
    """
    if not hasattr(leg, 'check_log'):
        leg.check_log = []
    
    leg.check_log.append(f"M11 Starting Oracle pipeline for {leg.match_id}")
    
    # Extract clean probabilities from leg features
    clean_model_prob = getattr(leg, 'model_prob', 0.50)
    if hasattr(leg, 'features'):
        clean_model_prob = leg.features.get('clean_win_prob', clean_model_prob)
    
    # Step 1: M4 Asymmetric Pre-filter
    pre_filter_passed = False
    pre_filter_score = 0.0
    
    if _M4_AVAILABLE:
        try:
            result = run_asymmetric_prefilter(leg, verbose=verbose)
            pre_filter_passed = result.passed
            pre_filter_score = result.weighted_pass_score
            leg.check_log.append(f"M11 M4 pre-filter: {'PASS' if pre_filter_passed else 'FAIL'} (score={pre_filter_score:.1%})")
        except Exception as e:
            leg.check_log.append(f"M11 M4 pre-filter error: {e}")
            pre_filter_passed = False
            pre_filter_score = 0.0
    else:
        leg.check_log.append("M11 M4 not available — skipping pre-filter")
        pre_filter_passed = True
        pre_filter_score = 0.5
    
    # Step 2: M5 Forensic Checks
    failure_score = 0.0
    failure_details = {}
    forensic_passed = False
    
    if _M5_AVAILABLE:
        try:
            failure_score = run_forensic_checks(leg, league_size)
            failure_details = getattr(leg, 'failure_details', {})
            forensic_passed = failure_score < REJECTION_THRESHOLD
            leg.check_log.append(f"M11 M5 forensic score: {failure_score:.2f} (threshold {REJECTION_THRESHOLD}) → {'PASS' if forensic_passed else 'FAIL'}")
        except Exception as e:
            leg.check_log.append(f"M11 M5 forensic error: {e}")
            failure_score = REJECTION_THRESHOLD + 1.0
            forensic_passed = False
    
    # Step 3: Determine Oracle Status
    model_prob = getattr(leg, 'model_prob', 0.50)
    edge = getattr(leg, 'edge', 0.0)
    clean_edge = model_prob - (1.0 / leg.odds if leg.odds > 0 else 0.0)
    
    if not pre_filter_passed:
        final_status = "REJECTED"
        confidence_tier = "LOW"
        leg.check_log.append("M11 Oracle: REJECTED (pre-filter failed)")
    elif failure_score >= REJECTION_THRESHOLD:
        final_status = "REJECTED"
        confidence_tier = "LOW"
        leg.check_log.append(f"M11 Oracle: REJECTED (failure score {failure_score:.2f} >= {REJECTION_THRESHOLD})")
    elif edge >= 0.08 and model_prob >= 0.57:
        final_status = "APPROVED"
        confidence_tier = "HIGH"
        leg.check_log.append(f"M11 Oracle: APPROVED (edge={edge:+.3f}, prob={model_prob:.1%})")
    elif edge >= 0.04 and model_prob >= 0.54:
        final_status = "CAUTION"
        confidence_tier = "MEDIUM"
        leg.check_log.append(f"M11 Oracle: CAUTION (edge={edge:+.3f}, prob={model_prob:.1%})")
    else:
        final_status = "REJECTED"
        confidence_tier = "LOW"
        leg.check_log.append(f"M11 Oracle: REJECTED (insufficient edge/prob)")
    
    leg.pre_verdict = final_status
    
    return OracleVerdict(
        leg=leg,
        pre_filter_passed=pre_filter_passed,
        pre_filter_score=pre_filter_score,
        failure_score=failure_score,
        forensic_passed=forensic_passed,
        final_status=final_status,
        model_prob=model_prob,
        clean_model_prob=clean_model_prob,
        edge=edge,
        clean_edge=clean_edge,
        confidence_tier=confidence_tier,
        failure_details=failure_details,
    )


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — MASTER AGGREGATION FUNCTION (ENHANCED v6)
# ═══════════════════════════════════════════════════════════════

def run_master_aggregation(
    leg: Leg,
    fav_is_home: bool = True,
    run_ai: bool = True,
    season_results: Optional[Dict[str, List[str]]] = None,
    league_size: int = 20,
    ai_keys: Optional[Dict[str, str]] = None,
    config: Optional[Any] = None,
    api_key: Optional[str] = None,
    use_clean: bool = True,
    standings: Dict[int, Dict] = None,
    verbose: bool = False,
) -> MasterVerdict:
    """
    Full aggregation pipeline orchestrator with weighted decision system.
    
    ENHANCED v6: Now HARD REJECT on conflict detection.
    
    Flow:
    1. Oracle pipeline (M4, M5)
    2. M6 Personnel analysis (if available)
    3. M8 Dual pattern engine (with clean option and conflict detection)
    4. M9 Underdog scanner (with clean option)
    5. M10 Season tally matrix (if data provided)
    6. M26 Match context
    7. M27 H2H deep analysis (with draw boost)
    8. M7 AI layer (if enabled)
    9. M28 Calibration correction
    10. Weighted decision aggregation with HARD REJECT on conflict
    
    Args:
        leg: Leg object with home_profile and away_profile
        fav_is_home: Whether favourite is home team
        run_ai: Whether to run AI intelligence layer
        season_results: Dict with 'home' and 'away' result lists
        league_size: Number of teams in league
        ai_keys: Optional dict with API keys for AI providers
        config: SystemConfig from M18 (for decision weights)
        api_key: API key for personnel data (M6)
        use_clean: If True, use clean (distortion-filtered) probabilities
        standings: Current standings for tier performance check
        verbose: Print detailed output
    
    Returns:
        MasterVerdict with all analyses aggregated
    """
    # Initialise leg features if needed
    if not hasattr(leg, 'features'):
        leg.features = {}
    if not hasattr(leg, 'check_log'):
        leg.check_log = []
    
    leg.check_log.append(f"\n{'='*50}")
    leg.check_log.append(f"M11 Starting master aggregation for {leg.match_id}")
    leg.check_log.append(f"{'='*50}")
    
    # ── Step 1: Oracle Pipeline (M4, M5) ──────────────────────────
    oracle_verdict = run_oracle_pipeline(leg, fav_is_home, league_size, verbose=verbose)
    leg.check_log.append(oracle_verdict.summary)
    
    # ── Step 2: M6 Personnel Forensics ────────────────────────────
    personnel_home = None
    personnel_away = None
    if _M6_AVAILABLE and api_key:
        try:
            personnel_home, personnel_away = analyze_personnel_for_leg(leg, api_key=api_key, verbose=verbose)
            leg.check_log.append(f"M11 M6 personnel: Home={personnel_home.overall_personnel_score:.0f}, Away={personnel_away.overall_personnel_score:.0f}")
        except Exception as e:
            leg.check_log.append(f"M11 M6 error: {e}")
    
    # ── Step 3: M8 Dual Pattern Engine (v6 with HARD REJECT) ──
    dual_verdict = None
    if _M8_AVAILABLE:
        try:
            dual_verdict = run_dual_pattern_engine(
                leg, fav_is_home=fav_is_home, store_in_leg=True, 
                use_clean=use_clean, standings=standings, league_size=league_size,
                verbose=verbose
            )
            if dual_verdict:
                leg.check_log.append(f"M11 M8 dual: risk={dual_verdict.dual_risk_level}, clean_risk={getattr(dual_verdict, 'clean_risk_score', 0.5):.2f}, reliable={getattr(dual_verdict, 'patterns_reliable', True)}")
                leg.check_log.append(f"M11 M8 conflict: detected={getattr(dual_verdict, 'h2h_conflict_detected', False)}, severity={getattr(dual_verdict, 'conflict_severity', 'NONE')}")
                
                # Update oracle with dual risk level
                oracle_verdict.dual_risk_level = dual_verdict.dual_risk_level
                
                # Store pattern reliability in leg features
                leg.features['patterns_reliable'] = getattr(dual_verdict, 'patterns_reliable', True)
                leg.features['distortion_warning'] = getattr(dual_verdict, 'distortion_warning', None)
                leg.features['clean_risk_score'] = getattr(dual_verdict, 'clean_risk_score', 0.5)
                # Store conflict flags (v6 - now HARD REJECT)
                leg.features['h2h_conflict_detected'] = getattr(dual_verdict, 'h2h_conflict_detected', False)
                leg.features['conflict_severity'] = getattr(dual_verdict, 'conflict_severity', 'NONE')
                leg.features['conflict_reasons'] = getattr(dual_verdict, 'conflict_reasons', [])
        except Exception as e:
            leg.check_log.append(f"M11 M8 error: {e}")
    
    # ── Step 4: M9 Underdog Scanner ────────────────────────────────
    underdog_res = None
    if _M9_AVAILABLE:
        try:
            underdog_res = run_underdog_scanner(
                leg, fav_is_home=fav_is_home, mode="OPPORTUNITY",
                use_clean=use_clean, verbose=verbose,
            )
            if underdog_res:
                leg.check_log.append(f"M11 M9 underdog: threat={underdog_res.threat_level}, clean_threat={getattr(underdog_res, 'clean_threat_level', 'NONE')}, reliable={getattr(underdog_res, 'patterns_reliable', True)}")
        except Exception as e:
            leg.check_log.append(f"M11 M9 error: {e}")
    
    # ── Step 5: M10 Season Tally Matrix ───────────────────────────
    season_matrix = None
    if season_results and _M10_AVAILABLE:
        try:
            home_results = season_results.get("home", [])
            away_results = season_results.get("away", [])
            if home_results and away_results:
                season_matrix = run_tally_matrix_analysis(
                    home_results=home_results,
                    away_results=away_results,
                    home_team_id=getattr(leg.home_profile, 'team_id', 'home') if leg.home_profile else 'home',
                    away_team_id=getattr(leg.away_profile, 'team_id', 'away') if leg.away_profile else 'away',
                    fav_odds=leg.odds if hasattr(leg, 'odds') else 0,
                    und_odds=None,
                    fav_is_home=fav_is_home,
                    verbose=verbose,
                )
                if season_matrix:
                    leg.check_log.append(f"M11 M10 matrix: useful={season_matrix.matrix_useful}, clean_useful={getattr(season_matrix, 'clean_matrix_useful', False)}, risk={season_matrix.combined_risk_flag}")
                    leg.features['distortion_impact'] = getattr(season_matrix, 'bilateral', None)
                    if leg.features['distortion_impact']:
                        leg.features['distortion_impact'] = getattr(leg.features['distortion_impact'], 'distortion_impact', 0.0)
        except Exception as e:
            leg.check_log.append(f"M11 M10 error: {e}")
    else:
        leg.check_log.append("M11 M10 skipped: no season_results provided")
    
    # ── Step 6: M26 Match Context ─────────────────────────────────
    match_context = None
    if _M26_AVAILABLE:
        try:
            match_context = context_from_leg(leg)
            if match_context:
                leg.check_log.append(f"M11 M26 context: {match_context.context_label} (importance={match_context.match_importance:.2f})")
                leg.features['season_progress'] = getattr(match_context, 'season_progress', 0.5)
                leg.features['is_dead_rubber'] = match_context.is_dead_rubber
        except Exception as e:
            leg.check_log.append(f"M11 M26 error: {e}")
    
    # ── Step 7: M27 H2H Deep Analysis ─────────────────────────────
    h2h_analysis = None
    if _M27_AVAILABLE and leg.h2h:
        try:
            h2h_fixtures = getattr(leg, 'features', {}).get('h2h_detail_fixtures', [])
            if h2h_fixtures:
                h2h_analysis = run_h2h_deep_analyzer(h2h_fixtures, fav_is_home, include_rtm=True, include_draw_boost=True)
                if h2h_analysis:
                    leg.check_log.append(f"M11 M27 H2H: score={h2h_analysis.h2h_score:.0f}, label={h2h_analysis.h2h_label}, draw_boost={h2h_analysis.draw_boost_factor:.2f}x")
                    leg.features['h2h_draw_boost_factor'] = h2h_analysis.draw_boost_factor
                    leg.features['h2h_draw_rate'] = h2h_analysis.overall.draw_rate
        except Exception as e:
            leg.check_log.append(f"M11 M27 error: {e}")
    
    # ── Step 8: M7 AI Layer ───────────────────────────────────────
    ai_verdict = None
    if run_ai and _M7_AVAILABLE and run_ai_intelligence:
        try:
            is_approved = oracle_verdict.final_status in ("APPROVED", "CAUTION")
            ai_verdict = run_ai_intelligence(
                leg, is_approved=is_approved,
                deepseek_key=ai_keys.get('deepseek') if ai_keys else None,
                claude_key=ai_keys.get('claude') if ai_keys else None,
                gemini_key=ai_keys.get('gemini') if ai_keys else None,
                gpt_key=ai_keys.get('gpt') if ai_keys else None,
            )
            if ai_verdict:
                leg.check_log.append(f"M11 M7 AI: {ai_verdict.final_status} via {ai_verdict.ai_chain_used}")
        except Exception as e:
            leg.check_log.append(f"M11 M7 error: {e}")
    else:
        leg.check_log.append("M11 M7 skipped: AI not enabled or unavailable")
    
    # ── Step 9: M28 Calibration Correction ─────────────────────────
    if _M28_AVAILABLE and oracle_verdict.model_prob > 0:
        try:
            feedback = None
            try:
                import module16 as db
                feedback = db.get_all_feedback()
            except (ImportError, AttributeError):
                pass
            
            if feedback and len(feedback) >= 10:
                cal_report = run_calibration_check(feedback)
                corrected_prob = apply_calibration_correction(oracle_verdict.model_prob, cal_report)
                if corrected_prob != oracle_verdict.model_prob:
                    leg.check_log.append(f"M28 Calibration correction: {oracle_verdict.model_prob:.1%} → {corrected_prob:.1%}")
                    oracle_verdict.model_prob = corrected_prob
                    if leg.odds > 0:
                        oracle_verdict.edge = corrected_prob - (1.0 / leg.odds)
        except Exception as e:
            leg.check_log.append(f"M11 M28 error: {e}")
    
    # ── Step 10: Collect all module outputs ───────────────────────
    leg_data = _collect_leg_data(
        leg, oracle_verdict, dual_verdict, underdog_res, season_matrix,
        ai_verdict, h2h_analysis, match_context, personnel_home, personnel_away
    )
    
    # ── Step 11: Check tier performance (critical override) ───────
    tier_critical, tier_penalty, tier_reason = _check_tier_performance(leg, fav_is_home, standings, league_size)
    if tier_critical:
        leg.check_log.append(f"M11 Tier performance CRITICAL: {tier_reason}")
        leg_data["dual_risk_score"] = max(leg_data["dual_risk_score"], 0.8)  # Increase risk
        if "conflict_reasons" in leg_data:
            leg_data["conflict_reasons"].append(tier_reason)
    
    # ── Step 12: Weighted decision calculation (v6 with HARD REJECT) ──
    weights = DecisionWeights()
    
    (weighted_score, clean_weighted_score, weighted_status, weighted_confidence, 
     contributions, conflict_detected, conflict_severity, conflict_reasons, override_applied) = _calculate_weighted_decision(
        leg_data, weights
    )
    
    # Store on oracle verdict
    oracle_verdict.normalized_scores = leg_data
    oracle_verdict.weighted_contributions = contributions
    oracle_verdict.weighted_total = weighted_score
    oracle_verdict.clean_weighted_total = clean_weighted_score
    
    # Get pattern reliability
    patterns_reliable = leg_data.get("patterns_reliable", True)
    distortion_warning = leg_data.get("distortion_warning", None)
    reliability_score = leg_data.get("pattern_reliability_score", 1.0)
    clean_bounce_back = leg_data.get("h2h_bounce_back", 0.0)
    
    # Get draw boost
    h2h_draw_boost = leg_data.get("h2h_draw_boost_factor", 1.0)
    h2h_draw_rate = leg_data.get("h2h_draw_rate", 0.0)
    
    # NEW v6: Calculate final stake multiplier (0.0 for conflicts)
    stake_multiplier = 1.0
    
    if conflict_detected:
        # HARD REJECT - no stake
        stake_multiplier = 0.0
    else:
        # No conflict - standard stake multiplier
        if tier_critical:
            stake_multiplier *= tier_penalty
    
    stake_multiplier = round(max(0.0, min(1.0, stake_multiplier)), 2)
    
    leg.check_log.append(f"M11 Weighted score: {weighted_score:.3f} (clean: {clean_weighted_score:.3f})")
    leg.check_log.append(f"M11 H2H draw boost: {h2h_draw_boost:.2f}x")
    leg.check_log.append(f"M11 Status: {weighted_status} (confidence: {weighted_confidence})")
    leg.check_log.append(f"M11 Patterns reliable: {patterns_reliable}")
    leg.check_log.append(f"M11 Conflict detected: {conflict_detected} (severity: {conflict_severity})")
    leg.check_log.append(f"M11 Final stake multiplier: {stake_multiplier:.2f}x")
    if conflict_reasons:
        leg.check_log.append(f"M11 Conflict reasons: {', '.join(conflict_reasons[:2])}")
    if distortion_warning:
        leg.check_log.append(f"M11 Distortion warning: {distortion_warning}")
    leg.check_log.append(f"M11 Contributions: {contributions}")
    
    # ── Step 13: Aggregate Final Status ───────────────────────────
    final_status = weighted_status
    final_confidence = weighted_confidence
    risk_flags = []
    decision_notes = []
    
    # Oracle notes
    if oracle_verdict.failure_score > 0:
        risk_flags.append(f"FailureScore: {oracle_verdict.failure_score:.2f}")
    decision_notes.append(f"Oracle: {oracle_verdict.final_status}")
    decision_notes.append(f"Weighted: {weighted_score:.3f} ({weighted_status})")
    if clean_weighted_score != weighted_score:
        decision_notes.append(f"Clean Weighted: {clean_weighted_score:.3f}")
    
    # H2H draw boost note
    if h2h_draw_boost > 1.05:
        decision_notes.append(f"H2H draw boost: {h2h_draw_boost:.2f}x (draw rate: {h2h_draw_rate:.0%})")
    
    # NEW v6: Conflict notes (HARD REJECT)
    if conflict_detected:
        risk_flags.append(f"H2H Conflict: {conflict_severity} - HARD REJECT")
        decision_notes.append(f"🚨 H2H vs current season conflict ({conflict_severity}) - HARD REJECT (NO BET)")
        if conflict_reasons:
            decision_notes.append(f"  Reason: {conflict_reasons[0][:80]}")
    
    # Tier performance note
    if tier_critical:
        risk_flags.append(f"Tier Performance: CRITICAL")
        decision_notes.append(f"⚠ {tier_reason}")
    
    # Pattern reliability flags
    if not patterns_reliable:
        risk_flags.append("Patterns unreliable - distortions detected")
        decision_notes.append(f"⚠ Pattern reliability: LOW (score: {reliability_score:.2f})")
    else:
        decision_notes.append(f"Pattern reliability: GOOD (score: {reliability_score:.2f})")
    
    if distortion_warning:
        risk_flags.append(f"Distortion: {distortion_warning}")
    
    # M6 personnel notes
    if personnel_home and personnel_away:
        if personnel_home.overall_personnel_score < 40:
            risk_flags.append(f"Personnel: {personnel_home.team_name} weak ({personnel_home.overall_personnel_score:.0f})")
        if personnel_away.overall_personnel_score < 40:
            risk_flags.append(f"Personnel: {personnel_away.team_name} weak ({personnel_away.overall_personnel_score:.0f})")
        decision_notes.append(f"Personnel: H={personnel_home.overall_personnel_score:.0f}, A={personnel_away.overall_personnel_score:.0f}")
    
    # M8 dual pattern flags
    if dual_verdict:
        risk_flags.append(f"DualRisk: {dual_verdict.dual_risk_level}")
        if hasattr(dual_verdict, 'clean_risk_score'):
            risk_flags.append(f"CleanDualRisk: {dual_verdict.clean_risk_score:.2f}")
        if hasattr(dual_verdict, 'underdog_threat_level') and dual_verdict.underdog_threat_level not in ("NONE", "LOW"):
            risk_flags.append(f"UnderdogThreat: {dual_verdict.underdog_threat_level}")
        decision_notes.append(f"Dual: {dual_verdict.dual_risk_level}")
        # NEW v6: Conflict already handled above
        if conflict_detected:
            risk_flags.append(f"REJECTED: H2H Conflict")
    
    # M9 underdog flags
    if underdog_res:
        if underdog_res.threat_level not in ("NONE", "LOW"):
            risk_flags.append(f"UnderdogScan: {underdog_res.threat_level}")
        if underdog_res.clean_threat_level not in ("NONE", "LOW"):
            risk_flags.append(f"CleanUnderdogScan: {underdog_res.clean_threat_level}")
        if underdog_res.pattern_count >= 2:
            risk_flags.append(f"PatternCount: {underdog_res.pattern_count}")
        decision_notes.append(f"Underdog: {underdog_res.recommendation}")
    
    # M10 matrix flags
    if season_matrix:
        if hasattr(season_matrix, 'combined_risk_flag') and season_matrix.combined_risk_flag not in ("NONE", "UNCERTAIN"):
            risk_flags.append(f"TrapValue: {season_matrix.combined_risk_flag}")
        if hasattr(season_matrix, 'clean_matrix_useful') and not season_matrix.clean_matrix_useful:
            risk_flags.append("Clean matrix unreliable")
    
    # M26 context flags
    if match_context:
        if match_context.is_dead_rubber:
            risk_flags.append("Dead Rubber - low intensity")
            decision_notes.append("Context: Dead rubber")
        if match_context.is_rivalry:
            risk_flags.append("Rivalry/Derby - high volatility")
            decision_notes.append("Context: Derby match")
        if match_context.is_six_pointer:
            risk_flags.append("Six-pointer - high stakes")
            decision_notes.append("Context: Six-pointer")
    
    # M27 H2H flags
    if h2h_analysis:
        if h2h_analysis.h2h_label in ("UND_EDGE", "UND_DOMINANT"):
            risk_flags.append(f"H2H: {h2h_analysis.h2h_label}")
        if h2h_analysis.draw_boost_factor > 1.10:
            risk_flags.append(f"H2H Draw Rate: {h2h_analysis.overall.draw_rate:.0%} (boost {h2h_analysis.draw_boost_factor:.2f}x)")
        decision_notes.append(f"H2H: {h2h_analysis.h2h_label} ({h2h_analysis.h2h_score:.0f})")
    
    # M29 Drawdown adjustment
    if hasattr(leg, 'drawdown_status') and leg.drawdown_status:
        dd = leg.drawdown_status
        if dd.stake_multiplier < 0.5:
            risk_flags.append(f"Drawdown active: {dd.drawdown_pct:.1%} below peak")
            decision_notes.append(f"Drawdown: {dd.health_label} (mult={dd.stake_multiplier:.0%})")
    
    # AI override for strong signals (only if no conflict)
    if not conflict_detected:
        if ai_verdict and ai_verdict.final_status in ("REJECTED (AI CONSENSUS)", "CAUTION (AI DISAGREEMENT)"):
            if weighted_status == "APPROVED":
                final_status = ai_verdict.final_status
                final_confidence = ai_verdict.final_confidence
                decision_notes.append(f"AI Override: {ai_verdict.final_status}")
        elif underdog_res and underdog_res.threat_level in ("HIGH", "MEDIUM") and weighted_status == "APPROVED":
            final_status = "CAUTION (UNDERDOG THREAT)"
            final_confidence = "MEDIUM"
            decision_notes.append("Underdog threat caused downgrade")
        elif underdog_res and underdog_res.clean_threat_level in ("HIGH", "MEDIUM") and weighted_status == "APPROVED":
            final_status = "CAUTION (CLEAN UNDERDOG THREAT)"
            final_confidence = "MEDIUM"
            decision_notes.append("Clean underdog threat caused downgrade")
    
    # Remove duplicate risk flags
    risk_flags = list(dict.fromkeys(risk_flags))
    
    leg.check_log.append(f"M11 Final: {final_status} ({final_confidence})")
    leg.check_log.append(f"M11 Weighted score: {weighted_score:.3f}")
    leg.check_log.append(f"M11 Clean score: {clean_weighted_score:.3f}")
    leg.check_log.append(f"M11 H2H draw boost: {h2h_draw_boost:.2f}x")
    leg.check_log.append(f"M11 Patterns reliable: {patterns_reliable}")
    leg.check_log.append(f"M11 Conflict detected: {conflict_detected}")
    leg.check_log.append(f"M11 Final stake multiplier: {stake_multiplier:.2f}x")
    leg.check_log.append(f"M11 Risk flags: {risk_flags}")
    leg.check_log.append(f"{'='*50}")
    
    if verbose:
        print(f"\n  M11 Final: {final_status} ({final_confidence}) | Score: {weighted_score:.3f} | Clean: {clean_weighted_score:.3f} | Reliable: {patterns_reliable}")
        if conflict_detected:
            print(f"  🚨 H2H CONFLICT: {conflict_severity} - HARD REJECT (NO BET)")
        if h2h_draw_boost != 1.0:
            print(f"  H2H Draw Boost: {h2h_draw_boost:.2f}x")
    
    return MasterVerdict(
        leg_id=getattr(leg, 'match_id', 'unknown'),
        oracle=oracle_verdict,
        ai=ai_verdict,
        dual_pattern=dual_verdict,
        underdog=underdog_res,
        season_matrix=season_matrix,
        match_context=match_context,
        h2h_analysis=h2h_analysis,
        final_status=final_status,
        final_confidence=final_confidence,
        risk_flags=risk_flags,
        decision_notes=decision_notes,
        weighted_score=weighted_score,
        clean_weighted_score=clean_weighted_score,
        weighted_contributions=contributions,
        patterns_reliable=patterns_reliable,
        reliability_score=reliability_score,
        distortion_warning=distortion_warning,
        clean_bounce_back=clean_bounce_back,
        h2h_draw_boost_factor=h2h_draw_boost,
        h2h_draw_rate=h2h_draw_rate,
        # NEW v5/v6 fields
        conflict_detected=conflict_detected,
        conflict_severity=conflict_severity,
        conflict_reasons=conflict_reasons,
        override_applied=override_applied,
        final_stake_multiplier=stake_multiplier,
    )


# ═══════════════════════════════════════════════════════════════
# SECTION 10 — BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════

def run_batch_aggregation(
    legs: List[Tuple[Leg, bool, Optional[float], Optional[float], Optional[float]]],
    run_ai: bool = True,
    league_size: int = 20,
    use_clean: bool = True,
    standings: Dict[int, Dict] = None,
    verbose: bool = True,
    config: Optional[Any] = None,
    api_key: Optional[str] = None,
) -> List[MasterVerdict]:
    """
    Run master aggregation for multiple legs.
    
    Args:
        legs: List of tuples from Module 1: (Leg, fav_is_home, h_odds, a_odds, d_odds)
        run_ai: Whether to run AI layer
        league_size: Number of teams in league
        use_clean: If True, use clean (distortion-filtered) probabilities
        standings: Current standings for tier performance check
        verbose: Print progress
        config: SystemConfig from M18 (for decision weights)
        api_key: API key for personnel data (M6)
    
    Returns:
        List of MasterVerdict objects
    """
    verdicts = []
    
    for i, (leg, fav_is_home, h_odds, a_odds, d_odds) in enumerate(legs, 1):
        if verbose:
            print(f"  Processing {i}/{len(legs)}: {getattr(leg, 'match_id', 'unknown')}")
        
        # Store odds on leg
        leg.home_odds = h_odds
        leg.away_odds = a_odds
        leg.draw_odds = d_odds
        
        try:
            verdict = run_master_aggregation(
                leg=leg,
                fav_is_home=fav_is_home,
                run_ai=run_ai,
                league_size=league_size,
                use_clean=use_clean,
                standings=standings,
                config=config,
                api_key=api_key,
                verbose=verbose,
            )
            verdicts.append(verdict)
            if verbose:
                print(f"    → {verdict.final_status} (weighted: {verdict.weighted_score:.3f}, reliable: {verdict.patterns_reliable}, conflict: {verdict.conflict_detected})")
        except Exception as e:
            if verbose:
                print(f"    ✘ Error: {e}")
            # Create error verdict
            oracle = OracleVerdict(leg=leg, final_status="REJECTED")
            verdicts.append(MasterVerdict(
                leg_id=getattr(leg, 'match_id', 'unknown'),
                oracle=oracle,
                final_status="REJECTED (PIPELINE ERROR)",
                decision_notes=[f"Error: {e}"],
                weighted_score=0.0,
                clean_weighted_score=0.0,
                patterns_reliable=False,
            ))
    
    if verbose:
        approved = sum(1 for v in verdicts if v.is_approved)
        cautioned = sum(1 for v in verdicts if v.is_caution)
        rejected = len(verdicts) - approved - cautioned
        reliable = sum(1 for v in verdicts if v.patterns_reliable)
        conflicts = sum(1 for v in verdicts if v.conflict_detected)
        draw_boost_active = sum(1 for v in verdicts if v.h2h_draw_boost_factor > 1.05)
        avg_weighted = sum(v.weighted_score for v in verdicts) / len(verdicts) if verdicts else 0
        avg_clean = sum(v.clean_weighted_score for v in verdicts) / len(verdicts) if verdicts else 0
        print(f"\nBatch complete: {approved} approved, {cautioned} cautioned, {rejected} rejected")
        print(f"Conflicts detected (REJECTED): {conflicts}/{len(verdicts)} ({conflicts/len(verdicts):.0%})")
        print(f"Reliable patterns: {reliable}/{len(verdicts)} ({reliable/len(verdicts):.0%})")
        print(f"Draw boost active: {draw_boost_active}/{len(verdicts)} ({draw_boost_active/len(verdicts):.0%})")
        print(f"Average weighted score: {avg_weighted:.3f}")
        print(f"Average clean score: {avg_clean:.3f}")
    
    return verdicts


# ═══════════════════════════════════════════════════════════════
# SECTION 11 — EXPORTS
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Data classes
    "DecisionWeights",
    "OracleVerdict",
    "MasterVerdict",
    # Core functions
    "run_oracle_pipeline",
    "run_master_aggregation",
    "run_batch_aggregation",
]


# ═══════════════════════════════════════════════════════════════
# SECTION 12 — EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from module2 import TeamProfile, TransitionMatrix
    
    print("\n" + "=" * 70)
    print("MODULE 11: MASTER AGGREGATION v6 - HARD REJECT ON CONFLICT")
    print("=" * 70)
    
    # Create mock team profiles
    liverpool = TeamProfile(team_id="4", team_name="Liverpool", is_mature=True)
    liverpool.update_metrics({
        "core.games": 36,
        "core.wins": 17,
        "core.draws": 8,
        "core.losses": 11,
        "core.xg": 60.0,
        "core.xga": 48.0,
        "position": 4,
        "points": 59,
    })
    liverpool.form = {"recent_results": ["L", "W", "L", "D", "W", "L"]}
    
    astonvilla = TeamProfile(team_id="5", team_name="Aston Villa", is_mature=True)
    astonvilla.update_metrics({
        "core.games": 36,
        "core.wins": 17,
        "core.draws": 8,
        "core.losses": 11,
        "core.xg": 50.0,
        "core.xga": 46.0,
        "position": 5,
        "points": 59,
    })
    astonvilla.form = {"recent_results": ["W", "W", "L", "W", "D", "W"]}
    
    # Create H2H record
    from module2 import H2HRecord
    h2h = H2HRecord(games=48, fav_wins=29, draws=11, und_wins=8)
    
    # Create leg with conflict detection data
    leg = Leg(
        match_id="test_liverpool_astonvilla",
        selection="Liverpool",
        odds=2.23,
        home_profile=astonvilla,
        away_profile=liverpool,
        h2h=h2h,
        home_odds=2.90,
        away_odds=2.23,
        draw_odds=3.75,
        model_prob=0.48,
        edge=0.032,
        league="Premier League",
        league_id=39,
        league_tier=1,
        league_country="england",
    )
    leg.check_log = []
    leg.features = {
        "patterns_reliable": True,
        "clean_risk_score": 0.35,
        "distortion_impact": 0.05,
        "season_progress": 0.95,
        # NEW v6: Conflict triggers HARD REJECT
        "h2h_conflict_detected": True,
        "conflict_severity": "HIGH",
        "conflict_reasons": ["Liverpool away vs top tier: 0/5 win rate (0%) conflicts with H2H at Villa Park (unbeaten 12)"],
    }
    
    # Mock standings for tier check
    mock_standings = {
        1: {"position": 1, "points": 79},   # Arsenal
        2: {"position": 2, "points": 74},   # Man City
        3: {"position": 3, "points": 65},   # Man United
        4: {"position": 4, "points": 59},   # Liverpool
        5: {"position": 5, "points": 59},   # Aston Villa
        6: {"position": 6, "points": 55},   # Bournemouth
    }
    
    def mock_detect_favourite():
        return "AWAY"
    leg.detect_favourite = mock_detect_favourite
    
    print("\n📊 ANALYSING: Aston Villa vs Liverpool (Conflict Test - Should REJECT)")
    print("-" * 40)
    print("H2H: Liverpool strong historically")
    print("Current season: Liverpool away vs top tier: 0 wins in 5 games (0%)")
    print("Expected: H2H CONFLICT DETECTED → REJECTED (NO BET)")
    
    # Run aggregation
    verdict = run_master_aggregation(
        leg=leg,
        fav_is_home=False,
        run_ai=False,
        league_size=20,
        use_clean=True,
        standings=mock_standings,
        verbose=True,
    )
    
    print(f"\n{'='*40}")
    print("RESULTS (v6)")
    print(f"{'='*40}")
    print(f"Leg ID: {verdict.leg_id}")
    print(f"Final Status: {verdict.final_status}")
    print(f"Final Confidence: {verdict.final_confidence}")
    print(f"Weighted Score: {verdict.weighted_score:.3f}")
    print(f"Clean Weighted Score: {verdict.clean_weighted_score:.3f}")
    print(f"H2H Draw Boost: {verdict.h2h_draw_boost_factor:.2f}x")
    print(f"H2H Draw Rate: {verdict.h2h_draw_rate:.0%}")
    print(f"Patterns Reliable: {verdict.patterns_reliable}")
    print(f"Conflict Detected: {verdict.conflict_detected}")
    print(f"Conflict Severity: {verdict.conflict_severity}")
    print(f"Final Stake Multiplier: {verdict.final_stake_multiplier:.2f}x")
    
    print(f"\nConflict Reasons:")
    for reason in verdict.conflict_reasons:
        print(f"  • {reason}")
    
    print(f"\nWeighted Contributions:")
    for module, contribution in sorted(verdict.weighted_contributions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {module}: {contribution:.3f}")
    
    print(f"\nRisk Flags:")
    for flag in verdict.risk_flags:
        print(f"  • {flag}")
    
    print(f"\nDecision Notes:")
    for note in verdict.decision_notes:
        print(f"  • {note}")
    
    print("\n" + "=" * 70)
    print("MODULE 11 v6 READY FOR PRODUCTION")
    print("=" * 70)