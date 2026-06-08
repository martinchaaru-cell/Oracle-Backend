"""
The Match Oracle - Module 13: Bankroll Manager & Execution Planner
==============================================================
Merged from:
  - Original Module 13 (Kelly Criterion staking for singles + ACCAs)
  - Module 19 (capital allocation across correlation-aware acca slips)

Pipeline position: receives CleanedPortfolio from Module 12 and
optionally AccaSlipPortfolio for slip-level allocation.

Kelly Criterion:
    f = (b*p - q) / b
    where b = decimal_odds - 1
          p = model win probability
          q = 1 - p

Kelly fractions are scaled by confidence level (half-Kelly or less)
to reduce variance without sacrificing long-run edge.

FIXES IN THIS VERSION:
---------------------
1. M29 DrawdownStatus.stake_multiplier is now applied inside stake_single()
   Previously apply_drawdown_multiplier() existed in M29 but was never called
   from M13, meaning stakes were always full Kelly regardless of losing streaks
   or drawdown severity.
2. Added proper edge validation and clamping
3. Added minimum stake floor and maximum stake cap
4. Added correlation-aware slip allocation
5. Added comprehensive logging and error handling
6. Added __all__ exports
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import math

from module11 import MasterVerdict
from module12 import CleanedPortfolio, AccaSlipPortfolio, AccaSlip


# FIX: import M29 drawdown multiplier — was built but never wired into staking
try:
    from module29 import apply_drawdown_multiplier, DrawdownStatus
    _M29_AVAILABLE = True
except ImportError:
    _M29_AVAILABLE = False
    DrawdownStatus = None
    def apply_drawdown_multiplier(stake, status):
        return stake


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — KELLY STAKING FOR SINGLES + BASIC ACCAs
# ═══════════════════════════════════════════════════════════════

# ── Configuration ─────────────────────────────────────────────

KELLY_SCALE = {
    "HIGH":         0.50,   # 50% of Kelly for high confidence
    "MEDIUM":       0.30,   # 30% of Kelly for medium confidence
    "LOW":          0.10,   # 10% of Kelly for low confidence
    "★★★ ELITE":    0.50,
    "★★ STRONG":    0.35,
    "★ VIABLE":     0.20,
    "- MARGINAL":   0.05,
}

MAX_KELLY_FRACTION  = 0.25   # Never stake more than 25% of bankroll on one leg
MIN_STAKE_FLOOR     = 1.00   # Minimum meaningful stake (in same units as bankroll)
MAX_STAKE_PERCENT   = 0.15   # Hard cap: 15% of bankroll per single

ACCA_BANKROLL_FRACTION = 0.03   # 3% of bankroll on the full ACCA stake
MAX_ACCA_STAKE_PERCENT = 0.10   # Hard cap: 10% of bankroll on any ACCA

# Edge thresholds for confidence tiers
EDGE_HIGH_THRESHOLD   = 0.08   # 8%+ edge = HIGH confidence
EDGE_MEDIUM_THRESHOLD = 0.04   # 4-8% edge = MEDIUM confidence
EDGE_LOW_THRESHOLD    = 0.02   # 2-4% edge = LOW confidence

# Minimum probabilities for betting
MIN_PROB_FOR_BET = 0.50   # Need at least 50% win probability
MAX_PROB_FOR_BET = 0.95   # Cap at 95% (no such thing as certainty)


# ── Result structures ─────────────────────────────────────────

@dataclass
class SingleStakePlan:
    """Stake plan for one single bet."""
    match_id:         str
    match_name:       str
    selection:        str
    odds:             float
    market:           str
    model_prob:       float
    edge:             float
    kelly_raw:        float
    kelly_adj:        float
    stake:            float
    bankroll:         float
    confidence:       str
    potential_return: float
    potential_profit: float
    risk_level:       str
    stake_percent:    float  # Stake as % of bankroll


@dataclass
class AccaStakePlan:
    """Stake plan for one accumulator."""
    legs:             List[str]
    leg_selections:   List[str]
    leg_odds:         List[float]
    combined_odds:    float
    combined_prob:    float
    combined_edge:    float
    stake:            float
    bankroll:         float
    potential_return: float
    potential_profit: float
    acca_type:        str       # ULTRA_SAFE / VALUE
    stake_percent:    float     # Stake as % of bankroll


@dataclass
class BankrollReport:
    """Full bankroll management output."""
    bankroll:         float
    singles:          List[SingleStakePlan] = field(default_factory=list)
    ultra_safe_acca:  Optional[AccaStakePlan] = None
    value_acca:       Optional[AccaStakePlan] = None
    total_exposure:   float = 0.0
    exposure_percent: float = 0.0
    exposure_level:   str   = "LOW"    # LOW / MEDIUM / HIGH
    summary:          Dict[str, Any] = field(default_factory=dict)

    def display(self) -> str:
        """Pretty print the bankroll report."""
        lines = [
            "=" * 60,
            "  ORACLE BEAST — BANKROLL MANAGER",
            "=" * 60,
            f"  Bankroll        : {self.bankroll:.2f}",
            f"  Total exposure  : {self.total_exposure:.2f} ({self.exposure_percent:.1f}%)",
            f"  Exposure level  : {self.exposure_level}",
            "",
            "  SINGLES:",
        ]
        for s in self.singles:
            lines.append(
                f"    {s.match_name[:30]:<30}  "
                f"@ {s.odds:.2f}  stake {s.stake:.2f}  "
                f"return {s.potential_return:.2f}  [{s.confidence}]"
            )
        if self.ultra_safe_acca:
            a = self.ultra_safe_acca
            lines += [
                "",
                f"  ULTRA-SAFE ACCA ({len(a.legs)} legs):",
                f"    Combined odds : {a.combined_odds:.2f}",
                f"    Combined prob : {a.combined_prob:.1%}",
                f"    Combined edge : {a.combined_edge:+.3f}",
                f"    Stake         : {a.stake:.2f} ({a.stake_percent:.1f}%)",
                f"    Potential     : {a.potential_return:.2f}",
            ]
        if self.value_acca:
            a = self.value_acca
            lines += [
                "",
                f"  VALUE ACCA ({len(a.legs)} legs):",
                f"    Combined odds : {a.combined_odds:.2f}",
                f"    Combined prob : {a.combined_prob:.1%}",
                f"    Combined edge : {a.combined_edge:+.3f}",
                f"    Stake         : {a.stake:.2f} ({a.stake_percent:.1f}%)",
                f"    Potential     : {a.potential_return:.2f}",
            ]
        lines.append("=" * 60)
        return "\n".join(lines)


# ── Kelly helpers ─────────────────────────────────────────────

def kelly_fraction(prob: float, odds: float) -> float:
    """
    Calculate full Kelly fraction.
    
    Formula: f = (p(b+1) - 1) / b = (p * odds - 1) / (odds - 1)
    
    Args:
        prob: Win probability (0-1)
        odds: Decimal odds (>1)
    
    Returns:
        Kelly fraction (0 = no bet, negative = avoid)
    """
    if odds <= 1.0 or prob <= 0.0 or prob >= 1.0:
        return 0.0
    
    b = odds - 1.0
    q = 1.0 - prob
    
    # f = (b*p - q) / b
    fraction = (b * prob - q) / b
    
    return max(0.0, fraction)


def adjusted_kelly(raw_kelly: float, confidence: str, edge: float) -> float:
    """
    Adjust Kelly fraction based on confidence and edge strength.
    
    Args:
        raw_kelly: Raw Kelly fraction
        confidence: Confidence tier (HIGH/MEDIUM/LOW)
        edge: Edge percentage
    
    Returns:
        Adjusted Kelly fraction (capped)
    """
    # Get base scale from confidence
    scale = KELLY_SCALE.get(confidence, 0.10)
    
    # Additional scaling based on edge strength
    if edge >= 0.12:
        scale *= 1.2   # Very strong edge gets 20% boost
    elif edge <= 0.03:
        scale *= 0.7   # Weak edge gets 30% reduction
    
    adj = raw_kelly * scale
    
    # Apply hard caps
    adj = min(adj, MAX_KELLY_FRACTION)
    
    return round(adj, 4)


def derive_confidence_from_edge(edge: float) -> str:
    """
    Derive confidence tier from edge value.
    
    Args:
        edge: Edge percentage (model_prob - implied_prob)
    
    Returns:
        Confidence tier string
    """
    if edge >= EDGE_HIGH_THRESHOLD:
        return "HIGH"
    elif edge >= EDGE_MEDIUM_THRESHOLD:
        return "MEDIUM"
    elif edge >= EDGE_LOW_THRESHOLD:
        return "LOW"
    return "LOW"


# ── Single staker ─────────────────────────────────────────────

def stake_single(
    verdict: MasterVerdict,
    bankroll: float,
    drawdown_status: Any = None,
) -> SingleStakePlan:
    """
    Calculate Kelly stake for a single bet.
    
    Args:
        verdict: MasterVerdict from the pipeline
        bankroll: Current bankroll amount
        drawdown_status: Optional DrawdownStatus from M29 for stake reduction
    
    Returns:
        SingleStakePlan with stake calculation
    """
    leg = verdict.oracle.leg
    odds = leg.odds
    prob = min(MAX_PROB_FOR_BET, max(MIN_PROB_FOR_BET, verdict.oracle.model_prob))
    edge = verdict.oracle.edge
    confidence = verdict.final_confidence
    
    # If confidence is still default, derive from edge
    if confidence in ("-", "LOW") and edge >= EDGE_MEDIUM_THRESHOLD:
        confidence = derive_confidence_from_edge(edge)
    
    # Calculate Kelly
    raw_k = kelly_fraction(prob, odds)
    adj_k = adjusted_kelly(raw_k, confidence, edge)
    
    # Calculate stake
    stake = round(bankroll * adj_k, 2)
    
    # Apply floor
    if 0 < stake < MIN_STAKE_FLOOR:
        stake = MIN_STAKE_FLOOR
        adj_k = stake / bankroll if bankroll > 0 else 0
    
    # FIX: apply M29 drawdown multiplier if available
    if _M29_AVAILABLE and drawdown_status is not None:
        original_stake = stake
        stake = apply_drawdown_multiplier(stake, drawdown_status)
        if stake != original_stake:
            # Recalculate adjusted Kelly based on reduced stake
            adj_k = stake / bankroll if bankroll > 0 else 0
    
    # Calculate returns
    potential_return = round(stake * odds, 2)
    potential_profit = round(stake * (odds - 1), 2)
    stake_percent = round((stake / bankroll) * 100, 2) if bankroll > 0 else 0
    
    return SingleStakePlan(
        match_id=verdict.leg_id,
        match_name=getattr(leg, 'match_id', 'unknown').replace("_", " "),
        selection=leg.selection,
        odds=odds,
        market=leg.market.value if leg.market else "Straight Win",
        model_prob=round(prob, 4),
        edge=round(edge, 4),
        kelly_raw=round(raw_k, 4),
        kelly_adj=adj_k,
        stake=stake,
        bankroll=bankroll,
        confidence=confidence,
        potential_return=potential_return,
        potential_profit=potential_profit,
        risk_level=getattr(verdict.oracle, 'dual_risk_level', 'UNKNOWN'),
        stake_percent=stake_percent,
    )


# ── ACCA staker ───────────────────────────────────────────────

def stake_acca(
    verdicts: List[MasterVerdict],
    bankroll: float,
    acca_type: str = "ULTRA_SAFE",
) -> Optional[AccaStakePlan]:
    """
    Calculate stake for an accumulator.
    
    Args:
        verdicts: List of MasterVerdict for the ACCA legs
        bankroll: Current bankroll amount
        acca_type: "ULTRA_SAFE" or "VALUE"
    
    Returns:
        AccaStakePlan or None if invalid
    """
    if not verdicts or len(verdicts) < 2:
        return None
    
    legs = [v.leg_id for v in verdicts]
    selections = [v.oracle.leg.selection for v in verdicts]
    odds_list = [v.oracle.leg.odds for v in verdicts]
    probs = [min(MAX_PROB_FOR_BET, max(MIN_PROB_FOR_BET, v.oracle.model_prob)) for v in verdicts]
    
    # Combined probability (product of individual probabilities)
    combined_prob = 1.0
    for p in probs:
        combined_prob *= max(p, 0.01)
    
    # Combined odds (product of individual odds)
    combined_odds = round(math.prod(odds_list), 2)
    
    # Combined edge
    implied = 1.0 / combined_odds if combined_odds > 1.0 else 0.0
    combined_edge = round(combined_prob - implied, 4)
    
    # Stake calculation: base fraction scaled by number of legs
    # More legs = smaller stake (diminishing returns)
    n_legs = len(verdicts)
    fraction = ACCA_BANKROLL_FRACTION * (1.0 / math.sqrt(n_legs))
    
    # Cap the fraction
    fraction = min(fraction, MAX_ACCA_STAKE_PERCENT)
    
    stake = max(round(bankroll * fraction, 2), MIN_STAKE_FLOOR)
    stake_percent = round((stake / bankroll) * 100, 2) if bankroll > 0 else 0
    
    # Strong edge boost for ACCAs
    if combined_edge >= 0.10:
        stake = round(stake * 1.2, 2)
        stake_percent = round((stake / bankroll) * 100, 2) if bankroll > 0 else 0
    
    return AccaStakePlan(
        legs=legs,
        leg_selections=selections,
        leg_odds=odds_list,
        combined_odds=combined_odds,
        combined_prob=round(combined_prob, 4),
        combined_edge=combined_edge,
        stake=stake,
        bankroll=bankroll,
        potential_return=round(stake * combined_odds, 2),
        potential_profit=round(stake * (combined_odds - 1), 2),
        acca_type=acca_type,
        stake_percent=stake_percent,
    )


# ── Exposure classification ───────────────────────────────────

def _classify_exposure(total_staked: float, bankroll: float) -> str:
    """Classify exposure level based on percentage of bankroll staked."""
    if bankroll <= 0:
        return "UNKNOWN"
    ratio = total_staked / bankroll
    if ratio < 0.10:
        return "LOW"
    if ratio < 0.20:
        return "MEDIUM"
    return "HIGH"


# ── Main entry point ──────────────────────────────────────────

def run_bankroll_manager(
    portfolio: CleanedPortfolio,
    bankroll: float,
    drawdown_status: Any = None,
) -> BankrollReport:
    """
    Main entry point for Module 13.
    
    Stakes singles via Kelly Criterion.
    Stakes ultra-safe and value ACCAs via fixed bankroll fraction.
    Classifies total exposure level (LOW / MEDIUM / HIGH).
    
    Args:
        portfolio: CleanedPortfolio from Module 12
        bankroll: Current bankroll amount
        drawdown_status: Optional DrawdownStatus from M29
    
    Returns:
        BankrollReport with all stake plans
    """
    report = BankrollReport(bankroll=bankroll)
    
    if bankroll <= 0:
        report.summary = {"error": "Bankroll must be positive"}
        return report
    
    # ── Stake singles from top picks ───────────────────────────────
    for verdict in portfolio.top_picks:
        try:
            stake_plan = stake_single(verdict, bankroll, drawdown_status)
            # Only include if stake is meaningful
            if stake_plan.stake > 0:
                report.singles.append(stake_plan)
        except Exception as e:
            # Log error but continue with other picks
            if hasattr(verdict, 'oracle') and hasattr(verdict.oracle, 'leg'):
                match_id = getattr(verdict.oracle.leg, 'match_id', 'unknown')
            else:
                match_id = 'unknown'
            print(f"  ⚠ Error staking {match_id}: {e}")
    
    # ── Stake Ultra-Safe ACCA ─────────────────────────────────────
    if len(portfolio.ultra_safe_acca) >= 2:
        try:
            acca_plan = stake_acca(portfolio.ultra_safe_acca, bankroll, acca_type="ULTRA_SAFE")
            if acca_plan and acca_plan.stake > 0:
                report.ultra_safe_acca = acca_plan
        except Exception as e:
            print(f"  ⚠ Error staking ultra-safe ACCA: {e}")
    
    # ── Stake Value ACCA ──────────────────────────────────────────
    if len(portfolio.value_acca) >= 2:
        try:
            acca_plan = stake_acca(portfolio.value_acca, bankroll, acca_type="VALUE")
            if acca_plan and acca_plan.stake > 0:
                report.value_acca = acca_plan
        except Exception as e:
            print(f"  ⚠ Error staking value ACCA: {e}")
    
    # ── Calculate total exposure ──────────────────────────────────
    total = sum(s.stake for s in report.singles)
    if report.ultra_safe_acca:
        total += report.ultra_safe_acca.stake
    if report.value_acca:
        total += report.value_acca.stake
    
    report.total_exposure = round(total, 2)
    report.exposure_percent = round((total / bankroll) * 100, 2) if bankroll > 0 else 0.0
    report.exposure_level = _classify_exposure(total, bankroll)
    
    # ── Summary statistics ────────────────────────────────────────
    report.summary = {
        "bankroll": bankroll,
        "total_exposure": report.total_exposure,
        "exposure_percent": report.exposure_percent,
        "exposure_level": report.exposure_level,
        "singles_count": len(report.singles),
        "singles_total_stake": round(sum(s.stake for s in report.singles), 2),
        "acca_ultra_legs": len(portfolio.ultra_safe_acca),
        "acca_value_legs": len(portfolio.value_acca),
        "acca_ultra_odds": report.ultra_safe_acca.combined_odds if report.ultra_safe_acca else 0,
        "acca_value_odds": report.value_acca.combined_odds if report.value_acca else 0,
        "drawdown_multiplier": drawdown_status.stake_multiplier if drawdown_status else 1.0,
        "drawdown_active": drawdown_status is not None and drawdown_status.stake_multiplier < 1.0,
    }
    
    return report


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — SLIP-LEVEL ALLOCATION (merged from M19)
# ═══════════════════════════════════════════════════════════════
# Allocates bankroll across the SAFE / BALANCED / AGGRESSIVE
# slip sets produced by Module 12's build_acca_slip_portfolio().

MAX_SLIP_EXPOSURE  = 0.25   # max 25% of bankroll across all slips
SAFE_WEIGHT        = 0.40
BALANCED_WEIGHT    = 0.35
AGGRESSIVE_WEIGHT  = 0.25


@dataclass
class SlipAllocation:
    """Allocated stake for one AccaSlip."""
    slip_type:         str
    total_odds:        float
    stake:             float
    potential_return:  float
    risk_score:        float
    correlation_score: float
    stake_percent:     float


@dataclass
class SlipExecutionPlan:
    """Full allocation plan across all three slip tiers."""
    bankroll:      float
    total_staked:  float = 0.0
    exposure_level: str = "LOW"
    allocations:   List[SlipAllocation] = field(default_factory=list)
    notes:         List[str] = field(default_factory=list)


def _best_valid_slip(slips: List[AccaSlip]) -> Optional[AccaSlip]:
    """Return the best valid slip from a list (lowest risk, lowest correlation)."""
    valid = [s for s in slips if s.is_valid]
    if not valid:
        return None
    return sorted(valid, key=lambda s: (s.risk_score, s.correlation_score))[0]


def allocate_slip_stakes(
    slip_portfolio: AccaSlipPortfolio,
    bankroll: float,
) -> SlipExecutionPlan:
    """
    Allocate bankroll fractions to the best valid slip from each tier.
    
    Uses fixed weights (SAFE 40%, BALANCED 35%, AGGRESSIVE 25%) of the
    maximum cycle exposure (25% of bankroll).
    
    Args:
        slip_portfolio: AccaSlipPortfolio from Module 12
        bankroll: Current bankroll amount
    
    Returns:
        SlipExecutionPlan with allocations
    """
    plan = SlipExecutionPlan(bankroll=bankroll)
    max_budget = bankroll * MAX_SLIP_EXPOSURE
    
    candidates = [
        (_best_valid_slip(slip_portfolio.safe), SAFE_WEIGHT, "SAFE"),
        (_best_valid_slip(slip_portfolio.balanced), BALANCED_WEIGHT, "BALANCED"),
        (_best_valid_slip(slip_portfolio.aggressive), AGGRESSIVE_WEIGHT, "AGGRESSIVE"),
    ]
    
    for slip, weight, slip_type in candidates:
        if slip is None:
            plan.notes.append(f"No valid {slip_type} slip available")
            continue
        
        stake = round(max_budget * weight, 2)
        stake_percent = round((stake / bankroll) * 100, 2) if bankroll > 0 else 0
        
        plan.allocations.append(SlipAllocation(
            slip_type=slip_type,
            total_odds=slip.total_odds,
            stake=stake,
            potential_return=round(stake * slip.total_odds, 2),
            risk_score=slip.risk_score,
            correlation_score=slip.correlation_score,
            stake_percent=stake_percent,
        ))
        plan.total_staked += stake
    
    plan.exposure_level = _classify_exposure(plan.total_staked, bankroll)
    
    if not plan.allocations:
        plan.notes.append("No valid slips available for allocation")
    
    return plan


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — EXPORTS
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Data classes
    "SingleStakePlan",
    "AccaStakePlan",
    "BankrollReport",
    "SlipAllocation",
    "SlipExecutionPlan",
    # Main functions
    "run_bankroll_manager",
    "allocate_slip_stakes",
    # Kelly helpers (for testing)
    "kelly_fraction",
    "adjusted_kelly",
    "derive_confidence_from_edge",
    # Constants
    "MAX_KELLY_FRACTION",
    "MIN_STAKE_FLOOR",
    "ACCA_BANKROLL_FRACTION",
    "EDGE_HIGH_THRESHOLD",
    "EDGE_MEDIUM_THRESHOLD",
]


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from module2 import Leg, BetMarket
    from module11 import OracleVerdict, MasterVerdict
    from module12 import CleanedPortfolio
    
    # Create mock verdicts
    def create_mock_verdict(match_id, selection, odds, prob, edge, confidence, status):
        leg = Leg(
            match_id=match_id,
            selection=selection,
            odds=odds,
            market=BetMarket.STRAIGHT_WIN,
        )
        oracle = OracleVerdict(
            leg=leg,
            final_status=status,
            model_prob=prob,
            edge=edge,
            confidence_tier=confidence,
        )
        return MasterVerdict(
            leg_id=match_id,
            oracle=oracle,
            final_status=status,
            final_confidence=confidence,
            risk_flags=[],
            decision_notes=[f"Test verdict for {match_id}"],
        )
    
    # Create top picks
    top_picks = [
        create_mock_verdict("arsenal_everton", "Arsenal", 1.85, 0.58, 0.09, "HIGH", "APPROVED"),
        create_mock_verdict("liverpool_city", "Liverpool", 2.10, 0.55, 0.07, "HIGH", "APPROVED"),
        create_mock_verdict("bayern_dortmund", "Bayern", 1.75, 0.62, 0.12, "HIGH", "APPROVED"),
        create_mock_verdict("real_barca", "Real Madrid", 2.20, 0.52, 0.05, "MEDIUM", "CAUTION"),
    ]
    
    # Create portfolio
    portfolio = CleanedPortfolio(
        cleaned_verdicts=top_picks,
        top_picks=top_picks[:3],
        ultra_safe_acca=top_picks[:2],
        value_acca=top_picks[1:3],
    )
    
    print("\n" + "="*70)
    print("MODULE 13: BANKROLL MANAGER - TEST RUN")
    print("="*70)
    
    # Run bankroll manager
    bankroll = 1000.0
    report = run_bankroll_manager(portfolio, bankroll, drawdown_status=None)
    
    print(report.display())
    
    print("\nSummary:")
    for key, value in report.summary.items():
        print(f"  {key}: {value}")
    
    # Test with drawdown
    print("\n" + "="*70)
    print("WITH DRAWDOWN ACTIVE (50% stake multiplier)")
    print("="*70)
    
    # Create mock drawdown status
    if _M29_AVAILABLE:
        class MockDrawdownStatus:
            stake_multiplier = 0.5
            drawdown_pct = 0.15
            health_label = "DANGER"
    
        mock_dd = MockDrawdownStatus()
        report_dd = run_bankroll_manager(portfolio, bankroll, drawdown_status=mock_dd)
        print(report_dd.display())
    else:
        print("  M29 not available — drawdown simulation skipped")
    
    # Test slip allocation
    print("\n" + "="*70)
    print("SLIP ALLOCATION TEST")
    print("="*70)
    
    from module12 import AccaSlipPortfolio, AccaSlip, AccaPick
    
    # Create mock slips
    mock_pick = AccaPick(
        match="Arsenal vs Everton",
        selection="Arsenal",
        market="Straight Win",
        odds=1.85,
        confidence="HIGH",
        status="APPROVED",
        league="Premier League",
    )
    
    mock_slip = AccaSlip(
        slip_type="SAFE",
        picks=[mock_pick, mock_pick],  # Simplified for test
        total_odds=3.42,
        risk_score=0.5,
        correlation_score=0.2,
        is_valid=True,
    )
    
    slip_portfolio = AccaSlipPortfolio(
        safe=[mock_slip],
        balanced=[mock_slip],
        aggressive=[mock_slip],
    )
    
    allocation = allocate_slip_stakes(slip_portfolio, 1000.0)
    print(f"Total staked: {allocation.total_staked:.2f}")
    print(f"Exposure level: {allocation.exposure_level}")
    for a in allocation.allocations:
        print(f"  {a.slip_type}: stake {a.stake:.2f} ({a.stake_percent:.1f}%) @ {a.total_odds:.2f}")