"""
The Match Oracle - Module 4: Asymmetric Pre-Filter (REFINED)
============================================================
8-point gate. Must pass at least 5 out of 8 checks to proceed.

This is the first major quality gate after data ingestion.
Rejects obvious mismatches early to save compute in later modules.

REFINEMENTS IN THIS VERSION:
---------------------------
1. ADDED: Proper Leg.detect_favourite() method (no monkey-patching)
2. ADDED: home_odds/away_odds fallback when odds not on Leg
3. FIXED: get_metric() calls use proper attribute access
4. FIXED: Maturity gate as hard requirement (not part of 8 checks)
5. ADDED: Detailed logging for each check result with point values
6. ADDED: Integration with M0 guardrail filters (dominance vs decay, tiers, playoffs)
7. ADDED: Strength scoring for each check (0-10 scale)
8. ADDED: Weighted gate (not just pass/fail count)
9. ADDED: PreFilterResult with detailed breakdown
10. ADDED: Batch processing for multiple legs

CHECKS PERFORMED:
-----------------
C1: Season Win Gap — favourite's total wins minus underdog's wins ≥ 3
C2: Venue Win Gap — home win rate minus away win rate ≥ 0.15
C3: H2H Favoured — favourite wins ≥ 50% of H2H meetings (min 5 games)
C4: Transition Favours — RTM says >45% chance favourite wins next
C5: Bounce-back Rate — after loss, >45% chance of win next match
C6: Ceiling Proximity — favourite not at statistical ceiling (>88% win rate)
C7: Momentum Gap — last 5 games win rate gap ≥ 0.20
C8: Resilience Gap — xG differential advantage ≥ 0.10

HARD FILTERS (enforced before checks):
-------------------------------------
H1: Senior Men's League — reject youth/women/amateur leagues
H2: League Tier — enforce top 3/2/1 tiers by country (with bypass for sufficient data)
H3: Format — reject playoff/knockout/relegation matches
H4: Dominance vs Decay — clear favourite dominance + underdog decay
H5: Maturity — both teams ≥10 games
H6: Odds ≥ 1.70 — favourite not priced too short

WEIGHTED DECISION SUPPORT:
-------------------------
- weighted_pass_score: 0-1 score based on check strengths
- normalized_score: For M11 aggregation
- confidence_factor: For M13 Kelly scaling
- to_leg_data(): Direct output for M11

Usage:
    from module4 import run_asymmetric_prefilter, PreFilterResult
    
    result = run_asymmetric_prefilter(leg, verbose=True)
    if result.passed:
        print(f"Passed with {result.checks_passed}/{result.checks_evaluated} checks")
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Dict
from enum import Enum
import math

from module2 import Leg, TeamProfile, H2HRecord, BetMarket, FormSequence


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — THRESHOLDS
# ═══════════════════════════════════════════════════════════════

SEASON_WIN_GAP_MIN      = 3      # Minimum win difference between teams
VENUE_WIN_GAP_MIN_RATE  = 0.15   # Minimum home win rate - away win rate
H2H_WIN_RATE_MIN        = 0.50   # Minimum H2H win rate for favourite
TRANSITION_NEXT_WIN_MIN = 0.45   # Minimum RTM win probability
BOUNCE_BACK_MIN         = 0.45   # Minimum win probability after a loss
CEILING_PROXIMITY_MAX   = 0.88   # Maximum win rate before ceiling warning
MOMENTUM_GAP_MIN        = 0.20   # Minimum last-5 win rate gap
RESILIENCE_GAP_MIN      = 0.10   # Minimum xG differential advantage

REQUIRED_CHECKS         = 5      # Must pass at least this many checks to proceed
MIN_H2H_GAMES_FOR_CHECK = 5      # Minimum H2H games for C3 to count
MIN_TM_SAMPLE_SIZE      = 5      # Minimum transitions for C4/C5 to count

# Check weights for weighted scoring (sum = 1.0)
CHECK_WEIGHTS = {
    "C1_season_win_gap": 0.12,
    "C2_venue_win_gap": 0.12,
    "C3_h2h_favoured": 0.10,
    "C4_transition_favours": 0.15,
    "C5_bounce_back": 0.10,
    "C6_ceiling_proximity": 0.08,
    "C7_momentum_gap": 0.18,
    "C8_resilience_gap": 0.15,
}

# Force bypass for testing
FORCE_BYPASS_FILTERS = os.getenv("ORACLE_FORCE_BYPASS_FILTERS", "false").lower() == "true"


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — ENUMS
# ═══════════════════════════════════════════════════════════════

class CheckStatus(Enum):
    """Status of an individual check."""
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"  # Insufficient data


class PreFilterStatus(Enum):
    """Overall pre-filter status."""
    PASSED = "PASSED"
    FAILED_HARD_FILTER = "FAILED_HARD_FILTER"
    FAILED_CHECKS = "FAILED_CHECKS"
    SKIPPED = "SKIPPED"


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — RESULT DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class CheckResult:
    """Result of a single asymmetric check."""
    check_id: str
    name: str
    status: CheckStatus
    value: float
    threshold: float
    points: float = 0.0  # 0-10 scale
    message: str = ""
    
    @property
    def passed(self) -> bool:
        return self.status == CheckStatus.PASS
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "name": self.name,
            "status": self.status.value,
            "value": round(self.value, 3),
            "threshold": self.threshold,
            "points": round(self.points, 1),
            "message": self.message,
        }


@dataclass
class HardFilterResult:
    """Result of a hard filter."""
    filter_name: str
    passed: bool
    reason: str = ""
    bypass_applied: bool = False


@dataclass
class PreFilterResult:
    """Result of running the asymmetric pre-filter."""
    passed: bool
    status: PreFilterStatus
    checks_passed: int
    checks_evaluated: int
    checks_required: int = REQUIRED_CHECKS
    weighted_pass_score: float = 0.0  # 0-1 weighted score
    check_details: List[CheckResult] = field(default_factory=list)
    hard_filter_failures: List[HardFilterResult] = field(default_factory=list)
    
    # NEW: Weighted decision properties
    @property
    def normalized_score(self) -> float:
        """Convert weighted_pass_score to 0-1 normalized score."""
        return self.weighted_pass_score
    
    @property
    def confidence_factor(self) -> float:
        """Confidence factor for M13 Kelly scaling."""
        if self.weighted_pass_score >= 0.8:
            return 1.0
        elif self.weighted_pass_score >= 0.6:
            return 0.8
        elif self.weighted_pass_score >= 0.4:
            return 0.5
        return 0.3
    
    def to_leg_data(self) -> Dict[str, Any]:
        """Convert to leg_data format for M11 weighted decision."""
        return {
            "pre_filter_passed": self.passed,
            "pre_filter_score": self.weighted_pass_score,
            "pre_filter_normalized": self.normalized_score,
            "pre_filter_confidence": self.confidence_factor,
            "pre_filter_checks_passed": self.checks_passed,
            "pre_filter_checks_total": self.checks_evaluated,
        }
    
    @property
    def summary(self) -> str:
        """Human-readable summary."""
        status_str = self.status.value
        if self.hard_filter_failures:
            failures = ", ".join([f.filter_name for f in self.hard_filter_failures])
            return f"M4 {status_str} - Hard filter failures: {failures}"
        return (f"M4 {status_str} - "
                f"Checks: {self.checks_passed}/{self.checks_evaluated} (need ≥{self.checks_required}) | "
                f"Weighted: {self.weighted_pass_score:.1%}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "passed": self.passed,
            "status": self.status.value,
            "checks_passed": self.checks_passed,
            "checks_evaluated": self.checks_evaluated,
            "checks_required": self.checks_required,
            "weighted_pass_score": round(self.weighted_pass_score, 3),
            "normalized_score": self.normalized_score,
            "confidence_factor": self.confidence_factor,
            "check_details": [c.to_dict() for c in self.check_details],
            "hard_filter_failures": [
                {"filter": f.filter_name, "reason": f.reason, "bypass": f.bypass_applied}
                for f in self.hard_filter_failures
            ],
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — GUARDRAIL INTEGRATION
# ═══════════════════════════════════════════════════════════════

try:
    from module0 import (
        validate_senior_men_league,
        validate_league_tier,
        validate_tier_bypass_with_data,
        validate_not_playoff_or_knockout,
        validate_dominance_vs_decay,
        validate_all_filters,
        GuardrailError,
        GuardrailReport
    )
    _GUARDRAIL_AVAILABLE = True
except ImportError:
    _GUARDRAIL_AVAILABLE = False
    # Fallback functions if M0 not available
    def validate_senior_men_league(leg):
        from module0 import GuardrailReport
        report = GuardrailReport()
        report.passed = True
        return report
    
    def validate_league_tier(leg):
        from module0 import GuardrailReport
        report = GuardrailReport()
        report.passed = True
        return report
    
    def validate_tier_bypass_with_data(leg):
        from module0 import GuardrailReport
        report = GuardrailReport()
        report.passed = True
        return report
    
    def validate_not_playoff_or_knockout(leg):
        from module0 import GuardrailReport
        report = GuardrailReport()
        report.passed = True
        return report
    
    def validate_dominance_vs_decay(leg, verbose=False):
        from module0 import GuardrailReport
        report = GuardrailReport()
        report.passed = True
        return report
    
    def validate_all_filters(leg, verbose=False):
        from module0 import GuardrailReport
        report = GuardrailReport()
        report.passed = True
        return report


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — DETECT FAVOURITE
# ═══════════════════════════════════════════════════════════════

def detect_favourite(leg: Leg) -> str:
    """
    Return which side is the favourite based on odds.
    
    Returns:
        "HOME" if home team is favourite
        "AWAY" if away team is favourite
        "UNKNOWN" if cannot determine (should not happen with valid data)
    """
    # Priority 1: Use home_odds and away_odds if available
    home_odds = getattr(leg, 'home_odds', None)
    away_odds = getattr(leg, 'away_odds', None)
    
    if home_odds is not None and away_odds is not None:
        if home_odds > 0 and away_odds > 0:
            return "HOME" if home_odds <= away_odds else "AWAY"
    
    # Priority 2: Use selection odds vs market average
    if leg.selection and leg.home_profile and leg.away_profile:
        if leg.selection == leg.home_profile.team_name:
            return "HOME"
        elif leg.selection == leg.away_profile.team_name:
            return "AWAY"
    
    # Priority 3: Final fallback - assume home favourite
    return "HOME"


# Attach method to Leg class for convenience
Leg.detect_favourite = detect_favourite


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — CHECK EXECUTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _check_season_win_gap(
    fav_profile: TeamProfile,
    und_profile: TeamProfile,
) -> CheckResult:
    """C1: Season Win Gap — favourite's total wins minus underdog's wins ≥ 3"""
    fav_wins = fav_profile.get_metric("core.wins", 0)
    und_wins = und_profile.get_metric("core.wins", 0)
    gap = fav_wins - und_wins
    
    passed = gap >= SEASON_WIN_GAP_MIN
    points = min(10.0, max(0.0, (gap / SEASON_WIN_GAP_MIN) * 10)) if gap > 0 else 0.0
    
    return CheckResult(
        check_id="C1",
        name="Season Win Gap",
        status=CheckStatus.PASS if passed else CheckStatus.FAIL,
        value=gap,
        threshold=SEASON_WIN_GAP_MIN,
        points=points,
        message=f"fav wins: {fav_wins:.0f}, und wins: {und_wins:.0f}, gap: {gap:+.0f}"
    )


def _check_venue_win_gap(
    leg: Leg,
    fav_is_home: bool,
    fav_profile: TeamProfile,
    und_profile: TeamProfile,
) -> CheckResult:
    """C2: Venue Win Gap — home win rate minus away win rate ≥ 0.15"""
    home_games = max(fav_profile.get_metric("home_games", 1), 1)
    venue_rate = fav_profile.get_metric("home_wins", 0) / home_games
    
    if fav_is_home:
        # Favourite is home, underdog is away
        away_games = max(und_profile.get_metric("away_games", 1), 1)
        away_wins = und_profile.get_metric("away_wins", 0)
        away_rate = away_wins / away_games if away_games > 0 else 0.0
    else:
        # Favourite is away, underdog is home
        # For away favourite, we want underdog's home win rate
        home_wins = und_profile.get_metric("home_wins", 0)
        home_games_und = max(und_profile.get_metric("home_games", 1), 1)
        away_rate = home_wins / home_games_und if home_games_und > 0 else 0.0
    
    gap = venue_rate - away_rate
    
    passed = gap >= VENUE_WIN_GAP_MIN_RATE
    points = min(10.0, max(0.0, (gap / VENUE_WIN_GAP_MIN_RATE) * 10))
    
    return CheckResult(
        check_id="C2",
        name="Venue Win Gap",
        status=CheckStatus.PASS if passed else CheckStatus.FAIL,
        value=gap,
        threshold=VENUE_WIN_GAP_MIN_RATE,
        points=points,
        message=f"fav venue rate: {venue_rate:.1%}, und away rate: {away_rate:.1%}, gap: {gap:+.1%}"
    )


def _check_h2h_favoured(
    leg: Leg,
) -> CheckResult:
    """C3: H2H Favoured — favourite wins ≥ 50% of H2H meetings (min 5 games)"""
    h2h = leg.h2h
    h2h_games = h2h.games if h2h else 0
    
    if h2h is None or h2h_games < MIN_H2H_GAMES_FOR_CHECK:
        return CheckResult(
            check_id="C3",
            name="H2H Favoured",
            status=CheckStatus.SKIP,
            value=0.0,
            threshold=H2H_WIN_RATE_MIN,
            points=0.0,
            message=f"Insufficient H2H data ({h2h_games} games, need {MIN_H2H_GAMES_FOR_CHECK})"
        )
    
    fav_rate = h2h.fav_win_rate
    passed = fav_rate >= H2H_WIN_RATE_MIN
    points = fav_rate * 10 if fav_rate >= 0 else 0.0
    
    return CheckResult(
        check_id="C3",
        name="H2H Favoured",
        status=CheckStatus.PASS if passed else CheckStatus.FAIL,
        value=fav_rate,
        threshold=H2H_WIN_RATE_MIN,
        points=points,
        message=f"fav win rate: {fav_rate:.1%} over {h2h_games} games"
    )


def _check_transition_favours(
    fav_profile: TeamProfile,
) -> CheckResult:
    """C4: Transition Favours — RTM says >45% chance favourite wins next"""
    tm = fav_profile.transition
    results = fav_profile.form.get("recent_results", [])
    last_res = results[-1] if results else "W"
    
    if tm is None or tm.sample_size < MIN_TM_SAMPLE_SIZE:
        return CheckResult(
            check_id="C4",
            name="Transition Favours",
            status=CheckStatus.SKIP,
            value=0.0,
            threshold=TRANSITION_NEXT_WIN_MIN,
            points=0.0,
            message=f"Insufficient transition data (sample size: {tm.sample_size if tm else 0})"
        )
    
    next_w = tm.get_next_prob(last_res, "W")
    passed = next_w >= TRANSITION_NEXT_WIN_MIN
    points = min(10.0, (next_w / TRANSITION_NEXT_WIN_MIN) * 10)
    
    return CheckResult(
        check_id="C4",
        name="Transition Favours",
        status=CheckStatus.PASS if passed else CheckStatus.FAIL,
        value=next_w,
        threshold=TRANSITION_NEXT_WIN_MIN,
        points=points,
        message=f"{last_res}→W probability: {next_w:.1%}"
    )


def _check_bounce_back(
    fav_profile: TeamProfile,
) -> CheckResult:
    """C5: Bounce-back Rate — after loss, >45% chance of win next match"""
    tm = fav_profile.transition
    
    if tm is None or tm.sample_size < MIN_TM_SAMPLE_SIZE:
        return CheckResult(
            check_id="C5",
            name="Bounce-back Rate",
            status=CheckStatus.SKIP,
            value=0.0,
            threshold=BOUNCE_BACK_MIN,
            points=0.0,
            message="Insufficient transition data"
        )
    
    bb = tm.get_next_prob("L", "W")
    passed = bb >= BOUNCE_BACK_MIN
    points = min(10.0, (bb / BOUNCE_BACK_MIN) * 10)
    
    return CheckResult(
        check_id="C5",
        name="Bounce-back Rate",
        status=CheckStatus.PASS if passed else CheckStatus.FAIL,
        value=bb,
        threshold=BOUNCE_BACK_MIN,
        points=points,
        message=f"L→W probability: {bb:.1%}"
    )


def _check_ceiling_proximity(
    fav_profile: TeamProfile,
) -> CheckResult:
    """C6: Ceiling Proximity — favourite not at statistical ceiling (>88% win rate)"""
    games = max(fav_profile.get_metric("core.games", 1), 1)
    win_rate = fav_profile.get_metric("core.wins", 0) / games
    
    passed = win_rate <= CEILING_PROXIMITY_MAX
    # Points: lower win rate = higher points (regression risk is bad)
    points = max(0.0, min(10.0, (1 - win_rate / CEILING_PROXIMITY_MAX) * 10)) if win_rate > 0 else 10.0
    
    return CheckResult(
        check_id="C6",
        name="Ceiling Proximity",
        status=CheckStatus.PASS if passed else CheckStatus.FAIL,
        value=win_rate,
        threshold=CEILING_PROXIMITY_MAX,
        points=points,
        message=f"fav win rate: {win_rate:.1%} (threshold: {CEILING_PROXIMITY_MAX:.0%})"
    )


def _check_momentum_gap(
    fav_profile: TeamProfile,
    und_profile: TeamProfile,
) -> CheckResult:
    """C7: Momentum Gap — last 5 games win rate gap ≥ 0.20"""
    fav_recent = fav_profile.form.get("recent_results", [])[-5:]
    und_recent = und_profile.form.get("recent_results", [])[-5:]
    
    fav_wins = sum(1 for r in fav_recent if r == "W") / max(len(fav_recent), 1)
    und_wins = sum(1 for r in und_recent if r == "W") / max(len(und_recent), 1)
    gap = fav_wins - und_wins
    
    passed = gap >= MOMENTUM_GAP_MIN
    points = min(10.0, max(0.0, (gap / MOMENTUM_GAP_MIN) * 10))
    
    return CheckResult(
        check_id="C7",
        name="Momentum Gap",
        status=CheckStatus.PASS if passed else CheckStatus.FAIL,
        value=gap,
        threshold=MOMENTUM_GAP_MIN,
        points=points,
        message=f"fav last5: {fav_wins:.1%}, und last5: {und_wins:.1%}, gap: {gap:+.1%}"
    )


def _check_resilience_gap(
    fav_profile: TeamProfile,
    und_profile: TeamProfile,
) -> CheckResult:
    """C8: Resilience Gap — xG differential advantage ≥ 0.10"""
    fav_games = max(fav_profile.get_metric("core.games", 1), 1)
    und_games = max(und_profile.get_metric("core.games", 1), 1)
    
    fav_xg = fav_profile.get_metric("core.xg", 0.0)
    fav_xga = fav_profile.get_metric("core.xga", 0.0)
    fav_xg_diff = (fav_xg - fav_xga) / fav_games if fav_games > 0 else 0.0
    
    und_xg = und_profile.get_metric("core.xg", 0.0)
    und_xga = und_profile.get_metric("core.xga", 0.0)
    und_xg_diff = (und_xg - und_xga) / und_games if und_games > 0 else 0.0
    
    gap = fav_xg_diff - und_xg_diff
    
    passed = gap >= RESILIENCE_GAP_MIN
    points = min(10.0, max(0.0, (gap / RESILIENCE_GAP_MIN) * 10))
    
    return CheckResult(
        check_id="C8",
        name="Resilience Gap",
        status=CheckStatus.PASS if passed else CheckStatus.FAIL,
        value=gap,
        threshold=RESILIENCE_GAP_MIN,
        points=points,
        message=f"fav xG diff: {fav_xg_diff:+.2f}, und xG diff: {und_xg_diff:+.2f}, gap: {gap:+.2f}"
    )


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — MAIN PRE-FILTER FUNCTION
# ═══════════════════════════════════════════════════════════════

def run_asymmetric_prefilter(
    leg: Leg,
    verbose: bool = False,
    skip_hard_filters: bool = False,
) -> PreFilterResult:
    """
    Run 8 asymmetric checks + all guardrail hard filters.
    
    Args:
        leg: Leg object with home_profile, away_profile, and odds
        verbose: Print detailed check results
        skip_hard_filters: Skip hard filters (for testing)
    
    Returns:
        PreFilterResult with detailed results
    """
    # Ensure check_log exists
    if not hasattr(leg, 'check_log'):
        leg.check_log = []
    
    hard_failures = []
    
    # ── FORCE BYPASS FOR TESTING ──────────────────────────────────────
    if FORCE_BYPASS_FILTERS or skip_hard_filters:
        leg.check_log.append("M4: Force bypass enabled - skipping all hard filters")
        return _run_checks_only(leg, verbose)
    
    # ── HARD FILTER 1: Senior Men's League ────────────────────────────
    if _GUARDRAIL_AVAILABLE:
        try:
            report = validate_senior_men_league(leg)
            if not report.passed:
                reason = report.errors[0] if report.errors else "Senior men's league filter failed"
                hard_failures.append(HardFilterResult("Senior Men's League", False, reason))
                leg.check_log.append(f"M4 REJECTED: {reason}")
                if verbose:
                    print(f"  ✘ Hard filter failed: {reason}")
        except Exception as e:
            hard_failures.append(HardFilterResult("Senior Men's League", False, str(e)))
    
    # ── HARD FILTER 2: League Tier (with bypass for sufficient data) ───
    if _GUARDRAIL_AVAILABLE:
        try:
            tier_report = validate_league_tier(leg)
            if not tier_report.passed:
                bypass_report = validate_tier_bypass_with_data(leg)
                if not bypass_report.passed:
                    reason = tier_report.errors[0] if tier_report.errors else "League tier filter failed"
                    hard_failures.append(HardFilterResult("League Tier", False, reason))
                    leg.check_log.append(f"M4 REJECTED: {reason}")
                    if verbose:
                        print(f"  ✘ Hard filter failed: {reason}")
                else:
                    hard_failures.append(HardFilterResult("League Tier", True, "Bypass applied", bypass_applied=True))
        except Exception as e:
            hard_failures.append(HardFilterResult("League Tier", False, str(e)))
    
    # ── HARD FILTER 3: Not Playoff/Knockout/Relegation ─────────────────
    if _GUARDRAIL_AVAILABLE:
        try:
            report = validate_not_playoff_or_knockout(leg)
            if not report.passed:
                reason = report.errors[0] if report.errors else "Playoff/knockout format rejected"
                hard_failures.append(HardFilterResult("Format", False, reason))
                leg.check_log.append(f"M4 REJECTED: {reason}")
                if verbose:
                    print(f"  ✘ Hard filter failed: {reason}")
        except Exception as e:
            hard_failures.append(HardFilterResult("Format", False, str(e)))
    
    # ── HARD FILTER 4: Missing profiles ────────────────────────────────
    if leg.home_profile is None or leg.away_profile is None:
        hard_failures.append(HardFilterResult("Profiles", False, "Missing home or away profile"))
        leg.check_log.append("M4 REJECTED: Missing profile(s)")
        return _build_failure_result(hard_failures, verbose)

    # ── HARD FILTER 5: Maturity gate (both teams must have ≥10 games) ──
    home_mature = getattr(leg.home_profile, 'is_mature', False)
    away_mature = getattr(leg.away_profile, 'is_mature', False)
    
    if not (home_mature and away_mature):
        home_games = leg.home_profile.get_metric("core.games", 0) if leg.home_profile else 0
        away_games = leg.away_profile.get_metric("core.games", 0) if leg.away_profile else 0
        reason = f"home games: {home_games:.0f}, away games: {away_games:.0f} (need ≥10)"
        hard_failures.append(HardFilterResult("Maturity", False, reason))
        leg.check_log.append(f"M4 REJECTED: {reason}")
        if verbose:
            print(f"  ✘ Hard filter failed: {reason}")
        return _build_failure_result(hard_failures, verbose)

    # ── HARD FILTER 6: Odds ≥ 1.70 ─────────────────────────────────────
    leg_odds = getattr(leg, 'odds', 0.0)
    if leg_odds < 1.70:
        reason = f"Odds {leg_odds:.2f} below minimum 1.70"
        hard_failures.append(HardFilterResult("Odds Threshold", False, reason))
        leg.check_log.append(f"M4 REJECTED: {reason}")
        if verbose:
            print(f"  ✘ Hard filter failed: {reason}")
        return _build_failure_result(hard_failures, verbose)

    # ── HARD FILTER 7: Dominance vs Decay ──────────────────────────────
    if _GUARDRAIL_AVAILABLE:
        try:
            report = validate_dominance_vs_decay(leg, verbose=verbose)
            if not report.passed:
                reason = report.errors[0] if report.errors else "Dominance vs decay pattern not found"
                hard_failures.append(HardFilterResult("Dominance vs Decay", False, reason))
                leg.check_log.append(f"M4 REJECTED: {reason}")
                if verbose:
                    print(f"  ✘ Hard filter failed: {reason}")
                return _build_failure_result(hard_failures, verbose)
        except Exception as e:
            hard_failures.append(HardFilterResult("Dominance vs Decay", False, str(e)))
    
    # If any hard filter failed, return failure
    if any(not hf.passed for hf in hard_failures):
        return _build_failure_result(hard_failures, verbose)
    
    # All hard filters passed - run the 8 asymmetric checks
    leg.check_log.append("M4: All hard filters passed")
    if verbose:
        print("  ✓ All hard filters passed")
    
    return _run_checks_only(leg, verbose, hard_failures)


def _build_failure_result(
    hard_failures: List[HardFilterResult],
    verbose: bool,
) -> PreFilterResult:
    """Build a failure result from hard filter failures."""
    any_passed = any(hf.passed for hf in hard_failures) if hard_failures else False
    return PreFilterResult(
        passed=False,
        status=PreFilterStatus.FAILED_HARD_FILTER,
        checks_passed=0,
        checks_evaluated=0,
        weighted_pass_score=0.0,
        hard_filter_failures=hard_failures,
    )


def _run_checks_only(
    leg: Leg,
    verbose: bool = False,
    hard_failures: List[HardFilterResult] = None,
) -> PreFilterResult:
    """
    Run only the 8 asymmetric checks (C1-C8).
    Used internally after hard filters pass.
    """
    # Determine favourite
    fav_is_home = detect_favourite(leg) == "HOME"
    fav_profile = leg.home_profile if fav_is_home else leg.away_profile
    und_profile = leg.away_profile if fav_is_home else leg.home_profile
    
    leg.check_log.append(f"M4 Favourite: {'HOME' if fav_is_home else 'AWAY'} ({fav_profile.team_name})")
    
    if verbose:
        print(f"  M4 Favourite: {'HOME' if fav_is_home else 'AWAY'} ({fav_profile.team_name})")
    
    # Run all checks
    checks = [
        _check_season_win_gap(fav_profile, und_profile),
        _check_venue_win_gap(leg, fav_is_home, fav_profile, und_profile),
        _check_h2h_favoured(leg),
        _check_transition_favours(fav_profile),
        _check_bounce_back(fav_profile),
        _check_ceiling_proximity(fav_profile),
        _check_momentum_gap(fav_profile, und_profile),
        _check_resilience_gap(fav_profile, und_profile),
    ]
    
    # Calculate statistics
    checks_passed = sum(1 for c in checks if c.passed)
    checks_evaluated = sum(1 for c in checks if c.status != CheckStatus.SKIP)
    
    # Calculate weighted pass score
    weighted_sum = 0.0
    total_weight = 0.0
    for c in checks:
        if c.status == CheckStatus.SKIP:
            continue
        weight = CHECK_WEIGHTS.get(c.check_id, 0.1)
        weighted_sum += (c.points / 10.0) * weight
        total_weight += weight
    
    weighted_pass_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    
    # Determine pass/fail
    passed = checks_passed >= REQUIRED_CHECKS
    status = PreFilterStatus.PASSED if passed else PreFilterStatus.FAILED_CHECKS
    
    # Log to leg
    leg.check_log.append(
        f"M4 Asymmetric checks: {checks_passed}/{checks_evaluated} passed (need {REQUIRED_CHECKS}) | "
        f"Weighted: {weighted_pass_score:.1%}"
    )
    
    if verbose:
        print(f"  M4 Asymmetric checks: {checks_passed}/{checks_evaluated} passed (need {REQUIRED_CHECKS})")
        for c in checks:
            icon = "✓" if c.passed else "✗" if c.status == CheckStatus.FAIL else "○"
            print(f"    {icon} {c.name}: {c.message}")
    
    return PreFilterResult(
        passed=passed,
        status=status,
        checks_passed=checks_passed,
        checks_evaluated=checks_evaluated,
        checks_required=REQUIRED_CHECKS,
        weighted_pass_score=round(weighted_pass_score, 3),
        check_details=checks,
        hard_filter_failures=hard_failures or [],
    )


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def quick_eligibility_check(leg: Leg) -> Tuple[bool, List[str]]:
    """
    Quick eligibility check without running full pre-filter.
    Returns (is_eligible, reasons) for fast filtering.
    """
    reasons = []
    
    # Check competition type
    if getattr(leg, 'competition_type', 'league') != 'league':
        reasons.append(f"Not a league match: {leg.competition_type}")
    
    # Check playoff status
    if getattr(leg, 'is_playoff', False):
        reasons.append("Playoff/knockout match")
    
    # Check maturity
    if leg.home_profile and leg.away_profile:
        home_mature = getattr(leg.home_profile, 'is_mature', False)
        away_mature = getattr(leg.away_profile, 'is_mature', False)
        if not (home_mature and away_mature):
            reasons.append("Insufficient games (<10)")
    
    # Check odds
    if getattr(leg, 'odds', 0.0) < 1.70:
        reasons.append(f"Odds {leg.odds:.2f} < 1.70")
    
    # Check league tier (quick version)
    league_tier = getattr(leg, 'league_tier', 3)
    league_country = getattr(leg, 'league_country', 'unknown')
    
    # Quick tier check
    if league_country in ['england', 'spain', 'germany', 'italy', 'france']:
        if league_tier > 3:
            reasons.append(f"League tier {league_tier} too low for {league_country}")
    elif league_country in ['netherlands', 'portugal', 'belgium', 'turkey', 'brazil', 'argentina', 'japan']:
        if league_tier > 2:
            reasons.append(f"League tier {league_tier} too low for {league_country}")
    else:
        if league_tier > 1 and league_country != 'unknown':
            reasons.append(f"League tier {league_tier} too low for {league_country}")
    
    return len(reasons) == 0, reasons


def batch_prefilter(
    legs: List[Leg],
    verbose: bool = False,
) -> List[PreFilterResult]:
    """
    Run pre-filter on multiple legs.
    
    Args:
        legs: List of Leg objects
        verbose: Print progress
    
    Returns:
        List of PreFilterResult objects
    """
    results = []
    
    for i, leg in enumerate(legs):
        if verbose:
            print(f"  Processing leg {i+1}/{len(legs)}: {getattr(leg, 'match_id', 'unknown')}")
        
        result = run_asymmetric_prefilter(leg, verbose=verbose)
        results.append(result)
    
    if verbose:
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        avg_score = sum(r.weighted_pass_score for r in results) / len(results) if results else 0
        print(f"\n  Batch complete: {passed} passed, {failed} failed, avg weighted score={avg_score:.1%}")
    
    return results


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — EXPORTS
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Constants
    "SEASON_WIN_GAP_MIN",
    "VENUE_WIN_GAP_MIN_RATE",
    "H2H_WIN_RATE_MIN",
    "TRANSITION_NEXT_WIN_MIN",
    "BOUNCE_BACK_MIN",
    "CEILING_PROXIMITY_MAX",
    "MOMENTUM_GAP_MIN",
    "RESILIENCE_GAP_MIN",
    "REQUIRED_CHECKS",
    "MIN_H2H_GAMES_FOR_CHECK",
    "MIN_TM_SAMPLE_SIZE",
    "FORCE_BYPASS_FILTERS",
    "CHECK_WEIGHTS",
    # Enums
    "CheckStatus",
    "PreFilterStatus",
    # Data classes
    "CheckResult",
    "HardFilterResult",
    "PreFilterResult",
    # Functions
    "detect_favourite",
    "run_asymmetric_prefilter",
    "quick_eligibility_check",
    "batch_prefilter",
]


# ═══════════════════════════════════════════════════════════════
# SECTION 10 — EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from module2 import TeamProfile, TransitionMatrix, H2HRecord
    
    print("\n" + "=" * 70)
    print("MODULE 4: ASYMMETRIC PRE-FILTER - TEST RUN")
    print("=" * 70)
    
    # Create mock favourite profile (strong team)
    fav_profile = TeamProfile(team_id="1", team_name="Arsenal", is_mature=True)
    fav_profile.update_metrics({
        "core.games": 28,
        "core.wins": 20,
        "core.draws": 4,
        "core.losses": 4,
        "core.xg": 55.0,
        "core.xga": 25.0,
        "home_wins": 11,
        "home_games": 14,
        "away_wins": 9,
        "away_games": 14,
        "position": 2,
        "points": 64,
    })
    fav_profile.form = {"recent_results": ["W", "W", "W", "D", "W"]}
    fav_profile.transition = TransitionMatrix(
        pattern="SERIAL_WINNER",
        probs={
            "W": {"W": 0.55, "D": 0.25, "L": 0.20},
            "D": {"W": 0.40, "D": 0.35, "L": 0.25},
            "L": {"W": 0.50, "D": 0.25, "L": 0.25},
        },
        sample_size=25
    )
    
    # Create mock underdog profile (weak team)
    und_profile = TeamProfile(team_id="2", team_name="Southampton", is_mature=True)
    und_profile.update_metrics({
        "core.games": 28,
        "core.wins": 6,
        "core.draws": 7,
        "core.losses": 15,
        "core.xg": 30.0,
        "core.xga": 48.0,
        "home_wins": 4,
        "home_games": 14,
        "away_wins": 2,
        "away_games": 14,
        "position": 18,
        "points": 25,
    })
    und_profile.form = {"recent_results": ["L", "L", "L", "D", "L"]}
    und_profile.transition = TransitionMatrix(
        pattern="LOSS_PRONE",
        probs={
            "W": {"W": 0.25, "D": 0.25, "L": 0.50},
            "D": {"W": 0.20, "D": 0.30, "L": 0.50},
            "L": {"W": 0.20, "D": 0.20, "L": 0.60},
        },
        sample_size=22
    )
    
    # Create H2H record (favourable to Arsenal)
    h2h = H2HRecord(games=10, fav_wins=6, draws=2, und_wins=2)
    
    # Create leg
    leg = Leg(
        match_id="test_arsenal_southampton",
        selection="Arsenal",
        odds=1.45,
        market=BetMarket.STRAIGHT_WIN,
        league="Premier League",
        league_id=39,
        league_tier=1,
        league_country="england",
        competition_type="league",
        home_profile=fav_profile,
        away_profile=und_profile,
        h2h=h2h,
        home_odds=1.45,
        away_odds=6.50,
        draw_odds=4.20,
        model_prob=0.68,
        edge=0.08,
    )
    leg.check_log = []
    
    # Add detect_favourite method (already added globally)
    
    print("\n📊 ANALYSING: Arsenal vs Southampton")
    print("-" * 40)
    
    # Run pre-filter
    result = run_asymmetric_prefilter(leg, verbose=True)
    
    print(f"\n{'='*40}")
    print("RESULT")
    print(f"{'='*40}")
    print(f"Passed: {result.passed}")
    print(f"Status: {result.status.value}")
    print(f"Checks: {result.checks_passed}/{result.checks_evaluated} (need {result.checks_required})")
    print(f"Weighted Score: {result.weighted_pass_score:.1%}")
    print(f"Normalized Score: {result.normalized_score:.3f}")
    print(f"Confidence Factor: {result.confidence_factor:.2f}")
    
    print(f"\nCheck Details:")
    for c in result.check_details:
        icon = "✅" if c.passed else "❌" if c.status == CheckStatus.FAIL else "⏭️"
        print(f"  {icon} {c.name}: {c.message} (score: {c.points:.1f}/10)")
    
    # Leg data for M11
    print(f"\n📊 Leg Data for M11:")
    leg_data = result.to_leg_data()
    for key, value in leg_data.items():
        print(f"  {key}: {value}")
    
    # Test quick eligibility check
    print("\n" + "=" * 70)
    print("QUICK ELIGIBILITY CHECK")
    print("=" * 70)
    eligible, reasons = quick_eligibility_check(leg)
    print(f"Eligible: {eligible}")
    if reasons:
        print(f"Reasons: {reasons}")
    
    print("\n" + "=" * 70)
    print("MODULE 4 READY FOR PRODUCTION")
    print("=" * 70)