"""
The Match Oracle - Module 12: Intelligent Cleaner, Portfolio Optimizer & Acca Builder (REFINED v4)
===============================================================================================
CRITICAL FIX v4 (May 2026):
---------------------------
Previous versions ranked picks by EDGE. This was mathematically incorrect and destroyed results.
- Edge tells you VALUE (how much to bet)
- Model Probability tells you LIKELIHOOD (whether to bet at all)

NEW HIERARCHY HARDCODED:
1. PRIMARY SORT: Model Probability (most likely to win)
2. SECONDARY SORT: Edge (for tie-breaking within same probability tier)
3. TERTIARY SORT: Confidence level

NEW IN VERSION 4:
----------------
1. ADDED: REJECTED (H2H CONFLICT) to REJECT_STATUSES
2. ADDED: REJECTED (COIN FLIP) to REJECT_STATUSES
3. ADDED: REJECTED (BOTH DECAYING) to REJECT_STATUSES
4. All conflict-related rejections are now filtered out of portfolio

DRAW PROBABILITY PROTECTION:
- Legs with draw probability > 30% are automatically flagged
- Safe ACCA excludes legs with draw probability > 25%
- DNB recommendation for high-draw matches

Portfolio position: receives List[MasterVerdict] from Module 11.

CUP & FRIENDLY POLICY (enforced here at portfolio level):
  Cups and friendlies are NEVER included in any parlay output — top_picks,
  ultra_safe_acca, value_acca, or any AccaSlip tier. Hard-blocked regardless
  of confidence or edge. The distinction is:
    Scan = YES  (algorithm sees them, M14 and M18 learn from their patterns)
    Parlay = NO (they never appear in the final betting output)

Usage:
    from module12 import run_intelligent_cleaner, CleanedPortfolio
    
    portfolio = run_intelligent_cleaner(verdicts)
    print(f"Top picks: {len(portfolio.top_picks)}")
    print(f"Rejected (including conflicts): {len(portfolio.rejected_legs)}")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from itertools import combinations
from datetime import datetime
import math
import logging

from module11 import MasterVerdict

# Set up logging
logger = logging.getLogger("oracle_beast.module12")


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — CORE PORTFOLIO CLEANER CONSTANTS
# ═══════════════════════════════════════════════════════════════

MIN_CONFIDENCE_ALLOWED = ["HIGH", "MEDIUM"]

# NEW v4: Expanded REJECT_STATUSES to include conflict rejections
REJECT_STATUSES = [
    "REJECTED",
    "REJECTED (AI SHADOW LOGGED)",
    "REJECTED (AI CONSENSUS)",
    "REJECTED (H2H CONFLICT)",     # NEW v4: H2H vs current season conflict
    "REJECTED (COIN FLIP)",         # NEW v4: Equal team strength / both poor
    "REJECTED (BOTH DECAYING)",     # NEW v4: Favourite also decaying
]

MAX_RISK_FLAGS         = 2
ULTRA_SAFE_CONFIDENCE  = "HIGH"
MAX_ACCUMULATOR_SIZE   = 4

# NEW: Draw probability thresholds
HIGH_DRAW_WARNING      = 0.30   # 30%+ draw probability = warning
SAFE_ACCA_MAX_DRAW     = 0.25   # 25% max draw probability for safe ACCA
DNB_RECOMMENDATION     = 0.28   # 28%+ draw probability = recommend DNB

# NEW: Minimum model probability for each tier
MIN_PROB_ULTRA_SAFE    = 0.60   # 60%+ for ultra-safe ACCA
MIN_PROB_BALANCED      = 0.55   # 55%+ for balanced ACCA
MIN_PROB_AGGRESSIVE    = 0.50   # 50%+ for aggressive ACCA

# FIX: PARLAY_BANNED_COMPETITION_TYPES imported from M31 (single source of truth)
try:
    from module31 import (
        PARLAY_BANNED_COMPETITION_TYPES,
        CUP_AND_FRIENDLY_IDS,
        PARLAY_ELIGIBLE_COMPETITION_TYPES
    )
    _M31_AVAILABLE = True
except ImportError:
    PARLAY_BANNED_COMPETITION_TYPES = {"cup", "friendly"}
    PARLAY_ELIGIBLE_COMPETITION_TYPES = {"league", "playoff"}
    CUP_AND_FRIENDLY_IDS = set()
    _M31_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — PORTFOLIO DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class CleanedPortfolio:
    """Filtered and ranked portfolio after cleaning."""
    cleaned_verdicts:   List[MasterVerdict] = field(default_factory=list)
    rejected_legs:      List[MasterVerdict] = field(default_factory=list)
    top_picks:          List[MasterVerdict] = field(default_factory=list)
    ultra_safe_acca:    List[MasterVerdict] = field(default_factory=list)
    value_acca:         List[MasterVerdict] = field(default_factory=list)
    dnb_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    summary:            Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — CUP/FRIENDLY DETECTION
# ═══════════════════════════════════════════════════════════════

def _is_parlay_banned(v: MasterVerdict) -> bool:
    """Returns True if this leg is from a cup or friendly competition."""
    leg = v.oracle.leg
    
    league_label = getattr(leg, "league", "") or ""
    cup_keywords = {"cup", "coupe", "copa", "pokal", "coppa", "taca",
                    "friendly", "amical", "amistoso", "test", "charity",
                    "supercup", "super cup", "community shield"}
    if any(word in league_label.lower() for word in cup_keywords):
        return True
    
    comp_type = getattr(leg, "competition_type", "league") or "league"
    if comp_type in PARLAY_BANNED_COMPETITION_TYPES:
        return True
    
    league_id = getattr(leg, "league_id", None)
    if league_id is not None and league_id in CUP_AND_FRIENDLY_IDS:
        return True
    
    return True if "cup" in league_label.lower() else False


def _get_draw_probability(v: MasterVerdict) -> float:
    """Extract draw probability from verdict if available."""
    if hasattr(v, 'oracle') and hasattr(v.oracle, 'draw_prob'):
        return v.oracle.draw_prob
    if hasattr(v, 'oracle') and hasattr(v.oracle, 'model_draw_prob'):
        return v.oracle.model_draw_prob
    if hasattr(v, 'features') and 'draw_prob' in v.features:
        return v.features['draw_prob']
    return 0.25  # Default fallback


def _get_model_probability(v: MasterVerdict) -> float:
    """Extract model probability from verdict."""
    if hasattr(v, 'oracle') and hasattr(v.oracle, 'model_prob'):
        return v.oracle.model_prob
    if hasattr(v, 'oracle') and hasattr(v.oracle, 'leg') and hasattr(v.oracle.leg, 'model_prob'):
        return v.oracle.leg.model_prob
    if hasattr(v, 'features') and 'model_prob' in v.features:
        return v.features['model_prob']
    return 0.50  # Default fallback


def _is_valid(v: MasterVerdict) -> bool:
    """
    Check verdict passes minimum requirements for parlay inclusion.
    
    FIXED v4: Now includes H2H CONFLICT and other reject statuses.
    """
    # NEW v4: Check if rejected due to conflict
    if v.final_status in REJECT_STATUSES:
        return False
    
    if v.final_confidence not in MIN_CONFIDENCE_ALLOWED:
        return False
    
    if len(v.risk_flags) > MAX_RISK_FLAGS:
        return False
    
    if not v.oracle.leg.selection:
        return False
    
    if _is_parlay_banned(v):
        return False
    
    # Minimum model probability check
    model_prob = _get_model_probability(v)
    if model_prob < 0.50:
        return False
    
    return True


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — PORTFOLIO CLEANING HELPERS (UPDATED SORTING)
# ═══════════════════════════════════════════════════════════════

def _probability_score(prob: float) -> int:
    """Convert probability to integer score for sorting (higher = better)."""
    return int(prob * 100)


def _confidence_score(conf: str) -> int:
    """Convert confidence string to numeric score for sorting."""
    return {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(conf, 0)


def _rank_verdicts(verdicts: List[MasterVerdict]) -> List[MasterVerdict]:
    """
    Rank verdicts by Model Probability FIRST, then Edge, then Confidence.
    
    FIXED v3: Primary sort is now MODEL PROBABILITY (most likely to win first)
    Edge is only used as tie-breaker.
    """
    def sort_key(v: MasterVerdict) -> Tuple[float, float, int]:
        model_prob = _get_model_probability(v)
        edge = getattr(v.oracle, "edge", 0.0)
        conf_score = _confidence_score(v.final_confidence)
        # Higher probability first (negative for descending)
        return (-model_prob, -edge, -conf_score)
    
    return sorted(verdicts, key=sort_key)


def _detect_conflicts(verdicts: List[MasterVerdict]) -> List[MasterVerdict]:
    """
    Remove duplicate matches, keeping the highest probability version.
    """
    seen: Dict[str, MasterVerdict] = {}
    cleaned: List[MasterVerdict] = []
    
    for v in verdicts:
        key = v.leg_id
        if key not in seen:
            seen[key] = v
            cleaned.append(v)
        else:
            existing = seen[key]
            # Keep the one with higher model probability
            if _get_model_probability(v) > _get_model_probability(existing):
                cleaned.remove(existing)
                cleaned.append(v)
                seen[key] = v
    
    return cleaned


def _build_top_picks(verdicts: List[MasterVerdict], limit: int = 3) -> List[MasterVerdict]:
    """
    Build top picks based on Model Probability (not edge).
    
    FIXED v3: Top picks are the highest probability bets, not highest edge.
    """
    return verdicts[:limit]


def _build_ultra_safe_acca(verdicts: List[MasterVerdict]) -> List[MasterVerdict]:
    """
    Build ultra-safe accumulator from HIGH probability legs.
    
    FIXED v3: Now requires model probability >= 60% (MIN_PROB_ULTRA_SAFE)
    AND draw probability <= 25% (SAFE_ACCA_MAX_DRAW)
    """
    acca = []
    for v in verdicts:
        model_prob = _get_model_probability(v)
        draw_prob = _get_draw_probability(v)
        
        if (v.final_confidence == ULTRA_SAFE_CONFIDENCE and 
            v.final_status == "APPROVED" and
            model_prob >= MIN_PROB_ULTRA_SAFE and
            draw_prob <= SAFE_ACCA_MAX_DRAW):
            acca.append(v)
        
        if len(acca) >= MAX_ACCUMULATOR_SIZE:
            break
    return acca


def _build_value_acca(verdicts: List[MasterVerdict]) -> List[MasterVerdict]:
    """
    Build value accumulator from APPROVED or CAUTION legs.
    
    FIXED v3: Now requires model probability >= 50%
    """
    acca = []
    for v in verdicts:
        model_prob = _get_model_probability(v)
        
        if v.final_status in ("APPROVED", "CAUTION") and model_prob >= MIN_PROB_AGGRESSIVE:
            acca.append(v)
        
        if len(acca) >= MAX_ACCUMULATOR_SIZE:
            break
    return acca


def _generate_dnb_recommendations(verdicts: List[MasterVerdict]) -> List[Dict[str, Any]]:
    """
    Generate DNB (Draw No Bet) recommendations for high-draw matches.
    
    NEW v3: When draw probability exceeds threshold, recommend DNB instead of straight win.
    """
    recommendations = []
    
    for v in verdicts:
        draw_prob = _get_draw_probability(v)
        model_prob = _get_model_probability(v)
        
        if draw_prob >= DNB_RECOMMENDATION and model_prob >= 0.50:
            leg = v.oracle.leg
            recommendations.append({
                "match_id": v.leg_id,
                "match": getattr(leg, 'match_id', 'unknown'),
                "selection": leg.selection,
                "straight_odds": leg.odds,
                "draw_probability": round(draw_prob, 3),
                "model_probability": round(model_prob, 3),
                "recommendation": "DRAW_NO_BET",
                "reason": f"Draw probability {draw_prob:.1%} exceeds {DNB_RECOMMENDATION:.0%} threshold",
                "estimated_dnb_odds": round(leg.odds * 0.85, 2),  # Approximate DNB odds
            })
    
    return recommendations


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — MAIN PORTFOLIO CLEANER (UPDATED v4)
# ═══════════════════════════════════════════════════════════════

def run_intelligent_cleaner(verdicts: List[MasterVerdict]) -> CleanedPortfolio:
    """
    Clean, filter, and rank the portfolio of MasterVerdicts.
    
    FIXED v4: Now filters out H2H CONFLICT, COIN FLIP, and BOTH DECAYING rejections.
    
    Process:
    1. Split into valid and rejected based on eligibility (including conflict statuses)
    2. Remove duplicate matches (keep highest probability)
    3. Rank by Model Probability (primary), Edge (secondary)
    4. Build top picks (top 3 by probability)
    5. Build ultra-safe and value ACCAs with probability thresholds
    6. Generate DNB recommendations for high-draw matches
    
    Args:
        verdicts: List of MasterVerdict from Module 11
    
    Returns:
        CleanedPortfolio with filtered and organized verdicts
    """
    portfolio = CleanedPortfolio()
    
    # Split into valid and rejected
    valid, rejected = [], []
    for v in verdicts:
        # NEW v4: Check if rejected (including conflict statuses)
        if _is_valid(v):
            valid.append(v)
        else:
            rejected.append(v)
    
    portfolio.rejected_legs = rejected
    
    # Remove conflicts (duplicate matches)
    valid = _detect_conflicts(valid)
    
    # Rank by Model Probability (primary), Edge (secondary)
    ranked = _rank_verdicts(valid)
    portfolio.cleaned_verdicts = ranked
    
    # Top picks (best 3 by probability, not edge)
    portfolio.top_picks = _build_top_picks(ranked, limit=3)
    
    # ACCA construction with probability thresholds
    portfolio.ultra_safe_acca = _build_ultra_safe_acca(ranked)
    portfolio.value_acca = _build_value_acca(ranked)
    
    # DNB recommendations for high-draw matches
    portfolio.dnb_recommendations = _generate_dnb_recommendations(ranked)
    
    # Count conflict rejections for summary
    conflict_rejections = sum(1 for v in rejected if "H2H CONFLICT" in v.final_status)
    coin_flip_rejections = sum(1 for v in rejected if "COIN FLIP" in v.final_status)
    both_decaying_rejections = sum(1 for v in rejected if "BOTH DECAYING" in v.final_status)
    
    # Summary statistics
    avg_prob = sum(_get_model_probability(v) for v in ranked) / max(len(ranked), 1)
    avg_edge = sum(getattr(v.oracle, "edge", 0.0) for v in ranked) / max(len(ranked), 1)
    
    portfolio.summary = {
        "total_input":          len(verdicts),
        "valid_after_cleaning": len(ranked),
        "rejected":             len(rejected),
        "rejected_cup_friendly": sum(1 for v in rejected if _is_parlay_banned(v)),
        "rejected_h2h_conflict": conflict_rejections,      # NEW v4
        "rejected_coin_flip":    coin_flip_rejections,      # NEW v4
        "rejected_both_decaying": both_decaying_rejections, # NEW v4
        "top_picks":            len(portfolio.top_picks),
        "ultra_safe_acca_size": len(portfolio.ultra_safe_acca),
        "value_acca_size":      len(portfolio.value_acca),
        "dnb_recommendations":  len(portfolio.dnb_recommendations),
        "avg_model_probability": round(avg_prob, 3),
        "avg_edge":             round(avg_edge, 4),
        "sort_method":          "MODEL_PROBABILITY_PRIMARY",
    }
    
    return portfolio


def export_portfolio(portfolio: CleanedPortfolio) -> Dict[str, Any]:
    """
    Convert CleanedPortfolio to JSON-serialisable format for frontend/API.
    
    FIXED v4: Now includes conflict rejection counts in summary.
    """
    def simplify(v: MasterVerdict) -> Dict[str, Any]:
        leg = v.oracle.leg
        return {
            "leg_id":            v.leg_id,
            "match":             getattr(leg, 'match_id', 'unknown'),
            "selection":         leg.selection,
            "market":            leg.market.value if leg.market else "Straight Win",
            "odds":              leg.odds,
            "confidence":        v.final_confidence,
            "status":            v.final_status,
            "model_probability": round(_get_model_probability(v), 3),
            "draw_probability":  round(_get_draw_probability(v), 3),
            "edge":              round(getattr(v.oracle, "edge", 0.0), 4),
            "risk_flags":        v.risk_flags[:5],
            "notes":             v.decision_notes[:3],
        }
    
    return {
        "top_picks":       [simplify(v) for v in portfolio.top_picks],
        "ultra_safe_acca": [simplify(v) for v in portfolio.ultra_safe_acca],
        "value_acca":      [simplify(v) for v in portfolio.value_acca],
        "dnb_recommendations": portfolio.dnb_recommendations,
        "summary":         portfolio.summary,
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — PROBABILITY-WEIGHTED ACCA SLIP BUILDER
# ═══════════════════════════════════════════════════════════════

# Slip-level targets
SAFE_TARGET       = (2.0,  3.5)
BALANCED_TARGET   = (3.5,  6.5)
AGGRESSIVE_TARGET = (6.5, 15.0)

MAX_SAFE          = 3
MAX_BALANCED      = 4
MAX_AGGRESSIVE    = 6

MAX_CORRELATION_ALLOWED = 1.5


@dataclass
class AccaPick:
    """Flat representation of one verdict for slip building."""
    match:        str
    selection:    str
    market:       str
    odds:         float
    confidence:   str
    status:       str
    model_prob:   float = 0.0
    draw_prob:    float = 0.0
    league:       str = ""
    kickoff_time: str = ""
    risk_tags:    List[str] = field(default_factory=list)


@dataclass
class AccaSlip:
    """One accumulator slip with picks and metadata."""
    slip_type:         str
    picks:             List[AccaPick] = field(default_factory=list)
    total_odds:        float = 1.0
    combined_prob:     float = 0.0
    risk_score:        float = 0.0
    correlation_score: float = 0.0
    is_valid:          bool = True
    notes:             List[str] = field(default_factory=list)


@dataclass
class AccaSlipPortfolio:
    """Collection of ACCA slips by risk tier."""
    safe:       List[AccaSlip] = field(default_factory=list)
    balanced:   List[AccaSlip] = field(default_factory=list)
    aggressive: List[AccaSlip] = field(default_factory=list)


def _verdict_to_pick(v: MasterVerdict) -> AccaPick:
    """Convert a MasterVerdict to a flat AccaPick with probability data."""
    leg = v.oracle.leg
    return AccaPick(
        match=getattr(leg, 'match_id', 'unknown'),
        selection=leg.selection,
        market=leg.market.value if leg.market else "Straight Win",
        odds=leg.odds,
        confidence=v.final_confidence,
        status=v.final_status,
        model_prob=_get_model_probability(v),
        draw_prob=_get_draw_probability(v),
        league=getattr(leg, "league", "unknown"),
        kickoff_time=getattr(leg, "kickoff", ""),
        risk_tags=v.risk_flags or [],
    )


def _filter_to_picks(verdicts: List[MasterVerdict]) -> List[AccaPick]:
    """
    Filter verdicts to picks eligible for ACCA slips.
    
    FIXED v4: Now filters out conflict-rejected verdicts.
    """
    picks = []
    for v in verdicts:
        # NEW v4: Skip conflict rejections
        if v.final_status in REJECT_STATUSES:
            continue
        
        if v.final_status not in ("APPROVED", "CAUTION"):
            continue
        if v.final_confidence not in ("HIGH", "MEDIUM"):
            continue
        if not v.oracle.leg.selection:
            continue
        if _is_parlay_banned(v):
            continue
        
        model_prob = _get_model_probability(v)
        if model_prob < MIN_PROB_AGGRESSIVE:
            continue
        
        picks.append(_verdict_to_pick(v))
    return picks


def _combined_odds(picks: List[AccaPick]) -> float:
    """Calculate combined odds for an accumulator."""
    total = 1.0
    for p in picks:
        total *= max(p.odds, 1.01)
    return round(total, 2)


def _combined_probability(picks: List[AccaPick]) -> float:
    """Calculate combined probability (product of model probabilities)."""
    total = 1.0
    for p in picks:
        total *= max(p.model_prob, 0.01)
    return round(total, 4)


def _risk_score(picks: List[AccaPick]) -> float:
    """Calculate risk score (higher = more risky)."""
    score = 0.0
    for p in picks:
        if p.status == "CAUTION":
            score += 0.5
        if p.confidence == "MEDIUM":
            score += 0.3
        score += len(p.risk_tags) * 0.1
        if p.draw_prob > HIGH_DRAW_WARNING:
            score += 0.2
    return round(score, 2)


def _time_bucket(kickoff: str) -> str:
    """Group kickoffs into 3-hour windows for clustering detection."""
    try:
        dt = datetime.fromisoformat(kickoff.replace('Z', '+00:00'))
        return str(dt.hour // 3)
    except Exception:
        return "unknown"


def _correlation_score(picks: List[AccaPick]) -> float:
    """
    Penalise same-league clustering, same-kickoff-window clustering,
    and stacked risk tags — all of which increase correlated loss risk.
    """
    score = 0.0
    leagues = {}
    time_groups = {}
    pattern_tags: Dict[str, int] = {}

    for p in picks:
        leagues[p.league] = leagues.get(p.league, 0) + 1
        tb = _time_bucket(p.kickoff_time)
        time_groups[tb] = time_groups.get(tb, 0) + 1
        for tag in p.risk_tags:
            if "risk" in tag.lower() or "danger" in tag.lower() or "threat" in tag.lower():
                pattern_tags[tag] = pattern_tags.get(tag, 0) + 1

    for count in leagues.values():
        if count > 1:
            score += (count - 1) * 0.5

    for count in time_groups.values():
        if count > 2:
            score += (count - 2) * 0.3

    for count in pattern_tags.values():
        if count > 1:
            score += (count - 1) * 0.4

    return round(score, 2)


def _validate_slip(picks: List[AccaPick], slip_type: str, corr: float) -> Tuple[bool, List[str]]:
    """Validate a slip against tier-specific rules."""
    notes = []
    
    caution_count = sum(1 for p in picks if p.status == "CAUTION")
    high_draw_count = sum(1 for p in picks if p.draw_prob > HIGH_DRAW_WARNING)
    
    if slip_type == "SAFE" and caution_count > 0:
        notes.append(f"SAFE slip cannot contain CAUTION picks")
        return False, notes
    
    if slip_type == "SAFE" and high_draw_count > 0:
        notes.append(f"SAFE slip cannot contain high-draw matches (>30%)")
        return False, notes
    
    if corr > MAX_CORRELATION_ALLOWED:
        notes.append(f"Correlation too high ({corr:.2f} > {MAX_CORRELATION_ALLOWED})")
        return False, notes
    
    total_odds = _combined_odds(picks)
    if slip_type == "SAFE" and total_odds > SAFE_TARGET[1]:
        notes.append(f"SAFE slip odds {total_odds:.2f} exceed target max {SAFE_TARGET[1]}")
        return False, notes
    
    if slip_type == "AGGRESSIVE" and total_odds < AGGRESSIVE_TARGET[0]:
        notes.append(f"AGGRESSIVE slip odds {total_odds:.2f} below target min {AGGRESSIVE_TARGET[0]}")
        return False, notes
    
    return True, notes


def _build_slips(
    picks: List[AccaPick],
    slip_type: str,
    max_picks: int,
    target: Tuple[float, float],
) -> List[AccaSlip]:
    """
    Build all valid slip combinations for a given tier.
    
    FIXED v3: Now prioritizes combinations with highest combined probability.
    """
    slips = []
    
    for r in range(2, min(len(picks), max_picks) + 1):
        for combo in combinations(picks, r):
            combo_list = list(combo)
            total_odds = _combined_odds(combo_list)
            
            if not (target[0] <= total_odds <= target[1]):
                continue
            
            risk = _risk_score(combo_list)
            corr = _correlation_score(combo_list)
            valid, notes = _validate_slip(combo_list, slip_type, corr)
            
            slips.append(AccaSlip(
                slip_type=slip_type,
                picks=combo_list,
                total_odds=total_odds,
                combined_prob=_combined_probability(combo_list),
                risk_score=risk,
                correlation_score=corr,
                is_valid=valid,
                notes=notes,
            ))
    
    # Sort by combined probability (highest first), then lowest correlation
    return sorted(slips, key=lambda s: (-s.combined_prob, s.correlation_score, s.risk_score))


def build_acca_slip_portfolio(verdicts: List[MasterVerdict]) -> AccaSlipPortfolio:
    """
    Correlation-aware acca slip builder.
    Produces SAFE / BALANCED / AGGRESSIVE slip sets from verdicts.
    
    FIXED v4: Now filters out conflict-rejected verdicts.
    """
    picks = _filter_to_picks(verdicts)
    
    if not picks:
        return AccaSlipPortfolio()
    
    return AccaSlipPortfolio(
        safe=_build_slips(picks, "SAFE", MAX_SAFE, SAFE_TARGET),
        balanced=_build_slips(picks, "BALANCED", MAX_BALANCED, BALANCED_TARGET),
        aggressive=_build_slips(picks, "AGGRESSIVE", MAX_AGGRESSIVE, AGGRESSIVE_TARGET),
    )


def export_acca_slips(portfolio: AccaSlipPortfolio, limit: int = 5) -> Dict[str, Any]:
    """Export ACCA slip portfolio to JSON-serializable format."""
    def slip_to_dict(s: AccaSlip) -> Dict:
        return {
            "slip_type":         s.slip_type,
            "total_odds":        s.total_odds,
            "combined_probability": s.combined_prob,
            "risk_score":        s.risk_score,
            "correlation_score": s.correlation_score,
            "is_valid":          s.is_valid,
            "picks": [
                {
                    "match": p.match,
                    "selection": p.selection,
                    "odds": p.odds,
                    "model_probability": round(p.model_prob, 3),
                    "draw_probability": round(p.draw_prob, 3),
                    "league": p.league,
                    "confidence": p.confidence,
                }
                for p in s.picks
            ],
            "notes": s.notes,
        }
    
    return {
        "safe":       [slip_to_dict(s) for s in portfolio.safe[:limit]],
        "balanced":   [slip_to_dict(s) for s in portfolio.balanced[:limit]],
        "aggressive": [slip_to_dict(s) for s in portfolio.aggressive[:limit]],
    }


def print_acca_slips(portfolio: AccaSlipPortfolio) -> None:
    """Pretty print ACCA slips for console output."""
    for slip_type, slips in [
        ("SAFE", portfolio.safe),
        ("BALANCED", portfolio.balanced),
        ("AGGRESSIVE", portfolio.aggressive),
    ]:
        print("\n" + "=" * 60)
        print(f"{slip_type} SLIPS  ({len(slips)} combinations)")
        print("=" * 60)
        for i, s in enumerate(slips[:5], 1):
            print(f"\nSlip #{i}  odds={s.total_odds:.2f}  prob={s.combined_prob:.1%}  risk={s.risk_score:.2f}  corr={s.correlation_score:.2f}  valid={s.is_valid}")
            for p in s.picks:
                print(f"  → {p.match} | {p.selection} @ {p.odds:.2f}  [{p.league}] (prob={p.model_prob:.1%}, draw={p.draw_prob:.1%})")
            for n in s.notes:
                print(f"  ⚠ {n}")


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — PROBABILITY-WEIGHTED DISJOINT PARLAY BUILDER
# ═══════════════════════════════════════════════════════════════

@dataclass
class ParlayLeg:
    """Single leg in a disjoint parlay."""
    leg_id: str
    match_name: str
    selection: str
    odds: float
    confidence: str
    model_prob: float = 0.0
    draw_prob: float = 0.0
    edge: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "leg_id": self.leg_id,
            "match": self.match_name,
            "selection": self.selection,
            "odds": self.odds,
            "confidence": self.confidence,
            "model_probability": round(self.model_prob, 3),
            "edge": round(self.edge, 4),
        }


@dataclass
class DisjointParlay:
    """One parlay slip with no leg reuse within its tier."""
    parlay_id: int
    legs: List[ParlayLeg]
    total_odds: float
    combined_prob: float
    confidence_tier: str
    risk_score: float = 0.0
    
    @property
    def leg_ids(self) -> List[str]:
        return [leg.leg_id for leg in self.legs]
    
    @property
    def leg_count(self) -> int:
        return len(self.legs)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "parlay_id": self.parlay_id,
            "legs": [leg.to_dict() for leg in self.legs],
            "total_odds": round(self.total_odds, 2),
            "combined_probability": round(self.combined_prob, 4),
            "confidence_tier": self.confidence_tier,
            "risk_score": round(self.risk_score, 4),
        }


@dataclass
class DisjointParlayResult:
    """Complete result from disjoint parlay builder."""
    ultra_safe_parlays: List[DisjointParlay] = field(default_factory=list)
    balanced_parlays: List[DisjointParlay] = field(default_factory=list)
    aggressive_parlays: List[DisjointParlay] = field(default_factory=list)
    
    used_legs_ultra: List[str] = field(default_factory=list)
    used_legs_balanced: List[str] = field(default_factory=list)
    used_legs_aggressive: List[str] = field(default_factory=list)
    
    total_parlays: int = 0
    total_legs_used: int = 0
    legs_remaining: int = 0
    
    def summary(self) -> str:
        lines = [
            "═" * 60,
            "  DISJOINT PARLAY BUILDER REPORT (Probability-Weighted)",
            "═" * 60,
            f"  Ultra Safe (prob≥60% & draw≤25%):   {len(self.ultra_safe_parlays)} parlays",
            f"  Balanced (prob≥55%):               {len(self.balanced_parlays)} parlays",
            f"  Aggressive (prob≥50%):             {len(self.aggressive_parlays)} parlays",
            f"  Total Parlays:            {self.total_parlays}",
            f"  Total Legs Used:          {self.total_legs_used}",
            f"  Legs Remaining:           {self.legs_remaining}",
            "═" * 60,
        ]
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ultra_safe": [p.to_dict() for p in self.ultra_safe_parlays],
            "balanced": [p.to_dict() for p in self.balanced_parlays],
            "aggressive": [p.to_dict() for p in self.aggressive_parlays],
            "summary": {
                "ultra_safe_count": len(self.ultra_safe_parlays),
                "balanced_count": len(self.balanced_parlays),
                "aggressive_count": len(self.aggressive_parlays),
                "total_parlays": self.total_parlays,
                "total_legs_used": self.total_legs_used,
                "legs_remaining": self.legs_remaining,
            }
        }


def _calculate_combined_odds(odds_list: List[float]) -> float:
    return math.prod(odds_list) if odds_list else 1.0


def _calculate_combined_probability(probs: List[float]) -> float:
    return math.prod(probs) if probs else 0.0


def _build_disjoint_parlays_internal(
    legs: List[ParlayLeg],
    legs_per_parlay: int = 3,
    sort_by: str = "probability",
    max_parlays: int = None,
) -> List[DisjointParlay]:
    """
    Build parlays with NO LEG REUSE within this tier.
    
    FIXED v3: Default sort is now by MODEL PROBABILITY (highest first),
    not confidence or edge.
    """
    if len(legs) < legs_per_parlay:
        return []
    
    # Sort by model probability (highest first) for leg selection
    if sort_by == "probability":
        sorted_legs = sorted(legs, key=lambda x: -x.model_prob)
    elif sort_by == "edge":
        sorted_legs = sorted(legs, key=lambda x: -x.edge)
    else:  # confidence then probability
        confidence_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        sorted_legs = sorted(legs, key=lambda x: (confidence_order.get(x.confidence, 3), -x.model_prob))
    
    parlays = []
    used_indices = set()
    parlay_counter = 0
    
    while len(used_indices) < len(sorted_legs):
        available = []
        for i, leg in enumerate(sorted_legs):
            if i not in used_indices:
                available.append((i, leg))
        
        if len(available) < legs_per_parlay:
            break
        
        selected = available[:legs_per_parlay]
        parlay_legs = [leg for _, leg in selected]
        
        odds_list = [leg.odds for leg in parlay_legs]
        prob_list = [leg.model_prob for leg in parlay_legs]
        
        total_odds = _calculate_combined_odds(odds_list)
        combined_prob = _calculate_combined_probability(prob_list)
        
        tier = "ULTRA_SAFE" if all(l.model_prob >= MIN_PROB_ULTRA_SAFE and l.draw_prob <= SAFE_ACCA_MAX_DRAW for l in parlay_legs) else \
               "BALANCED" if all(l.model_prob >= MIN_PROB_BALANCED for l in parlay_legs) else \
               "AGGRESSIVE"
        
        parlay_counter += 1
        parlay = DisjointParlay(
            parlay_id=parlay_counter,
            legs=parlay_legs,
            total_odds=round(total_odds, 2),
            combined_prob=round(combined_prob, 4),
            confidence_tier=tier,
            risk_score=round(1 - combined_prob, 3),
        )
        parlays.append(parlay)
        
        for idx, _ in selected:
            used_indices.add(idx)
        
        if max_parlays and len(parlays) >= max_parlays:
            break
    
    return parlays


def legs_from_master_verdicts(verdicts: List[MasterVerdict]) -> List[ParlayLeg]:
    """
    Extract ParlayLeg objects from MasterVerdicts.
    
    FIXED v4: Now filters out conflict-rejected verdicts.
    """
    legs = []
    
    for v in verdicts:
        # NEW v4: Skip conflict rejections
        if v.final_status in REJECT_STATUSES:
            continue
        
        if "APPROVED" not in v.final_status:
            continue
        
        leg = v.oracle.leg
        model_prob = _get_model_probability(v)
        draw_prob = _get_draw_probability(v)
        
        confidence = v.final_confidence
        if confidence not in ("HIGH", "MEDIUM", "LOW"):
            confidence = "MEDIUM"
        
        legs.append(ParlayLeg(
            leg_id=v.leg_id,
            match_name=getattr(leg, 'match_id', 'unknown').replace('_', ' '),
            selection=leg.selection,
            odds=leg.odds,
            confidence=confidence,
            model_prob=model_prob,
            draw_prob=draw_prob,
            edge=getattr(v.oracle, 'edge', 0.0),
        ))
    
    return legs


def legs_from_dict(legs_data: List[Dict]) -> List[ParlayLeg]:
    """Create ParlayLeg objects from dictionary data."""
    legs = []
    for data in legs_data:
        legs.append(ParlayLeg(
            leg_id=data.get("leg_id", f"leg_{len(legs)}"),
            match_name=data.get("match_name", "Unknown"),
            selection=data.get("selection", "?"),
            odds=data.get("odds", 2.0),
            confidence=data.get("confidence", "MEDIUM"),
            model_prob=data.get("model_prob", 0.5),
            draw_prob=data.get("draw_prob", 0.25),
            edge=data.get("edge", 0.0),
        ))
    return legs


def build_disjoint_parlays(
    legs: List[ParlayLeg],
    max_legs_per_parlay: int = 3,
    min_legs_per_parlay: int = 3,
    reuse_across_tiers: bool = False,
    ultra_safe_min_prob: float = MIN_PROB_ULTRA_SAFE,
    balanced_min_prob: float = MIN_PROB_BALANCED,
    aggressive_min_prob: float = MIN_PROB_AGGRESSIVE,
) -> DisjointParlayResult:
    """
    Build disjoint parlays across three tiers based on MODEL PROBABILITY.
    
    FIXED v4: Now filters out conflict legs at source.
    
    Args:
        legs: List of ParlayLeg objects
        max_legs_per_parlay: Maximum legs per parlay
        min_legs_per_parlay: Minimum legs per parlay
        reuse_across_tiers: If True, legs can appear in multiple tiers
        ultra_safe_min_prob: Minimum probability for Ultra Safe tier (default 60%)
        balanced_min_prob: Minimum probability for Balanced tier (default 55%)
        aggressive_min_prob: Minimum probability for Aggressive tier (default 50%)
    
    Returns:
        DisjointParlayResult with parlays organized by tier
    """
    ultra_legs = [leg for leg in legs if leg.model_prob >= ultra_safe_min_prob and leg.draw_prob <= SAFE_ACCA_MAX_DRAW]
    balanced_legs = [leg for leg in legs if leg.model_prob >= balanced_min_prob]
    aggressive_legs = [leg for leg in legs if leg.model_prob >= aggressive_min_prob]
    
    result = DisjointParlayResult()
    used_legs_tracker = {"ultra": set(), "balanced": set(), "aggressive": set()}
    
    if len(ultra_legs) >= min_legs_per_parlay:
        result.ultra_safe_parlays = _build_disjoint_parlays_internal(
            ultra_legs,
            legs_per_parlay=max_legs_per_parlay,
            sort_by="probability",
        )
        for p in result.ultra_safe_parlays:
            for leg in p.legs:
                used_legs_tracker["ultra"].add(leg.leg_id)
        result.used_legs_ultra = list(used_legs_tracker["ultra"])
    
    if len(balanced_legs) >= min_legs_per_parlay:
        if not reuse_across_tiers:
            balanced_legs = [leg for leg in balanced_legs if leg.leg_id not in used_legs_tracker["ultra"]]
        
        if len(balanced_legs) >= min_legs_per_parlay:
            result.balanced_parlays = _build_disjoint_parlays_internal(
                balanced_legs,
                legs_per_parlay=max_legs_per_parlay,
                sort_by="probability",
            )
            for p in result.balanced_parlays:
                for leg in p.legs:
                    used_legs_tracker["balanced"].add(leg.leg_id)
            result.used_legs_balanced = list(used_legs_tracker["balanced"])
    
    if len(aggressive_legs) >= min_legs_per_parlay:
        if not reuse_across_tiers:
            all_used = used_legs_tracker["ultra"] | used_legs_tracker["balanced"]
            aggressive_legs = [leg for leg in aggressive_legs if leg.leg_id not in all_used]
        
        if len(aggressive_legs) >= min_legs_per_parlay:
            result.aggressive_parlays = _build_disjoint_parlays_internal(
                aggressive_legs,
                legs_per_parlay=max_legs_per_parlay,
                sort_by="probability",
            )
            for p in result.aggressive_parlays:
                for leg in p.legs:
                    used_legs_tracker["aggressive"].add(leg.leg_id)
            result.used_legs_aggressive = list(used_legs_tracker["aggressive"])
    
    all_parlays = result.ultra_safe_parlays + result.balanced_parlays + result.aggressive_parlays
    result.total_parlays = len(all_parlays)
    
    all_used_legs = set()
    for p in all_parlays:
        for leg in p.legs:
            all_used_legs.add(leg.leg_id)
    result.total_legs_used = len(all_used_legs)
    result.legs_remaining = len(legs) - result.total_legs_used
    
    return result


def print_disjoint_parlays(result: DisjointParlayResult) -> None:
    """Pretty print disjoint parlays."""
    print("\n" + "=" * 70)
    print("  DISJOINT PARLAY BUILDER - RESULTS (Probability-Weighted)")
    print("=" * 70)
    
    if result.ultra_safe_parlays:
        print("\n🟢 ULTRA SAFE PARLAYS (prob≥60%, draw≤25%)")
        print("-" * 40)
        for p in result.ultra_safe_parlays:
            legs_str = " + ".join([f"{leg.selection} ({leg.odds:.2f})" for leg in p.legs])
            print(f"  #{p.parlay_id}: {legs_str} = {p.total_odds:.2f}x (prob={p.combined_prob:.1%})")
    
    if result.balanced_parlays:
        print("\n🟡 BALANCED PARLAYS (prob≥55%)")
        print("-" * 40)
        for p in result.balanced_parlays:
            legs_str = " + ".join([f"{leg.selection} ({leg.odds:.2f})" for leg in p.legs])
            print(f"  #{p.parlay_id}: {legs_str} = {p.total_odds:.2f}x (prob={p.combined_prob:.1%})")
    
    if result.aggressive_parlays:
        print("\n🔴 AGGRESSIVE PARLAYS (prob≥50%)")
        print("-" * 40)
        for p in result.aggressive_parlays:
            legs_str = " + ".join([f"{leg.selection} ({leg.odds:.2f})" for leg in p.legs])
            print(f"  #{p.parlay_id}: {legs_str} = {p.total_odds:.2f}x (prob={p.combined_prob:.1%})")
    
    print("\n" + result.summary())
    print("=" * 70)


def export_disjoint_parlays(result: DisjointParlayResult) -> Dict[str, Any]:
    """Export disjoint parlays to JSON-serializable format."""
    return result.to_dict()


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — PROBABILITY-WEIGHTED STAKE SUGGESTIONS
# ═══════════════════════════════════════════════════════════════

def get_probability_weighted_stakes(
    portfolio: CleanedPortfolio,
    bankroll: float,
    base_stake_percent: float = 0.03,
) -> List[Dict[str, Any]]:
    """
    Calculate stake suggestions based on Model Probability (not edge).
    
    FIXED v3: Stakes are proportional to (model_prob - 0.50) / 0.20,
    not edge. Higher probability = higher stake.
    
    Args:
        portfolio: CleanedPortfolio from run_intelligent_cleaner
        bankroll: Total bankroll amount
        base_stake_percent: Base stake percentage (3% default)
    
    Returns:
        List of stake suggestions for top picks
    """
    stakes = []
    
    for v in portfolio.top_picks:
        model_prob = _get_model_probability(v)
        
        # Probability weight: how much above 50%?
        prob_excess = max(0, model_prob - 0.50)
        weight = min(2.0, prob_excess / 0.20)  # 60% = 0.5x, 70% = 1.0x, 80% = 1.5x
        
        stake_percent = base_stake_percent * (0.5 + weight)
        stake_amount = bankroll * stake_percent
        
        leg = v.oracle.leg
        
        stakes.append({
            "match": getattr(leg, 'match_id', 'unknown'),
            "selection": leg.selection,
            "odds": leg.odds,
            "model_probability": round(model_prob, 3),
            "draw_probability": round(_get_draw_probability(v), 3),
            "edge": round(getattr(v.oracle, "edge", 0.0), 4),
            "stake_percent": round(stake_percent * 100, 2),
            "stake_amount": round(stake_amount, 2),
            "potential_return": round(stake_amount * leg.odds, 2),
            "reasoning": f"Probability weight: {weight:.1f}x base",
        })
    
    return stakes


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — EXPORTS
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Data classes (Portfolio)
    "CleanedPortfolio",
    "AccaPick",
    "AccaSlip",
    "AccaSlipPortfolio",
    # Data classes (Disjoint Parlay)
    "ParlayLeg",
    "DisjointParlay",
    "DisjointParlayResult",
    # Main functions
    "run_intelligent_cleaner",
    "export_portfolio",
    "build_acca_slip_portfolio",
    "export_acca_slips",
    "print_acca_slips",
    # Disjoint parlay functions
    "build_disjoint_parlays",
    "legs_from_master_verdicts",
    "legs_from_dict",
    "print_disjoint_parlays",
    "export_disjoint_parlays",
    # Probability-weighted stake suggestions
    "get_probability_weighted_stakes",
    # Constants
    "SAFE_TARGET",
    "BALANCED_TARGET", 
    "AGGRESSIVE_TARGET",
    "MIN_PROB_ULTRA_SAFE",
    "MIN_PROB_BALANCED",
    "MIN_PROB_AGGRESSIVE",
    "HIGH_DRAW_WARNING",
    "SAFE_ACCA_MAX_DRAW",
    "DNB_RECOMMENDATION",
    "REJECT_STATUSES",  # NEW v4: Export for other modules
]


# ═══════════════════════════════════════════════════════════════
# SECTION 10 — EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from module2 import Leg, BetMarket
    from module11 import OracleVerdict
    
    print("\n" + "=" * 70)
    print("MODULE 12: PORTFOLIO CLEANER v4 - CONFLICT REJECTION TEST")
    print("=" * 70)
    
    # Create mock verdicts including conflict rejections
    mock_verdicts = []
    
    test_data = [
        ("Inter vs Lazio", 1.75, "HIGH", "APPROVED", "Serie A", 0.68, 0.10),
        ("Liverpool vs Chelsea", 2.05, "HIGH", "APPROVED", "Premier League", 0.59, 0.25),
        ("Bournemouth vs Man City", 4.20, "MEDIUM", "REJECTED (H2H CONFLICT)", "Premier League", 0.37, 0.35),
        ("Chelsea vs Tottenham", 1.98, "MEDIUM", "REJECTED (H2H CONFLICT)", "Premier League", 0.36, 0.31),
        ("Leganés vs Huesca", 2.00, "MEDIUM", "REJECTED (COIN FLIP)", "Segunda", 0.45, 0.28),
        ("Marseille vs Le Havre", 1.85, "HIGH", "APPROVED", "Ligue 1", 0.62, 0.15),
    ]
    
    for i, (match, odds, conf, status, league, model_prob, draw_prob) in enumerate(test_data):
        leg = Leg(
            match_id=match.replace(" ", "_").lower(),
            selection=match.split(" vs ")[0],
            odds=odds,
            league=league,
            kickoff="2025-01-15T15:00:00Z",
        )
        
        # Create oracle with probability data
        class MockOracle:
            def __init__(self):
                self.leg = leg
                self.edge = 0.08 if model_prob > 0.60 else 0.05
                self.model_prob = model_prob
                self.draw_prob = draw_prob
                self.final_status = status
        
        oracle = MockOracle()
        
        verdict = MasterVerdict(
            leg_id=leg.match_id,
            oracle=oracle,
            final_status=status,
            final_confidence=conf,
            risk_flags=[],
            decision_notes=[f"Sample verdict {i+1}"],
        )
        
        # Add features for probability extraction
        verdict.features = {"model_prob": model_prob, "draw_prob": draw_prob}
        
        mock_verdicts.append(verdict)
    
    print("\n📊 Input Verdicts (including H2H CONFLICT rejections):")
    for v in mock_verdicts:
        print(f"  {v.leg_id}: {v.final_status} (prob={_get_model_probability(v):.1%})")
    
    # Run cleaner
    portfolio = run_intelligent_cleaner(mock_verdicts)
    
    print(f"\n📋 Portfolio Summary:")
    for key, value in portfolio.summary.items():
        print(f"  {key}: {value}")
    
    print(f"\n🏆 Top Picks (only APPROVED legs):")
    for i, v in enumerate(portfolio.top_picks, 1):
        print(f"  {i}. {v.leg_id}: prob={_get_model_probability(v):.1%}, status={v.final_status}")
    
    print(f"\n❌ Rejected Legs (including conflicts):")
    for v in portfolio.rejected_legs:
        print(f"  {v.leg_id}: {v.final_status}")
    
    # Verify conflict rejections are filtered
    h2h_conflicts = [v for v in portfolio.rejected_legs if "H2H CONFLICT" in v.final_status]
    print(f"\n  H2H Conflict rejections: {len(h2h_conflicts)}")
    
    print("\n" + "=" * 70)
    print("MODULE 12 v4 READY FOR PRODUCTION")
    print("=" * 70)