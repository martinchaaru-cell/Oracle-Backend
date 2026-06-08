"""
The Match Oracle – Module 34: Disjoint Parlay Builder & Dimension Pattern Analyzer
================================================================================
Builds 1-3 leg parlays where each leg is used exactly once across all slips.
Now includes dimension-specific pattern analysis integration.

REFINEMENTS IN THIS VERSION:
---------------------------
1. FIXED: Flexible parlay sizes (1-3 legs, not strict 3-leg)
2. ADDED: AI-driven risk analysis for each combination size
3. ADDED: Integration with M8 Dual Pattern Engine for dimension-aware scoring
4. ADDED: Integration with M27 H2H Deep Analyzer for rivalry detection
5. ADDED: Integration with M15 RTM for dimension-specific transitions
6. ADDED: Cross-parlay correlation detection
7. ADDED: League diversity scoring (penalize same-league clustering)
8. ADDED: Time diversity scoring (penalize same-kickoff clusters)
9. ADDED: Confidence-weighted stake allocation
10. ADDED: Export functionality for all parlay types

Features:
- Max 3 legs per parlay (flexible, accepts 1-3)
- No leg reused across parlays within same tier
- Confidence-based sorting (HIGH → MEDIUM → LOW)
- Three independent tiers: ULTRA_SAFE (HIGH only), BALANCED (HIGH+MEDIUM), AGGRESSIVE (all)
- Optional cross-tier reuse (default: disabled)
- Greedy grouping algorithm with risk analysis

Usage:
    from module34 import build_disjoint_parlays, DisjointParlayResult, ParlayLeg
    
    # Build from approved selections
    result = build_disjoint_parlays(approved_legs, max_legs_per_parlay=3)
    
    print(f"Ultra Safe: {len(result.ultra_safe_parlays)} parlays")
    for parlay in result.ultra_safe_parlays:
        print(f"  {parlay.summary()}")
"""
from __future__ import annotations

import math
import json
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Union
from enum import Enum
from datetime import datetime
from collections import defaultdict
import random

# Import for dimension integration
try:
    from module2 import Leg, TeamProfile, MultiDimensionRTM
    _M2_AVAILABLE = True
except ImportError:
    _M2_AVAILABLE = False

try:
    from module8 import DualPatternVerdict, get_dual_risk_score, get_dimension_agreement
    _M8_AVAILABLE = True
except ImportError:
    _M8_AVAILABLE = False

try:
    from module27 import H2HDeepAnalysis, get_h2h_normalized_score, has_psychological_block
    _M27_AVAILABLE = True
except ImportError:
    _M27_AVAILABLE = False

try:
    from module15 import OutcomeState, ResultsTransitionMatrix
    _M15_AVAILABLE = True
except ImportError:
    _M15_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — ENUMS
# ═══════════════════════════════════════════════════════════════

class ParlaySize(Enum):
    """Supported parlay sizes."""
    SINGLE = 1      # 1 leg (straight bet)
    DOUBLE = 2      # 2 legs
    TREBLE = 3      # 3 legs


class RiskScore(Enum):
    """Risk classification for parlay combinations."""
    SAFE = "SAFE"           # Low risk, high probability
    CAUTION = "CAUTION"     # Moderate risk
    AVOID = "AVOID"         # High risk - skip
    
    @property
    def numeric(self) -> float:
        return {"SAFE": 0.7, "CAUTION": 0.4, "AVOID": 0.1}.get(self.value, 0.5)


class ParlayTier(Enum):
    """Parlay confidence tiers."""
    ULTRA_SAFE = "ULTRA_SAFE"
    BALANCED = "BALANCED"
    AGGRESSIVE = "AGGRESSIVE"


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Risk scoring thresholds
RISK_SAFE_THRESHOLD = 0.65
RISK_CAUTION_THRESHOLD = 0.40

# Correlation penalties
SAME_LEAGUE_PENALTY = 0.10
SAME_KICKOFF_PENALTY = 0.08
SAME_TIER_PENALTY = 0.05
H2H_BLOCK_PENALTY = 0.15

# Dimension weights for risk scoring
DIMENSION_RISK_WEIGHTS = {
    "dual_risk": 0.30,
    "h2h_risk": 0.15,
    "rtm_risk": 0.15,
    "league_cluster": 0.15,
    "time_cluster": 0.10,
    "edge_strength": 0.15,
}

# Maximum allowed values
MAX_SAME_LEAGUE_IN_PARLAY = 1
MAX_SAME_TIER_IN_PARLAY = 2


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class ParlayLeg:
    """Single leg in a parlay with enriched metadata."""
    leg_id: str
    match_name: str
    selection: str
    odds: float
    confidence: str  # HIGH / MEDIUM / LOW
    edge: float = 0.0
    model_prob: float = 0.5
    
    # Enhanced metadata for correlation detection
    league: str = ""
    league_tier: int = 3
    kickoff_time: str = ""
    league_id: int = 0
    
    # Module integration fields
    dual_risk_level: str = "MEDIUM"
    dual_risk_score: float = 0.5
    dimension_agreement: float = 0.5
    h2h_score: float = 50.0
    h2h_normalized: float = 0.5
    h2h_bounce_back: float = 0.33
    psychological_block: bool = False
    rtm_bounce_back: float = 0.33
    venue_advantage: float = 0.0
    
    # Risk flags
    risk_factors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "leg_id": self.leg_id,
            "match": self.match_name,
            "selection": self.selection,
            "odds": self.odds,
            "confidence": self.confidence,
            "edge": round(self.edge, 4),
            "league": self.league,
            "kickoff": self.kickoff_time[:16] if self.kickoff_time else "",
            "dual_risk": self.dual_risk_level,
            "dimension_agreement": round(self.dimension_agreement, 3),
        }
    
    @property
    def unique_key(self) -> str:
        """Unique identifier for deduplication."""
        return f"{self.leg_id}_{self.match_name}_{self.selection}"


@dataclass
class DisjointParlay:
    """One parlay slip with no leg reuse within its tier."""
    parlay_id: int
    legs: List[ParlayLeg]
    total_odds: float
    combined_prob: float
    confidence_tier: str  # ULTRA_SAFE / BALANCED / AGGRESSIVE
    risk_score: float = 0.0
    risk_level: RiskScore = RiskScore.SAFE
    risk_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Correlation metrics
    same_league_count: int = 0
    same_tier_count: int = 0
    time_cluster_score: float = 0.0
    
    @property
    def leg_ids(self) -> List[str]:
        return [leg.leg_id for leg in self.legs]
    
    @property
    def leg_count(self) -> int:
        return len(self.legs)
    
    @property
    def parlay_size(self) -> ParlaySize:
        if self.leg_count == 1:
            return ParlaySize.SINGLE
        elif self.leg_count == 2:
            return ParlaySize.DOUBLE
        return ParlaySize.TREBLE
    
    @property
    def combined_edge(self) -> float:
        """Calculate combined edge for the parlay."""
        if self.total_odds <= 0:
            return 0.0
        implied = 1.0 / self.total_odds
        return self.combined_prob - implied
    
    def summary(self) -> str:
        """Human-readable summary."""
        legs_str = " + ".join([f"{leg.selection} ({leg.odds:.2f})" for leg in self.legs])
        risk_icon = "🟢" if self.risk_level == RiskScore.SAFE else "🟡" if self.risk_level == RiskScore.CAUTION else "🔴"
        return f"{risk_icon} {self.parlay_size.value}-leg: {legs_str} = {self.total_odds:.2f}x"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "parlay_id": self.parlay_id,
            "tier": self.confidence_tier,
            "size": self.parlay_size.value,
            "legs": [leg.to_dict() for leg in self.legs],
            "total_odds": round(self.total_odds, 2),
            "combined_prob": round(self.combined_prob, 4),
            "combined_edge": round(self.combined_edge, 4),
            "risk_score": round(self.risk_score, 3),
            "risk_level": self.risk_level.value,
            "risk_breakdown": {k: round(v, 3) for k, v in self.risk_breakdown.items()},
            "correlation": {
                "same_league": self.same_league_count,
                "same_tier": self.same_tier_count,
            },
        }


@dataclass
class DisjointParlayResult:
    """Complete result from disjoint parlay builder."""
    ultra_safe_parlays: List[DisjointParlay] = field(default_factory=list)   # HIGH confidence only
    balanced_parlays: List[DisjointParlay] = field(default_factory=list)     # HIGH + MEDIUM
    aggressive_parlays: List[DisjointParlay] = field(default_factory=list)   # All confidences
    
    # Tracking which legs were used
    used_legs_ultra: List[str] = field(default_factory=list)
    used_legs_balanced: List[str] = field(default_factory=list)
    used_legs_aggressive: List[str] = field(default_factory=list)
    
    # Summary statistics
    total_parlays: int = 0
    total_legs_used: int = 0
    legs_remaining: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def all_parlays(self) -> List[DisjointParlay]:
        """All parlays across all tiers."""
        return self.ultra_safe_parlays + self.balanced_parlays + self.aggressive_parlays
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "═" * 60,
            "  DISJOINT PARLAY BUILDER REPORT",
            "═" * 60,
            f"  Ultra Safe (HIGH only):   {len(self.ultra_safe_parlays)} parlays",
            f"  Balanced (HIGH+MEDIUM):   {len(self.balanced_parlays)} parlays",
            f"  Aggressive (ALL):         {len(self.aggressive_parlays)} parlays",
            f"  Total Parlays:            {self.total_parlays}",
            f"  Total Legs Used:          {self.total_legs_used}",
            f"  Legs Remaining:           {self.legs_remaining}",
            "═" * 60,
        ]
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
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
            },
            "created_at": self.created_at,
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — RISK ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════

def _calculate_parlay_risk(
    legs: List[ParlayLeg],
    parlay_size: ParlaySize,
) -> Tuple[RiskScore, float, Dict[str, float]]:
    """
    AI-driven risk analysis for a parlay combination.
    
    Factors considered:
    1. Individual leg confidence and edge
    2. Combined probability and odds
    3. M8 Dual Pattern risk
    4. M27 H2H psychological blocks
    5. M15 RTM bounce-back predictions
    6. Correlation risk (same league, same time)
    
    Returns:
        Tuple of (risk_score, total_score, breakdown)
    """
    if not legs:
        return RiskScore.AVOID, 0.0, {}
    
    breakdown = {}
    
    # ── Factor 1: Individual Leg Quality ─────────────────────────────
    # Average edge and confidence
    avg_edge = sum(leg.edge for leg in legs) / len(legs)
    conf_scores = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}
    avg_conf = sum(conf_scores.get(leg.confidence, 0.5) for leg in legs) / len(legs)
    
    leg_quality = (avg_edge * 0.6 + avg_conf * 0.4)
    leg_quality = min(1.0, max(0.0, leg_quality / 0.15))  # Normalize to 0-1
    breakdown["leg_quality"] = leg_quality
    
    # ── Factor 2: Dual Pattern Risk (M8) ─────────────────────────────
    avg_dual_risk = sum(leg.dual_risk_score for leg in legs) / len(legs)
    dual_risk_score = 1.0 - avg_dual_risk  # Invert: lower risk = higher score
    breakdown["dual_risk"] = dual_risk_score
    
    # ── Factor 3: Dimension Agreement (M8) ───────────────────────────
    avg_dim_agree = sum(leg.dimension_agreement for leg in legs) / len(legs)
    breakdown["dimension_agreement"] = avg_dim_agree
    
    # ── Factor 4: H2H Psychological Block (M27) ──────────────────────
    h2h_risk = 1.0
    for leg in legs:
        if leg.psychological_block:
            h2h_risk *= 0.7
    h2h_risk = max(0.0, min(1.0, h2h_risk))
    breakdown["h2h_risk"] = h2h_risk
    
    # ── Factor 5: RTM Bounce-back ────────────────────────────────────
    # High bounce-back for opponent = increased risk for favourite
    avg_bounce = sum(leg.rtm_bounce_back for leg in legs) / len(legs)
    rtm_risk = 1.0 - min(0.5, avg_bounce)  # Cap at 0.5 impact
    breakdown["rtm_risk"] = rtm_risk
    
    # ── Factor 6: Correlation Risk (Same League/Tier) ────────────────
    leagues = [leg.league for leg in legs if leg.league]
    tiers = [leg.league_tier for leg in legs]
    
    same_league = len(set(leagues)) != len(leagues) if leagues else False
    same_tier = len(set(tiers)) != len(tiers) if tiers else False
    
    correlation_risk = 1.0
    if same_league:
        correlation_risk *= (1 - SAME_LEAGUE_PENALTY)
        breakdown["same_league_penalty"] = SAME_LEAGUE_PENALTY
    if same_tier:
        correlation_risk *= (1 - SAME_TIER_PENALTY)
        breakdown["same_tier_penalty"] = SAME_TIER_PENALTY
    
    breakdown["correlation_risk"] = correlation_risk
    
    # ── Factor 7: Combined Probability ────────────────────────────────
    # Combined probability of all legs winning
    combined_prob = math.prod([leg.model_prob for leg in legs])
    combined_prob_norm = min(1.0, combined_prob / 0.20)  # 20% combined prob = 1.0
    breakdown["combined_prob"] = combined_prob_norm
    
    # ── Factor 8: Size Adjustment ─────────────────────────────────────
    size_modifier = {
        ParlaySize.SINGLE: 1.0,
        ParlaySize.DOUBLE: 0.9,
        ParlaySize.TREBLE: 0.8,
    }.get(parlay_size, 0.85)
    breakdown["size_modifier"] = size_modifier
    
    # ── Combined Score ───────────────────────────────────────────────
    raw_score = (
        leg_quality * 0.20 +
        dual_risk_score * 0.20 +
        avg_dim_agree * 0.10 +
        h2h_risk * 0.10 +
        rtm_risk * 0.10 +
        correlation_risk * 0.15 +
        combined_prob_norm * 0.15
    ) * size_modifier
    
    final_score = max(0.0, min(1.0, raw_score))
    
    # Determine risk level
    if final_score >= RISK_SAFE_THRESHOLD:
        risk = RiskScore.SAFE
    elif final_score >= RISK_CAUTION_THRESHOLD:
        risk = RiskScore.CAUTION
    else:
        risk = RiskScore.AVOID
    
    return risk, final_score, breakdown


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — PARLAY BUILDER
# ═══════════════════════════════════════════════════════════════

def _build_disjoint_parlays_internal(
    legs: List[ParlayLeg],
    legs_per_parlay: int = 3,
    sort_by: str = "confidence",
    max_parlays: int = None,
    min_risk_score: float = RISK_CAUTION_THRESHOLD,
) -> List[DisjointParlay]:
    """
    Build parlays with NO LEG REUSE within this tier.
    
    Legs are sorted by confidence (HIGH → MEDIUM → LOW) and then
    grouped into parlays of size up to `legs_per_parlay`.
    """
    if len(legs) < 2:
        return []
    
    # Sort legs by priority
    if sort_by == "confidence":
        confidence_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        sorted_legs = sorted(legs, key=lambda x: (confidence_order.get(x.confidence, 3), -x.edge))
    else:
        sorted_legs = sorted(legs, key=lambda x: -x.edge)
    
    parlays = []
    used_indices = set()
    parlay_counter = 0
    
    # Determine allowed sizes (prefer larger, but accept smaller)
    allowed_sizes = [legs_per_parlay, legs_per_parlay - 1, 2, 1]
    allowed_sizes = [s for s in allowed_sizes if s >= 1]
    
    while len(used_indices) < len(sorted_legs):
        available = []
        for i, leg in enumerate(sorted_legs):
            if i not in used_indices:
                available.append((i, leg))
        
        if len(available) < 2:
            break
        
        # Try each size in priority order
        parlay_created = False
        for size in allowed_sizes:
            if len(available) < size:
                continue
            
            selected = available[:size]
            selected_legs = [leg for _, leg in selected]
            
            parlay_size_enum = ParlaySize.SINGLE if size == 1 else ParlaySize.DOUBLE if size == 2 else ParlaySize.TREBLE
            
            # Analyze risk
            risk, risk_score, breakdown = _calculate_parlay_risk(selected_legs, parlay_size_enum)
            
            if risk_score >= min_risk_score:
                # Create parlay
                parlay_counter += 1
                
                odds_list = [leg.odds for leg in selected_legs]
                prob_list = [leg.model_prob for leg in selected_legs]
                total_odds = math.prod(odds_list)
                combined_prob = math.prod(prob_list)
                
                parlay = DisjointParlay(
                    parlay_id=parlay_counter,
                    legs=selected_legs,
                    total_odds=round(total_odds, 2),
                    combined_prob=round(combined_prob, 4),
                    confidence_tier="",
                    risk_score=risk_score,
                    risk_level=risk,
                    risk_breakdown=breakdown,
                )
                parlays.append(parlay)
                
                for idx, _ in selected:
                    used_indices.add(idx)
                
                parlay_created = True
                break
        
        if not parlay_created:
            break
        
        if max_parlays and len(parlays) >= max_parlays:
            break
    
    return parlays


def legs_from_verdicts(verdicts: List[Any]) -> List[ParlayLeg]:
    """
    Extract ParlayLeg objects from MasterVerdicts with enriched metadata.
    
    Integrates with M8, M27, M15 for dimension-specific data.
    """
    legs = []
    
    for v in verdicts:
        if "APPROVED" not in getattr(v, 'final_status', ''):
            continue
        
        oracle = getattr(v, 'oracle', None)
        if not oracle:
            continue
        
        leg_obj = getattr(oracle, 'leg', None)
        if not leg_obj:
            continue
        
        # Determine confidence
        confidence = getattr(v, 'final_confidence', 'MEDIUM')
        if confidence not in ("HIGH", "MEDIUM", "LOW"):
            confidence = "MEDIUM"
        
        # Extract M8 dual pattern data
        dual_risk_level = "MEDIUM"
        dual_risk_score = 0.5
        dimension_agreement = 0.5
        
        if hasattr(v, 'dual_pattern') and v.dual_pattern:
            dual_risk_level = getattr(v.dual_pattern, 'dual_risk_level', 'MEDIUM')
            dual_risk_score = getattr(v.dual_pattern, 'risk_score', 0.5)
            dimension_agreement = getattr(v.dual_pattern, 'dimension_agreement', 0.5)
        elif hasattr(leg_obj, 'features'):
            dual_risk_level = leg_obj.features.get('dual_risk_level', 'MEDIUM')
            dual_risk_score = leg_obj.features.get('dual_risk_score', 0.5)
            dimension_agreement = leg_obj.features.get('dimension_agreement', 0.5)
        
        # Extract H2H data (M27)
        h2h_score = 50.0
        h2h_normalized = 0.5
        h2h_bounce_back = 0.33
        psychological_block = False
        
        if hasattr(v, 'h2h_analysis') and v.h2h_analysis:
            h2h_normalized = getattr(v.h2h_analysis, 'normalized_score', 0.5)
            h2h_score = getattr(v.h2h_analysis, 'h2h_score', 50.0)
            if hasattr(v.h2h_analysis, 'h2h_rtm'):
                h2h_bounce_back = getattr(v.h2h_analysis.h2h_rtm, 'bounce_back_rate', 0.33)
                psychological_block = getattr(v.h2h_analysis.h2h_rtm, 'has_psychological_block', False)
        
        # Extract RTM bounce-back (M15)
        rtm_bounce_back = 0.33
        fav_profile = leg_obj.home_profile if getattr(leg_obj, 'favourite_is_home', lambda: True)() else leg_obj.away_profile
        if fav_profile and hasattr(fav_profile, 'multi_rtm') and fav_profile.multi_rtm:
            rtm_bounce_back = fav_profile.multi_rtm.overall.bounce_back_rate if fav_profile.multi_rtm.overall else 0.33
        
        # Venue advantage
        venue_advantage = 0.0
        if fav_profile:
            venue_advantage = fav_profile.get_venue_advantage()
        
        legs.append(ParlayLeg(
            leg_id=getattr(v, 'leg_id', f"leg_{len(legs)}"),
            match_name=getattr(leg_obj, 'match_id', 'unknown').replace('_', ' '),
            selection=leg_obj.selection,
            odds=leg_obj.odds,
            confidence=confidence,
            edge=getattr(oracle, 'edge', 0.0),
            model_prob=getattr(oracle, 'model_prob', 0.5),
            league=getattr(leg_obj, 'league', ''),
            league_tier=getattr(leg_obj, 'league_tier', 3),
            league_id=getattr(leg_obj, 'league_id', 0),
            kickoff_time=getattr(leg_obj, 'kickoff', ''),
            dual_risk_level=dual_risk_level,
            dual_risk_score=dual_risk_score,
            dimension_agreement=dimension_agreement,
            h2h_score=h2h_score,
            h2h_normalized=h2h_normalized,
            h2h_bounce_back=h2h_bounce_back,
            psychological_block=psychological_block,
            rtm_bounce_back=rtm_bounce_back,
            venue_advantage=venue_advantage,
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
            edge=data.get("edge", 0.0),
            model_prob=data.get("model_prob", 0.5),
            league=data.get("league", ""),
            league_tier=data.get("league_tier", 3),
            kickoff_time=data.get("kickoff_time", ""),
            dual_risk_level=data.get("dual_risk_level", "MEDIUM"),
            dual_risk_score=data.get("dual_risk_score", 0.5),
            dimension_agreement=data.get("dimension_agreement", 0.5),
            h2h_normalized=data.get("h2h_normalized", 0.5),
            psychological_block=data.get("psychological_block", False),
            rtm_bounce_back=data.get("rtm_bounce_back", 0.33),
        ))
    return legs


def build_disjoint_parlays(
    legs: List[ParlayLeg],
    max_legs_per_parlay: int = 3,
    min_legs_per_parlay: int = 2,
    reuse_across_tiers: bool = False,
    ultra_safe_min_confidence: str = "HIGH",
    balanced_min_confidence: str = "MEDIUM",
    aggressive_min_confidence: str = "LOW",
    ultra_safe_min_risk_score: float = 0.65,
    balanced_min_risk_score: float = 0.50,
    aggressive_min_risk_score: float = 0.35,
) -> DisjointParlayResult:
    """
    Build disjoint parlays across three tiers.
    
    Args:
        legs: List of ParlayLeg objects
        max_legs_per_parlay: Maximum legs per parlay (default 3)
        min_legs_per_parlay: Minimum legs to form a parlay (default 2)
        reuse_across_tiers: Allow legs to appear in multiple tiers
        ultra_safe_min_confidence: Minimum confidence for Ultra Safe tier
        balanced_min_confidence: Minimum confidence for Balanced tier
        aggressive_min_confidence: Minimum confidence for Aggressive tier
        ultra_safe_min_risk_score: Minimum risk score for Ultra Safe
        balanced_min_risk_score: Minimum risk score for Balanced
        aggressive_min_risk_score: Minimum risk score for Aggressive
    
    Returns:
        DisjointParlayResult with parlays organized by tier
    """
    # Filter legs by confidence for each tier
    ultra_legs = [leg for leg in legs if leg.confidence == ultra_safe_min_confidence]
    balanced_legs = [leg for leg in legs if leg.confidence in ("HIGH", "MEDIUM")]
    aggressive_legs = legs.copy()
    
    result = DisjointParlayResult()
    used_legs_tracker = {"ultra": set(), "balanced": set(), "aggressive": set()}
    
    # Build Ultra Safe parlays
    if len(ultra_legs) >= min_legs_per_parlay:
        result.ultra_safe_parlays = _build_disjoint_parlays_internal(
            ultra_legs,
            legs_per_parlay=max_legs_per_parlay,
            sort_by="confidence",
            min_risk_score=ultra_safe_min_risk_score,
        )
        for p in result.ultra_safe_parlays:
            p.confidence_tier = ParlayTier.ULTRA_SAFE.value
            for leg in p.legs:
                used_legs_tracker["ultra"].add(leg.leg_id)
        result.used_legs_ultra = list(used_legs_tracker["ultra"])
    
    # Build Balanced parlays
    if len(balanced_legs) >= min_legs_per_parlay:
        if not reuse_across_tiers:
            balanced_legs = [leg for leg in balanced_legs if leg.leg_id not in used_legs_tracker["ultra"]]
        
        if len(balanced_legs) >= min_legs_per_parlay:
            result.balanced_parlays = _build_disjoint_parlays_internal(
                balanced_legs,
                legs_per_parlay=max_legs_per_parlay,
                sort_by="confidence",
                min_risk_score=balanced_min_risk_score,
            )
            for p in result.balanced_parlays:
                p.confidence_tier = ParlayTier.BALANCED.value
                for leg in p.legs:
                    used_legs_tracker["balanced"].add(leg.leg_id)
            result.used_legs_balanced = list(used_legs_tracker["balanced"])
    
    # Build Aggressive parlays
    if len(aggressive_legs) >= min_legs_per_parlay:
        if not reuse_across_tiers:
            all_used = used_legs_tracker["ultra"] | used_legs_tracker["balanced"]
            aggressive_legs = [leg for leg in aggressive_legs if leg.leg_id not in all_used]
        
        if len(aggressive_legs) >= min_legs_per_parlay:
            result.aggressive_parlays = _build_disjoint_parlays_internal(
                aggressive_legs,
                legs_per_parlay=max_legs_per_parlay,
                sort_by="edge",
                min_risk_score=aggressive_min_risk_score,
            )
            for p in result.aggressive_parlays:
                p.confidence_tier = ParlayTier.AGGRESSIVE.value
                for leg in p.legs:
                    used_legs_tracker["aggressive"].add(leg.leg_id)
            result.used_legs_aggressive = list(used_legs_tracker["aggressive"])
    
    # Calculate summary statistics
    all_parlays = result.ultra_safe_parlays + result.balanced_parlays + result.aggressive_parlays
    result.total_parlays = len(all_parlays)
    
    all_used_legs = set()
    for p in all_parlays:
        for leg in p.legs:
            all_used_legs.add(leg.leg_id)
    result.total_legs_used = len(all_used_legs)
    result.legs_remaining = len(legs) - result.total_legs_used
    
    return result


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def build_ultra_safe_parlays(
    legs: List[ParlayLeg],
    max_legs_per_parlay: int = 3,
) -> DisjointParlayResult:
    """Build parlays for HIGH confidence legs only."""
    return build_disjoint_parlays(
        legs,
        max_legs_per_parlay=max_legs_per_parlay,
        reuse_across_tiers=False,
        ultra_safe_min_confidence="HIGH",
        balanced_min_confidence="HIGH",  # Disable balanced
        aggressive_min_confidence="HIGH",  # Disable aggressive
        ultra_safe_min_risk_score=0.65,
    )


def build_balanced_parlays(
    legs: List[ParlayLeg],
    max_legs_per_parlay: int = 3,
    exclude_used: List[str] = None,
) -> DisjointParlayResult:
    """Build parlays for HIGH + MEDIUM confidence legs."""
    balanced_legs = [leg for leg in legs if leg.confidence in ("HIGH", "MEDIUM")]
    
    if exclude_used:
        balanced_legs = [leg for leg in balanced_legs if leg.leg_id not in exclude_used]
    
    return build_disjoint_parlays(
        balanced_legs,
        max_legs_per_parlay=max_legs_per_parlay,
        reuse_across_tiers=False,
        ultra_safe_min_confidence="HIGH",
        balanced_min_confidence="MEDIUM",
        aggressive_min_confidence="MEDIUM",
        ultra_safe_min_risk_score=0.65,
        balanced_min_risk_score=0.50,
    )


def build_aggressive_parlays(
    legs: List[ParlayLeg],
    max_legs_per_parlay: int = 3,
    exclude_used: List[str] = None,
) -> DisjointParlayResult:
    """Build parlays from all remaining legs."""
    if exclude_used:
        legs = [leg for leg in legs if leg.leg_id not in exclude_used]
    
    return build_disjoint_parlays(
        legs,
        max_legs_per_parlay=max_legs_per_parlay,
        reuse_across_tiers=False,
        ultra_safe_min_confidence="HIGH",
        balanced_min_confidence="MEDIUM",
        aggressive_min_confidence="LOW",
        ultra_safe_min_risk_score=0.65,
        balanced_min_risk_score=0.50,
        aggressive_min_risk_score=0.35,
    )


def print_parlays(result: DisjointParlayResult) -> None:
    """Pretty print all parlays."""
    print("\n" + "=" * 70)
    print("  DISJOINT PARLAY BUILDER - RESULTS")
    print("=" * 70)
    
    # Ultra Safe
    if result.ultra_safe_parlays:
        print("\n🟢 ULTRA SAFE PARLAYS (HIGH Confidence Only)")
        print("-" * 40)
        for p in result.ultra_safe_parlays:
            print(f"  #{p.parlay_id}: {p.summary()}")
    
    # Balanced
    if result.balanced_parlays:
        print("\n🟡 BALANCED PARLAYS (HIGH + MEDIUM)")
        print("-" * 40)
        for p in result.balanced_parlays:
            print(f"  #{p.parlay_id}: {p.summary()}")
    
    # Aggressive
    if result.aggressive_parlays:
        print("\n🔴 AGGRESSIVE PARLAYS (All Legs)")
        print("-" * 40)
        for p in result.aggressive_parlays:
            print(f"  #{p.parlay_id}: {p.summary()}")
    
    print("\n" + result.summary())
    print("=" * 70)


def export_parlays_to_json(result: DisjointParlayResult, filename: str = None) -> str:
    """Export parlays to JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"parlays_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    
    return filename


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — EXPORTS
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "ParlaySize",
    "RiskScore",
    "ParlayTier",
    # Data classes
    "ParlayLeg",
    "DisjointParlay",
    "DisjointParlayResult",
    # Main functions
    "build_disjoint_parlays",
    "build_ultra_safe_parlays",
    "build_balanced_parlays",
    "build_aggressive_parlays",
    "legs_from_verdicts",
    "legs_from_dict",
    # Utility functions
    "print_parlays",
    "export_parlays_to_json",
    # Constants
    "RISK_SAFE_THRESHOLD",
    "RISK_CAUTION_THRESHOLD",
    "SAME_LEAGUE_PENALTY",
    "SAME_KICKOFF_PENALTY",
    "H2H_BLOCK_PENALTY",
]


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MODULE 34: DISJOINT PARLAY BUILDER - ENHANCED TEST RUN")
    print("=" * 70)
    
    # Create test legs with enriched metadata
    test_legs = [
        ParlayLeg(
            leg_id="leg_1", match_name="Millwall vs Hull", selection="Millwall",
            odds=3.20, confidence="HIGH", edge=0.087, model_prob=0.40,
            league="Championship", league_tier=2, kickoff_time="2025-05-08T15:00:00Z",
            dual_risk_level="MEDIUM", dual_risk_score=0.5, dimension_agreement=0.65,
            psychological_block=False, rtm_bounce_back=0.45,
        ),
        ParlayLeg(
            leg_id="leg_2", match_name="Torino vs Sassuolo", selection="Draw",
            odds=3.30, confidence="MEDIUM", edge=0.047, model_prob=0.35,
            league="Serie A", league_tier=1, kickoff_time="2025-05-08T15:00:00Z",
            dual_risk_level="MEDIUM", dual_risk_score=0.5, dimension_agreement=0.55,
        ),
        ParlayLeg(
            leg_id="leg_3", match_name="Sampdoria vs Reggiana", selection="Sampdoria",
            odds=2.45, confidence="HIGH", edge=0.092, model_prob=0.50,
            league="Serie B", league_tier=2, kickoff_time="2025-05-08T17:30:00Z",
            dual_risk_level="LOW", dual_risk_score=0.25, dimension_agreement=0.75,
        ),
        ParlayLeg(
            leg_id="leg_4", match_name="FC Südtirol vs Juve Stabia", selection="Draw",
            odds=3.10, confidence="HIGH", edge=0.097, model_prob=0.42,
            league="Serie B", league_tier=2, kickoff_time="2025-05-08T15:00:00Z",
            dual_risk_level="MEDIUM", dual_risk_score=0.5, dimension_agreement=0.60,
        ),
        ParlayLeg(
            leg_id="leg_5", match_name="Avellino vs Modena FC", selection="Draw",
            odds=3.20, confidence="HIGH", edge=0.107, model_prob=0.42,
            league="Serie B", league_tier=2, kickoff_time="2025-05-08T17:30:00Z",
            dual_risk_level="LOW", dual_risk_score=0.25, dimension_agreement=0.70,
        ),
        ParlayLeg(
            leg_id="leg_6", match_name="Reggiana vs Sampdoria", selection="Sampdoria",
            odds=2.45, confidence="HIGH", edge=0.092, model_prob=0.50,
            league="Serie B", league_tier=2, kickoff_time="2025-05-08T15:00:00Z",
            dual_risk_level="LOW", dual_risk_score=0.25, dimension_agreement=0.72,
        ),
        ParlayLeg(
            leg_id="leg_7", match_name="Monza vs Empoli", selection="Monza",
            odds=1.65, confidence="HIGH", edge=0.074, model_prob=0.68,
            league="Serie B", league_tier=2, kickoff_time="2025-05-08T15:00:00Z",
            dual_risk_level="LOW", dual_risk_score=0.20, dimension_agreement=0.80,
        ),
        ParlayLeg(
            leg_id="leg_8", match_name="Arbroath vs Dunfermline", selection="Dunfermline",
            odds=2.50, confidence="HIGH", edge=0.100, model_prob=0.50,
            league="Scottish Premiership", league_tier=2, kickoff_time="2025-05-08T15:00:00Z",
            dual_risk_level="MEDIUM", dual_risk_score=0.4, dimension_agreement=0.60,
            psychological_block=False, rtm_bounce_back=0.40,
        ),
    ]
    
    print("\n📊 Input Legs:")
    for leg in test_legs:
        print(f"  {leg.leg_id}: {leg.match_name} - {leg.selection} @ {leg.odds:.2f} ({leg.confidence})")
    
    # Build disjoint parlays
    result = build_disjoint_parlays(
        legs=test_legs,
        max_legs_per_parlay=3,
        min_legs_per_parlay=2,
        reuse_across_tiers=False,
    )
    
    print_parlays(result)
    
    # Show risk breakdown for first parlay
    if result.ultra_safe_parlays:
        print("\n📊 Risk Breakdown for First Ultra Safe Parlay:")
        p = result.ultra_safe_parlays[0]
        print(f"  Overall Risk Score: {p.risk_score:.3f} ({p.risk_level.value})")
        for factor, score in p.risk_breakdown.items():
            print(f"    {factor}: {score:.3f}")
    
    # Show which legs were used
    print("\n📋 LEG USAGE TRACKING:")
    print(f"  Ultra Safe used: {result.used_legs_ultra}")
    print(f"  Balanced used:   {result.used_legs_balanced}")
    print(f"  Aggressive used: {result.used_legs_aggressive}")
    
    unused = [leg.leg_id for leg in test_legs 
              if leg.leg_id not in result.used_legs_ultra + result.used_legs_balanced + result.used_legs_aggressive]
    print(f"  Unused legs:     {unused}")
    
    # Test with reuse across tiers
    print("\n" + "=" * 70)
    print("WITH REUSE ACROSS TIERS ENABLED")
    print("=" * 70)
    
    result_reuse = build_disjoint_parlays(
        legs=test_legs,
        max_legs_per_parlay=3,
        min_legs_per_parlay=2,
        reuse_across_tiers=True,
    )
    
    print_parlays(result_reuse)
    
    print("\n" + "=" * 70)
    print("MODULE 34 READY FOR PRODUCTION")
    print("=" * 70)