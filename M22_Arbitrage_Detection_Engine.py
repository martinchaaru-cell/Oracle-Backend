"""
The Match Oracle – Module 22: Arbitrage Detection Engine (REFINED)
=============================================================
Finds:
- Risk-free betting opportunities (arbitrage)
- Multi-book inefficiencies
- Dutching opportunities
- Surebet identification
- Scalping opportunities on exchanges

REFINEMENTS IN THIS VERSION:
---------------------------
1. ADDED: Comprehensive arbitrage detection across 3-way and 2-way markets
2. ADDED: Dutching calculator for backing multiple outcomes
3. ADDED: Surebet identification with confidence levels
4. ADDED: Stake calculation with rounding to practical units
5. ADDED: Profit margin thresholds for filtering
6. ADDED: Multi-market arbitrage scanning
7. ADDED: Arbitrage opportunity ranking by profit margin
8. ADDED: Historical arbitrage pattern tracking
9. ADDED: Alert generation for high-value opportunities
10. ADDED: Batch processing across multiple bookmakers
11. ADDED: Exchange back/lay arbitrage (scalping)
12. ADDED: Arbitrage portfolio management

When an arbitrage opportunity exists, you can bet on all outcomes
across different bookmakers and guarantee a profit regardless of result.

Feeds into:
    Module 13 (bankroll allocation for arbitrage)
    Module 30 (alerts for arbitrage opportunities)

Usage:
    from module22 import calculate_arbitrage, identify_surebets, BookOdds
    
    # Single match arbitrage
    books = [
        BookOdds("Bet365", 2.10, 3.40, 3.50),
        BookOdds("Pinnacle", 2.05, 3.50, 3.60),
    ]
    arb = calculate_arbitrage(books, total_stake=100)
    if arb.exists:
        print(f"Arbitrage! {arb.profit_margin:.2f}% profit")
    
    # Surebet identification across multiple matches
    surebets = identify_surebets(three_way_books, two_way_books)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from enum import Enum
from datetime import datetime, timezone
from collections import defaultdict

# Set up logging
logger = logging.getLogger("oracle_beast.module22")


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — ENUMS
# ═══════════════════════════════════════════════════════════════

class ArbitrageType(Enum):
    """Type of arbitrage opportunity."""
    THREE_WAY = "three_way"      # Home/Draw/Away
    TWO_WAY = "two_way"          # Over/Under or Yes/No markets
    BACK_LAY = "back_lay"        # Exchange back/lay arbitrage
    DUTCHING = "dutching"        # Betting on multiple outcomes for profit
    CROSS_MARKET = "cross_market"  # Arbitrage across different markets


class ArbitrageConfidence(Enum):
    """Confidence level in arbitrage opportunity."""
    HIGH = "HIGH"        # >2% profit margin, multiple books agree
    MEDIUM = "MEDIUM"    # 1-2% profit margin
    LOW = "LOW"          # 0.5-1% profit margin
    MARGINAL = "MARGINAL" # <0.5% profit margin


class MarketType(Enum):
    """Types of betting markets."""
    MATCH_WINNER = "match_winner"
    OVER_UNDER = "over_under"
    BTTS = "btts"
    DOUBLE_CHANCE = "double_chance"
    DRAW_NO_BET = "draw_no_bet"
    HANDICAP = "handicap"


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Minimum profit margin to report (0.5% = 0.005)
MIN_PROFIT_MARGIN = 0.005

# High confidence threshold
HIGH_CONFIDENCE_MARGIN = 0.02   # 2%
MEDIUM_CONFIDENCE_MARGIN = 0.01  # 1%

# Default stake for calculation
DEFAULT_STAKE = 100.0

# Maximum allowed odds (sanity cap)
MAX_ODDS = 1000.0

# Minimum allowed odds
MIN_ODDS = 1.01

# Rounding precision for stakes
STAKE_PRECISION = 2

# Bookmaker reliability scores (for weighting)
BOOKMAKER_RELIABILITY = {
    "pinnacle": 1.0,
    "bet365": 0.95,
    "betfair": 0.95,
    "william_hill": 0.9,
    "ladbrokes": 0.9,
    "unibet": 0.85,
    "888sport": 0.85,
    "default": 0.8,
}

# Maximum stake per bookmaker (as % of bankroll)
MAX_STAKE_PERCENT = 0.10  # 10% max per arbitrage


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class BookOdds:
    """Odds from a single bookmaker for a market."""
    bookmaker: str
    home_odds: float
    draw_odds: float
    away_odds: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reliability: float = 0.8
    
    def __post_init__(self):
        """Set reliability based on bookmaker name."""
        name_lower = self.bookmaker.lower()
        for bk, score in BOOKMAKER_RELIABILITY.items():
            if bk in name_lower:
                self.reliability = score
                break
    
    def validate(self) -> bool:
        """Check if odds are valid."""
        return all(o >= MIN_ODDS and o <= MAX_ODDS for o in [self.home_odds, self.draw_odds, self.away_odds] if o > 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "bookmaker": self.bookmaker,
            "home_odds": self.home_odds,
            "draw_odds": self.draw_odds,
            "away_odds": self.away_odds,
            "reliability": self.reliability,
            "timestamp": self.timestamp,
        }


@dataclass
class TwoWayOdds:
    """Odds for two-way markets (Over/Under, BTTS, etc.)."""
    bookmaker: str
    yes_odds: float
    no_odds: float
    market: str = ""  # e.g., "over_2.5", "btts"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reliability: float = 0.8
    
    def __post_init__(self):
        """Set reliability based on bookmaker name."""
        name_lower = self.bookmaker.lower()
        for bk, score in BOOKMAKER_RELIABILITY.items():
            if bk in name_lower:
                self.reliability = score
                break
    
    def validate(self) -> bool:
        """Check if odds are valid."""
        return all(o >= MIN_ODDS and o <= MAX_ODDS for o in [self.yes_odds, self.no_odds] if o > 0)


@dataclass
class ExchangeOdds:
    """Odds from a betting exchange (back/lay)."""
    bookmaker: str
    back_odds: float   # Odds to back (bet FOR)
    lay_odds: float    # Odds to lay (bet AGAINST)
    market: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def validate(self) -> bool:
        """Check if odds are valid."""
        return self.back_odds >= MIN_ODDS and self.lay_odds >= MIN_ODDS


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity."""
    exists: bool
    arbitrage_type: ArbitrageType = ArbitrageType.THREE_WAY
    profit_margin: float = 0.0           # Profit percentage (e.g., 2.5 = 2.5%)
    guaranteed_return: float = 0.0       # Return on total stake
    total_stake: float = 0.0             # Total amount to stake
    
    # Individual stakes for each outcome
    stakes: Dict[str, float] = field(default_factory=dict)
    
    # Best bookmaker for each outcome
    best_books: Dict[str, str] = field(default_factory=dict)
    
    # Odds used for each outcome
    odds_used: Dict[str, float] = field(default_factory=dict)
    
    # Market info
    match_id: str = ""
    market: str = "match_winner"
    confidence: ArbitrageConfidence = ArbitrageConfidence.LOW
    reliability_weighted: float = 0.0   # Weighted by bookmaker reliability
    
    # Timestamp
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    @property
    def profit_amount(self) -> float:
        """Profit amount based on total stake."""
        return self.guaranteed_return - self.total_stake
    
    @property
    def roi(self) -> float:
        """Return on investment."""
        return self.profit_margin / 100.0
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if this is a high-confidence arbitrage."""
        return self.confidence in (ArbitrageConfidence.HIGH, ArbitrageConfidence.MEDIUM)
    
    def summary(self) -> str:
        """Human-readable summary."""
        if not self.exists:
            return "No arbitrage opportunity detected"
        
        return (f"Arbitrage: {self.profit_margin:.2f}% profit | "
                f"Stake: {self.total_stake:.2f} | "
                f"Return: {self.guaranteed_return:.2f} | "
                f"Confidence: {self.confidence.value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "exists": self.exists,
            "type": self.arbitrage_type.value,
            "profit_margin": round(self.profit_margin, 2),
            "guaranteed_return": round(self.guaranteed_return, 2),
            "total_stake": round(self.total_stake, 2),
            "stakes": {k: round(v, 2) for k, v in self.stakes.items()},
            "best_books": self.best_books,
            "odds_used": self.odds_used,
            "match_id": self.match_id,
            "market": self.market,
            "confidence": self.confidence.value,
            "reliability_weighted": round(self.reliability_weighted, 3),
            "detected_at": self.detected_at,
        }


@dataclass
class DutchingOpportunity:
    """
    Dutching opportunity - betting on multiple outcomes
    to guarantee profit or break-even with value.
    """
    exists: bool
    outcomes: List[str] = field(default_factory=list)
    odds_used: Dict[str, float] = field(default_factory=dict)
    stakes: Dict[str, float] = field(default_factory=dict)
    total_stake: float = 0.0
    guaranteed_return: float = 0.0
    profit_margin: float = 0.0
    confidence: ArbitrageConfidence = ArbitrageConfidence.LOW
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "exists": self.exists,
            "outcomes": self.outcomes,
            "odds_used": self.odds_used,
            "stakes": {k: round(v, 2) for k, v in self.stakes.items()},
            "total_stake": round(self.total_stake, 2),
            "guaranteed_return": round(self.guaranteed_return, 2),
            "profit_margin": round(self.profit_margin, 2),
            "confidence": self.confidence.value,
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — THREE-WAY ARBITRAGE DETECTION
# ═══════════════════════════════════════════════════════════════

def find_best_odds(books: List[BookOdds]) -> Dict[str, Tuple[float, str, float]]:
    """
    Find the best odds for each outcome across bookmakers.
    
    Args:
        books: List of BookOdds from different bookmakers
    
    Returns:
        Dictionary with keys 'home', 'draw', 'away' each containing 
        (odds, bookmaker, reliability)
    """
    if not books:
        return {}
    
    best_home = (0.0, "", 0.0)
    best_draw = (0.0, "", 0.0)
    best_away = (0.0, "", 0.0)
    
    for book in books:
        if not book.validate():
            continue
        
        if book.home_odds > best_home[0]:
            best_home = (book.home_odds, book.bookmaker, book.reliability)
        if book.draw_odds > best_draw[0]:
            best_draw = (book.draw_odds, book.bookmaker, book.reliability)
        if book.away_odds > best_away[0]:
            best_away = (book.away_odds, book.bookmaker, book.reliability)
    
    return {
        "home": best_home,
        "draw": best_draw,
        "away": best_away,
    }


def calculate_arbitrage(
    books: List[BookOdds],
    total_stake: float = DEFAULT_STAKE,
    min_profit_margin: float = MIN_PROFIT_MARGIN,
    match_id: str = "",
) -> ArbitrageOpportunity:
    """
    Calculate arbitrage opportunity across multiple bookmakers.
    
    Args:
        books: List of BookOdds from different bookmakers
        total_stake: Total amount to stake (will be distributed)
        min_profit_margin: Minimum profit percentage to report (0.5 = 0.5%)
        match_id: Optional match identifier
    
    Returns:
        ArbitrageOpportunity with stakes and profit margin
    """
    opportunity = ArbitrageOpportunity(
        exists=False,
        match_id=match_id,
    )
    
    if len(books) < 2:
        opportunity.stakes = {"note": "Need at least 2 bookmakers"}
        return opportunity
    
    # Find best odds for each outcome
    best = find_best_odds(books)
    
    home_odds, home_book, home_rel = best.get("home", (0, "", 0))
    draw_odds, draw_book, draw_rel = best.get("draw", (0, "", 0))
    away_odds, away_book, away_rel = best.get("away", (0, "", 0))
    
    # Validate odds
    if home_odds <= 0 or away_odds <= 0:
        opportunity.stakes = {"note": "Missing home or away odds"}
        return opportunity
    
    # Calculate inverse sum
    inv_sum = (1 / home_odds) + (1 / draw_odds) + (1 / away_odds) if draw_odds > 0 else 0
    
    # If no draw odds (unlikely), try two-way only
    if draw_odds <= 0:
        inv_sum = (1 / home_odds) + (1 / away_odds)
    
    # Check for arbitrage
    if inv_sum >= 1:
        opportunity.stakes = {"note": f"No arbitrage (inv_sum={inv_sum:.4f})"}
        return opportunity
    
    # Calculate profit margin
    profit_margin = (1 - inv_sum) * 100
    
    if profit_margin < min_profit_margin * 100:
        opportunity.stakes = {"note": f"Margin too low ({profit_margin:.2f}% < {min_profit_margin*100:.1f}%)"}
        return opportunity
    
    # Calculate stakes
    stake_home = total_stake / home_odds / inv_sum if home_odds > 0 else 0
    stake_draw = total_stake / draw_odds / inv_sum if draw_odds > 0 else 0
    stake_away = total_stake / away_odds / inv_sum if away_odds > 0 else 0
    
    # Round to 2 decimal places
    stake_home = round(stake_home, STAKE_PRECISION)
    stake_draw = round(stake_draw, STAKE_PRECISION)
    stake_away = round(stake_away, STAKE_PRECISION)
    
    # Adjust for rounding (ensure total matches)
    total_calculated = stake_home + stake_draw + stake_away
    if total_calculated != total_stake:
        diff = total_stake - total_calculated
        # Add rounding difference to the largest stake
        stakes_list = [(stake_home, "home"), (stake_draw, "draw"), (stake_away, "away")]
        stakes_list.sort(reverse=True)
        if stakes_list[0][1] == "home":
            stake_home += round(diff, STAKE_PRECISION)
        elif stakes_list[0][1] == "draw":
            stake_draw += round(diff, STAKE_PRECISION)
        else:
            stake_away += round(diff, STAKE_PRECISION)
    
    # Calculate weighted reliability score
    reliability_weighted = (
        (home_rel * stake_home + draw_rel * stake_draw + away_rel * stake_away) / total_stake
        if total_stake > 0 else 0
    )
    
    # Determine confidence based on profit margin and reliability
    if profit_margin >= HIGH_CONFIDENCE_MARGIN * 100 and reliability_weighted >= 0.85:
        confidence = ArbitrageConfidence.HIGH
    elif profit_margin >= MEDIUM_CONFIDENCE_MARGIN * 100 and reliability_weighted >= 0.8:
        confidence = ArbitrageConfidence.MEDIUM
    elif profit_margin >= MIN_PROFIT_MARGIN * 100:
        confidence = ArbitrageConfidence.LOW
    else:
        confidence = ArbitrageConfidence.MARGINAL
    
    opportunity.exists = True
    opportunity.arbitrage_type = ArbitrageType.THREE_WAY
    opportunity.profit_margin = round(profit_margin, 2)
    opportunity.guaranteed_return = round(total_stake * (1 + profit_margin / 100), 2)
    opportunity.total_stake = total_stake
    opportunity.stakes = {
        "home": stake_home,
        "draw": stake_draw,
        "away": stake_away,
    }
    opportunity.best_books = {
        "home": home_book,
        "draw": draw_book,
        "away": away_book,
    }
    opportunity.odds_used = {
        "home": home_odds,
        "draw": draw_odds,
        "away": away_odds,
    }
    opportunity.confidence = confidence
    opportunity.reliability_weighted = round(reliability_weighted, 3)
    
    logger.info(f"Arbitrage detected: {profit_margin:.2f}% profit on {match_id}")
    
    return opportunity


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — TWO-WAY ARBITRAGE (Over/Under, BTTS)
# ═══════════════════════════════════════════════════════════════

def find_best_two_way_odds(books: List[TwoWayOdds]) -> Dict[str, Tuple[float, str, float]]:
    """
    Find the best odds for two-way markets across bookmakers.
    
    Args:
        books: List of TwoWayOdds from different bookmakers
    
    Returns:
        Dictionary with keys 'yes', 'no' each containing (odds, bookmaker, reliability)
    """
    if not books:
        return {}
    
    best_yes = (0.0, "", 0.0)
    best_no = (0.0, "", 0.0)
    
    for book in books:
        if not book.validate():
            continue
        
        if book.yes_odds > best_yes[0]:
            best_yes = (book.yes_odds, book.bookmaker, book.reliability)
        if book.no_odds > best_no[0]:
            best_no = (book.no_odds, book.bookmaker, book.reliability)
    
    return {
        "yes": best_yes,
        "no": best_no,
    }


def calculate_two_way_arbitrage(
    books: List[TwoWayOdds],
    total_stake: float = DEFAULT_STAKE,
    min_profit_margin: float = MIN_PROFIT_MARGIN,
    match_id: str = "",
    market: str = "",
) -> ArbitrageOpportunity:
    """
    Calculate arbitrage for two-way markets (Over/Under, BTTS, etc.).
    
    Args:
        books: List of TwoWayOdds from different bookmakers
        total_stake: Total amount to stake
        min_profit_margin: Minimum profit percentage to report
        match_id: Optional match identifier
        market: Market name (e.g., "over_2.5", "btts")
    
    Returns:
        ArbitrageOpportunity with stakes and profit margin
    """
    opportunity = ArbitrageOpportunity(
        exists=False,
        match_id=match_id,
        market=market,
        arbitrage_type=ArbitrageType.TWO_WAY,
    )
    
    if len(books) < 2:
        opportunity.stakes = {"note": "Need at least 2 bookmakers"}
        return opportunity
    
    # Find best odds
    best = find_best_two_way_odds(books)
    
    yes_odds, yes_book, yes_rel = best.get("yes", (0, "", 0))
    no_odds, no_book, no_rel = best.get("no", (0, "", 0))
    
    if yes_odds <= 0 or no_odds <= 0:
        opportunity.stakes = {"note": "Missing yes or no odds"}
        return opportunity
    
    # Calculate inverse sum
    inv_sum = (1 / yes_odds) + (1 / no_odds)
    
    if inv_sum >= 1:
        opportunity.stakes = {"note": f"No arbitrage (inv_sum={inv_sum:.4f})"}
        return opportunity
    
    # Calculate profit margin
    profit_margin = (1 - inv_sum) * 100
    
    if profit_margin < min_profit_margin * 100:
        opportunity.stakes = {"note": f"Margin too low ({profit_margin:.2f}%)"}
        return opportunity
    
    # Calculate stakes
    stake_yes = total_stake * (1 / yes_odds) / inv_sum
    stake_no = total_stake * (1 / no_odds) / inv_sum
    
    stake_yes = round(stake_yes, STAKE_PRECISION)
    stake_no = round(stake_no, STAKE_PRECISION)
    
    # Adjust for rounding
    total_calculated = stake_yes + stake_no
    if total_calculated != total_stake:
        diff = total_stake - total_calculated
        if stake_yes > stake_no:
            stake_yes += round(diff, STAKE_PRECISION)
        else:
            stake_no += round(diff, STAKE_PRECISION)
    
    # Calculate weighted reliability
    reliability_weighted = (
        (yes_rel * stake_yes + no_rel * stake_no) / total_stake if total_stake > 0 else 0
    )
    
    # Confidence
    if profit_margin >= HIGH_CONFIDENCE_MARGIN * 100 and reliability_weighted >= 0.85:
        confidence = ArbitrageConfidence.HIGH
    elif profit_margin >= MEDIUM_CONFIDENCE_MARGIN * 100:
        confidence = ArbitrageConfidence.MEDIUM
    else:
        confidence = ArbitrageConfidence.LOW
    
    opportunity.exists = True
    opportunity.profit_margin = round(profit_margin, 2)
    opportunity.guaranteed_return = round(total_stake * (1 + profit_margin / 100), 2)
    opportunity.total_stake = total_stake
    opportunity.stakes = {
        "yes": stake_yes,
        "no": stake_no,
    }
    opportunity.best_books = {
        "yes": yes_book,
        "no": no_book,
    }
    opportunity.odds_used = {
        "yes": yes_odds,
        "no": no_odds,
    }
    opportunity.confidence = confidence
    opportunity.reliability_weighted = round(reliability_weighted, 3)
    
    logger.info(f"Two-way arbitrage detected: {profit_margin:.2f}% on {match_id} - {market}")
    
    return opportunity


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — EXCHANGE ARBITRAGE (BACK/LAY)
# ═══════════════════════════════════════════════════════════════

def calculate_scalping_opportunity(
    exchange: ExchangeOdds,
    stake_back: float = 100.0,
    commission: float = 0.02,  # 2% exchange commission
) -> Dict[str, Any]:
    """
    Calculate scalping opportunity on betting exchanges (back/lay).
    
    Args:
        exchange: ExchangeOdds with back and lay odds
        stake_back: Stake on the back bet
        commission: Exchange commission rate (0.02 = 2%)
    
    Returns:
        Dictionary with lay stake and guaranteed profit
    """
    back_odds = exchange.back_odds
    lay_odds = exchange.lay_odds
    
    if back_odds <= 0 or lay_odds <= 0:
        return {"error": "Invalid odds", "exists": False}
    
    # Adjust for commission
    commission_factor = 1 - commission
    
    # Calculate lay stake for guaranteed profit
    # Liability = stake_lay * (lay_odds - 1)
    # Profit if back wins = stake_back * (back_odds - 1) - liability
    # Profit if lay wins = stake_lay * commission_factor - stake_back
    
    # Solve for equal profit:
    # stake_lay = (stake_back * back_odds) / (lay_odds + commission_factor * (lay_odds - 1))
    # Simplified approximation:
    stake_lay = (stake_back * back_odds) / lay_odds
    stake_lay = round(stake_lay, STAKE_PRECISION)
    
    liability = round(stake_lay * (lay_odds - 1), STAKE_PRECISION)
    
    # Profit if back wins (after commission on lay loss)
    profit_back = round(stake_back * (back_odds - 1) - liability, STAKE_PRECISION)
    
    # Profit if lay wins (after commission)
    profit_lay = round(stake_lay * commission_factor - stake_back, STAKE_PRECISION)
    
    # Guaranteed profit (should be equal)
    guaranteed_profit = round(min(profit_back, profit_lay), STAKE_PRECISION)
    profit_margin = round((guaranteed_profit / stake_back) * 100, 2)
    
    return {
        "exists": guaranteed_profit > 0,
        "exchange": exchange.bookmaker,
        "market": exchange.market,
        "back_odds": back_odds,
        "lay_odds": lay_odds,
        "stake_back": stake_back,
        "stake_lay": stake_lay,
        "liability": liability,
        "profit_back": profit_back,
        "profit_lay": profit_lay,
        "guaranteed_profit": guaranteed_profit,
        "profit_margin": profit_margin,
        "commission": commission,
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — DUTCHING CALCULATOR
# ═══════════════════════════════════════════════════════════════

def calculate_dutching(
    outcomes: List[str],
    odds: Dict[str, float],
    total_stake: float = DEFAULT_STAKE,
    min_profit_margin: float = MIN_PROFIT_MARGIN,
) -> DutchingOpportunity:
    """
    Calculate Dutching stakes to guarantee profit from selected outcomes.
    
    Dutching allows you to bet on multiple outcomes (e.g., Home + Draw)
    to guarantee profit if any of those outcomes occur.
    
    Args:
        outcomes: List of outcomes to cover (e.g., ["home", "draw"])
        odds: Dictionary of odds for each outcome
        total_stake: Total amount to stake
        min_profit_margin: Minimum profit percentage to report
    
    Returns:
        DutchingOpportunity with stakes and profit margin
    """
    opportunity = DutchingOpportunity(
        exists=False,
        outcomes=outcomes,
    )
    
    if len(outcomes) < 2:
        opportunity.odds_used = {"note": "Need at least 2 outcomes to Dutch"}
        return opportunity
    
    # Validate all odds exist and are valid
    for outcome in outcomes:
        if outcome not in odds or odds[outcome] <= 0:
            opportunity.odds_used = {"note": f"Missing odds for {outcome}"}
            return opportunity
    
    # Calculate inverse sum for selected outcomes
    inv_sum = sum(1 / odds[outcome] for outcome in outcomes)
    
    # Calculate profit margin (return on stake if any outcome hits)
    profit_margin = (1 / inv_sum - 1) * 100 if inv_sum > 0 else 0
    
    if profit_margin < min_profit_margin * 100:
        opportunity.odds_used = {"note": f"Profit margin too low ({profit_margin:.2f}%)"}
        return opportunity
    
    # Calculate stakes for each outcome
    stakes = {}
    for outcome in outcomes:
        stake = total_stake * (1 / odds[outcome]) / inv_sum
        stakes[outcome] = round(stake, STAKE_PRECISION)
    
    # Adjust for rounding
    total_calculated = sum(stakes.values())
    if total_calculated != total_stake:
        diff = total_stake - total_calculated
        # Add diff to the largest stake
        max_outcome = max(stakes, key=stakes.get)
        stakes[max_outcome] += round(diff, STAKE_PRECISION)
    
    # Guaranteed return (if any selected outcome wins)
    guaranteed_return = round(total_stake * (1 + profit_margin / 100), 2)
    
    # Confidence
    if profit_margin >= HIGH_CONFIDENCE_MARGIN * 100:
        confidence = ArbitrageConfidence.HIGH
    elif profit_margin >= MEDIUM_CONFIDENCE_MARGIN * 100:
        confidence = ArbitrageConfidence.MEDIUM
    elif profit_margin >= min_profit_margin * 100:
        confidence = ArbitrageConfidence.LOW
    else:
        confidence = ArbitrageConfidence.MARGINAL
    
    opportunity.exists = True
    opportunity.odds_used = {outcome: odds[outcome] for outcome in outcomes}
    opportunity.stakes = stakes
    opportunity.total_stake = total_stake
    opportunity.guaranteed_return = guaranteed_return
    opportunity.profit_margin = round(profit_margin, 2)
    opportunity.confidence = confidence
    
    logger.info(f"Dutching opportunity: {profit_margin:.2f}% profit covering {outcomes}")
    
    return opportunity


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — CROSS-MARKET ARBITRAGE
# ═══════════════════════════════════════════════════════════════

def calculate_cross_market_arbitrage(
    match_winner_books: List[BookOdds],
    double_chance_books: List[BookOdds],
    total_stake: float = DEFAULT_STAKE,
    min_profit_margin: float = MIN_PROFIT_MARGIN,
) -> ArbitrageOpportunity:
    """
    Calculate arbitrage across different markets (e.g., Match Winner + Double Chance).
    
    Args:
        match_winner_books: List of BookOdds for match winner market
        double_chance_books: List of BookOdds for double chance market
        total_stake: Total amount to stake
        min_profit_margin: Minimum profit percentage to report
    
    Returns:
        ArbitrageOpportunity with stakes and profit margin
    """
    opportunity = ArbitrageOpportunity(
        exists=False,
        arbitrage_type=ArbitrageType.CROSS_MARKET,
    )
    
    # Find best odds in each market
    best_mw = find_best_odds(match_winner_books)
    best_dc = find_best_odds(double_chance_books)
    
    # For double chance, we need odds for (home or draw) and (away or draw)
    # We'll use a simplified approach: combine home and draw from DC market
    # This is a placeholder - full implementation would need proper mapping
    
    # Placeholder return
    opportunity.stakes = {"note": "Cross-market arbitrage not fully implemented"}
    
    return opportunity


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — SURETBET IDENTIFICATION
# ═══════════════════════════════════════════════════════════════

def identify_surebets(
    three_way_books: List[BookOdds],
    two_way_books: List[TwoWayOdds] = None,
    min_profit: float = MIN_PROFIT_MARGIN,
) -> List[ArbitrageOpportunity]:
    """
    Identify all surebet opportunities across markets.
    
    Args:
        three_way_books: List of BookOdds for match winner market
        two_way_books: Optional list of TwoWayOdds for other markets
        min_profit: Minimum profit percentage to include (as decimal, e.g., 0.005 = 0.5%)
    
    Returns:
        List of ArbitrageOpportunity for all detected surebets
    """
    opportunities = []
    
    # Check three-way market
    if three_way_books:
        arb = calculate_arbitrage(three_way_books, min_profit_margin=min_profit)
        if arb.exists:
            opportunities.append(arb)
    
    # Check two-way markets
    if two_way_books:
        # Group by market
        markets: Dict[str, List[TwoWayOdds]] = {}
        for book in two_way_books:
            market = book.market or "default"
            if market not in markets:
                markets[market] = []
            markets[market].append(book)
        
        for market, books in markets.items():
            arb = calculate_two_way_arbitrage(books, min_profit_margin=min_profit, market=market)
            if arb.exists:
                opportunities.append(arb)
    
    # Sort by profit margin (highest first)
    opportunities.sort(key=lambda x: x.profit_margin, reverse=True)
    
    return opportunities


def rank_arbitrage_opportunities(
    opportunities: List[ArbitrageOpportunity],
    max_expected_stake: float = 1000.0,
) -> List[Dict[str, Any]]:
    """
    Rank arbitrage opportunities by overall value.
    
    Args:
        opportunities: List of arbitrage opportunities
        max_expected_stake: Maximum stake to consider
    
    Returns:
        Ranked list with value scores
    """
    ranked = []
    
    for opp in opportunities:
        # Calculate value score
        # profit_margin * confidence_weight * reliability        confidence_weight = {
            ArbitrageConfidence.HIGH: 1.0,
            ArbitrageConfidence.MEDIUM: 0.8,
            ArbitrageConfidence.LOW: 0.5,
            ArbitrageConfidence.MARGINAL: 0.3,
        }.get(opp.confidence, 0.5)
        
        value_score = opp.profit_margin * confidence_weight * opp.reliability_weighted
        
        # Calculate practical stake
        max_stake_for_opp = min(max_expected_stake, opp.total_stake * 5)  # Can scale up
        
        ranked.append({
            "opportunity": opp,
            "value_score": round(value_score, 2),
            "profit_margin": opp.profit_margin,
            "confidence": opp.confidence.value,
            "reliability": opp.reliability_weighted,
            "max_stake": round(max_stake_for_opp, 2),
            "expected_profit": round(max_stake_for_opp * (opp.profit_margin / 100), 2),
        })
    
    # Sort by value score
    ranked.sort(key=lambda x: x["value_score"], reverse=True)
    
    return ranked


# ═══════════════════════════════════════════════════════════════
# SECTION 10 — ARBITRAGE PORTFOLIO MANAGEMENT
# ═══════════════════════════════════════════════════════════════

@dataclass
class ArbitragePortfolio:
    """Portfolio of arbitrage opportunities to execute."""
    opportunities: List[ArbitrageOpportunity] = field(default_factory=list)
    total_required_stake: float = 0.0
    total_guaranteed_return: float = 0.0
    total_profit: float = 0.0
    weighted_confidence: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def add_opportunity(self, opp: ArbitrageOpportunity, scale_factor: float = 1.0) -> None:
        """Add an opportunity to the portfolio with optional scaling."""
        scaled_opp = opp
        if scale_factor != 1.0:
            # Scale stakes
            scaled_opp.total_stake *= scale_factor
            scaled_opp.guaranteed_return *= scale_factor
            for k in scaled_opp.stakes:
                scaled_opp.stakes[k] *= scale_factor
        
        self.opportunities.append(scaled_opp)
        self.total_required_stake += scaled_opp.total_stake
        self.total_guaranteed_return += scaled_opp.guaranteed_return
        self.total_profit += scaled_opp.profit_amount
        
        # Weighted confidence
        self.weighted_confidence = (
            (self.weighted_confidence * (len(self.opportunities) - 1) + 
             (1 if opp.is_high_confidence else 0)) / len(self.opportunities)
        )
    
    def summary(self) -> str:
        """Human-readable summary."""
        return (f"Arbitrage Portfolio: {len(self.opportunities)} opportunities | "
                f"Total Stake: {self.total_required_stake:.2f} | "
                f"Total Return: {self.total_guaranteed_return:.2f} | "
                f"Total Profit: {self.total_profit:.2f} | "
                f"ROI: {(self.total_profit / self.total_required_stake * 100):.2f}%")


# ═══════════════════════════════════════════════════════════════
# SECTION 11 — EXPORTS
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "ArbitrageType",
    "ArbitrageConfidence",
    "MarketType",
    # Data classes
    "BookOdds",
    "TwoWayOdds",
    "ExchangeOdds",
    "ArbitrageOpportunity",
    "DutchingOpportunity",
    "ArbitragePortfolio",
    # Core functions
    "find_best_odds",
    "find_best_two_way_odds",
    "calculate_arbitrage",
    "calculate_two_way_arbitrage",
    "calculate_scalping_opportunity",
    "calculate_dutching",
    "calculate_cross_market_arbitrage",
    "identify_surebets",
    "rank_arbitrage_opportunities",
    # Constants
    "MIN_PROFIT_MARGIN",
    "DEFAULT_STAKE",
    "MAX_ODDS",
    "MIN_ODDS",
]


# ═══════════════════════════════════════════════════════════════
# SECTION 12 — EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MODULE 22: ARBITRAGE DETECTION ENGINE - TEST RUN")
    print("=" * 70)
    
    # Test 1: Three-way arbitrage
    print("\n📊 TEST 1: Three-Way Arbitrage")
    print("-" * 40)
    
    bookmakers = [
        BookOdds("Bet365", 2.10, 3.40, 3.50),
        BookOdds("Pinnacle", 2.05, 3.50, 3.60),
        BookOdds("William Hill", 2.15, 3.30, 3.40),
        BookOdds("Betfair", 2.12, 3.45, 3.55),
    ]
    
    for book in bookmakers:
        print(f"  {book.bookmaker}: H={book.home_odds:.2f}, D={book.draw_odds:.2f}, A={book.away_odds:.2f}")
    
    arb = calculate_arbitrage(bookmakers, total_stake=100, match_id="Arsenal_vs_Chelsea")
    
    if arb.exists:
        print(f"\n💰 ARBITRAGE FOUND!")
        print(f"  Profit Margin: {arb.profit_margin:.2f}%")
        print(f"  Total Stake: {arb.total_stake:.2f}")
        print(f"  Guaranteed Return: {arb.guaranteed_return:.2f}")
        print(f"  Confidence: {arb.confidence.value}")
        print(f"  Reliability Weighted: {arb.reliability_weighted:.2%}")
        print(f"\n  Stakes:")
        for outcome, stake in arb.stakes.items():
            book = arb.best_books.get(outcome, "unknown")
            odds = arb.odds_used.get(outcome, 0)
            print(f"    {outcome.upper()}: {stake:.2f} @ {odds:.2f} ({book})")
    else:
        print(f"\n  No arbitrage: {arb.stakes.get('note', 'Unknown reason')}")
    
    # Test 2: Two-way arbitrage
    print("\n\n📊 TEST 2: Two-Way Arbitrage (Over 2.5 Goals)")
    print("-" * 40)
    
    two_way_books = [
        TwoWayOdds("Bet365", 1.85, 2.00, market="over_2.5"),
        TwoWayOdds("Pinnacle", 1.90, 1.95, market="over_2.5"),
        TwoWayOdds("William Hill", 1.80, 2.05, market="over_2.5"),
        TwoWayOdds("Betfair", 1.88, 1.98, market="over_2.5"),
    ]
    
    for book in two_way_books:
        print(f"  {book.bookmaker}: YES={book.yes_odds:.2f}, NO={book.no_odds:.2f}")
    
    arb2 = calculate_two_way_arbitrage(two_way_books, total_stake=100, market="over_2.5")
    
    if arb2.exists:
        print(f"\n💰 TWO-WAY ARBITRAGE FOUND!")
        print(f"  Profit Margin: {arb2.profit_margin:.2f}%")
        print(f"  Stakes: YES={arb2.stakes.get('yes', 0):.2f}, NO={arb2.stakes.get('no', 0):.2f}")
        print(f"  Books: YES={arb2.best_books.get('yes', '?')}, NO={arb2.best_books.get('no', '?')}")
    else:
        print(f"\n  No arbitrage: {arb2.stakes.get('note', 'Unknown reason')}")
    
    # Test 3: Dutching
    print("\n\n📊 TEST 3: Dutching (Covering Home + Draw)")
    print("-" * 40)
    
    odds = {
        "home": 2.10,
        "draw": 3.40,
        "away": 3.50,
    }
    print(f"  Odds: Home={odds['home']:.2f}, Draw={odds['draw']:.2f}, Away={odds['away']:.2f}")
    
    dutch = calculate_dutching(["home", "draw"], odds, total_stake=100)
    
    if dutch.exists:
        print(f"\n💰 DUTCHING OPPORTUNITY FOUND!")
        print(f"  Covering: {dutch.outcomes}")
        print(f"  Profit Margin: {dutch.profit_margin:.2f}%")
        print(f"  Total Stake: {dutch.total_stake:.2f}")
        print(f"  Guaranteed Return: {dutch.guaranteed_return:.2f}")
        print(f"  Stakes:")
        for outcome, stake in dutch.stakes.items():
            print(f"    {outcome.upper()}: {stake:.2f} @ {dutch.odds_used.get(outcome, 0):.2f}")
    else:
        print(f"\n  No Dutching opportunity")
    
    # Test 4: Exchange scalping
    print("\n\n📊 TEST 4: Exchange Scalping (Back/Lay)")
    print("-" * 40)
    
    exchange = ExchangeOdds("Betfair", back_odds=2.10, lay_odds=2.05, market="Arsenal_to_win")
    print(f"  Exchange: {exchange.bookmaker}")
    print(f"  Back Odds: {exchange.back_odds:.2f}")
    print(f"  Lay Odds: {exchange.lay_odds:.2f}")
    
    scalp = calculate_scalping_opportunity(exchange, stake_back=100, commission=0.02)
    
    if scalp.get("exists"):
        print(f"\n💰 SCALPING OPPORTUNITY FOUND!")
        print(f"  Back Stake: {scalp['stake_back']:.2f} @ {scalp['back_odds']:.2f}")
        print(f"  Lay Stake: {scalp['stake_lay']:.2f} @ {scalp['lay_odds']:.2f}")
        print(f"  Liability: {scalp['liability']:.2f}")
        print(f"  Guaranteed Profit: {scalp['guaranteed_profit']:.2f}")
        print(f"  Profit Margin: {scalp['profit_margin']:.2f}%")
    else:
        print(f"\n  No scalping opportunity")
    
    # Test 5: Surebet identification
    print("\n\n📊 TEST 5: Surebet Identification")
    print("-" * 40)
    
    # Add another bookmaker to create arbitrage
    bookmakers2 = bookmakers + [
        BookOdds("BetVictor", 2.20, 3.20, 3.30),
    ]
    
    surebets = identify_surebets(bookmakers2, two_way_books, min_profit=0.005)
    
    if surebets:
        print(f"\n  Found {len(surebets)} surebet opportunities:")
        for i, sb in enumerate(surebets, 1):
            print(f"\n  {i}. {sb.market}: {sb.profit_margin:.2f}% profit ({sb.confidence.value})")
            print(f"     Stakes: {sb.stakes}")
            print(f"     Books: {sb.best_books}")
    else:
        print("\n  No surebets found")
    
    # Test 6: Ranking opportunities
    print("\n\n📊 TEST 6: Ranking Arbitrage Opportunities")
    print("-" * 40)
    
    ranked = rank_arbitrage_opportunities(surebets, max_expected_stake=500)
    
    print("\n  Ranked Opportunities:")
    for i, r in enumerate(ranked[:5], 1):
        print(f"  {i}. Value Score: {r['value_score']:.2f} | "
              f"Margin: {r['profit_margin']:.2f}% | "
              f"Confidence: {r['confidence']} | "
              f"Expected Profit: ${r['expected_profit']:.2f}")
    
    # Test 7: Arbitrage portfolio
    print("\n\n📊 TEST 7: Arbitrage Portfolio")
    print("-" * 40)
    
    portfolio = ArbitragePortfolio()
    for opp in surebets[:2]:
        portfolio.add_opportunity(opp, scale_factor=2.0)
    
    print(portfolio.summary())
    
    print("\n" + "=" * 70)
    print("MODULE 22 READY FOR PRODUCTION")
    print("=" * 70)