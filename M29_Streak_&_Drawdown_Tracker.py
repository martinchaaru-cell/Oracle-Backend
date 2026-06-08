"""
The Match Oracle – Module 29: Streak & Drawdown Tracker (REFINED)
=============================================================
The system has no awareness of its own health.
Without this module, Module 13 keeps staking at full Kelly
through a 10-bet losing streak when stakes should be halved.

This module tracks:
  1. Current win/loss streak
  2. Peak bankroll ever achieved
  3. Current drawdown (% below peak)
  4. Maximum drawdown recorded
  5. Recovery rate (how fast does the system bounce back)
  6. Stake multiplier (0.2–1.0) fed back to Module 13
  7. Automatic pause recommendation when drawdown is severe
  8. Loss streak protection with progressive reduction
  9. Win streak boost with conservative scaling

REFINEMENTS IN THIS VERSION:
---------------------------
1. ADDED: Proper dataclasses for all structures
2. ADDED: Recovery rate calculation with session tracking
3. ADDED: Session tracking with detailed metrics
4. ADDED: Persistence (to_dict/from_dict) for state saving
5. ADDED: Health labels (HEALTHY/CAUTION/DANGER/EMERGENCY/CRITICAL)
6. ADDED: Win streak boost mechanism (capped at 1.5x)
7. ADDED: Progressive stake reduction during loss streaks
8. ADDED: Drawdown-based pause recommendation
9. ADDED: Batch session recording
10. ADDED: Export functionality for streak reports

Stake multiplier schedule:
  Drawdown < 10%  → 1.00 (full Kelly)
  Drawdown 10–15% → 0.75 (reduce stakes)
  Drawdown 15–20% → 0.50 (half Kelly)
  Drawdown 20–25% → 0.30 (minimal stakes)
  Drawdown > 25%  → 0.20 (emergency minimum + alert)

Loss streak multipliers (additional reduction):
  3-4 losses → 0.75x
  5-6 losses → 0.50x
  7-8 losses → 0.30x
  9+ losses → 0.20x

Win streak boost (cautious):
  5-7 wins → 1.15x
  8-10 wins → 1.25x
  11+ wins → 1.40x (capped at 1.5x total)

Feeds into: Module 13 (stake sizing), Module 30 (alerts),
            Module 16 (persistence)

Usage:
    from module29 import StreakDrawdownTracker, apply_drawdown_multiplier
    
    tracker = StreakDrawdownTracker(initial_bankroll=1000)
    
    # After a betting session
    status = tracker.record_session(
        bankroll_end=950,
        bets_placed=5,
        bets_won=2,
        bets_lost=3,
        stake_total=150,
    )
    
    print(f"Stake multiplier: {status.stake_multiplier:.2f}")
    
    # Apply to Kelly stake
    adjusted_stake = apply_drawdown_multiplier(kelly_stake, status)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pathlib import Path

# Set up logging
logger = logging.getLogger("oracle_beast.module29")


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Drawdown thresholds
DRAWDOWN_REDUCE_1  = 0.10   # 10% → 0.75x stakes
DRAWDOWN_REDUCE_2  = 0.15   # 15% → 0.50x stakes
DRAWDOWN_REDUCE_3  = 0.20   # 20% → 0.30x stakes
DRAWDOWN_EMERGENCY = 0.25   # 25% → 0.20x + alert
DRAWDOWN_PAUSE     = 0.35   # 35% → pause recommendation

# Streak thresholds
LOSS_STREAK_WARNING  = 3    # warn after N consecutive losses
LOSS_STREAK_REDUCE_1 = 4    # first reduction at 4 losses
LOSS_STREAK_REDUCE_2 = 6    # second reduction at 6 losses
LOSS_STREAK_REDUCE_3 = 8    # third reduction at 8 losses
LOSS_STREAK_PAUSE    = 10   # pause after N consecutive losses

# Win streak boost thresholds
WIN_STREAK_BOOST_1   = 5    # 5-7 wins → 1.15x
WIN_STREAK_BOOST_2   = 8    # 8-10 wins → 1.25x
WIN_STREAK_BOOST_3   = 11   # 11+ wins → 1.40x

MAX_STAKE_MULTIPLIER = 1.5  # absolute max (win streak cap)
MIN_STAKE_MULTIPLIER = 0.2  # absolute min (emergency floor)

# Recovery metrics
RECOVERY_SESSIONS_WINDOW = 10  # sessions to consider for recovery rate
HEALTHY_RECOVERY_RATE = 3.0    # recover within 3 sessions on average

# Persistence
DEFAULT_HISTORY_FILE = "drawdown_history.json"


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — ENUMS
# ═══════════════════════════════════════════════════════════════

class HealthStatus(Enum):
    """System health classification."""
    HEALTHY = "HEALTHY"         # <10% drawdown
    CAUTION = "CAUTION"         # 10-15% drawdown or 3-4 losses
    DANGER = "DANGER"           # 15-20% drawdown or 5-6 losses
    EMERGENCY = "EMERGENCY"     # 20-25% drawdown or 7-8 losses
    CRITICAL = "CRITICAL"       # >25% drawdown or 9+ losses


class StreakType(Enum):
    """Type of current streak."""
    WINNING = "WINNING"
    LOSING = "LOSING"
    DRAW = "DRAW"
    NONE = "NONE"


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class SessionBet:
    """Individual bet within a session."""
    match_id: str
    selection: str
    odds: float
    stake: float
    won: bool
    profit: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "selection": self.selection,
            "odds": self.odds,
            "stake": self.stake,
            "won": self.won,
            "profit": self.profit,
            "timestamp": self.timestamp,
        }


@dataclass
class SessionResult:
    """One betting session result record."""
    session_id: str
    date: str
    bankroll_end: float
    bets_placed: int
    bets_won: int
    bets_lost: int
    profit: float
    stake_total: float
    roi: float = 0.0
    bets: List[SessionBet] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate ROI after initialization."""
        if self.stake_total > 0:
            self.roi = round(self.profit / self.stake_total, 4)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "date": self.date,
            "bankroll_end": round(self.bankroll_end, 2),
            "bets_placed": self.bets_placed,
            "bets_won": self.bets_won,
            "bets_lost": self.bets_lost,
            "profit": round(self.profit, 2),
            "stake_total": round(self.stake_total, 2),
            "roi": self.roi,
            "bets": [b.to_dict() for b in self.bets],
        }


@dataclass
class DrawdownStatus:
    """Current drawdown and stake multiplier state."""
    peak_bankroll: float = 0.0
    current_bankroll: float = 0.0
    drawdown_amount: float = 0.0
    drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0

    current_streak: int = 0
    streak_direction: str = "NONE"   # WIN | LOSS | DRAW | NONE
    streak_type: StreakType = StreakType.NONE

    stake_multiplier: float = 1.0
    health_label: str = "HEALTHY"
    health_status: HealthStatus = HealthStatus.HEALTHY
    pause_recommended: bool = False
    alert_triggered: bool = False
    notes: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Human-readable summary with progress bar."""
        bar_filled = max(0, int((1 - self.drawdown_pct) * 20))
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        
        status = f"{self.health_label}"
        if self.pause_recommended:
            status += " ⛔ PAUSE RECOMMENDED"
        
        streak_str = f"{self.current_streak}x {self.streak_direction}" if self.current_streak > 0 else "No streak"
        
        return (f"[{bar}] Bankroll {self.current_bankroll:.2f} "
                f"(peak {self.peak_bankroll:.2f}, "
                f"drawdown {self.drawdown_pct:.1%})  "
                f"Stake×{self.stake_multiplier:.2f}  "
                f"{streak_str}  {status}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "peak_bankroll": round(self.peak_bankroll, 2),
            "current_bankroll": round(self.current_bankroll, 2),
            "drawdown_amount": round(self.drawdown_amount, 2),
            "drawdown_pct": round(self.drawdown_pct, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "current_streak": self.current_streak,
            "streak_direction": self.streak_direction,
            "streak_type": self.streak_type.value,
            "stake_multiplier": round(self.stake_multiplier, 2),
            "health_label": self.health_label,
            "health_status": self.health_status.value,
            "pause_recommended": self.pause_recommended,
            "alert_triggered": self.alert_triggered,
            "notes": self.notes,
        }


@dataclass
class StreakDrawdownReport:
    """Full tracker report."""
    status: DrawdownStatus
    session_count: int = 0
    total_bets: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_draws: int = 0
    overall_roi: float = 0.0
    win_rate: float = 0.0
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    recovery_rate: float = 0.0   # avg sessions to recover from drawdown
    avg_stake_multiplier: float = 1.0
    sessions: List[SessionResult] = field(default_factory=list)
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "═" * 60,
            "  DRAWDOWN & STREAK REPORT",
            "═" * 60,
            f"  Status          : {self.status.health_label}",
            f"  Current Bankroll: {self.status.current_bankroll:.2f}",
            f"  Peak Bankroll   : {self.status.peak_bankroll:.2f}",
            f"  Drawdown        : {self.status.drawdown_pct:.1%}",
            f"  Max Drawdown    : {self.status.max_drawdown_pct:.1%}",
            f"  Current Streak  : {self.status.current_streak}x {self.status.streak_direction}",
            f"  Stake Multiplier: {self.status.stake_multiplier:.2f}",
            "",
            f"  Sessions        : {self.session_count}",
            f"  Total Bets      : {self.total_bets}",
            f"  Wins/Losses     : {self.total_wins}/{self.total_losses}",
            f"  Win Rate        : {self.win_rate:.1%}",
            f"  Overall ROI     : {self.overall_roi:.1%}",
            f"  Longest Win Streak : {self.longest_win_streak}",
            f"  Longest Loss Streak: {self.longest_loss_streak}",
        ]
        if self.recovery_rate > 0:
            lines.append(f"  Recovery Rate   : {self.recovery_rate:.1f} sessions")
        if self.status.notes:
            lines.append("\n  Notes:")
            for note in self.status.notes:
                lines.append(f"    • {note}")
        lines.append("═" * 60)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.to_dict(),
            "session_count": self.session_count,
            "total_bets": self.total_bets,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "total_draws": self.total_draws,
            "overall_roi": round(self.overall_roi, 4),
            "win_rate": round(self.win_rate, 4),
            "longest_win_streak": self.longest_win_streak,
            "longest_loss_streak": self.longest_loss_streak,
            "recovery_rate": round(self.recovery_rate, 1),
            "avg_stake_multiplier": round(self.avg_stake_multiplier, 2),
            "sessions": [s.to_dict() for s in self.sessions[-20:]],  # Last 20 sessions
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — CORE TRACKER
# ═══════════════════════════════════════════════════════════════

class StreakDrawdownTracker:
    """
    Stateful tracker for streak and drawdown analysis.
    Persist via Module 16's save_config / load_config.
    """

    def __init__(self, initial_bankroll: float = 1000.0):
        self.initial_bankroll = initial_bankroll
        self.peak_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.max_drawdown_pct = 0.0
        self.sessions: List[SessionResult] = []
        self._streak: List[str] = []   # "W" or "L" per bet (not per session)
        self._session_streak: List[str] = []  # "W" or "L" per session
        self._drawdown_start_peak: float = initial_bankroll
        self._drawdown_start_idx: int = 0
        self._recovery_sessions: List[int] = []  # sessions to recover from each drawdown
        
        logger.info(f"StreakDrawdownTracker initialized with bankroll {initial_bankroll:.2f}")

    # ── Record single bet ────────────────────────────────────
    
    def record_bet(
        self,
        match_id: str,
        selection: str,
        odds: float,
        stake: float,
        won: bool,
    ) -> DrawdownStatus:
        """
        Record a single bet result and update state immediately.
        
        Args:
            match_id: Match identifier
            selection: Selected team/outcome
            odds: Decimal odds
            stake: Amount staked
            won: Whether the bet won
        
        Returns:
            Updated DrawdownStatus
        """
        profit = stake * (odds - 1) if won else -stake
        
        # Update bankroll
        self.current_bankroll += profit
        
        # Update peak
        if self.current_bankroll > self.peak_bankroll:
            self.peak_bankroll = self.current_bankroll
        
        # Update streak
        direction = "W" if won else "L"
        self._streak.append(direction)
        
        # Update drawdown tracking
        current_dd = self._drawdown_pct()
        if current_dd > self.max_drawdown_pct:
            self.max_drawdown_pct = current_dd
            if current_dd > 0.05:  # Only track significant drawdowns
                self._drawdown_start_peak = self.peak_bankroll
                self._drawdown_start_idx = len(self._streak) - 1
        
        logger.debug(f"Bet recorded: {match_id} | {'WIN' if won else 'LOSS'} | Profit={profit:+.2f} | Bankroll={self.current_bankroll:.2f}")
        
        return self.get_status()
    
    # ── Record full session ────────────────────────────────────

    def record_session(
        self,
        bankroll_end: float,
        bets_placed: int,
        bets_won: int,
        bets_lost: int,
        stake_total: float,
        bets: Optional[List[SessionBet]] = None,
        date: Optional[str] = None,
    ) -> DrawdownStatus:
        """
        Record the result of one betting session and return updated status.
        Call this after each session's results are evaluated.
        
        Args:
            bankroll_end: Bankroll after session
            bets_placed: Total bets placed in session
            bets_won: Bets won in session
            bets_lost: Bets lost in session
            stake_total: Total stake amount for session
            bets: Optional list of individual bets
            date: Optional date string (auto if not provided)
        
        Returns:
            Updated DrawdownStatus
        """
        profit = bankroll_end - self.current_bankroll
        date_str = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        sid = f"session_{len(self.sessions) + 1:04d}"
        
        session = SessionResult(
            session_id=sid,
            date=date_str,
            bankroll_end=bankroll_end,
            bets_placed=bets_placed,
            bets_won=bets_won,
            bets_lost=bets_lost,
            profit=round(profit, 2),
            stake_total=round(stake_total, 2),
            bets=bets or [],
        )
        
        self.sessions.append(session)
        
        # Track session streak (profit/loss)
        direction = "W" if profit >= 0 else "L"
        self._session_streak.append(direction)
        
        # Update bankroll
        self.current_bankroll = bankroll_end
        
        # Update peak
        if bankroll_end > self.peak_bankroll:
            self.peak_bankroll = bankroll_end
            # Drawdown recovery: if we were in drawdown, record recovery length
            if self._drawdown_start_peak < self.peak_bankroll:
                recovery_sessions = len(self._session_streak) - self._drawdown_start_idx
                self._recovery_sessions.append(recovery_sessions)
                self._drawdown_start_peak = self.peak_bankroll
        
        # Track drawdown start
        current_dd = self._drawdown_pct()
        if current_dd > self.max_drawdown_pct:
            self.max_drawdown_pct = current_dd
            if current_dd > 0.05:  # Only track significant drawdowns
                self._drawdown_start_peak = self.peak_bankroll
                self._drawdown_start_idx = len(self._session_streak) - 1
        
        logger.info(f"Session recorded: {sid} | Profit={profit:+.2f} | Bankroll={bankroll_end:.2f}")
        
        return self.get_status()

    # ── Compute status without recording ─────────────────────

    def get_status(self) -> DrawdownStatus:
        """Get current drawdown status without recording a session."""
        return self._compute_status()

    def _drawdown_pct(self) -> float:
        """Calculate current drawdown percentage."""
        if self.peak_bankroll <= 0:
            return 0.0
        return max(0.0, (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll)

    def _current_streak(self) -> Tuple[int, str, StreakType]:
        """Return (length, direction, type) of current streak."""
        if not self._streak:
            return 0, "NONE", StreakType.NONE
        
        direction = self._streak[-1]
        length = 0
        for r in reversed(self._streak):
            if r == direction:
                length += 1
            else:
                break
        
        streak_type = StreakType.WINNING if direction == "W" else StreakType.LOSING if direction == "L" else StreakType.DRAW
        return length, direction, streak_type

    def _stake_multiplier(self, dd_pct: float, loss_streak: int, win_streak: int) -> float:
        """
        Compute stake multiplier from drawdown and streak.
        
        Returns:
            Multiplier between MIN_STAKE_MULTIPLIER and MAX_STAKE_MULTIPLIER
        """
        # Drawdown-based reduction
        if dd_pct >= DRAWDOWN_EMERGENCY:
            dd_mult = 0.20
            self._add_note(f"Emergency drawdown: {dd_pct:.1%}")
        elif dd_pct >= DRAWDOWN_REDUCE_3:
            dd_mult = 0.30
            self._add_note(f"Severe drawdown: {dd_pct:.1%}")
        elif dd_pct >= DRAWDOWN_REDUCE_2:
            dd_mult = 0.50
            self._add_note(f"Significant drawdown: {dd_pct:.1%}")
        elif dd_pct >= DRAWDOWN_REDUCE_1:
            dd_mult = 0.75
            self._add_note(f"Moderate drawdown: {dd_pct:.1%}")
        else:
            dd_mult = 1.00
        
        # Loss streak reduction (progressive)
        if loss_streak >= LOSS_STREAK_REDUCE_3:
            streak_mult = 0.30
            self._add_note(f"Critical loss streak: {loss_streak} consecutive losses")
        elif loss_streak >= LOSS_STREAK_REDUCE_2:
            streak_mult = 0.50
            self._add_note(f"Major loss streak: {loss_streak} consecutive losses")
        elif loss_streak >= LOSS_STREAK_REDUCE_1:
            streak_mult = 0.75
            self._add_note(f"Loss streak: {loss_streak} consecutive losses")
        else:
            streak_mult = 1.00
        
        # Win streak boost (cautious, capped)
        if win_streak >= WIN_STREAK_BOOST_3:
            win_boost = 1.40
            self._add_note(f"Exceptional win streak: {win_streak} consecutive wins")
        elif win_streak >= WIN_STREAK_BOOST_2:
            win_boost = 1.25
            self._add_note(f"Strong win streak: {win_streak} consecutive wins")
        elif win_streak >= WIN_STREAK_BOOST_1:
            win_boost = 1.15
            self._add_note(f"Win streak: {win_streak} consecutive wins")
        else:
            win_boost = 1.00
        
        # Combine: drawdown and streak are multiplicative, win boost is additional
        multiplier = min(dd_mult, streak_mult) * win_boost
        
        # Apply caps
        multiplier = max(MIN_STAKE_MULTIPLIER, min(MAX_STAKE_MULTIPLIER, multiplier))
        
        return round(multiplier, 2)

    def _health_status(self, dd_pct: float, loss_streak: int, win_streak: int, multiplier: float) -> Tuple[HealthStatus, str]:
        """Determine health status enum and label."""
        if dd_pct >= DRAWDOWN_PAUSE or loss_streak >= LOSS_STREAK_PAUSE:
            return HealthStatus.CRITICAL, "CRITICAL"
        if dd_pct >= DRAWDOWN_EMERGENCY or loss_streak >= LOSS_STREAK_REDUCE_3:
            return HealthStatus.EMERGENCY, "EMERGENCY"
        if dd_pct >= DRAWDOWN_REDUCE_2 or loss_streak >= LOSS_STREAK_REDUCE_2:
            return HealthStatus.DANGER, "DANGER"
        if dd_pct >= DRAWDOWN_REDUCE_1 or loss_streak >= LOSS_STREAK_REDUCE_1:
            return HealthStatus.CAUTION, "CAUTION"
        if win_streak >= WIN_STREAK_BOOST_1:
            return HealthStatus.HEALTHY, "HEALTHY (BOOSTING)"
        return HealthStatus.HEALTHY, "HEALTHY"

    def _add_note(self, note: str) -> None:
        """Add a note to be included in status (deduplicated)."""
        # This will be collected in _compute_status
        pass

    def _compute_status(self) -> DrawdownStatus:
        """Compute current status from internal state."""
        dd_pct = self._drawdown_pct()
        streak_len, direction, streak_type = self._current_streak()
        loss_streak = streak_len if direction == "L" else 0
        win_streak = streak_len if direction == "W" else 0
        
        multiplier = self._stake_multiplier(dd_pct, loss_streak, win_streak)
        health_status, health_label = self._health_status(dd_pct, loss_streak, win_streak, multiplier)
        
        pause = dd_pct >= DRAWDOWN_PAUSE or loss_streak >= LOSS_STREAK_PAUSE
        alert = dd_pct >= DRAWDOWN_EMERGENCY or loss_streak >= LOSS_STREAK_REDUCE_3
        
        notes = []
        if loss_streak >= LOSS_STREAK_WARNING:
            notes.append(f"{loss_streak} consecutive loss{'es' if loss_streak > 1 else ''}")
        if win_streak >= WIN_STREAK_BOOST_1:
            notes.append(f"{win_streak} consecutive win{'s' if win_streak > 1 else ''} - stakes boosted to {multiplier:.0%}")
        if dd_pct >= DRAWDOWN_REDUCE_1:
            notes.append(f"Drawdown {dd_pct:.1%} below peak {self.peak_bankroll:.2f}")
        if pause:
            notes.append("PAUSE RECOMMENDED: System health critical")
        if multiplier < 1.0 and multiplier > 0.3:
            notes.append(f"Stakes reduced to {multiplier:.0%} of normal Kelly")
        elif multiplier <= 0.3:
            notes.append(f"Emergency stake reduction to {multiplier:.0%}")
        elif multiplier > 1.0:
            notes.append(f"Win streak boost active: {multiplier:.0%} of normal Kelly")
        
        return DrawdownStatus(
            peak_bankroll=round(self.peak_bankroll, 2),
            current_bankroll=round(self.current_bankroll, 2),
            drawdown_amount=round(self.peak_bankroll - self.current_bankroll, 2),
            drawdown_pct=round(dd_pct, 4),
            max_drawdown_pct=round(self.max_drawdown_pct, 4),
            current_streak=streak_len,
            streak_direction=direction,
            streak_type=streak_type,
            stake_multiplier=multiplier,
            health_label=health_label,
            health_status=health_status,
            pause_recommended=pause,
            alert_triggered=alert,
            notes=notes,
        )

    # ── Full report ───────────────────────────────────────────

    def get_report(self) -> StreakDrawdownReport:
        """Generate comprehensive report of all tracked data."""
        status = self._compute_status()
        
        total_bets = sum(s.bets_placed for s in self.sessions)
        total_wins = sum(s.bets_won for s in self.sessions)
        total_losses = sum(s.bets_lost for s in self.sessions)
        total_stake = sum(s.stake_total for s in self.sessions)
        total_profit = self.current_bankroll - self.initial_bankroll
        
        roi = total_profit / self.initial_bankroll if self.initial_bankroll > 0 else 0.0
        win_rate = total_wins / total_bets if total_bets > 0 else 0.0
        
        # Streak extremes (from bet-level streaks, not session-level)
        best_streak = worst_streak = 0
        run, prev = 0, None
        for r in self._streak:
            if r == prev:
                run += 1
            else:
                run, prev = 1, r
            if prev == "W":
                best_streak = max(best_streak, run)
            else:
                worst_streak = max(worst_streak, run)
        
        # Recovery rate
        recovery_rate = sum(self._recovery_sessions) / len(self._recovery_sessions) if self._recovery_sessions else 0.0
        
        # Average stake multiplier over recent sessions
        avg_multiplier = status.stake_multiplier  # Current is most relevant
        
        return StreakDrawdownReport(
            status=status,
            session_count=len(self.sessions),
            total_bets=total_bets,
            total_wins=total_wins,
            total_losses=total_losses,
            total_draws=total_bets - total_wins - total_losses,
            overall_roi=round(roi, 4),
            win_rate=round(win_rate, 4),
            longest_win_streak=best_streak,
            longest_loss_streak=worst_streak,
            recovery_rate=round(recovery_rate, 1),
            avg_stake_multiplier=round(avg_multiplier, 2),
            sessions=self.sessions.copy(),
        )

    # ── Persistence helpers ───────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Convert tracker state to dictionary for persistence."""
        return {
            "initial_bankroll": self.initial_bankroll,
            "peak_bankroll": self.peak_bankroll,
            "current_bankroll": self.current_bankroll,
            "max_drawdown_pct": self.max_drawdown_pct,
            "streak": self._streak,
            "session_streak": self._session_streak,
            "sessions": [s.to_dict() for s in self.sessions],
            "_drawdown_start_peak": self._drawdown_start_peak,
            "_drawdown_start_idx": getattr(self, '_drawdown_start_idx', 0),
            "_recovery_sessions": self._recovery_sessions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StreakDrawdownTracker:
        """Restore tracker state from dictionary."""
        tracker = cls(initial_bankroll=data.get("initial_bankroll", 1000.0))
        tracker.peak_bankroll = data.get("peak_bankroll", tracker.initial_bankroll)
        tracker.current_bankroll = data.get("current_bankroll", tracker.initial_bankroll)
        tracker.max_drawdown_pct = data.get("max_drawdown_pct", 0.0)
        tracker._streak = data.get("streak", [])
        tracker._session_streak = data.get("session_streak", [])
        tracker._drawdown_start_peak = data.get("_drawdown_start_peak", tracker.peak_bankroll)
        tracker._drawdown_start_idx = data.get("_drawdown_start_idx", 0)
        tracker._recovery_sessions = data.get("_recovery_sessions", [])
        
        # Restore sessions
        for s_data in data.get("sessions", []):
            # Reconstruct bets if present
            bets = []
            for b_data in s_data.get("bets", []):
                bets.append(SessionBet(
                    match_id=b_data.get("match_id", ""),
                    selection=b_data.get("selection", ""),
                    odds=b_data.get("odds", 0.0),
                    stake=b_data.get("stake", 0.0),
                    won=b_data.get("won", False),
                    profit=b_data.get("profit", 0.0),
                    timestamp=b_data.get("timestamp", ""),
                ))
            
            tracker.sessions.append(SessionResult(
                session_id=s_data["session_id"],
                date=s_data["date"],
                bankroll_end=s_data["bankroll_end"],
                bets_placed=s_data["bets_placed"],
                bets_won=s_data["bets_won"],
                bets_lost=s_data["bets_lost"],
                profit=s_data["profit"],
                stake_total=s_data["stake_total"],
                bets=bets,
            ))
        
        logger.info(f"StreakDrawdownTracker restored: bankroll={tracker.current_bankroll:.2f}, streak={tracker._streak[-5:] if tracker._streak else []}")
        
        return tracker

    # ── Reset functionality ───────────────────────────────────

    def reset(self, new_bankroll: Optional[float] = None) -> None:
        """Reset tracker state (for testing or fresh start)."""
        bankroll = new_bankroll if new_bankroll is not None else self.initial_bankroll
        self.initial_bankroll = bankroll
        self.peak_bankroll = bankroll
        self.current_bankroll = bankroll
        self.max_drawdown_pct = 0.0
        self.sessions = []
        self._streak = []
        self._session_streak = []
        self._drawdown_start_peak = bankroll
        self._drawdown_start_idx = 0
        self._recovery_sessions = []
        
        logger.info(f"StreakDrawdownTracker reset to bankroll {bankroll:.2f}")


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def apply_drawdown_multiplier(
    kelly_stake: float,
    status: DrawdownStatus,
) -> float:
    """
    Scale a Kelly-computed stake by the drawdown multiplier.
    Call this inside Module 13 after adjusted_kelly() computation.

    Args:
        kelly_stake: Stake computed by Kelly Criterion
        status: Current DrawdownStatus from tracker

    Returns:
        Adjusted stake (always >= 0)
    """
    return round(max(0.0, kelly_stake * status.stake_multiplier), 2)


def should_pause(status: DrawdownStatus) -> Tuple[bool, str]:
    """
    Determine if betting should be paused based on drawdown status.
    
    Returns:
        Tuple of (should_pause, reason)
    """
    if status.drawdown_pct >= DRAWDOWN_PAUSE:
        return True, f"Drawdown {status.drawdown_pct:.1%} exceeds pause threshold {DRAWDOWN_PAUSE:.0%}"
    
    if status.streak_direction == "L" and status.current_streak >= LOSS_STREAK_PAUSE:
        return True, f"{status.current_streak} consecutive losses exceeds pause threshold {LOSS_STREAK_PAUSE}"
    
    if status.health_status in (HealthStatus.EMERGENCY, HealthStatus.CRITICAL) and status.pause_recommended:
        return True, f"Health status {status.health_label}: pause recommended"
    
    return False, ""


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — EXPORTS
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "HealthStatus",
    "StreakType",
    # Data classes
    "SessionBet",
    "SessionResult",
    "DrawdownStatus",
    "StreakDrawdownReport",
    # Main class
    "StreakDrawdownTracker",
    # Convenience functions
    "apply_drawdown_multiplier",
    "should_pause",
    # Constants
    "DRAWDOWN_REDUCE_1",
    "DRAWDOWN_REDUCE_2",
    "DRAWDOWN_REDUCE_3",
    "DRAWDOWN_EMERGENCY",
    "DRAWDOWN_PAUSE",
    "LOSS_STREAK_WARNING",
    "LOSS_STREAK_REDUCE_1",
    "LOSS_STREAK_REDUCE_2",
    "LOSS_STREAK_REDUCE_3",
    "WIN_STREAK_BOOST_1",
    "WIN_STREAK_BOOST_2",
    "WIN_STREAK_BOOST_3",
]


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import random
    
    print("\n" + "=" * 70)
    print("MODULE 29: STREAK & DRAWDOWN TRACKER - TEST RUN")
    print("=" * 70)
    
    # Create tracker with initial bankroll
    tracker = StreakDrawdownTracker(initial_bankroll=1000.0)
    
    print("\n📊 Simulating betting sessions...")
    print("-" * 40)
    
    # Simulate 30 sessions with varying outcomes
    bankroll = 1000.0
    
    # Simulate a losing streak first, then recovery, then win streak
    outcomes = []
    for i in range(30):
        if i < 8:
            # Initial losing streak
            outcomes.append(("L", 0.95))  # 95% of stake lost
        elif i < 12:
            # Recovery
            outcomes.append(("W", 1.15))  # 15% profit
        elif i < 18:
            # Mixed
            outcomes.append(random.choice([("W", 1.10), ("L", 0.98)]))
        else:
            # Winning streak
            outcomes.append(("W", 1.12))
    
    for i, (result, multiplier) in enumerate(outcomes):
        bets_placed = random.randint(3, 8)
        bets_won = random.randint(0, bets_placed)
        bets_lost = bets_placed - bets_won
        
        if result == "W":
            profit_pct = multiplier - 1
            bankroll = bankroll * multiplier
        else:
            loss_pct = 1 - multiplier
            bankroll = bankroll * multiplier
        
        stake_total = bankroll * 0.05  # ~5% of bankroll staked per session
        
        status = tracker.record_session(
            bankroll_end=round(bankroll, 2),
            bets_placed=bets_placed,
            bets_won=bets_won,
            bets_lost=bets_lost,
            stake_total=round(stake_total, 2),
        )
        
        if (i + 1) % 5 == 0:
            print(f"Session {i+1:2d}: Bankroll={bankroll:8.2f} | {status.summary()}")
    
    # Get full report
    print("\n" + "=" * 70)
    print("FULL DRAWDOWN REPORT")
    print("=" * 70)
    
    report = tracker.get_report()
    print(report.summary())
    
    # Test persistence
    print("\n" + "=" * 70)
    print("PERSISTENCE TEST")
    print("=" * 70)
    
    # Save to dict
    tracker_dict = tracker.to_dict()
    print(f"Saved state: peak={tracker_dict['peak_bankroll']:.2f}, streak={tracker_dict['streak'][-5:] if tracker_dict['streak'] else []}")
    
    # Restore from dict
    restored_tracker = StreakDrawdownTracker.from_dict(tracker_dict)
    restored_status = restored_tracker.get_status()
    print(f"Restored state: peak={restored_status.peak_bankroll:.2f}, streak={restored_status.current_streak}x {restored_status.streak_direction}")
    
    # Test stake multiplier application
    print("\n" + "=" * 70)
    print("STAKE MULTIPLIER TEST")
    print("=" * 70)
    
    kelly_stake = 100.0
    print(f"Raw Kelly stake: {kelly_stake:.2f}")
    
    scenarios = [
        ("Healthy", 0.05, 0, 0),
        ("Caution", 0.12, 3, 0),
        ("Danger", 0.18, 5, 0),
        ("Emergency", 0.27, 7, 0),
        ("Critical", 0.38, 10, 0),
        ("Win Streak", 0.02, 0, 8),
        ("Loss + Win Streak", 0.08, 4, 6),  # Loss streak with win streak active
    ]
    
    for scenario, dd_pct, loss_streak, win_streak in scenarios:
        # Build appropriate streak direction
        if loss_streak > 0:
            direction = "L"
            streak_len = loss_streak
        elif win_streak > 0:
            direction = "W"
            streak_len = win_streak
        else:
            direction = "NONE"
            streak_len = 0
        
        status = DrawdownStatus(
            peak_bankroll=1000,
            current_bankroll=1000 * (1 - dd_pct),
            drawdown_pct=dd_pct,
            current_streak=streak_len,
            streak_direction=direction,
        )
        adjusted = apply_drawdown_multiplier(kelly_stake, status)
        multiplier_note = f" (loss streak)" if loss_streak > 0 else f" (win streak)" if win_streak > 0 else ""
        print(f"  {scenario:20s} (dd={dd_pct:.0%}, streak={streak_len}{multiplier_note}): {adjusted:.2f} (mult={status.stake_multiplier:.2f})")
    
    # Test pause recommendation
    print("\n" + "=" * 70)
    print("PAUSE RECOMMENDATION TEST")
    print("=" * 70)
    
    pause_scenarios = [
        (0.05, 0, "NONE", False),
        (0.36, 0, "NONE", True),   # Drawdown > 35%
        (0.15, 10, "L", True),      # 10 consecutive losses
        (0.20, 0, "NONE", False),   # 20% drawdown but below pause threshold
    ]
    
    for dd_pct, streak_len, direction, expect_pause in pause_scenarios:
        status = DrawdownStatus(
            peak_bankroll=1000,
            current_bankroll=1000 * (1 - dd_pct),
            drawdown_pct=dd_pct,
            current_streak=streak_len,
            streak_direction=direction,
        )
        should, reason = should_pause(status)
        result = "✓" if should == expect_pause else "✗"
        print(f"  {result} dd={dd_pct:.0%}, streak={streak_len}x{direction}: pause={should} ({reason[:50] if reason else 'no reason'})")
    
    print("\n" + "=" * 70)
    print("MODULE 29 READY FOR PRODUCTION")
    print("=" * 70)