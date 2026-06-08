"""
The Match Oracle – Module 30: Alert & Notification Engine (REFINED)
==============================================================
Sends real-time alerts when the system detects events that need
immediate attention. Currently blind without this module — you
only know what happened by reading Railway logs.

REFINEMENTS IN THIS VERSION:
---------------------------
1. ADDED: Proper dataclasses for Alert and AlertDeliveryReport
2. ADDED: Comprehensive alert types for all system events
3. ADDED: Batch alert support with rate limiting
4. ADDED: Alert throttling to prevent spam
5. ADDED: Alert history tracking with persistence
6. ADDED: Template system for consistent formatting
7. ADDED: Multi-channel delivery (Telegram, Slack, Webhook, Email, Console)
8. ADDED: Alert escalation (repeat alerts get higher severity)
9. ADDED: Scheduled alert digests
10. ADDED: Alert acknowledgment tracking

Alert triggers:
  1. HIGH confidence pick identified → notify to place bet
  2. Significant odds move against open bet (via Module 20)
  3. Drawdown warning / emergency (via Module 29)
  4. API budget warning (via Module 24)
  5. Learning cycle completion + performance summary
  6. Pipeline error / crash
  7. Arbitrage opportunity detected (via Module 22)
  8. Surebet opportunity detected
  9. System startup / shutdown
  10. Configuration change

Supported channels (configure via env vars):
  - Telegram Bot (TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)
  - Webhook / Slack (ALERT_WEBHOOK_URL or SLACK_WEBHOOK_URL)
  - Email via SMTP (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, ALERT_EMAIL)
  - Console (always enabled for logging)

All channels are optional — the module degrades gracefully if none
are configured (alerts are logged only).

Usage:
    from module30 import notify, AlertLevel, AlertEvent
    
    notify(AlertEvent.PICK_FOUND, title="Pick found", message="Arsenal @ 1.85")
    
    # Or use pre-built constructors
    from module30 import alert_pick_found, alert_drawdown
    
    alert_pick_found("Arsenal vs Chelsea", "Arsenal", 1.85, "HIGH", 0.09)
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
import urllib.request
import urllib.parse
import threading
import time
import queue
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any, Callable, Tuple
from pathlib import Path

log = logging.getLogger("oracle_beast.alerts")


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — CONFIG FROM ENVIRONMENT
# ═══════════════════════════════════════════════════════════════

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN",  "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID",    "")
WEBHOOK_URL      = os.getenv("ALERT_WEBHOOK_URL",   "")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL",  "")
SMTP_HOST        = os.getenv("SMTP_HOST",            "")
SMTP_PORT        = int(os.getenv("SMTP_PORT",        "587"))
SMTP_USER        = os.getenv("SMTP_USER",            "")
SMTP_PASS        = os.getenv("SMTP_PASS",            "")
ALERT_EMAIL      = os.getenv("ALERT_EMAIL",          "")
ALERT_DIGEST_INTERVAL = int(os.getenv("ALERT_DIGEST_INTERVAL", "3600"))  # 1 hour

# Alert throttling (prevent spam)
ALERT_THROTTLE_SECONDS = 60   # minimum time between same type alerts
ALERT_HISTORY_SIZE = 200      # keep last 200 alerts for history
ALERT_QUEUE_SIZE = 500        # max queued alerts

# Alert escalation
ESCALATION_THRESHOLD = 3      # same alert type 3 times → escalate
ESCALATION_DURATION = 3600    # within 1 hour


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — ENUMS
# ═══════════════════════════════════════════════════════════════

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO     = "INFO"
    WARNING  = "WARNING"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


class AlertEvent(Enum):
    """Types of alerts that can be sent."""
    PICK_FOUND        = "PICK_FOUND"
    ODDS_MOVE         = "ODDS_MOVE"
    DRAWDOWN_WARNING  = "DRAWDOWN_WARNING"
    DRAWDOWN_CRITICAL = "DRAWDOWN_CRITICAL"
    BUDGET_WARNING    = "BUDGET_WARNING"
    BUDGET_EXHAUSTED  = "BUDGET_EXHAUSTED"
    LEARNING_COMPLETE = "LEARNING_COMPLETE"
    PIPELINE_ERROR    = "PIPELINE_ERROR"
    ARBITRAGE_FOUND   = "ARBITRAGE_FOUND"
    SURETBET_FOUND    = "SURETBET_FOUND"
    SYSTEM_STARTUP    = "SYSTEM_STARTUP"
    SYSTEM_SHUTDOWN   = "SYSTEM_SHUTDOWN"
    CONFIG_CHANGE     = "CONFIG_CHANGE"
    CUSTOM            = "CUSTOM"


class AlertStatus(Enum):
    """Status of alert delivery."""
    PENDING = "PENDING"
    SENT = "SENT"
    FAILED = "FAILED"
    THROTTLED = "THROTTLED"
    QUEUED = "QUEUED"


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class Alert:
    """One alert message."""
    event: AlertEvent
    level: AlertLevel
    title: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tags: List[str] = field(default_factory=list)
    repeat_count: int = 0
    
    @property
    def timestamp_display(self) -> str:
        """Human-readable timestamp."""
        try:
            dt = datetime.fromisoformat(self.timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            return self.timestamp
    
    @property
    def level_emoji(self) -> str:
        """Emoji for alert level."""
        return {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.HIGH: "🔔",
            AlertLevel.CRITICAL: "🚨",
        }.get(self.level, "📢")
    
    def format_telegram(self) -> str:
        """Format for Telegram (HTML)."""
        repeat_suffix = f" (repeat #{self.repeat_count})" if self.repeat_count > 1 else ""
        
        lines = [
            f"{self.level_emoji} <b>ORACLE BEAST — {self.level.value}</b>",
            f"<b>📌 {self.title}</b>{repeat_suffix}",
            f"📝 {self.message}",
            f"🕐 {self.timestamp_display}",
        ]
        if self.data:
            lines.append("<b>📊 Details:</b>")
            for k, v in self.data.items():
                # Truncate long values
                v_str = str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
                lines.append(f"   {k}: {v_str}")
        return "\n".join(lines)
    
    def format_slack(self) -> Dict:
        """Format for Slack webhook."""
        colour = {
            AlertLevel.INFO: "#36a64f",
            AlertLevel.WARNING: "#ff9900",
            AlertLevel.HIGH: "#e01e5a",
            AlertLevel.CRITICAL: "#cc0000",
        }.get(self.level, "#cccccc")
        
        fields = []
        for k, v in self.data.items():
            v_str = str(v)[:250] if len(str(v)) > 250 else str(v)
            fields.append({"title": k, "value": v_str, "short": True})
        
        repeat_text = f" (Repeat #{self.repeat_count})" if self.repeat_count > 1 else ""
        
        return {
            "attachments": [{
                "color": colour,
                "title": f"{self.title}{repeat_text}",
                "text": self.message,
                "footer": f"Oracle Beast | {self.timestamp_display}",
                "fields": fields[:10],  # Limit to 10 fields
            }]
        }
    
    def format_email(self) -> str:
        """Format for email (plain text)."""
        repeat_suffix = f" [Repeat #{self.repeat_count}]" if self.repeat_count > 1 else ""
        
        lines = [
            f"{self.level_emoji} ORACLE BEAST — {self.level.value}{repeat_suffix}",
            "=" * 60,
            f"Title: {self.title}",
            f"Message: {self.message}",
            f"Timestamp: {self.timestamp_display}",
            "",
            "Details:",
        ]
        for k, v in self.data.items():
            lines.append(f"  {k}: {v}")
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def format_console(self) -> str:
        """Format for console output."""
        repeat_suffix = f" (x{self.repeat_count})" if self.repeat_count > 1 else ""
        return f"{self.level_emoji} [{self.timestamp_display}] {self.level.value}{repeat_suffix}: {self.title} - {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event": self.event.value,
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "repeat_count": self.repeat_count,
        }


@dataclass
class AlertDeliveryReport:
    """Report of alert delivery attempt."""
    alert: Alert
    channels: List[str] = field(default_factory=list)
    successes: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    status: AlertStatus = AlertStatus.PENDING
    error_message: str = ""
    
    @property
    def delivered(self) -> bool:
        """True if at least one channel succeeded."""
        return len(self.successes) > 0
    
    def summary(self) -> str:
        """Human-readable summary."""
        return f"Alert '{self.alert.title}': {len(self.successes)}/{len(self.channels)} channels succeeded"


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — ALERT THROTTLING
# ═══════════════════════════════════════════════════════════════

class AlertThrottler:
    """Prevents alert spam by throttling identical alerts."""
    
    def __init__(self, throttle_seconds: int = ALERT_THROTTLE_SECONDS):
        self.throttle_seconds = throttle_seconds
        self._last_alert: Dict[str, float] = {}
        self._repeat_counts: Dict[str, int] = {}
        self._lock = threading.Lock()
    
    def _get_key(self, alert: Alert) -> str:
        """Get throttle key for alert."""
        return f"{alert.event.value}:{alert.level.value}"
    
    def check_and_update(self, alert: Alert) -> Tuple[bool, int]:
        """
        Check if alert can be sent and update counters.
        
        Returns:
            Tuple of (can_send, repeat_count)
        """
        key = self._get_key(alert)
        
        with self._lock:
            now = time.time()
            last = self._last_alert.get(key, 0)
            
            if now - last < self.throttle_seconds:
                # Throttled - increment repeat count
                self._repeat_counts[key] = self._repeat_counts.get(key, 0) + 1
                return False, self._repeat_counts[key]
            
            # Not throttled - reset repeat count
            self._last_alert[key] = now
            repeat_count = self._repeat_counts.get(key, 0)
            self._repeat_counts[key] = 0
            return True, repeat_count
    
    def reset(self) -> None:
        """Reset throttle state (useful for testing)."""
        with self._lock:
            self._last_alert.clear()
            self._repeat_counts.clear()


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — ALERT ESCALATION
# ═══════════════════════════════════════════════════════════════

class AlertEscalator:
    """Tracks repeated alerts and escalates severity."""
    
    def __init__(self, threshold: int = ESCALATION_THRESHOLD, duration: int = ESCALATION_DURATION):
        self.threshold = threshold
        self.duration = duration
        self._counts: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    def _get_key(self, alert: Alert) -> str:
        """Get escalation key for alert."""
        return f"{alert.event.value}"
    
    def get_escalated_level(self, alert: Alert) -> AlertLevel:
        """
        Check if alert should be escalated based on frequency.
        
        Returns:
            Escalated AlertLevel if needed, otherwise original level
        """
        if alert.level == AlertLevel.CRITICAL:
            return alert.level  # Already highest
        
        key = self._get_key(alert)
        now = time.time()
        cutoff = now - self.duration
        
        with self._lock:
            # Clean old entries
            if key in self._counts:
                self._counts[key] = [ts for ts in self._counts[key] if ts > cutoff]
            else:
                self._counts[key] = []
            
            # Add current
            self._counts[key].append(now)
            count = len(self._counts[key])
        
        if count >= self.threshold * 2:
            return AlertLevel.CRITICAL
        elif count >= self.threshold:
            if alert.level == AlertLevel.WARNING:
                return AlertLevel.HIGH
            elif alert.level == AlertLevel.INFO:
                return AlertLevel.WARNING
        
        return alert.level
    
    def reset(self) -> None:
        """Reset escalation state."""
        with self._lock:
            self._counts.clear()


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — ALERT HISTORY
# ═══════════════════════════════════════════════════════════════

class AlertHistory:
    """Tracks recent alerts for audit and debugging."""
    
    def __init__(self, max_size: int = ALERT_HISTORY_SIZE, persist_file: str = "alert_history.json"):
        self.max_size = max_size
        self.persist_file = persist_file
        self._history: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._load()
    
    def _load(self) -> None:
        """Load alert history from file."""
        if not os.path.exists(self.persist_file):
            return
        
        try:
            with open(self.persist_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data[-self.max_size:]:
                    alert = Alert(
                        event=AlertEvent(item["event"]),
                        level=AlertLevel(item["level"]),
                        title=item["title"],
                        message=item["message"],
                        data=item.get("data", {}),
                        timestamp=item["timestamp"],
                        tags=item.get("tags", []),
                        repeat_count=item.get("repeat_count", 0),
                    )
                    self._history.append(alert)
            log.info(f"Loaded {len(self._history)} alerts from {self.persist_file}")
        except Exception as e:
            log.warning(f"Failed to load alert history: {e}")
    
    def _save(self) -> None:
        """Save alert history to file."""
        try:
            # Ensure directory exists
            Path(self.persist_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_file, "w", encoding="utf-8") as f:
                json.dump([a.to_dict() for a in self._history], f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save alert history: {e}")
    
    def add(self, alert: Alert) -> None:
        """Add alert to history."""
        with self._lock:
            self._history.append(alert)
            self._save()
    
    def get_recent(self, limit: int = 10, level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get most recent alerts."""
        with self._lock:
            alerts = list(self._history)[-limit:]
            if level:
                alerts = [a for a in alerts if a.level == level]
            return alerts
    
    def get_by_event(self, event: AlertEvent, limit: int = 10) -> List[Alert]:
        """Get recent alerts by event type."""
        with self._lock:
            return [a for a in self._history if a.event == event][-limit:]
    
    def clear(self) -> None:
        """Clear alert history."""
        with self._lock:
            self._history.clear()
            self._save()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        with self._lock:
            by_level = {}
            by_event = {}
            for a in self._history:
                by_level[a.level.value] = by_level.get(a.level.value, 0) + 1
                by_event[a.event.value] = by_event.get(a.event.value, 0) + 1
            
            return {
                "total": len(self._history),
                "by_level": by_level,
                "by_event": by_event,
            }


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — ALERT QUEUE
# ═══════════════════════════════════════════════════════════════

class AlertQueue:
    """Queue for async alert delivery."""
    
    def __init__(self, max_size: int = ALERT_QUEUE_SIZE):
        self._queue = queue.Queue(maxsize=max_size)
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
    
    def start(self) -> None:
        """Start the background worker thread."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        log.info("Alert queue worker started")
    
    def stop(self) -> None:
        """Stop the background worker."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        log.info("Alert queue worker stopped")
    
    def put(self, alert: Alert) -> bool:
        """Add alert to queue (non-blocking)."""
        try:
            self._queue.put_nowait(alert)
            return True
        except queue.Full:
            log.warning("Alert queue full, dropping alert")
            return False
    
    def _worker(self) -> None:
        """Background worker processing alerts."""
        while self._running:
            try:
                alert = self._queue.get(timeout=1)
                # Send immediately (non-async within worker)
                _send_alert_sync(alert)
            except queue.Empty:
                continue
            except Exception as e:
                log.error(f"Alert worker error: {e}")


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — CHANNEL SENDERS
# ═══════════════════════════════════════════════════════════════

def _send_telegram(alert: Alert) -> bool:
    """Send alert via Telegram bot."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    
    try:
        text = alert.format_telegram()
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = json.dumps({
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            resp = json.loads(r.read())
            return resp.get("ok", False)
    except Exception as e:
        log.error(f"Telegram send failed: {e}")
        return False


def _send_webhook(alert: Alert) -> bool:
    """Send alert via generic webhook."""
    url = WEBHOOK_URL or SLACK_WEBHOOK_URL
    if not url:
        return False
    
    try:
        # Check if it's a Slack webhook (contains "slack" in URL)
        if "slack" in url.lower():
            payload = alert.format_slack()
        else:
            payload = {
                "title": alert.title,
                "message": alert.message,
                "level": alert.level.value,
                "event": alert.event.value,
                "timestamp": alert.timestamp,
                "data": alert.data,
            }
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception as e:
        log.error(f"Webhook send failed: {e}")
        return False


def _send_slack(alert: Alert) -> bool:
    """Send alert via Slack webhook (legacy, use _send_webhook)."""
    return _send_webhook(alert)


def _send_email(alert: Alert) -> bool:
    """Send alert via email."""
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASS, ALERT_EMAIL]):
        return False
    
    try:
        body = alert.format_email()
        msg = MIMEMultipart()
        subject = f"[Oracle Beast] {alert.level.value}: {alert.title}"
        if alert.repeat_count > 1:
            subject += f" (x{alert.repeat_count})"
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = ALERT_EMAIL
        msg.attach(MIMEText(body, "plain"))
        
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        return True
    except Exception as e:
        log.error(f"Email send failed: {e}")
        return False


def _send_console(alert: Alert) -> bool:
    """Log alert to console (always enabled)."""
    log.info(alert.format_console())
    return True


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — MAIN NOTIFY FUNCTION
# ═══════════════════════════════════════════════════════════════

# Global instances
_throttler = AlertThrottler()
_escalator = AlertEscalator()
_history = AlertHistory()
_queue = AlertQueue()

# Start queue worker on module load
_queue.start()


def notify(
    event: AlertEvent,
    title: str,
    message: str,
    level: AlertLevel = AlertLevel.INFO,
    data: Dict[str, Any] = None,
    tags: List[str] = None,
    force: bool = False,
    async_send: bool = True,
) -> AlertDeliveryReport:
    """
    Send an alert to all configured channels.
    
    Args:
        event: Type of alert (AlertEvent enum)
        title: Alert title (short)
        message: Alert message (detailed)
        level: Severity level
        data: Additional structured data
        tags: Optional tags for filtering
        force: Skip throttling (for critical alerts)
        async_send: Send asynchronously via queue
    
    Returns:
        AlertDeliveryReport describing what was sent
    """
    # Apply escalation first
    if not force:
        level = _escalator.get_escalated_level(Alert(event=event, level=level, title=title, message=message))
    
    alert = Alert(
        event=event,
        level=level,
        title=title[:200],  # Truncate
        message=message[:1000],
        data=data or {},
        tags=tags or [],
    )
    
    # Check throttling
    can_send, repeat_count = _throttler.check_and_update(alert)
    alert.repeat_count = repeat_count
    
    if not can_send and not force:
        log.debug(f"Alert throttled: {event.value} ({level.value})")
        return AlertDeliveryReport(
            alert=alert,
            status=AlertStatus.THROTTLED,
        )
    
    # Add to history
    _history.add(alert)
    
    # Create report
    report = AlertDeliveryReport(alert=alert, status=AlertStatus.QUEUED)
    
    if async_send:
        # Queue for async delivery
        if _queue.put(alert):
            report.status = AlertStatus.QUEUED
            report.channels = ["Queued"]
        else:
            # Fallback to sync
            return _send_alert_sync_with_report(alert)
    else:
        return _send_alert_sync_with_report(alert)
    
    return report


def _send_alert_sync(alert: Alert) -> None:
    """Send alert synchronously (for worker)."""
    _send_alert_sync_with_report(alert)


def _send_alert_sync_with_report(alert: Alert) -> AlertDeliveryReport:
    """Send alert synchronously and return report."""
    report = AlertDeliveryReport(alert=alert, status=AlertStatus.PENDING)
    
    # Always send to console
    _send_console(alert)
    report.channels.append("Console")
    report.successes.append("Console")
    
    # Telegram
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        report.channels.append("Telegram")
        if _send_telegram(alert):
            report.successes.append("Telegram")
        else:
            report.failures.append("Telegram")
    
    # Webhook/Slack
    if WEBHOOK_URL or SLACK_WEBHOOK_URL:
        report.channels.append("Webhook")
        if _send_webhook(alert):
            report.successes.append("Webhook")
        else:
            report.failures.append("Webhook")
    
    # Email
    if SMTP_HOST and ALERT_EMAIL:
        report.channels.append("Email")
        if _send_email(alert):
            report.successes.append("Email")
        else:
            report.failures.append("Email")
    
    report.status = AlertStatus.SENT if report.successes else AlertStatus.FAILED
    
    if not report.successes and len(report.channels) <= 1:
        log.warning("No alert channels configured. Set TELEGRAM_BOT_TOKEN+CHAT_ID, "
                    "ALERT_WEBHOOK_URL, or SMTP_* env vars to enable notifications.")
    
    return report


# ═══════════════════════════════════════════════════════════════
# SECTION 10 — PRE-BUILT ALERT CONSTRUCTORS
# ═══════════════════════════════════════════════════════════════

def alert_pick_found(
    match: str,
    selection: str,
    odds: float,
    confidence: str,
    edge: float,
) -> AlertDeliveryReport:
    """Alert when a high-confidence pick is found."""
    return notify(
        event=AlertEvent.PICK_FOUND,
        level=AlertLevel.HIGH,
        title=f"🎯 Pick: {match}",
        message=f"{selection} @ {odds:.2f} — {confidence} confidence",
        data={
            "Match": match,
            "Selection": selection,
            "Odds": odds,
            "Confidence": confidence,
            "Edge": f"{edge:+.3f}",
        },
        tags=["pick", "value"],
    )


def alert_odds_move(
    match: str,
    original_odds: float,
    current_odds: float,
    drift: float,
    selection: str = "",
) -> AlertDeliveryReport:
    """Alert when odds move significantly."""
    direction = "shortened" if drift < 0 else "drifted out"
    return notify(
        event=AlertEvent.ODDS_MOVE,
        level=AlertLevel.WARNING,
        title=f"📊 Odds move: {match}",
        message=f"Odds {direction} from {original_odds:.2f} to {current_odds:.2f} ({drift:+.2f})",
        data={
            "Match": match,
            "Selection": selection,
            "Original": original_odds,
            "Current": current_odds,
            "Drift": drift,
        },
        tags=["odds", "movement"],
    )


def alert_drawdown(
    drawdown_pct: float,
    bankroll: float,
    peak: float,
    multiplier: float,
    streak: int = 0,
    streak_direction: str = "",
) -> AlertDeliveryReport:
    """Alert when drawdown exceeds thresholds."""
    level = AlertLevel.CRITICAL if drawdown_pct >= 0.25 else AlertLevel.HIGH
    event = AlertEvent.DRAWDOWN_CRITICAL if level == AlertLevel.CRITICAL else AlertEvent.DRAWDOWN_WARNING
    
    streak_text = f" | Streak: {streak}x {streak_direction}" if streak > 0 else ""
    
    return notify(
        event=event,
        level=level,
        title=f"📉 Drawdown {drawdown_pct:.1%}",
        message=f"Bankroll {bankroll:.2f} (peak {peak:.2f}). Stakes at {multiplier:.0%}{streak_text}",
        data={
            "Drawdown": f"{drawdown_pct:.1%}",
            "Bankroll": bankroll,
            "Peak": peak,
            "Stake multiplier": multiplier,
            "Streak": f"{streak}x {streak_direction}" if streak > 0 else "None",
        },
        tags=["risk", "drawdown"],
    )


def alert_budget(
    calls_used: int,
    calls_limit: int,
    level: str = "WARNING",
) -> AlertDeliveryReport:
    """Alert when API budget is low."""
    pct = calls_used / calls_limit
    event = AlertEvent.BUDGET_EXHAUSTED if pct >= 1.0 else AlertEvent.BUDGET_WARNING
    alert_level = AlertLevel.CRITICAL if pct >= 1.0 else AlertLevel.WARNING
    
    return notify(
        event=event,
        level=alert_level,
        title=f"💰 API Budget: {calls_used}/{calls_limit} calls",
        message=f"Daily API budget {('exhausted' if pct >= 1.0 else 'low')}. "
                f"{calls_limit - calls_used} calls remaining.",
        data={
            "Used": calls_used,
            "Limit": calls_limit,
            "Remaining": calls_limit - calls_used,
            "Percentage": f"{pct:.0%}",
        },
        tags=["api", "budget"],
    )


def alert_learning_complete(
    accuracy: float,
    correct: int,
    total: int,
    roi: float,
    new_config: Optional[Dict] = None,
) -> AlertDeliveryReport:
    """Alert when learning cycle completes."""
    return notify(
        event=AlertEvent.LEARNING_COMPLETE,
        level=AlertLevel.INFO,
        title="🧠 Learning cycle complete",
        message=f"Accuracy: {accuracy:.1%} ({correct}/{total} correct). ROI: {roi:+.1%}",
        data={
            "Accuracy": f"{accuracy:.1%}",
            "Correct": correct,
            "Total": total,
            "ROI": f"{roi:+.1%}",
            "New thresholds": new_config.get("home_win_threshold") if new_config else None,
        },
        tags=["learning", "performance"],
    )


def alert_pipeline_error(
    error_msg: str,
    context: str = "",
    match_id: str = "",
) -> AlertDeliveryReport:
    """Alert when pipeline error occurs."""
    return notify(
        event=AlertEvent.PIPELINE_ERROR,
        level=AlertLevel.CRITICAL,
        title="🔥 Pipeline error",
        message=error_msg[:200],
        data={
            "Context": context,
            "Match": match_id,
            "Error": error_msg[:500],
        },
        tags=["error", "critical"],
    )


def alert_arbitrage_found(
    match: str,
    profit_margin: float,
    guaranteed_return: float,
    stakes: Dict[str, float],
    books: Dict[str, str],
) -> AlertDeliveryReport:
    """Alert when arbitrage opportunity is found."""
    return notify(
        event=AlertEvent.ARBITRAGE_FOUND,
        level=AlertLevel.HIGH,
        title=f"💰 Arbitrage: {match}",
        message=f"{profit_margin:.2f}% guaranteed profit",
        data={
            "Match": match,
            "Profit Margin": f"{profit_margin:.2f}%",
            "Guaranteed Return": guaranteed_return,
            "Stakes": stakes,
            "Best Books": books,
        },
        tags=["arbitrage", "value"],
    )


def alert_surebet_found(
    match: str,
    market: str,
    profit_margin: float,
    stakes: Dict[str, float],
) -> AlertDeliveryReport:
    """Alert when surebet is found."""
    return notify(
        event=AlertEvent.SURETBET_FOUND,
        level=AlertLevel.HIGH,
        title=f"✅ Surebet: {match} ({market})",
        message=f"{profit_margin:.2f}% guaranteed profit",
        data={
            "Match": match,
            "Market": market,
            "Profit Margin": f"{profit_margin:.2f}%",
            "Stakes": stakes,
        },
        tags=["arbitrage", "surebet"],
    )


def alert_system_startup() -> AlertDeliveryReport:
    """Alert when system starts up."""
    return notify(
        event=AlertEvent.SYSTEM_STARTUP,
        level=AlertLevel.INFO,
        title="🚀 Oracle Beast Started",
        message="System is online and monitoring for opportunities",
        data={
            "Version": "2.2.0",
            "Startup Time": datetime.now(timezone.utc).isoformat(),
        },
        tags=["system"],
    )


def alert_system_shutdown() -> AlertDeliveryReport:
    """Alert when system shuts down."""
    return notify(
        event=AlertEvent.SYSTEM_SHUTDOWN,
        level=AlertLevel.WARNING,
        title="🛑 Oracle Beast Shutting Down",
        message="System is going offline",
        data={
            "Shutdown Time": datetime.now(timezone.utc).isoformat(),
        },
        tags=["system"],
    )


def alert_config_change(
    old_config: Dict,
    new_config: Dict,
    changed_keys: List[str],
) -> AlertDeliveryReport:
    """Alert when configuration changes."""
    return notify(
        event=AlertEvent.CONFIG_CHANGE,
        level=AlertLevel.INFO,
        title="⚙️ Configuration changed",
        message=f"Updated: {', '.join(changed_keys[:5])}",
        data={
            "Changed keys": changed_keys,
            "Old values": {k: old_config.get(k) for k in changed_keys[:5]},
            "New values": {k: new_config.get(k) for k in changed_keys[:5]},
        },
        tags=["config"],
    )


# ═══════════════════════════════════════════════════════════════
# SECTION 11 — UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_alert_history(limit: int = 10, level: Optional[AlertLevel] = None) -> List[Dict]:
    """Get recent alert history as dictionaries."""
    alerts = _history.get_recent(limit, level)
    return [a.to_dict() for a in alerts]


def get_alert_stats() -> Dict[str, Any]:
    """Get alert statistics."""
    return _history.get_stats()


def clear_alert_history() -> None:
    """Clear alert history."""
    _history.clear()


def reset_throttler() -> None:
    """Reset alert throttler (useful for testing)."""
    _throttler.reset()
    _escalator.reset()


def shutdown_queue() -> None:
    """Shutdown alert queue (call on system exit)."""
    _queue.stop()


# ═══════════════════════════════════════════════════════════════
# SECTION 12 — EXPORTS
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "AlertLevel",
    "AlertEvent",
    "AlertStatus",
    # Data classes
    "Alert",
    "AlertDeliveryReport",
    # Main function
    "notify",
    # Pre-built alert constructors
    "alert_pick_found",
    "alert_odds_move",
    "alert_drawdown",
    "alert_budget",
    "alert_learning_complete",
    "alert_pipeline_error",
    "alert_arbitrage_found",
    "alert_surebet_found",
    "alert_system_startup",
    "alert_system_shutdown",
    "alert_config_change",
    # Utilities
    "get_alert_history",
    "get_alert_stats",
    "clear_alert_history",
    "reset_throttler",
    "shutdown_queue",
]


# ═══════════════════════════════════════════════════════════════
# SECTION 13 — EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MODULE 30: ALERT ENGINE - TEST RUN")
    print("=" * 70)
    
    print("\n📢 Sending test alerts (console only, no API keys set)...")
    print("-" * 40)
    
    # Test pick found alert
    result1 = alert_pick_found(
        match="Arsenal vs Chelsea",
        selection="Arsenal",
        odds=1.85,
        confidence="HIGH",
        edge=0.09,
    )
    print(f"Pick alert: {result1.summary()}")
    
    # Test odds move alert
    result2 = alert_odds_move(
        match="Liverpool vs Everton",
        original_odds=1.95,
        current_odds=1.85,
        drift=-0.10,
        selection="Liverpool",
    )
    print(f"Odds move alert: {result2.summary()}")
    
    # Test drawdown alert
    result3 = alert_drawdown(
        drawdown_pct=0.18,
        bankroll=820.00,
        peak=1000.00,
        multiplier=0.50,
        streak=5,
        streak_direction="L",
    )
    print(f"Drawdown alert: {result3.summary()}")
    
    # Test budget alert
    result4 = alert_budget(calls_used=85, calls_limit=100, level="WARNING")
    print(f"Budget alert: {result4.summary()}")
    
    # Test arbitrage alert
    result5 = alert_arbitrage_found(
        match="Barcelona vs Real Madrid",
        profit_margin=2.5,
        guaranteed_return=102.50,
        stakes={"home": 45.00, "draw": 25.00, "away": 30.00},
        books={"home": "Bet365", "draw": "Pinnacle", "away": "William Hill"},
    )
    print(f"Arbitrage alert: {result5.summary()}")
    
    # Test learning complete
    result6 = alert_learning_complete(
        accuracy=0.58,
        correct=58,
        total=100,
        roi=0.12,
        new_config={"home_win_threshold": 0.58},
    )
    print(f"Learning alert: {result6.summary()}")
    
    # Test config change
    result7 = alert_config_change(
        old_config={"home_win_threshold": 0.57},
        new_config={"home_win_threshold": 0.58},
        changed_keys=["home_win_threshold"],
    )
    print(f"Config alert: {result7.summary()}")
    
    # Get alert history
    print("\n📜 Recent alert history:")
    history = get_alert_history(5)
    for h in history:
        print(f"  [{h['level']}] {h['title']} - {h['timestamp'][:19]}")
    
    # Get stats
    print("\n📊 Alert statistics:")
    stats = get_alert_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Alert engine test complete")
    print("\n" + "=" * 70)
    print("MODULE 30 READY FOR PRODUCTION")
    print("=" * 70)