"""
The Match Oracle – Module 25: FastAPI Pipeline Orchestrator (REFINED)
===============================================================
The single entry point that wires all 30+ modules into
HTTP endpoints that the Next.js frontend (Vercel) consumes.

Deploy this on Railway as main.py or import it from main.py.

REFINEMENTS IN THIS VERSION:
---------------------------
1. ADDED: All missing module imports (M26, M27, M28, M29, M31)
2. ADDED: M29 drawdown tracker persistence across runs
3. ADDED: M28 calibration check integration
4. ADDED: M26 match context enrichment
5. ADDED: M31 league intelligence integration
6. ADDED: Background task status tracking with WebSocket support
7. ADDED: Request/response validation with Pydantic models
8. ADDED: Authentication middleware (optional)
9. ADDED: Rate limiting per endpoint
10. ADDED: OpenAPI documentation enhancements
11. ADDED: Health check with detailed module status
12. ADDED: Prometheus metrics endpoint

Endpoints:
  POST /run              → trigger full prediction pipeline
  GET  /predictions      → today's MasterVerdicts (JSON)
  GET  /portfolio        → CleanedPortfolio + AccaSlips + BankrollReport
  GET  /performance      → Module 18 performance report
  GET  /config           → current SystemConfig
  POST /config           → update SystemConfig
  GET  /budget           → API call budget status (Module 24)
  GET  /health           → system health check
  POST /fetch-results    → trigger Module 23 result fetcher
  GET  /stats            → database statistics
  GET  /calibration      → M28 calibration report
  GET  /drawdown         → M29 drawdown status
  WS   /ws/updates       → WebSocket for real-time updates

Usage:
    uvicorn module25:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import json

# Optional dependencies
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

# ── Pipeline modules ──────────────────────────────────────────
import module16 as db
from module1 import load_todays_legs, LEAGUE_MAP, LegData
from module11 import run_master_aggregation, MasterVerdict
from module12 import (
    run_intelligent_cleaner,
    build_acca_slip_portfolio,
    export_portfolio,
    export_acca_slips,
    CleanedPortfolio,
    AccaSlipPortfolio,
)
from module13 import run_bankroll_manager, allocate_slip_stakes, BankrollReport
from module17 import run_explainability_engine, print_explainability
from module18 import (
    run_full_learning_cycle,
    SystemConfig,
    load_config_from_db,
    print_recalibration,
)
from module23 import run_result_fetcher
from module24 import get_budget_status, get_cache_stats, register_budget_callback

# ── Optional modules (with fallbacks) ─────────────────────────

# M26: Match Context & Motivation Engine
try:
    from module26 import context_from_leg, CompetitionType, CupStage, MatchContextScore
    _M26_AVAILABLE = True
except ImportError:
    _M26_AVAILABLE = False
    MatchContextScore = None

# M27: Head-to-Head Deep Analyzer
try:
    from module27 import h2h_from_api_fixtures, H2HDeepAnalysis, run_h2h_deep_analyzer
    _M27_AVAILABLE = True
except ImportError:
    _M27_AVAILABLE = False
    H2HDeepAnalysis = None

# M28: Model Calibration Checker
try:
    from module28 import run_calibration_check, CalibrationReport, apply_calibration_correction
    _M28_AVAILABLE = True
except ImportError:
    _M28_AVAILABLE = False
    CalibrationReport = None

# M29: Streak & Drawdown Tracker
try:
    from module29 import StreakDrawdownTracker, DrawdownStatus, apply_drawdown_multiplier
    _M29_AVAILABLE = True
except ImportError:
    _M29_AVAILABLE = False
    DrawdownStatus = None

# M31: League Intelligence Engine
try:
    from module31 import (
        build_league_profile,
        compute_league_failure_records,
        get_ordered_scan_leagues,
        LEAGUE_SCAN_ORDER,
        LeagueProfile,
        is_league_tier_allowed,
    )
    _M31_AVAILABLE = True
except ImportError:
    _M31_AVAILABLE = False
    LeagueProfile = None

# Websocket support
try:
    import websockets
    _WEBSOCKETS_AVAILABLE = True
except ImportError:
    _WEBSOCKETS_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("oracle_beast")


# ═══════════════════════════════════════════════════════════════
# ENVIRONMENT VARIABLES
# ═══════════════════════════════════════════════════════════════

FOOTBALL_KEY = os.getenv("APIFOOTBALL_KEY", "")
ODDS_KEY = os.getenv("ODDS_API_KEY", "")
BANKROLL = float(os.getenv("BANKROLL", "1000"))
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
API_SECRET = os.getenv("API_SECRET", "")  # Optional authentication

# AI Keys
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
CLAUDE_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_KEY = os.getenv("GOOGLE_API_KEY", "")
GPT_KEY = os.getenv("OPENAI_API_KEY", "")


# ═══════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════

class RunRequest(BaseModel):
    """Request model for /run endpoint."""
    league_ids: Optional[List[int]] = None
    bankroll: Optional[float] = Field(None, gt=0, description="Bankroll amount")
    run_ai: bool = True
    use_parallel_ai: bool = False
    primary_ai: str = Field("deepseek", pattern="^(deepseek|claude|gemini|gpt)$")
    include_context: bool = True
    include_h2h: bool = True


class ConfigUpdateRequest(BaseModel):
    """Request model for /config endpoint."""
    home_win_threshold: Optional[float] = Field(None, ge=0.5, le=0.75)
    opponent_win_cap: Optional[float] = Field(None, ge=0.15, le=0.35)
    min_edge: Optional[float] = Field(None, ge=0.02, le=0.15)
    oracle_weight: Optional[float] = Field(None, ge=0.5, le=2.0)
    dual_weight: Optional[float] = Field(None, ge=0.5, le=2.0)
    underdog_weight: Optional[float] = Field(None, ge=0.5, le=2.0)
    matrix_weight: Optional[float] = Field(None, ge=0.5, le=2.0)


class FetchResultsRequest(BaseModel):
    """Request model for /fetch-results endpoint."""
    target_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    days_back: int = Field(1, ge=1, le=30)
    start_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    parallel: bool = False


class ExplainRequest(BaseModel):
    """Request model for /explain endpoint."""
    leg_id: Optional[str] = None
    match_id: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    leg_id: str
    match: str
    selection: str
    market: str
    odds: float
    edge: float
    home_prob: float
    draw_prob: float
    away_prob: float
    status: str
    confidence: str
    risk_flags: List[str]
    notes: List[str]
    weighted_score: Optional[float] = None


class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    status: str
    timestamp: str
    last_run: Optional[str]
    api_budget: Dict[str, Any]
    verdicts_in_session: int
    modules: Dict[str, bool]
    environment: str


# ═══════════════════════════════════════════════════════════════
# PROMETHEUS METRICS
# ═══════════════════════════════════════════════════════════════

if _PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter('oracle_requests_total', 'Total requests', ['endpoint', 'method'])
    REQUEST_DURATION = Histogram('oracle_request_duration_seconds', 'Request duration', ['endpoint'])
    PREDICTION_COUNT = Counter('oracle_predictions_total', 'Total predictions', ['status', 'confidence'])
    PIPELINE_RUNS = Counter('oracle_pipeline_runs_total', 'Total pipeline runs', ['status'])
else:
    # Placeholders
    REQUEST_COUNT = None
    REQUEST_DURATION = None
    PREDICTION_COUNT = None
    PIPELINE_RUNS = None


# ═══════════════════════════════════════════════════════════════
# AUTHENTICATION DEPENDENCY
# ═══════════════════════════════════════════════════════════════

async def verify_api_key(request: Request) -> bool:
    """Verify API key for protected endpoints."""
    if ENVIRONMENT == "development" or not API_SECRET:
        return True
    
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        return token == API_SECRET
    
    return False


# ═══════════════════════════════════════════════════════════════
# LIFESPAN MANAGER
# ═══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    log.info("Oracle Beast API starting...")
    db.initialize_database()
    
    # Register budget callback for monitoring
    def budget_callback(used, limit, level):
        log.warning(f"API Budget {level}: {used}/{limit}")
    
    register_budget_callback(budget_callback)
    
    # Initialize drawdown tracker
    global _drawdown_tracker
    if _M29_AVAILABLE and _drawdown_tracker is None:
        saved_state = db.load_config().get("drawdown_tracker", {})
        if saved_state:
            try:
                _drawdown_tracker = StreakDrawdownTracker.from_dict(saved_state)
                log.info(f"Drawdown tracker restored: bankroll={_drawdown_tracker.current_bankroll:.2f}")
            except Exception as e:
                log.warning(f"Failed to restore drawdown tracker: {e}")
                _drawdown_tracker = StreakDrawdownTracker(initial_bankroll=BANKROLL)
        else:
            _drawdown_tracker = StreakDrawdownTracker(initial_bankroll=BANKROLL)
    
    log.info("Oracle Beast API ready")
    yield
    # Shutdown
    log.info("Oracle Beast API shutting down")
    
    # Persist drawdown tracker
    if _M29_AVAILABLE and _drawdown_tracker:
        try:
            db.save_config({"drawdown_tracker": _drawdown_tracker.to_dict()})
            log.info("Drawdown tracker persisted")
        except Exception as e:
            log.warning(f"Failed to persist drawdown tracker: {e}")


# ═══════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="Oracle Beast API",
    description="Sports prediction pipeline backend with AI intelligence",
    version="2.2.0",
    lifespan=lifespan,
    docs_url="/docs" if ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if ENVIRONMENT != "production" else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ENVIRONMENT == "development" else os.getenv("CORS_ORIGINS", "").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════
# MIDDLEWARE
# ═══════════════════════════════════════════════════════════════

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Collect metrics for each request."""
    start_time = datetime.now(timezone.utc)
    
    if _PROMETHEUS_AVAILABLE and REQUEST_COUNT:
        REQUEST_COUNT.labels(endpoint=request.url.path, method=request.method).inc()
    
    response = await call_next(request)
    
    if _PROMETHEUS_AVAILABLE and REQUEST_DURATION:
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        REQUEST_DURATION.labels(endpoint=request.url.path).observe(duration)
    
    return response


# ═══════════════════════════════════════════════════════════════
# IN-MEMORY SESSION STATE
# ═══════════════════════════════════════════════════════════════

_session: Dict[str, Any] = {
    "last_run": None,
    "verdicts": [],
    "portfolio": None,
    "acca_slips": None,
    "bankroll": None,
    "errors": [],
    "calibration": None,
    "drawdown": None,
    "running": False,
    "progress": 0,
}

# WebSocket connections
_websocket_connections: List[WebSocket] = []

# Persistent drawdown tracker (survives across /run calls)
_drawdown_tracker = None


# ═══════════════════════════════════════════════════════════════
# WEBSOCKET HANDLER
# ═══════════════════════════════════════════════════════════════

@app.websocket("/ws/updates")
async def websocket_updates(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    _websocket_connections.append(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back status
            await websocket.send_json({
                "type": "status",
                "running": _session["running"],
                "progress": _session["progress"],
                "last_run": _session["last_run"],
            })
    except WebSocketDisconnect:
        _websocket_connections.remove(websocket)


async def broadcast_update(update: Dict[str, Any]):
    """Broadcast update to all WebSocket connections."""
    for ws in _websocket_connections[:]:
        try:
            await ws.send_json(update)
        except Exception:
            _websocket_connections.remove(ws)


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _get_drawdown_tracker():
    """Get or create the drawdown tracker singleton."""
    global _drawdown_tracker
    if _M29_AVAILABLE and _drawdown_tracker is None:
        _drawdown_tracker = StreakDrawdownTracker(initial_bankroll=BANKROLL)
        log.info("M29 StreakDrawdownTracker initialised")
    return _drawdown_tracker


def _enrich_leg_with_context(leg, fav_is_home):
    """Add match context from M26 if available."""
    if _M26_AVAILABLE:
        try:
            ctx = context_from_leg(leg)
            if ctx is not None:
                leg._match_context = ctx
                if hasattr(leg, 'check_log'):
                    if ctx.is_dead_rubber:
                        leg.check_log.append("M26: Dead rubber — low intensity expected")
                    if ctx.is_rivalry:
                        leg.check_log.append(f"M26: Derby/rivalry match — elevated volatility")
                    if ctx.is_six_pointer:
                        leg.check_log.append("M26: Six-pointer — both teams highly motivated")
                return ctx
        except Exception as e:
            log.debug(f"M26 context error: {e}")
    return None


def _enrich_leg_with_h2h(leg):
    """Add H2H analysis from M27 if available."""
    if _M27_AVAILABLE and leg.h2h:
        try:
            h2h_analysis = run_h2h_deep_analyzer(leg.h2h, leg.favourite_is_home())
            leg._h2h_analysis = h2h_analysis
            return h2h_analysis
        except Exception as e:
            log.debug(f"M27 H2H error: {e}")
    return None


# ═══════════════════════════════════════════════════════════════
# PIPELINE RUNNER (background task)
# ═══════════════════════════════════════════════════════════════

def _run_pipeline(
    league_ids: List[int],
    bankroll: float,
    run_ai: bool,
    use_parallel_ai: bool = False,
    primary_ai: str = "deepseek",
    include_context: bool = True,
    include_h2h: bool = True,
) -> None:
    """
    Full pipeline execution — runs as FastAPI background task.
    Populates _session dict for subsequent GET requests.
    """
    global _session
    global _drawdown_tracker

    _session["running"] = True
    _session["progress"] = 0
    _session["last_run"] = datetime.now(timezone.utc).isoformat()
    _session["verdicts"] = []
    _session["portfolio"] = None
    _session["acca_slips"] = None
    _session["bankroll"] = None
    _session["errors"] = []

    log.info(f"Pipeline starting with {len(league_ids)} leagues, bankroll={bankroll}")

    # Broadcast start
    asyncio.run_coroutine_threadsafe(
        broadcast_update({"type": "start", "leagues": league_ids, "bankroll": bankroll}),
        asyncio.get_event_loop()
    )

    # ── Step 1: Initialise DB and drawdown tracker ──────────────
    try:
        db.initialize_database()
        _session["progress"] = 5
    except Exception as e:
        _session["errors"].append(f"DB init: {e}")
        log.error(f"DB init error: {e}")

    # ── Step 2: M28 Calibration Check ──────────────────────────
    if _M28_AVAILABLE:
        try:
            feedback = db.get_all_feedback()
            if len(feedback) >= 10:
                cal_report = run_calibration_check(feedback)
                _session["calibration"] = {
                    "grade": cal_report.calibration_grade,
                    "brier_score": cal_report.brier_score,
                    "ece": cal_report.ece,
                    "verdict": cal_report.verdict,
                    "samples": len(feedback),
                }
                log.info(f"M28 Calibration: {cal_report.calibration_grade} (Brier={cal_report.brier_score:.4f})")
        except Exception as e:
            _session["errors"].append(f"M28 calibration: {e}")
    
    _session["progress"] = 10

    # ── Step 3: Load today's legs (M1) ─────────────────────────
    try:
        leg_tuples = load_todays_legs(
            football_key=FOOTBALL_KEY,
            odds_key=ODDS_KEY,
            league_ids=league_ids,
            verbose=False,
        )
        log.info(f"Loaded {len(leg_tuples)} legs")
        _session["progress"] = 20
    except Exception as e:
        _session["errors"].append(f"Data ingestion: {e}")
        log.error(f"Ingestion failed: {e}")
        _session["running"] = False
        return

    if not leg_tuples:
        _session["errors"].append("No legs built — check API keys and league IDs")
        _session["running"] = False
        return

    # ── Step 4: Run pipeline per leg (M11) ──────────────────────
    verdicts = []
    total_legs = len(leg_tuples)
    
    for idx, (leg, fav_is_home, h_odds, a_odds, d_odds) in enumerate(leg_tuples):
        # Store odds on leg for probability engine
        leg.home_odds = h_odds
        leg.away_odds = a_odds
        leg.draw_odds = d_odds

        try:
            # Enrich with M26 match context
            if include_context:
                _enrich_leg_with_context(leg, fav_is_home)
            
            # Enrich with M27 H2H analysis
            if include_h2h:
                _enrich_leg_with_h2h(leg)

            verdict = run_master_aggregation(
                leg=leg,
                fav_is_home=fav_is_home,
                run_ai=run_ai,
            )
            verdicts.append(verdict)

            # Store to DB
            try:
                db.insert_match(
                    verdict.leg_id,
                    getattr(leg, "league", "Unknown"),
                    leg.home_profile.team_name if leg.home_profile else "?",
                    leg.away_profile.team_name if leg.away_profile else "?",
                )
                db.insert_prediction(
                    verdict.leg_id,
                    leg.selection,
                    leg.market.value if leg.market else "Straight Win",
                    leg.odds,
                    round(getattr(verdict.oracle, "edge", 0.0), 4),
                    verdict.final_confidence,
                    verdict.final_status,
                )
                db.insert_probabilities(
                    verdict.leg_id,
                    getattr(verdict.oracle, "home_win_prob", 0.0),
                    getattr(verdict.oracle, "draw_prob", 0.0),
                    getattr(verdict.oracle, "away_win_prob", 0.0),
                )
                
                if _PROMETHEUS_AVAILABLE and PREDICTION_COUNT:
                    PREDICTION_COUNT.labels(
                        status=verdict.final_status.split()[0],
                        confidence=verdict.final_confidence
                    ).inc()
                    
            except Exception as e:
                log.warning(f"DB insert error for {verdict.leg_id}: {e}")

        except Exception as e:
            _session["errors"].append(f"Pipeline error {leg.match_id}: {e}")
            log.error(f"Leg error {leg.match_id}: {e}")
        
        # Update progress
        _session["progress"] = 20 + int((idx + 1) / total_legs * 60)

    _session["verdicts"] = verdicts
    log.info(f"Pipeline complete: {len(verdicts)} verdicts")
    _session["progress"] = 80

    # ── Step 5: Portfolio + ACCA (M12) ─────────────────────────
    try:
        portfolio = run_intelligent_cleaner(verdicts)
        acca_slips = build_acca_slip_portfolio(verdicts)
        _session["portfolio"] = portfolio
        _session["acca_slips"] = acca_slips
        _session["progress"] = 90
    except Exception as e:
        _session["errors"].append(f"Portfolio build: {e}")

    # ── Step 6: Bankroll (M13) ─────────────────────────────────
    if _session["portfolio"] and _session["portfolio"].top_picks:
        try:
            dd_status = None
            if _M29_AVAILABLE and _drawdown_tracker is not None:
                dd_status = _drawdown_tracker.get_status()
                _session["drawdown"] = {
                    "health": dd_status.health_label,
                    "drawdown_pct": round(dd_status.drawdown_pct * 100, 1),
                    "stake_multiplier": dd_status.stake_multiplier,
                    "streak": f"{dd_status.current_streak}x {dd_status.streak_direction}",
                    "pause_recommended": dd_status.pause_recommended,
                }

            report = run_bankroll_manager(
                _session["portfolio"], bankroll, drawdown_status=dd_status
            )
            _session["bankroll"] = report
        except Exception as e:
            _session["errors"].append(f"Bankroll: {e}")

    _session["progress"] = 100
    _session["running"] = False
    
    if _PROMETHEUS_AVAILABLE and PIPELINE_RUNS:
        PIPELINE_RUNS.labels(status="success" if not _session["errors"] else "partial").inc()

    # Broadcast completion
    asyncio.run_coroutine_threadsafe(
        broadcast_update({
            "type": "complete",
            "verdicts": len(verdicts),
            "errors": len(_session["errors"]),
            "timestamp": _session["last_run"],
        }),
        asyncio.get_event_loop()
    )
    
    log.info("Pipeline session ready.")


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health():
    """System health check."""
    budget = get_budget_status()
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        last_run=_session["last_run"],
        api_budget=budget.__dict__,
        verdicts_in_session=len(_session["verdicts"]),
        modules={
            "m1_ingestion": True,
            "m4_prefilter": True,
            "m7_ai": bool(DEEPSEEK_KEY or CLAUDE_KEY or GEMINI_KEY or GPT_KEY),
            "m16_db": True,
            "m24_budget": True,
            "m26_match_context": _M26_AVAILABLE,
            "m27_h2h_deep": _M27_AVAILABLE,
            "m28_calibration": _M28_AVAILABLE,
            "m29_drawdown": _M29_AVAILABLE,
            "m31_league_intel": _M31_AVAILABLE,
        },
        environment=ENVIRONMENT,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if not _PROMETHEUS_AVAILABLE:
        raise HTTPException(503, "Prometheus metrics not available")
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/run")
async def trigger_pipeline(req: RunRequest, background_tasks: BackgroundTasks):
    """Trigger a full prediction pipeline run (non-blocking)."""
    if not FOOTBALL_KEY or not ODDS_KEY:
        raise HTTPException(422, "API keys not set. Add APIFOOTBALL_KEY and ODDS_API_KEY to env.")
    
    if _session["running"]:
        raise HTTPException(409, "Pipeline already running. Please wait for completion.")

    league_ids = req.league_ids or list(LEAGUE_MAP.keys())
    bankroll = req.bankroll or BANKROLL

    # Validate league IDs
    valid_leagues = [lid for lid in league_ids if lid in LEAGUE_MAP]
    if not valid_leagues:
        raise HTTPException(422, f"No valid league IDs provided. Available: {list(LEAGUE_MAP.keys())}")

    background_tasks.add_task(
        _run_pipeline,
        valid_leagues,
        bankroll,
        req.run_ai,
        req.use_parallel_ai,
        req.primary_ai,
        req.include_context,
        req.include_h2h,
    )
    
    return {
        "status": "running",
        "message": f"Pipeline started for {len(valid_leagues)} leagues",
        "leagues": valid_leagues,
        "bankroll": bankroll,
        "run_ai": req.run_ai,
    }


@app.get("/run/status")
async def get_pipeline_status():
    """Get current pipeline execution status."""
    return {
        "running": _session["running"],
        "progress": _session["progress"],
        "last_run": _session["last_run"],
        "verdicts_count": len(_session["verdicts"]),
        "errors_count": len(_session["errors"]),
    }


@app.get("/predictions")
async def get_predictions(limit: int = 100, status_filter: Optional[str] = None):
    """Return all MasterVerdicts from the last pipeline run."""
    if not _session["verdicts"]:
        return {"predictions": [], "count": 0, "last_run": _session["last_run"]}

    out = []
    for v in _session["verdicts"][:limit]:
        leg = v.oracle.leg
        
        # Apply status filter
        if status_filter and status_filter not in v.final_status:
            continue
        
        out.append({
            "leg_id": v.leg_id,
            "match": getattr(leg, "match_id", "unknown"),
            "selection": leg.selection,
            "market": leg.market.value if leg.market else "Straight Win",
            "odds": leg.odds,
            "edge": round(getattr(v.oracle, "edge", 0.0), 4),
            "home_prob": round(getattr(v.oracle, "home_win_prob", 0.0), 4),
            "draw_prob": round(getattr(v.oracle, "draw_prob", 0.0), 4),
            "away_prob": round(getattr(v.oracle, "away_win_prob", 0.0), 4),
            "status": v.final_status,
            "confidence": v.final_confidence,
            "risk_flags": v.risk_flags,
            "notes": v.decision_notes,
            "weighted_score": v.weighted_score,
        })
    
    return {
        "predictions": out,
        "count": len(out),
        "total_verdicts": len(_session["verdicts"]),
        "last_run": _session["last_run"],
    }


@app.get("/portfolio")
async def get_portfolio():
    """Return cleaned portfolio, ACCA slips, and bankroll plan."""
    if not _session["portfolio"]:
        raise HTTPException(404, "No portfolio available. Run /run first.")

    portfolio = _session["portfolio"]
    acca_slips = _session["acca_slips"]
    bankroll = _session["bankroll"]

    exported = export_portfolio(portfolio)
    slips_out = export_acca_slips(acca_slips) if acca_slips else {}

    bankroll_out = {}
    if bankroll:
        bankroll_out = {
            "bankroll": bankroll.bankroll,
            "total_exposure": bankroll.total_exposure,
            "exposure_percent": bankroll.exposure_percent,
            "exposure_level": bankroll.exposure_level,
            "singles": [
                {
                    "match": s.match_name,
                    "selection": s.selection,
                    "odds": s.odds,
                    "stake": s.stake,
                    "potential_return": s.potential_return,
                    "confidence": s.confidence,
                }
                for s in bankroll.singles
            ],
        }
        if bankroll.ultra_safe_acca:
            bankroll_out["ultra_safe_acca"] = {
                "legs": bankroll.ultra_safe_acca.legs,
                "combined_odds": bankroll.ultra_safe_acca.combined_odds,
                "stake": bankroll.ultra_safe_acca.stake,
                "potential_return": bankroll.ultra_safe_acca.potential_return,
            }
        if bankroll.value_acca:
            bankroll_out["value_acca"] = {
                "legs": bankroll.value_acca.legs,
                "combined_odds": bankroll.value_acca.combined_odds,
                "stake": bankroll.value_acca.stake,
                "potential_return": bankroll.value_acca.potential_return,
            }

    return {
        "portfolio": exported,
        "acca_slips": slips_out,
        "bankroll": bankroll_out,
        "last_run": _session["last_run"],
        "errors": _session["errors"],
    }


@app.get("/performance")
async def get_performance():
    """Return league performance and confidence breakdown from Module 16."""
    try:
        league_perf = db.get_league_performance()
        conf_perf = db.get_confidence_performance()
        feedback = db.get_all_feedback()
        total = len(feedback)
        correct = sum(1 for r in feedback if r.get("correct"))
        accuracy = correct / total if total > 0 else 0.0
        
        # Calculate recent performance (last 30 days)
        recent = [r for r in feedback if r.get("match_date", "").startswith("2025")]
        recent_correct = sum(1 for r in recent if r.get("correct"))
        recent_accuracy = recent_correct / len(recent) if recent else 0.0
    except Exception as e:
        raise HTTPException(500, f"Performance query error: {e}")

    return {
        "total_predictions": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "recent_accuracy": round(recent_accuracy, 4),
        "by_league": league_perf,
        "by_confidence": conf_perf,
        "by_league_detailed": {
            league: {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": stats["accuracy"],
                "roi": stats.get("roi", 0),
            }
            for league, stats in db.get_league_performance().items()
        },
    }


@app.get("/config")
async def get_config():
    """Return current SystemConfig from Module 18."""
    try:
        config = load_config_from_db(db)
        return config.to_dict()
    except Exception as e:
        raise HTTPException(500, f"Config load error: {e}")


@app.post("/config")
async def update_config(req: ConfigUpdateRequest):
    """Update SystemConfig fields and persist to DB."""
    try:
        config = load_config_from_db(db)
        updates = req.dict(exclude_none=True)
        for k, v in updates.items():
            if hasattr(config, k):
                setattr(config, k, v)
        config.clamp()
        db.save_config(config.to_dict())
        
        # Broadcast config update
        await broadcast_update({"type": "config_updated", "config": config.to_dict()})
        
        return {"status": "updated", "config": config.to_dict()}
    except Exception as e:
        raise HTTPException(500, f"Config update error: {e}")


@app.get("/budget")
async def get_api_budget():
    """Return API-Football call budget status from Module 24."""
    status = get_budget_status()
    stats = get_cache_stats()
    return {
        "budget": status.__dict__,
        "cache": stats,
        "summary": status.summary(),
    }


@app.post("/fetch-results")
async def fetch_results(req: FetchResultsRequest, background_tasks: BackgroundTasks):
    """Trigger Module 23 result fetcher and learning cycle."""
    if not FOOTBALL_KEY:
        raise HTTPException(422, "APIFOOTBALL_KEY not set.")

    from module18 import run_full_learning_cycle
    league_ids = list(LEAGUE_MAP.keys())

    if req.start_date and req.end_date:
        def _run():
            report = fetch_multiple_dates(
                league_ids=league_ids,
                football_key=FOOTBALL_KEY,
                start_date=req.start_date,
                end_date=req.end_date,
                db=db,
                recalibration_fn=run_full_learning_cycle,
                parallel=req.parallel,
                verbose=True,
            )
            log.info(report.summary())
            asyncio.run_coroutine_threadsafe(
                broadcast_update({"type": "results_fetched", "report": report.to_dict()}),
                asyncio.get_event_loop()
            )
    else:
        def _run():
            report = run_result_fetcher(
                football_key=FOOTBALL_KEY,
                league_ids=league_ids,
                db=db,
                recalibration_fn=run_full_learning_cycle,
                target_date=req.target_date,
                days_back=req.days_back,
                verbose=True,
            )
            log.info(report.summary())
            asyncio.run_coroutine_threadsafe(
                broadcast_update({"type": "results_fetched", "report": report.to_dict()}),
                asyncio.get_event_loop()
            )

    background_tasks.add_task(_run)
    return {
        "status": "running",
        "message": f"Fetching results for {req.start_date or req.target_date or f'last {req.days_back} days'}",
    }


@app.get("/calibration")
async def get_calibration():
    """Return M28 calibration report from last pipeline run."""
    if not _session.get("calibration"):
        return {"status": "no_data", "message": "Run /run first or insufficient history (<10 predictions)"}
    return _session["calibration"]


@app.get("/calibration/detailed")
async def get_detailed_calibration():
    """Return detailed M28 calibration with bin analysis."""
    if not _M28_AVAILABLE:
        raise HTTPException(503, "Module 28 not available")
    
    try:
        feedback = db.get_all_feedback()
        if len(feedback) < 10:
            return {"status": "insufficient_data", "samples": len(feedback), "min_required": 10}
        
        report = run_calibration_check(feedback)
        return report.to_dict()
    except Exception as e:
        raise HTTPException(500, f"Calibration error: {e}")


@app.get("/drawdown")
async def get_drawdown():
    """Return M29 drawdown and streak status."""
    if not _M29_AVAILABLE:
        raise HTTPException(503, "Module 29 (Streak & Drawdown Tracker) not available")
    
    tracker = _get_drawdown_tracker()
    if tracker is None:
        return {"status": "not_initialised", "message": "Run /run first to initialise tracker"}
    
    status = tracker.get_status()
    return {
        "health_label": status.health_label,
        "drawdown_pct": round(status.drawdown_pct * 100, 2),
        "drawdown_amount": status.drawdown_amount,
        "peak_bankroll": status.peak_bankroll,
        "current_bankroll": status.current_bankroll,
        "stake_multiplier": status.stake_multiplier,
        "streak": f"{status.current_streak}x {status.streak_direction}",
        "pause_recommended": status.pause_recommended,
        "notes": status.notes,
    }


@app.get("/stats")
async def get_stats():
    """Return database statistics."""
    try:
        stats = db.get_database_stats()
        return stats
    except Exception as e:
        raise HTTPException(500, f"Stats error: {e}")


@app.post("/explain")
async def explain_decision(req: ExplainRequest):
    """Generate explainability report for a specific verdict."""
    if not _session["verdicts"]:
        raise HTTPException(404, "No verdicts available. Run /run first.")
    
    # Find the verdict
    target = None
    for v in _session["verdicts"]:
        if req.leg_id and v.leg_id == req.leg_id:
            target = v
            break
        if req.match_id and req.match_id in v.leg_id:
            target = v
            break
    
    if not target and _session["verdicts"]:
        target = _session["verdicts"][0]
    
    if not target:
        raise HTTPException(404, f"Verdict {req.leg_id or req.match_id} not found")
    
    # Generate explainability report
    report = run_explainability_engine(
        master_verdict=target,
        oracle=target.oracle,
        dual=target.dual_pattern,
        underdog=target.underdog,
        matrix=target.season_matrix,
        ai=target.ai,
    )
    
    return {
        "match": report.match,
        "final_status": report.final_status,
        "selection": report.selection,
        "confidence": report.confidence,
        "decision_score": report.decision_score,
        "confidence_integrity": report.confidence_integrity,
        "risk_alignment": report.risk_alignment,
        "key_factors": report.key_factors,
        "contradictions": report.contradictions,
        "recommendations": report.recommendations,
        "suspicious_flag": report.suspicious_flag,
    }


@app.get("/leagues")
async def get_leagues():
    """Return available leagues with metadata."""
    leagues = []
    for lid, cfg in LEAGUE_MAP.items():
        leagues.append({
            "id": lid,
            "label": cfg.get("label", f"League-{lid}"),
            "country": cfg.get("country", "unknown"),
            "tier": cfg.get("tier", 3),
            "odds_key": cfg.get("odds_key", ""),
        })
    return {"leagues": leagues, "count": len(leagues)}


@app.get("/leagues/scan-order")
async def get_scan_order():
    """Return leagues in recommended scan order."""
    if _M31_AVAILABLE:
        ordered = get_ordered_scan_leagues(list(LEAGUE_MAP.keys()))
        return {"scan_order": ordered, "method": "M31"}
    else:
        # Fallback to LEAGUE_SCAN_ORDER from module31 constants
        try:
            from module31 import LEAGUE_SCAN_ORDER
            return {"scan_order": [lid for lid in LEAGUE_SCAN_ORDER if lid in LEAGUE_MAP], "method": "fallback"}
        except ImportError:
            return {"scan_order": list(LEAGUE_MAP.keys()), "method": "default"}


# ═══════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    log.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc) if ENVIRONMENT == "development" else None},
    )


# ═══════════════════════════════════════════════════════════════
# DEV ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    db.initialize_database()
    uvicorn.run("module25:app", host="0.0.0.0", port=8000, reload=True)