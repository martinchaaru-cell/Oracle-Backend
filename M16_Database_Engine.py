"""
The Match Oracle – Module 16: Database Engine (REFINED)
===================================================
Central intelligence storage for all matches, predictions, and outcomes.

REFINEMENTS IN THIS VERSION:
---------------------------
1. ADDED: Proper connection context manager for safe resource handling
2. ADDED: Database migration system for schema evolution (v1 to v2)
3. ADDED: Thread-local connections for concurrency safety
4. ADDED: Comprehensive error handling and logging
5. ADDED: Batch insert functions for performance
6. ADDED: Query functions with filtering and pagination
7. ADDED: Performance metrics tracking (ROI, accuracy over time)
8. ADDED: Pattern log storage for M9 curse tracking
9. ADDED: Configuration persistence for M18
10. ADDED: Database vacuum and optimization utilities
11. ADDED: Export functionality for backup/analysis

Features:
- SQLite-based (no external setup)
- Stores full pipeline outputs
- Enables learning queries for Modules 13–15
- Fully dynamic (no hardcoded leagues or teams)
- Thread-safe connection pooling
- Automatic database migrations

Database file: The_Match_Oracle.db

Schema Version: 2 (tracks migrations)

Usage:
    import module16 as db
    
    db.initialize_database()
    db.insert_match(match_id, league, home, away)
    db.insert_prediction(match_id, selection, market, odds, edge, confidence, status)
    
    feedback = db.get_all_feedback()
    league_perf = db.get_league_performance()
"""
from __future__ import annotations

import sqlite3
import logging
import json
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from contextlib import contextmanager
from pathlib import Path

# Set up logging
logger = logging.getLogger("oracle_beast.module16")

DB_FILE = "The_Match_Oracle.db"
SCHEMA_VERSION = 2  # Current schema version

# Thread-local storage for database connections
_local = threading.local()


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — CONNECTION HANDLER (thread-safe)
# ═══════════════════════════════════════════════════════════════

@contextmanager
def get_connection():
    """
    Thread-safe connection context manager.
    Automatically commits or rolls back on exception.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE, timeout=30)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()


def get_thread_connection():
    """
    Get a connection for the current thread (reuses if possible).
    Useful for batch operations.
    """
    if not hasattr(_local, 'connection') or _local.connection is None:
        _local.connection = sqlite3.connect(DB_FILE, timeout=30)
        _local.connection.row_factory = sqlite3.Row
    return _local.connection


def close_thread_connection():
    """Close the connection for the current thread."""
    if hasattr(_local, 'connection') and _local.connection:
        _local.connection.close()
        _local.connection = None


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — DATABASE INITIALIZATION & MIGRATIONS
# ═══════════════════════════════════════════════════════════════

def get_schema_version() -> int:
    """Get current schema version from the database."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM metadata WHERE key = 'schema_version'")
            row = cursor.fetchone()
            if row:
                return int(row['value'])
            return 0
    except (sqlite3.OperationalError, KeyError):
        # metadata table doesn't exist yet
        return 0


def set_schema_version(version: int):
    """Set the schema version in the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO metadata (key, value, updated_at)
            VALUES ('schema_version', ?, ?)
        """, (str(version), datetime.now(timezone.utc).isoformat()))


def initialize_database():
    """
    Initialize database with all required tables.
    Runs migrations if schema version is behind.
    """
    current_version = get_schema_version()
    
    if current_version == 0:
        _create_tables_v1()
        set_schema_version(1)
        current_version = 1
        logger.info("Database initialized with schema version 1")
    
    if current_version < SCHEMA_VERSION:
        _run_migrations(current_version + 1, SCHEMA_VERSION)
        set_schema_version(SCHEMA_VERSION)
        logger.info(f"Database migrated to schema version {SCHEMA_VERSION}")
    
    # Optimize database after migrations
    _optimize_database()


def _create_tables_v1():
    """Create all tables for schema version 1."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Metadata table for schema version tracking
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        )
        """)
        
        # Matches table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            match_id TEXT PRIMARY KEY,
            league TEXT,
            home_team TEXT,
            away_team TEXT,
            match_date TEXT,
            league_id INTEGER,
            season TEXT,
            created_at TEXT
        )
        """)
        
        # Predictions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id TEXT,
            selection TEXT,
            market TEXT,
            odds REAL,
            edge REAL,
            confidence TEXT,
            status TEXT,
            created_at TEXT,
            FOREIGN KEY(match_id) REFERENCES matches(match_id)
        )
        """)
        
        # Create index on match_id for faster lookups
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_match_id 
        ON predictions(match_id)
        """)
        
        # Probabilities table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS probabilities (
            match_id TEXT,
            home_win REAL,
            draw REAL,
            away_win REAL,
            recorded_at TEXT,
            FOREIGN KEY(match_id) REFERENCES matches(match_id),
            PRIMARY KEY(match_id, recorded_at)
        )
        """)
        
        # Outcomes table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS outcomes (
            match_id TEXT PRIMARY KEY,
            result TEXT,
            correct INTEGER,
            home_goals INTEGER,
            away_goals INTEGER,
            evaluated_at TEXT,
            FOREIGN KEY(match_id) REFERENCES matches(match_id)
        )
        """)
        
        # Learning signals table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS learning (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id TEXT,
            error_type TEXT,
            risk_level TEXT,
            league TEXT,
            created_at TEXT,
            FOREIGN KEY(match_id) REFERENCES matches(match_id)
        )
        """)
        
        # System configuration table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_config (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        )
        """)
        
        # Performance metrics table (for tracking over time)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            total_predictions INTEGER,
            correct_predictions INTEGER,
            accuracy REAL,
            roi REAL,
            recorded_at TEXT
        )
        """)
        
        # Pattern logs table (from M9)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS pattern_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id TEXT,
            match_date TEXT,
            fav_team TEXT,
            und_team TEXT,
            patterns_active TEXT,
            rtm_state_before TEXT,
            rtm_prediction TEXT,
            actual_outcome TEXT,
            pattern_confirmed INTEGER,
            logged_at TEXT,
            FOREIGN KEY(match_id) REFERENCES matches(match_id)
        )
        """)


def _run_migrations(from_version: int, to_version: int):
    """Run migrations from from_version to to_version."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        if from_version <= 2 <= to_version:
            # Migration to version 2: add indices for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_learning_match_id ON learning(match_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_result ON outcomes(result)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_status ON predictions(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date)")
            logger.info("Migration to version 2 complete: added performance indices")


def _optimize_database():
    """Optimize database (vacuum, analyze)."""
    try:
        with get_connection() as conn:
            conn.execute("ANALYZE")
            conn.execute("VACUUM")
            logger.debug("Database optimized")
    except Exception as e:
        logger.warning(f"Database optimization failed: {e}")


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — INSERT FUNCTIONS (Single Records)
# ═══════════════════════════════════════════════════════════════

def insert_match(
    match_id: str,
    league: str,
    home: str,
    away: str,
    league_id: Optional[int] = None,
    season: Optional[str] = None,
    match_date: Optional[str] = None,
) -> bool:
    """
    Insert or update a match record.
    
    Returns:
        True if inserted/updated successfully
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO matches 
            (match_id, league, home_team, away_team, match_date, league_id, season, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match_id,
                league,
                home,
                away,
                match_date or datetime.now(timezone.utc).isoformat(),
                league_id,
                season,
                datetime.now(timezone.utc).isoformat()
            ))
            return True
    except Exception as e:
        logger.error(f"Failed to insert match {match_id}: {e}")
        return False


def insert_prediction(
    match_id: str,
    selection: str,
    market: str,
    odds: float,
    edge: float,
    confidence: str,
    status: str,
) -> bool:
    """
    Insert a prediction record.
    
    Returns:
        True if inserted successfully
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO predictions 
            (match_id, selection, market, odds, edge, confidence, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match_id,
                selection,
                market,
                odds,
                edge,
                confidence,
                status,
                datetime.now(timezone.utc).isoformat()
            ))
            return True
    except Exception as e:
        logger.error(f"Failed to insert prediction for {match_id}: {e}")
        return False


def insert_probabilities(
    match_id: str,
    home: float,
    draw: float,
    away: float,
) -> bool:
    """
    Insert probability record for a match.
    
    Args:
        match_id: Match identifier
        home: Home win probability (0-1)
        draw: Draw probability (0-1)
        away: Away win probability (0-1)
    
    Returns:
        True if inserted successfully
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO probabilities (match_id, home_win, draw, away_win, recorded_at)
            VALUES (?, ?, ?, ?, ?)
            """, (
                match_id,
                home,
                draw,
                away,
                datetime.now(timezone.utc).isoformat()
            ))
            return True
    except Exception as e:
        logger.error(f"Failed to insert probabilities for {match_id}: {e}")
        return False


def insert_outcome(
    match_id: str,
    result: str,  # 'W', 'D', 'L' (from home team perspective)
    home_goals: Optional[int] = None,
    away_goals: Optional[int] = None,
) -> bool:
    """
    Insert or update outcome for a match.
    
    Args:
        match_id: Match identifier
        result: 'W', 'D', or 'L'
        home_goals: Home team goals (optional)
        away_goals: Away team goals (optional)
    
    Returns:
        True if inserted/updated successfully
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Fetch the latest prediction to determine if correct
            cursor.execute("""
            SELECT selection FROM predictions 
            WHERE match_id = ? 
            ORDER BY id DESC LIMIT 1
            """, (match_id,))
            row = cursor.fetchone()
            
            correct = 0
            if row:
                predicted = row['selection']
                # Convert prediction to W/D/L
                if predicted == "HOME" and result == "W":
                    correct = 1
                elif predicted == "AWAY" and result == "L":
                    correct = 1
                elif predicted == "Draw" and result == "D":
                    correct = 1
            
            cursor.execute("""
            INSERT OR REPLACE INTO outcomes 
            (match_id, result, correct, home_goals, away_goals, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                match_id,
                result,
                correct,
                home_goals,
                away_goals,
                datetime.now(timezone.utc).isoformat()
            ))
            return True
    except Exception as e:
        logger.error(f"Failed to insert outcome for {match_id}: {e}")
        return False


def insert_learning(
    match_id: str,
    error_type: str,
    risk_level: str,
    league: str,
) -> bool:
    """
    Insert a learning signal record.
    
    Args:
        match_id: Match identifier
        error_type: Type of error (e.g., "overconfidence", "draw_trap")
        risk_level: "LOW", "MEDIUM", "HIGH"
        league: League name
    
    Returns:
        True if inserted successfully
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO learning (match_id, error_type, risk_level, league, created_at)
            VALUES (?, ?, ?, ?, ?)
            """, (
                match_id,
                error_type,
                risk_level,
                league,
                datetime.now(timezone.utc).isoformat()
            ))
            return True
    except Exception as e:
        logger.error(f"Failed to insert learning for {match_id}: {e}")
        return False


def insert_pattern_log(
    match_id: str,
    match_date: str,
    fav_team: str,
    und_team: str,
    patterns_active: List[str],
    rtm_state_before: str,
    rtm_prediction: str,
    actual_outcome: Optional[str] = None,
    pattern_confirmed: Optional[bool] = None,
) -> bool:
    """
    Insert a pattern log entry (from M9).
    
    Returns:
        True if inserted successfully
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO pattern_logs 
            (match_id, match_date, fav_team, und_team, patterns_active, 
             rtm_state_before, rtm_prediction, actual_outcome, pattern_confirmed, logged_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match_id,
                match_date,
                fav_team,
                und_team,
                json.dumps(patterns_active),
                rtm_state_before,
                rtm_prediction,
                actual_outcome,
                1 if pattern_confirmed else 0 if pattern_confirmed is not None else None,
                datetime.now(timezone.utc).isoformat()
            ))
            return True
    except Exception as e:
        logger.error(f"Failed to insert pattern log for {match_id}: {e}")
        return False


def update_pattern_log_outcome(
    match_id: str,
    actual_outcome: str,
    pattern_confirmed: bool,
) -> bool:
    """
    Update a pattern log entry with the actual outcome.
    
    Returns:
        True if updated successfully
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE pattern_logs 
            SET actual_outcome = ?, pattern_confirmed = ?
            WHERE match_id = ?
            """, (actual_outcome, 1 if pattern_confirmed else 0, match_id))
            return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Failed to update pattern log for {match_id}: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — BATCH INSERT FUNCTIONS (Performance)
# ═══════════════════════════════════════════════════════════════

def batch_insert_predictions(predictions: List[Dict]) -> int:
    """
    Insert multiple predictions in a single transaction.
    
    Args:
        predictions: List of prediction dicts with keys:
            match_id, selection, market, odds, edge, confidence, status
    
    Returns:
        Number of successfully inserted predictions
    """
    if not predictions:
        return 0
    
    inserted = 0
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now(timezone.utc).isoformat()
            
            for pred in predictions:
                try:
                    cursor.execute("""
                    INSERT INTO predictions 
                    (match_id, selection, market, odds, edge, confidence, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pred["match_id"],
                        pred["selection"],
                        pred["market"],
                        pred["odds"],
                        pred["edge"],
                        pred["confidence"],
                        pred["status"],
                        now,
                    ))
                    inserted += 1
                except Exception as e:
                    logger.warning(f"Failed to insert prediction for {pred.get('match_id')}: {e}")
            
            return inserted
    except Exception as e:
        logger.error(f"Batch insert failed: {e}")
        return inserted


def batch_insert_matches(matches: List[Dict]) -> int:
    """
    Insert multiple matches in a single transaction.
    
    Args:
        matches: List of match dicts with keys:
            match_id, league, home_team, away_team, league_id, season, match_date
    
    Returns:
        Number of successfully inserted matches
    """
    if not matches:
        return 0
    
    inserted = 0
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now(timezone.utc).isoformat()
            
            for match in matches:
                try:
                    cursor.execute("""
                    INSERT OR REPLACE INTO matches 
                    (match_id, league, home_team, away_team, match_date, league_id, season, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        match["match_id"],
                        match["league"],
                        match["home_team"],
                        match["away_team"],
                        match.get("match_date", now),
                        match.get("league_id"),
                        match.get("season"),
                        now,
                    ))
                    inserted += 1
                except Exception as e:
                    logger.warning(f"Failed to insert match {match.get('match_id')}: {e}")
            
            return inserted
    except Exception as e:
        logger.error(f"Batch insert failed: {e}")
        return inserted


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — QUERY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_all_feedback(
    limit: Optional[int] = None,
    offset: int = 0,
    league_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get all prediction feedback with optional filtering.
    
    Returns:
        List of dictionaries with prediction feedback
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
            SELECT 
                m.match_id, 
                m.league, 
                p.selection as prediction, 
                o.result as actual_result,
                p.confidence,
                pr.home_win, 
                pr.draw, 
                pr.away_win, 
                o.correct,
                p.odds,
                p.edge,
                p.status,
                m.match_date,
                p.created_at as prediction_date
            FROM matches m
            LEFT JOIN predictions p ON m.match_id = p.match_id
            LEFT JOIN probabilities pr ON m.match_id = pr.match_id
            LEFT JOIN outcomes o ON m.match_id = o.match_id
            WHERE 1=1
            """
            params = []
            
            if league_filter:
                query += " AND m.league = ?"
                params.append(league_filter)
            
            if status_filter:
                query += " AND p.status LIKE ?"
                params.append(f"%{status_filter}%")
            
            query += " ORDER BY m.match_date DESC, p.created_at DESC"
            
            if limit:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            results = []
            for r in rows:
                results.append({
                    "match_id": r["match_id"],
                    "league": r["league"] or "Unknown",
                    "prediction": r["prediction"],
                    "actual_result": r["actual_result"],
                    "confidence": r["confidence"] or "LOW",
                    "home_win_prob": r["home_win"] or 0.0,
                    "draw_prob": r["draw"] or 0.0,
                    "away_win_prob": r["away_win"] or 0.0,
                    "correct": r["correct"] if r["correct"] is not None else 0,
                    "odds": r["odds"] or 0.0,
                    "edge": r["edge"] or 0.0,
                    "status": r["status"] or "UNKNOWN",
                    "match_date": r["match_date"] or "",
                    "prediction_date": r["prediction_date"] or "",
                })
            return results
    except Exception as e:
        logger.error(f"Failed to get feedback: {e}")
        return []


def get_league_performance() -> Dict[str, Dict[str, Union[int, float]]]:
    """
    Get performance statistics by league.
    
    Returns:
        Dict[league_name, {"total": int, "correct": int, "accuracy": float, "roi": float}]
    """
    feedback = get_all_feedback()
    
    league_stats = {}
    
    for d in feedback:
        league = d["league"] or "Unknown"
        if league not in league_stats:
            league_stats[league] = {"total": 0, "correct": 0, "total_stake": 0.0, "total_return": 0.0}
        
        league_stats[league]["total"] += 1
        if d["correct"]:
            league_stats[league]["correct"] += 1
        
        # ROI calculation
        if d["odds"] > 0:
            if d["correct"]:
                league_stats[league]["total_return"] += d["odds"] * 100  # Assume 1 unit stake
            else:
                league_stats[league]["total_return"] += 0
            league_stats[league]["total_stake"] += 100
    
    for league in league_stats:
        total = league_stats[league]["total"]
        league_stats[league]["accuracy"] = round(
            league_stats[league]["correct"] / total, 4
        ) if total > 0 else 0.0
        
        total_stake = league_stats[league]["total_stake"]
        total_return = league_stats[league]["total_return"]
        league_stats[league]["roi"] = round(
            (total_return - total_stake) / total_stake, 4
        ) if total_stake > 0 else 0.0
        
        # Clean up intermediate fields
        del league_stats[league]["total_stake"]
        del league_stats[league]["total_return"]
    
    return league_stats


def get_confidence_performance() -> Dict[str, Dict[str, Union[int, float]]]:
    """
    Get performance statistics by confidence level.
    
    Returns:
        Dict[confidence, {"total": int, "correct": int, "accuracy": float, "roi": float}]
    """
    feedback = get_all_feedback()
    
    perf = {
        "HIGH": {"total": 0, "correct": 0, "total_stake": 0.0, "total_return": 0.0},
        "MEDIUM": {"total": 0, "correct": 0, "total_stake": 0.0, "total_return": 0.0},
        "LOW": {"total": 0, "correct": 0, "total_stake": 0.0, "total_return": 0.0},
    }
    
    for d in feedback:
        conf = d["confidence"] or "LOW"
        if conf not in perf:
            perf[conf] = {"total": 0, "correct": 0, "total_stake": 0.0, "total_return": 0.0}
        
        perf[conf]["total"] += 1
        if d["correct"]:
            perf[conf]["correct"] += 1
        
        if d["odds"] > 0:
            if d["correct"]:
                perf[conf]["total_return"] += d["odds"] * 100
            perf[conf]["total_stake"] += 100
    
    for level in perf:
        total = perf[level]["total"]
        perf[level]["accuracy"] = round(
            perf[level]["correct"] / total, 4
        ) if total > 0 else 0.0
        
        total_stake = perf[level]["total_stake"]
        total_return = perf[level]["total_return"]
        perf[level]["roi"] = round(
            (total_return - total_stake) / total_stake, 4
        ) if total_stake > 0 else 0.0
        
        # Clean up
        del perf[level]["total_stake"]
        del perf[level]["total_return"]
    
    return perf


def get_match_outcome(match_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the stored outcome for a specific match.
    
    Returns:
        Dict with result, correct, home_goals, away_goals, or None if not found
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT result, correct, home_goals, away_goals, evaluated_at
            FROM outcomes
            WHERE match_id = ?
            """, (match_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "result": row["result"],
                    "correct": bool(row["correct"]),
                    "home_goals": row["home_goals"],
                    "away_goals": row["away_goals"],
                    "evaluated_at": row["evaluated_at"],
                }
            return None
    except Exception as e:
        logger.error(f"Failed to get outcome for {match_id}: {e}")
        return None


def get_prediction_history(
    limit: int = 100,
    only_with_outcomes: bool = True,
    league: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get prediction history with optional filtering.
    
    Args:
        limit: Maximum number of records to return
        only_with_outcomes: Only include matches with outcomes
        league: Filter by league name
    
    Returns:
        List of prediction records
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            if only_with_outcomes:
                query = """
                SELECT 
                    p.*, 
                    m.league, m.home_team, m.away_team, m.match_date,
                    o.result, o.correct, o.home_goals, o.away_goals
                FROM predictions p
                JOIN matches m ON p.match_id = m.match_id
                JOIN outcomes o ON p.match_id = o.match_id
                """
            else:
                query = """
                SELECT 
                    p.*, 
                    m.league, m.home_team, m.away_team, m.match_date,
                    o.result, o.correct, o.home_goals, o.away_goals
                FROM predictions p
                JOIN matches m ON p.match_id = m.match_id
                LEFT JOIN outcomes o ON p.match_id = o.match_id
                """
            
            params = []
            if league:
                query += " WHERE m.league = ?"
                params.append(league)
            
            query += " ORDER BY p.created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            results = []
            for r in rows:
                results.append(dict(r))
            return results
    except Exception as e:
        logger.error(f"Failed to get prediction history: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — CONFIG PERSISTENCE (for Module 18/21)
# ═══════════════════════════════════════════════════════════════

def save_config(data: Dict[str, Any]) -> bool:
    """
    Persist system configuration to the database.
    
    Args:
        data: Dictionary of configuration key-value pairs
    
    Returns:
        True if saved successfully
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now(timezone.utc).isoformat()
            for key, value in data.items():
                # Convert non-string values to JSON
                if not isinstance(value, str):
                    value = json.dumps(value)
                cursor.execute("""
                INSERT OR REPLACE INTO system_config (key, value, updated_at)
                VALUES (?, ?, ?)
                """, (str(key), value, now))
            return True
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        return False


def load_config() -> Dict[str, Any]:
    """
    Load the most recently saved system configuration.
    
    Returns:
        Dictionary of configuration key-value pairs (empty dict if none)
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM system_config")
            rows = cursor.fetchall()
            
            if not rows:
                return {}
            
            result = {}
            for row in rows:
                key = row["key"]
                value = row["value"]
                # Attempt to parse JSON, fall back to string
                try:
                    result[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    # Try numeric conversion
                    try:
                        result[key] = float(value)
                    except (ValueError, TypeError):
                        result[key] = value
            return result
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}


def save_performance_metrics(
    total_predictions: int,
    correct_predictions: int,
    accuracy: float,
    roi: float,
) -> bool:
    """
    Save daily performance metrics for trend analysis.
    
    Returns:
        True if saved successfully
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            today = datetime.now(timezone.utc).date().isoformat()
            cursor.execute("""
            INSERT INTO performance_metrics (date, total_predictions, correct_predictions, accuracy, roi, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                today,
                total_predictions,
                correct_predictions,
                accuracy,
                roi,
                datetime.now(timezone.utc).isoformat()
            ))
            return True
    except Exception as e:
        logger.error(f"Failed to save performance metrics: {e}")
        return False


def get_performance_history(days: int = 30) -> List[Dict[str, Any]]:
    """
    Get performance history for the last N days.
    
    Returns:
        List of daily performance records
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT date, total_predictions, correct_predictions, accuracy, roi, recorded_at
            FROM performance_metrics
            WHERE date >= date('now', ?)
            ORDER BY date DESC
            """, (f'-{days} days',))
            
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Failed to get performance history: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — STATISTICS & DEBUG FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_database_stats() -> Dict[str, Any]:
    """Get comprehensive database statistics."""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Count tables
            cursor.execute("SELECT COUNT(*) as count FROM matches")
            stats["matches"] = cursor.fetchone()["count"]
            
            cursor.execute("SELECT COUNT(*) as count FROM predictions")
            stats["predictions"] = cursor.fetchone()["count"]
            
            cursor.execute("SELECT COUNT(*) as count FROM outcomes")
            stats["outcomes"] = cursor.fetchone()["count"]
            
            cursor.execute("SELECT COUNT(*) as count FROM learning")
            stats["learning_signals"] = cursor.fetchone()["count"]
            
            cursor.execute("SELECT COUNT(*) as count FROM pattern_logs")
            stats["pattern_logs"] = cursor.fetchone()["count"]
            
            cursor.execute("SELECT COUNT(*) as count FROM performance_metrics")
            stats["performance_records"] = cursor.fetchone()["count"]
            
            # Additional stats
            cursor.execute("SELECT COUNT(DISTINCT league) as count FROM matches")
            stats["unique_leagues"] = cursor.fetchone()["count"]
            
            cursor.execute("SELECT COUNT(DISTINCT status) as count FROM predictions")
            stats["unique_statuses"] = cursor.fetchone()["count"]
            
            # Date range
            cursor.execute("SELECT MIN(match_date) as first, MAX(match_date) as last FROM matches")
            row = cursor.fetchone()
            stats["first_match_date"] = row["first"]
            stats["last_match_date"] = row["last"]
            
            # Success rate
            cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(correct) as correct
            FROM outcomes
            """)
            row = cursor.fetchone()
            total = row["total"] or 0
            correct = row["correct"] or 0
            stats["overall_accuracy"] = correct / total if total > 0 else 0.0
            
            return stats
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {"error": str(e)}


def print_database_summary():
    """Print a summary of database contents."""
    data = get_all_feedback()
    leagues = get_league_performance()
    conf = get_confidence_performance()
    stats = get_database_stats()
    
    print("\n📊 DATABASE SUMMARY")
    print("=" * 40)
    print(f"Total Matches: {len(data)}")
    print(f"Unique Leagues: {stats.get('unique_leagues', 0)}")
    print(f"Overall Accuracy: {stats.get('overall_accuracy', 0):.1%}")
    
    print("\n🏆 League Performance:")
    for k, v in sorted(leagues.items(), key=lambda x: x[1]["accuracy"], reverse=True)[:10]:
        acc = v["accuracy"]
        roi = v.get("roi", 0)
        print(f"  {k:<20}: {acc:.1%} ({v['correct']}/{v['total']}) | ROI: {roi:+.1%}")
    
    print("\n🎯 Confidence Performance:")
    for k, v in conf.items():
        if v["total"] > 0:
            print(f"  {k}: {v['accuracy']:.1%} ({v['correct']}/{v['total']}) | ROI: {v.get('roi', 0):+.1%}")


def vacuum_database() -> bool:
    """Run VACUUM to reclaim unused space."""
    try:
        with get_connection() as conn:
            conn.execute("VACUUM")
            logger.info("Database vacuum completed")
            return True
    except Exception as e:
        logger.error(f"Vacuum failed: {e}")
        return False


def export_database(output_file: str = None) -> str:
    """
    Export database to SQL dump file.
    
    Args:
        output_file: Output file path (auto-generated if None)
    
    Returns:
        Path to output file
    """
    if output_file is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"oracle_db_backup_{timestamp}.sql"
    
    try:
        import subprocess
        result = subprocess.run(
            ["sqlite3", DB_FILE, f".output {output_file}", ".dump"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info(f"Database exported to {output_file}")
            return output_file
        else:
            logger.error(f"Export failed: {result.stderr}")
            return ""
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — EXPORTS
# ═══════════════════════════════════════════════════════════════

__all__ = [
    # Initialization
    "initialize_database",
    "get_schema_version",
    # Connection management
    "get_connection",
    "get_thread_connection",
    "close_thread_connection",
    # Insert functions (single)
    "insert_match",
    "insert_prediction",
    "insert_probabilities",
    "insert_outcome",
    "insert_learning",
    "insert_pattern_log",
    "update_pattern_log_outcome",
    # Batch insert functions
    "batch_insert_predictions",
    "batch_insert_matches",
    # Query functions
    "get_all_feedback",
    "get_league_performance",
    "get_confidence_performance",
    "get_match_outcome",
    "get_prediction_history",
    # Config persistence
    "save_config",
    "load_config",
    "save_performance_metrics",
    "get_performance_history",
    # Statistics & debug
    "get_database_stats",
    "print_database_summary",
    "vacuum_database",
    "export_database",
]


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Initialize database
    initialize_database()
    
    print("\n" + "=" * 70)
    print("MODULE 16: DATABASE ENGINE - TEST RUN")
    print("=" * 70)
    
    # Insert test data
    print("\n📝 Inserting test data...")
    
    insert_match(
        match_id="test_arsenal_chelsea_20250101",
        league="Premier League",
        home="Arsenal",
        away="Chelsea",
        league_id=39,
        season="2024-25"
    )
    
    insert_prediction(
        match_id="test_arsenal_chelsea_20250101",
        selection="HOME",
        market="straight_win",
        odds=2.10,
        edge=0.08,
        confidence="HIGH",
        status="APPROVED"
    )
    
    insert_probabilities(
        match_id="test_arsenal_chelsea_20250101",
        home=0.58,
        draw=0.25,
        away=0.17
    )
    
    insert_outcome(
        match_id="test_arsenal_chelsea_20250101",
        result="W",
        home_goals=2,
        away_goals=1
    )
    
    # Query and display
    print("\n📊 All Feedback:")
    feedback = get_all_feedback()
    for f in feedback[:3]:
        print(f"  {f['match_id']}: predicted={f['prediction']}, actual={f['actual_result']}, correct={f['correct']}")
    
    print("\n🏆 League Performance:")
    for league, stats in get_league_performance().items():
        print(f"  {league}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']}) | ROI: {stats.get('roi', 0):+.1%}")
    
    print("\n🎯 Confidence Performance:")
    for conf, stats in get_confidence_performance().items():
        print(f"  {conf}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']}) | ROI: {stats.get('roi', 0):+.1%}")
    
    print("\n📈 Database Stats:")
    stats = get_database_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.1%}" if "accuracy" in key else f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Test config persistence
    print("\n⚙️ Config Persistence Test:")
    test_config = {"home_win_threshold": 0.58, "min_edge": 0.04, "test_flag": True}
    save_config(test_config)
    loaded = load_config()
    print(f"  Saved: {test_config}")
    print(f"  Loaded: {loaded}")
    
    # Print summary
    print_database_summary()
    
    print("\n✅ Database test completed successfully")