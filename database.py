# =============================================================================
# database.py — Работа с SQLite: предупреждения и статистика
# =============================================================================

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import config

logger = logging.getLogger("antispam.database")


@contextmanager
def get_conn():
    """Контекстный менеджер для подключения к SQLite."""
    conn = sqlite3.connect(config.DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """
    Создаёт таблицы базы данных если они не существуют.
    Вызывается при запуске бота.
    """
    Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    with get_conn() as conn:
        conn.executescript("""
            -- Таблица предупреждений (составной ключ: пользователь + сервер)
            -- Каждый сервер хранит предупреждения независимо!
            CREATE TABLE IF NOT EXISTS warnings (
                user_id     INTEGER NOT NULL,
                guild_id    INTEGER NOT NULL,
                count       INTEGER NOT NULL DEFAULT 0,
                updated_at  TIMESTAMP NOT NULL,
                PRIMARY KEY (user_id, guild_id)
            );

            -- Таблица лога нарушений
            CREATE TABLE IF NOT EXISTS violations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                guild_id    INTEGER NOT NULL,
                username    TEXT NOT NULL,
                filename    TEXT,
                confidence  REAL,
                method      TEXT,
                action      TEXT,
                created_at  TIMESTAMP NOT NULL
            );

            -- Таблица настроек сервера (лог-канал для каждого сервера свой)
            CREATE TABLE IF NOT EXISTS guild_settings (
                guild_id        INTEGER PRIMARY KEY,
                log_channel_id  INTEGER,
                updated_at      TIMESTAMP NOT NULL
            );

            -- Таблица статистики
            CREATE TABLE IF NOT EXISTS stats (
                key     TEXT PRIMARY KEY,
                value   INTEGER NOT NULL DEFAULT 0
            );

            -- Начальные значения статистики
            INSERT OR IGNORE INTO stats (key, value) VALUES ('images_checked', 0);
            INSERT OR IGNORE INTO stats (key, value) VALUES ('violations_found', 0);
            INSERT OR IGNORE INTO stats (key, value) VALUES ('users_muted', 0);
        """)

    logger.info("База данных инициализирована: %s", config.DB_PATH)


# ---------------------------------------------------------------------------
# Работа с предупреждениями
# ---------------------------------------------------------------------------

def get_warnings(user_id: int, guild_id: int) -> int:
    """Возвращает текущее количество предупреждений пользователя."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT count FROM warnings WHERE user_id = ? AND guild_id = ?",
            (user_id, guild_id),
        ).fetchone()
    return row["count"] if row else 0


def add_warning(user_id: int, guild_id: int) -> int:
    """
    Добавляет предупреждение пользователю.

    Returns:
        Новое общее количество предупреждений.
    """
    now = datetime.utcnow()
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO warnings (user_id, guild_id, count, updated_at)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(user_id, guild_id) DO UPDATE SET
                count      = count + 1,
                updated_at = excluded.updated_at
        """, (user_id, guild_id, now))

        # Фильтруем по обоим полям — предупреждения независимы на каждом сервере
        row = conn.execute(
            "SELECT count FROM warnings WHERE user_id = ? AND guild_id = ?",
            (user_id, guild_id),
        ).fetchone()

    return row["count"]


def reset_warnings(user_id: int, guild_id: int) -> None:
    """Сбрасывает предупреждения пользователя (используется после мута или командой)."""
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM warnings WHERE user_id = ? AND guild_id = ?",
            (user_id, guild_id),
        )
    logger.info("Предупреждения сброшены для user_id=%s", user_id)


# ---------------------------------------------------------------------------
# Лог нарушений
# ---------------------------------------------------------------------------

def log_violation(
    user_id: int,
    guild_id: int,
    username: str,
    filename: str,
    confidence: float,
    method: str,
    action: str,
) -> None:
    """Записывает нарушение в таблицу violations."""
    now = datetime.utcnow()
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO violations
                (user_id, guild_id, username, filename, confidence, method, action, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, guild_id, username, filename, confidence, method, action, now))


# ---------------------------------------------------------------------------
# Статистика
# ---------------------------------------------------------------------------

def increment_stat(key: str, amount: int = 1) -> None:
    """Увеличивает значение счётчика статистики."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE stats SET value = value + ? WHERE key = ?",
            (amount, key),
        )


def get_stats() -> dict[str, int]:
    """Возвращает словарь со всеми значениями статистики."""
    with get_conn() as conn:
        rows = conn.execute("SELECT key, value FROM stats").fetchall()
    return {row["key"]: row["value"] for row in rows}


# ---------------------------------------------------------------------------
# Настройки сервера (per-guild)
# ---------------------------------------------------------------------------

def get_log_channel(guild_id: int) -> int | None:
    """
    Возвращает ID лог-канала для конкретного сервера.
    Администратор каждого сервера устанавливает свой канал через !setlogchannel.

    Returns:
        ID канала или None если не настроен.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT log_channel_id FROM guild_settings WHERE guild_id = ?",
            (guild_id,),
        ).fetchone()
    return row["log_channel_id"] if row else None


def set_log_channel(guild_id: int, channel_id: int) -> None:
    """Сохраняет ID лог-канала для сервера."""
    now = datetime.utcnow()
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO guild_settings (guild_id, log_channel_id, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(guild_id) DO UPDATE SET
                log_channel_id = excluded.log_channel_id,
                updated_at     = excluded.updated_at
        """, (guild_id, channel_id, now))
    logger.info("Лог-канал сервера %s → канал %s", guild_id, channel_id)
