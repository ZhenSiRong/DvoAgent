"""数据访问层 - aiosqlite 连接池 + 自动建表"""
import sqlite3
from pathlib import Path
import logging

import aiosqlite

logger = logging.getLogger(__name__)

# 项目根目录（相对于此文件向上 3 级）
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "devops_agent.db"


# ============================================================
#  建表 SQL：5 张新表 + nanoclaw-py 原有 2 表保留
# ============================================================

CREATE_TABLES_SQL = """
-- ========================================
--  1. 会话表
-- ========================================
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,           -- UUID v4 字符串
    title       TEXT NOT NULL DEFAULT '新对话',
    user_id     TEXT NOT NULL DEFAULT 'default',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ========================================
--  2. 消息表
-- ========================================
CREATE TABLE IF NOT EXISTS messages (
    id            TEXT PRIMARY KEY,         -- UUID
    session_id    TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role          TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system', 'tool')),
    content       TEXT NOT NULL DEFAULT '',
    tool_calls    TEXT DEFAULT '[]',         -- JSON 数组字符串
    audit_trail   TEXT DEFAULT '[]',        -- JSON 数组: ["received","sense",...]
    token_count   INTEGER,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);

-- ========================================
--  3. 审计日志表（赛题核心：五段式闭环溯源）
-- ========================================
CREATE TABLE IF NOT EXISTS audit_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    message_id      TEXT,                    -- 可为空：探针阶段可能无 message
    phase           TEXT NOT NULL CHECK(
        phase IN ('received', 'sense', 'inference', 'security_check', 'execution', 'response_ready')
    ),
    content         TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'ok' CHECK(status IN ('ok', 'warning', 'error', 'blocked')),
    security_result TEXT,                     -- PASSED | BLOCKED | WARNING | ESCALATE
    blocked_reason  TEXT,
    raw_input       TEXT,                     -- 安全校验时记录 LLM 原始输出
    raw_output      TEXT,                     -- 记录实际执行输出
    duration_ms     INTEGER NOT NULL DEFAULT 0,
    timestamp       TEXT NOT NULL DEFAULT (datetime('now')),
    command         TEXT,                     -- 执行的命令（execution 阶段专用）
    exit_code       INTEGER NOT NULL DEFAULT 0,
    executed_by     TEXT,                     -- 执行用户（如 devops-runner）
    source_ip       TEXT                      -- 请求来源 IP
);
CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_logs(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_phase ON audit_logs(phase);
CREATE INDEX IF NOT EXISTS idx_audit_message ON audit_logs(message_id);

-- ========================================
--  4. 全局配置键值对表
-- ========================================
CREATE TABLE IF NOT EXISTS configs (
    key         TEXT PRIMARY KEY,             -- 如 "llm.model_name", "llm.temperature"
    value       TEXT NOT NULL DEFAULT '',      -- JSON 或纯文本
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ========================================
--  5. 对话状态表（快速恢复上下文）
-- ========================================
CREATE TABLE IF NOT EXISTS conversation_state (
    session_id       TEXT PRIMARY KEY,
    last_message_id  TEXT,
    context_summary  TEXT,                   -- 长对话压缩摘要
    total_turns      INTEGER NOT NULL DEFAULT 0,
    updated_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ========================================
--  [保留] nanoclaw-py 原有 2 表
-- ========================================
CREATE TABLE IF NOT EXISTS scheduled_tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    cron_expr   TEXT NOT NULL DEFAULT '* * * * *',
    command     TEXT NOT NULL DEFAULT '',
    enabled     INTEGER NOT NULL DEFAULT 1,
    last_run_at TEXT,
    next_run_at TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS task_run_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     INTEGER NOT NULL REFERENCES scheduled_tasks(id),
    status      TEXT NOT NULL DEFAULT 'pending',
    output      TEXT DEFAULT '',
    error       TEXT DEFAULT '',
    started_at  TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_task_runs ON task_run_logs(task_id, started_at);

-- ========================================
--  6. 记忆表（跨会话长期记忆）
-- ========================================
CREATE TABLE IF NOT EXISTS memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    type            TEXT NOT NULL CHECK(type IN ('fact', 'summary', 'preference', 'system_state')),
    content         TEXT NOT NULL,
    source_session_id TEXT,
    importance      REAL NOT NULL DEFAULT 1.0,
    access_count    INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);

-- ========================================
--  7. 推理链路日志表（五段式闭环溯源）
-- ========================================
CREATE TABLE IF NOT EXISTS reasoning_chains (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    round_number  INTEGER NOT NULL DEFAULT 1,     -- 第几轮 tool-use loop
    stage         TEXT NOT NULL CHECK(
        stage IN ('SENSE', 'ANALYZE', 'PLAN', 'EXECUTE', 'OUTPUT')
    ),
    content       TEXT NOT NULL DEFAULT '',      -- 该阶段详细内容（JSON 或文本）
    metadata      TEXT,                          -- 补充信息 JSON（token 用量、耗时等）
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_reasoning_session ON reasoning_chains(session_id, round_number);
CREATE INDEX IF NOT EXISTS idx_reasoning_stage ON reasoning_chains(stage);

-- ========================================
--  8. 动态工具表（用户运行时注册自定义 MCP 工具）
-- ========================================
CREATE TABLE IF NOT EXISTS dynamic_tools (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT UNIQUE NOT NULL,         -- 工具标识符（LLM function name）
    description TEXT NOT NULL,                -- 工具描述（给 LLM 看）
    tool_type   TEXT NOT NULL,                -- shell | http | mcp_stdio | mcp_sse
    config      TEXT NOT NULL DEFAULT '{}',   -- JSON 配置（根据 type 不同结构）
    schema_json TEXT NOT NULL DEFAULT '{}',   -- OpenAI function schema JSON
    is_active   INTEGER NOT NULL DEFAULT 1,   -- 是否启用（0=禁用，1=启用）
    created_by  TEXT DEFAULT 'system',        -- 创建者标识
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_dynamic_tools_active ON dynamic_tools(is_active);
"""


class DatabaseManager:
    """aiosqlite 连接池管理器 — 单例模式"""

    _instance: "DatabaseManager | None" = None
    _db: aiosqlite.Connection | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def get_db(self) -> aiosqlite.Connection:
        if self._db is None:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            self._db = await aiosqlite.connect(str(DATABASE_PATH))
            # 启用外键约束
            await self._db.execute("PRAGMA foreign_keys=ON")
            # WAL 模式提升并发性能
            await self._db.execute("PRAGMA journal_mode=WAL")
            logger.info("数据库连接已建立: %s", DATABASE_PATH)
        return self._db

    async def init_tables(self) -> None:
        """执行建表 SQL（幂等，可重复调用）+ 自动迁移"""
        db = await self.get_db()
        await db.executescript(CREATE_TABLES_SQL)
        # 自动迁移：为 audit_logs 补充分区执行审计所需的列（兼容已有数据库）
        await self._migrate_audit_logs(db)
        await db.commit()
        logger.info("数据库表初始化完成（sessions, messages, audit_logs, configs, conversation_state, memories, reasoning_chains, dynamic_tools + 2 张保留表）")

    async def _migrate_audit_logs(self, db: aiosqlite.Connection) -> None:
        """检查并补充 audit_logs 缺失的列（命令执行审计专用）"""
        try:
            cursor = await db.execute("PRAGMA table_info(audit_logs)")
            rows = await cursor.fetchall()
            existing_cols = {r[1] for r in rows}
            migrations = [
                ("command", "TEXT"),
                ("exit_code", "INTEGER NOT NULL DEFAULT 0"),
                ("executed_by", "TEXT"),
                ("source_ip", "TEXT"),
            ]
            for col_name, col_def in migrations:
                if col_name not in existing_cols:
                    await db.execute(f"ALTER TABLE audit_logs ADD COLUMN {col_name} {col_def}")
                    logger.info("迁移 audit_logs: 新增列 %s", col_name)
        except Exception as e:
            logger.warning("audit_logs 迁移检查失败(非阻塞): %s", e)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
            logger.info("数据库连接已关闭")


# 全局单例
db_manager = DatabaseManager()


async def get_db() -> aiosqlite.Connection:
    """FastAPI 依赖注入用：获取数据库连接"""
    return await db_manager.get_db()


async def fetchall_as_dicts(
    db: aiosqlite.Connection,
    sql: str,
    params: tuple = (),
) -> list[dict]:
    """执行查询并返回 dict 列表。

    aiosqlite 的 execute_fetchall 默认返回 tuple，
    此函数自动将 tuple 转为 dict（用 cursor.description 映射列名）。
    """
    cursor = await db.execute(sql, params)
    rows = await cursor.fetchall()
    if not rows:
        await cursor.close()
        return []
    columns = [col[0] for col in cursor.description] if cursor.description else []
    await cursor.close()
    return [dict(zip(columns, row)) for row in rows]
