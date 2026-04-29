"""全局配置 - 从 .env 读取所有环境变量

动态配置机制：
1. Settings 提供 .env 默认值（启动时固定）
2. DB configs 表提供运行时覆盖值（通过 /api/v1/config 修改）
3. agent/core.py 使用 get_llm_runtime_config() 获取合并后的配置
"""
from dataclasses import dataclass

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # === 应用 ===
    app_name: str = "DevOps-Agent"
    app_debug: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # === LLM（OpenAI 协议） ===
    llm_protocol: str = "openai"  # openai | anthropic
    llm_base_url: str = "https://api.minimaxi.com/v1"
    llm_api_key: str = ""  # 请在 .env 中配置 LLM_API_KEY
    llm_model: str = "MiniMax-M2.1"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096

    # === LLM（Anthropic 协议备用） ===
    anthropic_base_url: str = "https://api.minimaxi.com/anthropic"
    anthropic_api_key: str = ""  # 请在 .env 中配置 ANTHROPIC_API_KEY
    anthropic_model: str = "MiniMax-M2.1"

    # === 数据库 ===
    database_url: str = "sqlite+aiosqlite:///./data/devops_agent.db"

    # === 安全层 ===
    safe_exec_user: str = "devops-runner"
    exec_timeout: int = 30
    sudo_whitelist: str = (
        "logrotate,systemctl status,systemctl restart,"
        "journalctl,df,du,ps,netstat,ss,cat,ls,find,"
        "grep,head,tail,wc,sort,uniq,cut,awk,sed"
    )

    # === 认证 ===
    jwt_secret_key: str = ""  # 请在 .env 中配置 JWT_SECRET_KEY（至少32字符随机字符串）
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440

    # === 探针 ===
    probe_timeout: int = 10
    probe_max_rounds: int = 5

    @property
    def sudo_whitelist_list(self) -> list[str]:
        """解析 sudo 白名单为列表"""
        return [cmd.strip() for cmd in self.sudo_whitelist.split(",") if cmd.strip()]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()


# ============================================================
#  运行时动态 LLM 配置 —— 支持通过 API 热切换模型
# ============================================================

@dataclass
class LLMRuntimeConfig:
    """LLM 运行时配置（Settings 默认值 + DB 覆盖值合并结果）"""
    protocol: str
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    anthropic_base_url: str
    anthropic_api_key: str
    anthropic_model: str


async def get_llm_runtime_config() -> LLMRuntimeConfig:
    """
    获取 LLM 运行时配置。

    读取策略：
    1. 先从 Settings（.env）获取默认值
    2. 再从 DB configs 表读取覆盖值（如果存在）
    3. 合并返回 LLMRuntimeConfig

    此函数在每次 Agent 对话请求开始时调用一次，
    确保配置变更即时生效，同时避免循环内的重复 IO。
    """
    settings = get_settings()

    # 延迟导入，避免循环依赖（config.py 被很多模块导入）
    from .db.config import get_config

    # 尝试从 DB 读取覆盖值，不存在则使用 Settings 默认值
    # 使用 try/except 包裹，防止 DB 未初始化时崩溃（启动阶段可能调用）
    try:
        protocol = await get_config("llm.protocol", settings.llm_protocol)
        base_url = await get_config("llm.base_url", settings.llm_base_url)
        api_key = await get_config("llm.api_key", settings.llm_api_key)
        model = await get_config("llm.model", settings.llm_model)

        temp_str = await get_config("llm.temperature")
        temperature = float(temp_str) if temp_str is not None else settings.llm_temperature

        max_tok_str = await get_config("llm.max_tokens")
        max_tokens = int(max_tok_str) if max_tok_str is not None else settings.llm_max_tokens

        anthropic_base = await get_config("llm.anthropic_base_url", settings.anthropic_base_url)
        anthropic_key = await get_config("llm.anthropic_api_key", settings.anthropic_api_key)
        anthropic_model = await get_config("llm.anthropic_model", settings.anthropic_model)
    except Exception:
        # DB 未就绪时回退到 Settings 默认值
        protocol = settings.llm_protocol
        base_url = settings.llm_base_url
        api_key = settings.llm_api_key
        model = settings.llm_model
        temperature = settings.llm_temperature
        max_tokens = settings.llm_max_tokens
        anthropic_base = settings.anthropic_base_url
        anthropic_key = settings.anthropic_api_key
        anthropic_model = settings.anthropic_model

    return LLMRuntimeConfig(
        protocol=protocol or settings.llm_protocol,
        base_url=base_url or settings.llm_base_url,
        api_key=api_key or settings.llm_api_key,
        model=model or settings.llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        anthropic_base_url=anthropic_base or settings.anthropic_base_url,
        anthropic_api_key=anthropic_key or settings.anthropic_api_key,
        anthropic_model=anthropic_model or settings.anthropic_model,
    )
