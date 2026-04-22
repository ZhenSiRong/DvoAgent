"""全局配置 - 从 .env 读取所有环境变量"""
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
    llm_api_key: str = "sk-cp-WuUWdO9Vm74-DeEurDgH2FC9wpW3HoWQtVO11gxYHrLwQ3_bCLE2GYx4tqrV0gqunmd_ri1Id5Tu2z0CcgEy-dP-Pdg2FaTXIPQoY3nvaMiUJ3eoC-v3fI4"
    llm_model: str = "MiniMax-M2.1"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096

    # === LLM（Anthropic 协议备用） ===
    anthropic_base_url: str = "https://api.minimaxi.com/anthropic"
    anthropic_api_key: str = "sk-cp-WuUWdO9Vm74-DeEurDgH2FC9wpW3HoWQtVO11gxYHrLwQ3_bCLE2GYx4tqrV0gqunmd_ri1Id5Tu2z0CcgEy-dP-Pdg2FaTXIPQoY3nvaMiUJ3eoC-v3fI4"
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
    jwt_secret_key: str = "change-this-secret-key"
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
