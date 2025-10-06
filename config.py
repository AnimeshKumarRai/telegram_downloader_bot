from __future__ import annotations
import os
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_settings = None

class Settings(BaseModel):
    telegram_token: str = Field(description="Telegram Bot token")
    bot_name: str = Field(default="mediadownloader_byani_bot", description="Bot name")
    admins: list[int] = Field(default_factory=list, description="Comma-separated admin user IDs")
    postgres_dsn: str = Field(description="PostgreSQL connection string")
    redis_url: str = Field(description="Redis connection string")
    download_dir: Path = Field(default=Path("./downloads"), description="Directory for downloaded files")  # Changed to relative Path
    max_filesize_mb: int = Field(default=1900, description="Compatible file size limit in MB")
    max_telegram_upload_mb: int = Field(default=1950, description="Hard safety ceiling for uploads in MB")
    per_user_per_minute: int = Field(default=5)
    concurrent_per_user: int = Field(default=1)
    ytdlp_cookies_path: str | None = Field(default=None)
    sentry_dsn: str | None = Field(default=None)
    health_port: int = Field(default=8080)
    environment: str = Field(default="production")
    group_link_listen: bool = Field(default=True)
    max_format_buttons: int = Field(default=10)
    ttl_download_hours: int = Field(default=6)
    allow_large_downloads: bool = Field(default=True)
    default_large_media_strategy: str = Field(default="split")
    split_segment_duration_sec: int = Field(default=600)
    default_quality: str = Field(default="best")
    prefer_document_default: bool = Field(default=False)
    webhook_enabled: bool = Field(default=False)
    webhook_url: str | None = Field(default=None)
    webhook_secret_token: str | None = Field(default=None)
    webhook_listen_ip: str = Field(default="0.0.0.0")
    webhook_listen_port: int = Field(default=8443)
    webhook_path: str = Field(default="/webhook")
    admin_pass: str = Field(default="ani123", description="Admin panel password")

    @field_validator("telegram_token", "postgres_dsn", "redis_url", "download_dir", mode="after")
    @classmethod
    def ensure_non_empty(cls, v, info):
        if not v:
            raise ValueError(f"{info.field_name} cannot be empty; must be set in .env file")
        return v

    @field_validator("download_dir", mode="after")
    @classmethod
    def resolve_download_dir(cls, v):
        """Resolve download_dir to an absolute path relative to the project directory."""
        if isinstance(v, str):
            v = Path(v)
        # Make it absolute relative to the project root (where config.py is)
        project_root = Path(__file__).parent
        resolved = (project_root / v).resolve()
        # Ensure it exists on validation
        resolved.mkdir(parents=True, exist_ok=True)
        logger.info(f"Download directory resolved to: {resolved}")
        return resolved

    @field_validator("telegram_token", mode="before")
    @classmethod
    def read_token_from_secret_if_empty(cls, v):
        if v:
            logger.info("TELEGRAM_TOKEN provided via environment variable")
            return v
        secret_path = os.getenv("TELEGRAM_TOKEN_FILE", "/run/secrets/telegram_token")
        if os.path.exists(secret_path):
            token = Path(secret_path).read_text().strip()
            logger.info("Loaded TELEGRAM_TOKEN from secret file: %s", secret_path)
            return token
        raise ValueError("TELEGRAM_TOKEN not provided via env or secret")

    @field_validator("admins", mode="before")
    @classmethod
    def parse_admins(cls, v):
        if not v:
            return []
        if isinstance(v, list):
            return [int(x) for x in v]
        return [int(x.strip()) for x in str(v).split(",") if x.strip()]

def get_settings() -> Settings:
    global _settings
    if _settings is not None:
        return _settings

    # Explicitly load .env file
    env_file = ".env"
    if not os.path.exists(env_file):
        logger.error(".env file not found at %s; required for configuration", env_file)
        raise FileNotFoundError(f".env file not found at {env_file}")
    
    load_dotenv(env_file, override=True)
    logger.info("Successfully loaded .env file: %s", env_file)
    
    # Get DOWNLOAD_DIR and force it to be relative by stripping leading '/'
    download_dir_raw = os.getenv("DOWNLOAD_DIR", "./downloads")
    download_dir = download_dir_raw.lstrip('/')  # Remove leading '/' to make relative (e.g., /downloads -> downloads)
    if download_dir_raw.startswith('/'):
        logger.warning("Converted absolute DOWNLOAD_DIR '%s' to relative './%s' to avoid permission issues", download_dir_raw, download_dir)
    
    # Manually pass environment variables to Settings
    _settings = Settings(
        telegram_token=os.getenv("TELEGRAM_TOKEN", "telegram_token_here"),  # Replace with your actual token
        postgres_dsn=os.getenv("POSTGRES_DSN", "postgresql://user:pass@localhost/dbname"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:port/0"),
        download_dir=download_dir,  
        bot_name=os.getenv("BOT_NAME", "bot_name_here"),
        admins=os.getenv("ADMINS", ""),
        max_filesize_mb=int(os.getenv("MAX_FILESIZE_MB", 1900)),
        max_telegram_upload_mb=int(os.getenv("MAX_TELEGRAM_UPLOAD_MB", 1950)),
        per_user_per_minute=int(os.getenv("PER_USER_PER_MINUTE", 5)),
        concurrent_per_user=int(os.getenv("CONCURRENT_PER_USER", 1)),
        health_port=int(os.getenv("HEALTH_PORT", 8080)),
        environment=os.getenv("ENVIRONMENT", "production"),
        group_link_listen=os.getenv("GROUP_LINK_LISTEN", "true").lower() == "true",
        max_format_buttons=int(os.getenv("MAX_FORMAT_BUTTONS", 10)),
        ttl_download_hours=int(os.getenv("TTL_DOWNLOAD_HOURS", 6)),
        allow_large_downloads=os.getenv("ALLOW_LARGE_DOWNLOADS", "true").lower() == "true",
        default_large_media_strategy=os.getenv("DEFAULT_LARGE_MEDIA_STRATEGY", "split"),
        split_segment_duration_sec=int(os.getenv("SPLIT_SEGMENT_DURATION_SEC", 600)),
        default_quality=os.getenv("DEFAULT_QUALITY", "best"),
        prefer_document_default=os.getenv("PREFER_DOCUMENT_DEFAULT", "false").lower() == "false",
        webhook_enabled=os.getenv("WEBHOOK_ENABLED", "false").lower() == "true",
        webhook_url=os.getenv("WEBHOOK_URL"),
        webhook_secret_token=os.getenv("WEBHOOK_SECRET_TOKEN"),
        webhook_listen_ip=os.getenv("WEBHOOK_LISTEN_IP", "0.0.0.0"),
        webhook_listen_port=int(os.getenv("WEBHOOK_LISTEN_PORT", 8443)),
        webhook_path=os.getenv("WEBHOOK_PATH", "/webhook"),
        admin_pass= os.getenv("ADMIN_PASS", "changeme")
    )
    logger.info("TELEGRAM_TOKEN loaded successfully: %s", "*" * len(_settings.telegram_token))
    return _settings