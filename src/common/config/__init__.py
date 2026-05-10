from functools import lru_cache
from pathlib import Path
from typing import Any
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    host: str = "localhost"
    port: int = 5432
    name: str = "annotation_db"
    user: str = "postgres"
    password: str = "postgres"
    pool_size: int = 20
    max_overflow: int = 10
    echo: bool = False


class RedisConfig(BaseSettings):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None


class KafkaConfig(BaseSettings):
    bootstrap_servers: str = "localhost:9092"
    consumer_group: str = "annotation-platform"


class MinioConfig(BaseSettings):
    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    bucket: str = "annotation-data"
    secure: bool = False


class ModelConfig(BaseSettings):
    name: str = "Qwen/Qwen3-4B"
    max_tokens: int = 8192
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 50


class VLLMConfig(BaseSettings):
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_num_seqs: int = 256
    enforce_eager: bool = False
    dtype: str = "auto"
    trust_remote_code: bool = True


class AnnotationConfig(BaseSettings):
    num_branches: int = 3
    confidence_threshold: float = 0.8
    max_retries: int = 3
    timeout_seconds: int = 120


class RetrievalConfig(BaseSettings):
    top_k: int = 5
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    similarity_threshold: float = 0.7
    rerank_top_n: int = 3


class LoggingConfig(BaseSettings):
    level: str = "INFO"
    format: str = "json"
    output: str = "stdout"


class MonitoringConfig(BaseSettings):
    enabled: bool = True
    metrics_port: int = 9090
    tracing_enabled: bool = True


class PlatformConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PLATFORM_")

    name: str = "Annotation Platform"
    version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    platform: PlatformConfig = Field(default_factory=PlatformConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    minio: MinioConfig = Field(default_factory=MinioConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)


@lru_cache
def get_settings() -> Settings:
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "base.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config_data = yaml.safe_load(f) or {}
        return Settings(**config_data)
    return Settings()


def load_config_from_yaml(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f) or {}