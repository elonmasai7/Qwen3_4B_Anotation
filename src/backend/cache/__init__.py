import redis.asyncio as redis
from typing import Any
import json
from common.config import get_settings
from common.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class CacheManager:
    def __init__(self):
        self._client: redis.Redis | None = None

    async def connect(self) -> None:
        if self._client is None:
            self._client = redis.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                db=settings.redis.db,
                password=settings.redis.password,
                decode_responses=True,
            )
            logger.info("redis_connected")

    async def disconnect(self) -> None:
        if self._client:
            await self._client.close()
            logger.info("redis_disconnected")

    async def get(self, key: str) -> Any | None:
        if not self._client:
            return None

        try:
            value = await self._client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.warning("cache_get_failed", key=key, error=str(e))

        return None

    async def set(
        self,
        key: str,
        value: Any,
        expire: int = 3600,
    ) -> bool:
        if not self._client:
            return False

        try:
            serialized = json.dumps(value)
            await self._client.set(key, serialized, ex=expire)
            return True
        except Exception as e:
            logger.warning("cache_set_failed", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        if not self._client:
            return False

        try:
            await self._client.delete(key)
            return True
        except Exception as e:
            logger.warning("cache_delete_failed", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        if not self._client:
            return False

        try:
            return await self._client.exists(key) > 0
        except Exception as e:
            logger.warning("cache_exists_failed", key=key, error=str(e))
            return False

    async def get_pattern(self, pattern: str) -> list[str]:
        if not self._client:
            return []

        try:
            return await self._client.keys(pattern)
        except Exception as e:
            logger.warning("cache_pattern_failed", pattern=pattern, error=str(e))
            return []

    async def increment(self, key: str, amount: int = 1) -> int:
        if not self._client:
            return 0

        try:
            return await self._client.incrby(key, amount)
        except Exception as e:
            logger.warning("cache_increment_failed", key=key, error=str(e))
            return 0


cache_manager = CacheManager()


async def get_cache() -> CacheManager:
    if cache_manager._client is None:
        await cache_manager.connect()
    return cache_manager