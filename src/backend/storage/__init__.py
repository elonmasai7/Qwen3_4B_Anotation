import io
from minio import Minio
from datetime import timedelta
from common.config import get_settings
from common.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class StorageClient:
    def __init__(self):
        self._client: Minio | None = None

    def _get_client(self) -> Minio:
        if self._client is None:
            self._client = Minio(
                settings.minio.endpoint,
                access_key=settings.minio.access_key,
                secret_key=settings.minio.secret_key,
                secure=settings.minio.secure,
            )
            self._ensure_bucket()
        return self._client

    def _ensure_bucket(self) -> None:
        client = self._client
        if client:
            if not client.bucket_exists(settings.minio.bucket):
                client.make_bucket(settings.minio.bucket)
                logger.info("minio_bucket_created", bucket=settings.minio.bucket)

    async def upload_file(
        self,
        object_name: str,
        data: bytes | io.BytesIO,
        content_type: str = "application/octet-stream",
    ) -> bool:
        try:
            client = self._get_client()
            client.put_object(
                settings.minio.bucket,
                object_name,
                data,
                length=len(data) if isinstance(data, bytes) else -1,
                content_type=content_type,
            )
            logger.info("file_uploaded", object_name=object_name)
            return True
        except Exception as e:
            logger.error("upload_failed", object_name=object_name, error=str(e))
            return False

    async def download_file(self, object_name: str) -> bytes | None:
        try:
            client = self._get_client()
            response = client.get_object(settings.minio.bucket, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except Exception as e:
            logger.error("download_failed", object_name=object_name, error=str(e))
            return None

    async def delete_file(self, object_name: str) -> bool:
        try:
            client = self._get_client()
            client.remove_object(settings.minio.bucket, object_name)
            logger.info("file_deleted", object_name=object_name)
            return True
        except Exception as e:
            logger.error("delete_failed", object_name=object_name, error=str(e))
            return False

    async def list_objects(self, prefix: str = "") -> list[str]:
        try:
            client = self._get_client()
            objects = client.list_objects(settings.minio.bucket, prefix=prefix)
            return [obj.object_name for obj in objects]
        except Exception as e:
            logger.error("list_failed", prefix=prefix, error=str(e))
            return []

    def get_presigned_url(self, object_name: str, expires: int = 3600) -> str | None:
        try:
            client = self._get_client()
            url = client.presigned_get_object(
                settings.minio.bucket,
                object_name,
                expires=timedelta(seconds=expires),
            )
            return url
        except Exception as e:
            logger.error("presigned_url_failed", object_name=object_name, error=str(e))
            return None


storage_client = StorageClient()