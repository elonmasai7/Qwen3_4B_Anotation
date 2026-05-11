from typing import Any, Callable
import json
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from common.config import get_settings
from common.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class MessagePublisher:
    def __init__(self) -> None:
        self._producer: AIOKafkaProducer | None = None

    async def connect(self) -> None:
        if self._producer is None:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=settings.kafka.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            await self._producer.start()
            logger.info("kafka_producer_connected")

    async def disconnect(self) -> None:
        if self._producer:
            await self._producer.stop()
            logger.info("kafka_producer_disconnected")

    async def publish(
        self,
        topic: str,
        message: dict[str, Any],
        key: str | None = None,
    ) -> bool:
        if not self._producer:
            await self.connect()

        try:
            key_bytes = key.encode("utf-8") if key else None
            await self._producer.send_and_wait(topic, message, key=key_bytes)
            return True
        except Exception as e:
            logger.error("kafka_publish_failed", topic=topic, error=str(e))
            return False


class MessageConsumer:
    def __init__(self, topic: str, group_id: str | None = None) -> None:
        self.topic = topic
        self.group_id = group_id or settings.kafka.consumer_group
        self._consumer: AIOKafkaConsumer | None = None
        self._running = False
        self._handlers: list[Callable] = []

    async def connect(self) -> None:
        if self._consumer is None:
            self._consumer = AIOKafkaConsumer(
                self.topic,
                bootstrap_servers=settings.kafka.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            )
            await self._consumer.start()
            logger.info("kafka_consumer_connected", topic=self.topic)

    async def disconnect(self) -> None:
        self._running = False
        if self._consumer:
            await self._consumer.stop()
            logger.info("kafka_consumer_disconnected")

    def add_handler(self, handler: Callable[[dict[str, Any]], None]) -> None:
        self._handlers.append(handler)

    async def start_consuming(self) -> None:
        if not self._consumer:
            await self.connect()

        self._running = True

        async for msg in self._consumer:
            if not self._running:
                break

            try:
                for handler in self._handlers:
                    handler(msg.value)
            except Exception as e:
                logger.error("message_handler_failed", error=str(e))

    def stop(self) -> None:
        self._running = False


class AnnotationTaskProducer:
    def __init__(self) -> None:
        self.publisher = MessagePublisher()

    async def send_task(self, task: dict[str, Any]) -> bool:
        return await self.publisher.publish("annotation-tasks", task)


class AnnotationResultConsumer:
    def __init__(self) -> None:
        self.consumer = MessageConsumer("annotation-results")

    async def start(self, handler: Callable[[dict[str, Any]], None]) -> None:
        self.consumer.add_handler(handler)
        await self.consumer.start_consuming()


publisher = MessagePublisher()