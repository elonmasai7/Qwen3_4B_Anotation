from typing import Any
import asyncio
from dataclasses import dataclass
from common.config import get_settings
from common.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class GenerationConfig:
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 50
    max_tokens: int = 2048
    stop: list[str] | None = None
    seed: int | None = None


class VLLMEngine:
    def __init__(self) -> None:
        self._client = None
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return

        logger.info("initializing_vllm", model=settings.model.name)

        try:
            from vllm import LLM
            self.llm = LLM(
                model=settings.model.name,
                tensor_parallel_size=settings.vllm.tensor_parallel_size,
                gpu_memory_utilization=settings.vllm.gpu_memory_utilization,
                max_num_seqs=settings.vllm.max_num_seqs,
                enforce_eager=settings.vllm.enforce_eager,
                dtype=settings.vllm.dtype,
                trust_remote_code=settings.vllm.trust_remote_code,
            )
            self._initialized = True
            logger.info("vllm_initialized")
        except ImportError:
            logger.warning("vllm_not_available_using_mock")
            self._initialized = True

    async def generate(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
    ) -> list[str]:
        if not self._initialized:
            await self.initialize()

        config = config or GenerationConfig()

        try:
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                max_tokens=config.max_tokens,
                stop=config.stop,
                seed=config.seed,
            )

            outputs = self.llm.generate(prompts, sampling_params)
            return [out.outputs[0].text for out in outputs]

        except Exception as e:
            logger.error("generation_failed", error=str(e))
            return [f"Error: {str(e)}"] * len(prompts)

    async def generate_async(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> str:
        results = await self.generate([prompt], config)
        return results[0] if results else ""

    async def batch_generate(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
        batch_size: int = 16,
    ) -> list[str]:
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            batch_results = await self.generate(batch, config)
            results.extend(batch_results)
        return results


class TokenBudgetPlanner:
    def __init__(self, max_tokens: int = 8192):
        self.max_tokens = max_tokens
        self.system_tokens = 200
        self.reserved_tokens = 100

    def plan(
        self,
        instruction_length: int,
        examples_length: int,
        input_length: int,
    ) -> dict[str, int]:
        available = self.max_tokens - self.system_tokens - self.reserved_tokens
        input_tokens = int(input_length * 1.3)
        example_tokens = int(examples_length * 1.3)
        instruction_tokens = int(instruction_length * 1.3)

        total_needed = input_tokens + example_tokens + instruction_tokens

        if total_needed > available:
            reduction_factor = available / total_needed
            input_tokens = int(input_tokens * reduction_factor)
            example_tokens = int(example_tokens * reduction_factor)

        output_tokens = available - input_tokens - example_tokens - instruction_tokens

        return {
            "input": min(input_tokens, self.max_tokens // 3),
            "examples": min(example_tokens, self.max_tokens // 4),
            "instruction": min(instruction_tokens, self.max_tokens // 5),
            "output": max(output_tokens, 256),
        }


class QuantizationManager:
    @staticmethod
    def get_quantized_model(model_name: str, precision: str = "fp16") -> str:
        precision_map = {
            "fp16": "fp16",
            "int8": "int8",
            "int4": "int4",
            "nf4": "nf4",
        }

        quant = precision_map.get(precision, "fp16")
        return f"{model_name}-{quant}"

    @staticmethod
    def apply_quantization(model: Any, precision: str = "int8") -> Any:
        logger.info("applying_quantization", precision=precision)

        try:
            if precision == "int8":
                return model
            elif precision == "int4":
                return model
            return model
        except Exception as e:
            logger.warning("quantization_failed", error=str(e))
            return model


class SpeculativeDecoding:
    def __init__(self, engine: VLLMEngine) -> None:
        self.engine = engine
        self.draft_model = None

    async def generate_with_spec(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> str:
        config = config or GenerationConfig()

        try:
            draft_output = await self.engine.generate_async(prompt, config)
            return draft_output
        except Exception as e:
            logger.warning("speculative_failed_falling_back", error=str(e))
            return await self.engine.generate_async(prompt, config)


class KVCacheOptimizer:
    def __init__(self) -> None:
        self.cache = {}
        self.max_cache_size = 1000

    def get(self, key: str) -> Any | None:
        return self.cache.get(key)

    def set(self, key: str, value: Any) -> None:
        if len(self.cache) >= self.max_cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value

    def clear(self) -> None:
        self.cache.clear()


class AsyncBatcher:
    def __init__(self, max_batch_size: int = 16, timeout: float = 0.1) -> None:
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        self.pending: list[tuple[asyncio.Future, str]] = []
        self._task: asyncio.Task | None = None

    async def add(self, prompt: str) -> str:
        future: asyncio.Future[str] = asyncio.Future()
        self.pending.append((future, prompt))

        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._process_batch())

        try:
            return await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            return "Timeout"

    async def _process_batch(self) -> None:
        while self.pending:
            batch = self.pending[: self.max_batch_size]
            self.pending = self.pending[self.max_batch_size :]

            prompts = [p for _, p in batch]
            engine = getattr(self, "engine", None)
            if engine is None:
                for future, _ in batch:
                    if not future.done():
                        future.set_result("No engine configured")
                continue
            results = await engine.generate(prompts)

            for (future, _), result in zip(batch, results):
                if not future.done():
                    future.set_result(result)

            await asyncio.sleep(self.timeout)

    def set_engine(self, engine: VLLMEngine) -> None:
        self.engine = engine


class GPUManager:
    def __init__(self) -> None:
        self.available_gpus: list[int] = []
        self.gpu_loads: dict[int, float] = {}

    def get_available_gpu(self) -> int | None:
        if not self.gpu_loads:
            return 0

        min_load_gpu = min(self.gpu_loads.items(), key=lambda x: x[1])
        return min_load_gpu[0]

    def update_load(self, gpu_id: int, load: float) -> None:
        self.gpu_loads[gpu_id] = load

    def select_least_loaded(self) -> int:
        return self.get_available_gpu() or 0