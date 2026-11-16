"""Direct adapter for the pinned vLLM 0.11.0 top-k/top-p sampler."""

from __future__ import annotations

import torch


class VLLMDirectSampler:
    def __init__(self, device: str = "cuda") -> None:
        try:
            import vllm
            from vllm.v1.sample.sampler import Sampler
            from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
        except ImportError as error:
            raise RuntimeError(
                "Install the pinned dependencies to use the direct vLLM sampler"
            ) from error

        if vllm.__version__ != "0.11.0":
            raise RuntimeError(
                f"Expected vLLM 0.11.0, found {vllm.__version__}; "
                "the internal sampler API is version-specific"
            )
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

        self.device = torch.device(device)
        self.sampler = TopKTopPSampler(logprobs_mode="processed_logprobs")
        self.vllm_sampler = Sampler(logprobs_mode="raw_logprobs")
        backend = getattr(self.sampler.forward, "__name__", "unknown")
        if backend != "forward_native":
            raise RuntimeError(f"Expected native sampling backend, got {backend}")
        self.backend = backend

    @torch.inference_mode()
    def greedy(self, logits: torch.Tensor) -> tuple[int, torch.Tensor]:
        source = torch.as_tensor(logits, dtype=torch.float32, device=self.device)
        if source.ndim != 1:
            raise ValueError("logits must be a one-dimensional vector")
        batched = source.unsqueeze(0)
        token_id = int(self.vllm_sampler.greedy_sample(batched)[0].item())
        raw_logprobs = self.vllm_sampler.compute_logprobs(batched)[0].cpu()
        return token_id, raw_logprobs

    @torch.inference_mode()
    def sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        sample_count: int,
        seed: int,
        batch_size: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("temperature must be positive and top_p must be in (0, 1]")
        if sample_count < 1 or batch_size < 1:
            raise ValueError("sample_count and batch_size must be positive")

        source = torch.as_tensor(logits, dtype=torch.float32, device=self.device)
        if source.ndim != 1:
            raise ValueError("logits must be a one-dimensional vector")
        raw_logprobs = source.log_softmax(dim=-1, dtype=torch.float32).cpu()

        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        sample_batches = []
        reported_processed = None
        for start in range(0, sample_count, batch_size):
            current_batch = min(batch_size, sample_count - start)
            batch_logits = (source / temperature).expand(current_batch, -1).contiguous()
            top_p_values = torch.full(
                (current_batch,), top_p, dtype=torch.float32, device=self.device
            )
            sampled, processed_logprobs = self.sampler(
                batch_logits, {}, None, top_p_values
            )
            if processed_logprobs is None:
                raise RuntimeError("vLLM did not return processed logprobs")
            if reported_processed is None:
                reported_processed = processed_logprobs[0].detach().cpu()
            sample_batches.append(sampled.detach().cpu())

        return torch.cat(sample_batches), raw_logprobs, reported_processed