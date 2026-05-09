#!/usr/bin/env python3
"""
Standalone inference profiler for HuggingFace causal language models.

It reports:
1. Static model cost
   - total params
   - trainable params
   - forward MACs / FLOPs for a single-sample forward pass
2. Single-sample generation latency
   - TTFT mean / P95
   - TPOT mean / P95
   - generated tokens / second
   - generate latency distribution
3. Dataset throughput and batch latency
   - generate batch latency
   - end-to-end batch latency
   - mean sample latency
   - eval samples / second
4. CUDA peak memory
   - max memory allocated
   - max memory reserved

Example:
  python profile_inference_efficiency.py \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --prompt-file prompts.jsonl \
    --text-key prompt \
    --batch-size 4 \
    --max-new-tokens 128 \
    --output-json profiler_report.json
"""

import argparse
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)


DEFAULT_PROMPTS = [
    "Explain why attention mechanisms help large language models reason over long contexts.",
    "Write a concise summary of the tradeoff between latency and throughput during autoregressive decoding.",
    "A train leaves station A at 9:00 and arrives at station B at 11:30. What is the travel time in minutes?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile inference efficiency of a causal LM.")
    parser.add_argument("--model-path", type=str, required=True, help="Local path or HF model id.")
    parser.add_argument("--tokenizer-path", type=str, default="", help="Tokenizer path; defaults to model path.")
    parser.add_argument("--device", type=str, default="auto", help="Device, e.g. auto/cpu/cuda/cuda:0.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code.")
    parser.add_argument("--attn-implementation", type=str, default="", help="Optional attention implementation.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for throughput profiling.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Number of output tokens to generate.")
    parser.add_argument("--max-input-length", type=int, default=512, help="Prompt truncation length.")
    parser.add_argument("--num-warmup", type=int, default=1, help="Warmup runs before measurement.")
    parser.add_argument(
        "--single-latency-samples",
        type=int,
        default=16,
        help="How many single-sample prompts to use for TTFT/TPOT profiling.",
    )
    parser.add_argument("--limit-samples", type=int, default=0, help="Limit number of prompts loaded from file.")
    parser.add_argument("--prompt", action="append", default=[], help="Inline prompt; may be provided multiple times.")
    parser.add_argument("--prompt-file", type=str, default="", help="txt/json/jsonl file containing prompts.")
    parser.add_argument(
        "--text-key",
        type=str,
        default="",
        help="Field name to read from json/jsonl prompt files. If empty, common keys are tried.",
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Wrap prompts with tokenizer.apply_chat_template when available.",
    )
    parser.add_argument("--system-prompt", type=str, default="", help="Optional system prompt for chat formatting.")
    parser.add_argument("--do-sample", action="store_true", help="Use sampling instead of greedy decoding.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Sampling top-p.")
    parser.add_argument(
        "--profile-flops",
        action="store_true",
        help="Try static MAC/FLOP analysis via thop/fvcore on one forward pass.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-json", type=str, default="", help="Optional path to save the report as JSON.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def resolve_dtype(dtype_str: str, device: torch.device) -> torch.dtype:
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def collect_peak_memory(device: torch.device) -> Dict[str, Optional[float]]:
    if device.type != "cuda":
        return {
            "cuda_max_memory_allocated_mb": None,
            "cuda_max_memory_reserved_mb": None,
        }
    allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return {
        "cuda_max_memory_allocated_mb": allocated,
        "cuda_max_memory_reserved_mb": reserved,
    }


def _read_prompt_item(obj: Any, text_key: str) -> Optional[str]:
    if isinstance(obj, str):
        return obj.strip()
    if not isinstance(obj, dict):
        return None

    candidate_keys = [text_key] if text_key else []
    candidate_keys.extend(["prompt", "text", "input", "instruction", "question"])
    for key in candidate_keys:
        if key and key in obj and isinstance(obj[key], str):
            return obj[key].strip()
    return None


def load_prompts(prompt_args: Sequence[str], prompt_file: str, text_key: str, limit: int) -> List[str]:
    prompts: List[str] = []
    prompts.extend([p for p in prompt_args if p and p.strip()])

    if prompt_file:
        ext = os.path.splitext(prompt_file)[1].lower()
        with open(prompt_file, "r", encoding="utf-8") as f:
            if ext in {".jsonl", ".json"}:
                if ext == ".jsonl":
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        item = _read_prompt_item(json.loads(line), text_key)
                        if item:
                            prompts.append(item)
                else:
                    payload = json.load(f)
                    if isinstance(payload, list):
                        for item in payload:
                            text = _read_prompt_item(item, text_key)
                            if text:
                                prompts.append(text)
                    else:
                        text = _read_prompt_item(payload, text_key)
                        if text:
                            prompts.append(text)
            else:
                for line in f:
                    line = line.strip()
                    if line:
                        prompts.append(line)

    if not prompts:
        prompts = list(DEFAULT_PROMPTS)

    if limit > 0:
        prompts = prompts[:limit]
    return prompts


def chunked(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(float(v) for v in values)
    idx = (len(sorted_values) - 1) * q
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return sorted_values[lower]
    weight = idx - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize(values: Sequence[float]) -> Dict[str, Optional[float]]:
    values = [float(v) for v in values if v is not None]
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "p50": None,
            "p90": None,
            "p95": None,
        }
    return {
        "count": len(values),
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
    }


def format_chat_prompt(tokenizer: AutoTokenizer, prompt: str, system_prompt: str) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_prompt_batch(
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    use_chat_template: bool,
    system_prompt: str,
) -> List[str]:
    if not use_chat_template:
        return list(prompts)
    return [format_chat_prompt(tokenizer, prompt, system_prompt) for prompt in prompts]


def tokenize_batch(
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    max_input_length: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    batch = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )
    return {k: v.to(device) for k, v in batch.items()}


@dataclass
class SingleSampleLatency:
    generate_latency_s: float
    ttft_s: Optional[float]
    tpot_samples_s: List[float]
    generated_tokens: int
    generated_tokens_per_s: Optional[float]


class LatencyProfiler(LogitsProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.start_time: Optional[float] = None
        self.step_timestamps: List[float] = []

    def reset(self) -> None:
        self.start_time = None
        self.step_timestamps = []

    def start(self) -> None:
        self.start_time = time.perf_counter()
        self.step_timestamps = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.step_timestamps.append(time.perf_counter())
        return scores

    def finalize(self, generated_tokens: int) -> Tuple[Optional[float], List[float]]:
        if self.start_time is None or not self.step_timestamps or generated_tokens <= 0:
            return None, []
        ttft = self.step_timestamps[0] - self.start_time
        decode_step_times: List[float] = []
        for prev, cur in zip(self.step_timestamps, self.step_timestamps[1:generated_tokens]):
            decode_step_times.append(cur - prev)
        return ttft, decode_step_times


class ForwardWrapper(nn.Module):
    def __init__(self, model: AutoModelForCausalLM) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits


def human_count(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    abs_value = abs(float(value))
    for unit, scale in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs_value >= scale:
            return f"{value / scale:.3f}{unit}"
    return f"{value:.3f}"


def maybe_profile_flops(
    model: AutoModelForCausalLM,
    sample_inputs: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    wrapper = ForwardWrapper(model).eval()
    flops_report: Dict[str, Any] = {
        "backend": None,
        "forward_macs": None,
        "forward_flops": None,
        "error": None,
    }

    try:
        from thop import profile as thop_profile

        macs, _ = thop_profile(
            wrapper,
            inputs=(sample_inputs["input_ids"], sample_inputs["attention_mask"]),
            verbose=False,
        )
        flops_report["backend"] = "thop"
        flops_report["forward_macs"] = float(macs)
        flops_report["forward_flops"] = float(macs) * 2.0
        return flops_report
    except Exception as exc:
        flops_report["error"] = f"thop failed: {exc}"

    try:
        from fvcore.nn import FlopCountAnalysis

        analysis = FlopCountAnalysis(
            wrapper,
            (sample_inputs["input_ids"], sample_inputs["attention_mask"]),
        )
        flops = float(analysis.total())
        flops_report["backend"] = "fvcore"
        flops_report["forward_flops"] = flops
        flops_report["forward_macs"] = flops / 2.0
        flops_report["error"] = None
    except Exception as exc:
        flops_report["error"] = f"{flops_report['error']}; fvcore failed: {exc}"

    return flops_report


def build_generate_kwargs(args: argparse.Namespace, tokenizer: AutoTokenizer) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "return_dict_in_generate": True,
    }
    if args.do_sample:
        kwargs["temperature"] = args.temperature
        kwargs["top_p"] = args.top_p
    return kwargs


def warmup_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if not prompts or args.num_warmup <= 0:
        return
    generate_kwargs = build_generate_kwargs(args, tokenizer)
    warmup_prompts = build_prompt_batch(
        tokenizer,
        [prompts[0]] * min(args.batch_size, len(prompts)),
        args.use_chat_template,
        args.system_prompt,
    )
    for _ in range(args.num_warmup):
        inputs = tokenize_batch(tokenizer, warmup_prompts, args.max_input_length, device)
        sync_if_needed(device)
        with torch.inference_mode():
            _ = model.generate(**inputs, **generate_kwargs)
        sync_if_needed(device)


def profile_single_sample_latency(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    latency_probe = LatencyProfiler()
    generate_kwargs = build_generate_kwargs(args, tokenizer)
    generate_kwargs["logits_processor"] = LogitsProcessorList([latency_probe])

    samples: List[SingleSampleLatency] = []
    for prompt in prompts:
        formatted = build_prompt_batch(
            tokenizer,
            [prompt],
            args.use_chat_template,
            args.system_prompt,
        )
        inputs = tokenize_batch(tokenizer, formatted, args.max_input_length, device)
        prompt_len = int(inputs["attention_mask"][0].sum().item())

        latency_probe.reset()
        sync_if_needed(device)
        latency_probe.start()
        with torch.inference_mode():
            output = model.generate(**inputs, **generate_kwargs)
        sync_if_needed(device)
        total_latency = time.perf_counter() - latency_probe.start_time

        sequences = output.sequences
        generated_tokens = max(int(sequences.shape[1] - prompt_len), 0)
        ttft, tpot_samples = latency_probe.finalize(generated_tokens)
        tokens_per_s = (generated_tokens / total_latency) if total_latency > 0 and generated_tokens > 0 else None
        samples.append(
            SingleSampleLatency(
                generate_latency_s=total_latency,
                ttft_s=ttft,
                tpot_samples_s=tpot_samples,
                generated_tokens=generated_tokens,
                generated_tokens_per_s=tokens_per_s,
            )
        )

    generate_latencies = [s.generate_latency_s for s in samples]
    ttft_values = [s.ttft_s for s in samples if s.ttft_s is not None]
    tpot_values = [value for s in samples for value in s.tpot_samples_s]
    tokens_per_s_values = [s.generated_tokens_per_s for s in samples if s.generated_tokens_per_s is not None]
    generated_tokens_values = [float(s.generated_tokens) for s in samples]

    return {
        "num_samples": len(samples),
        "ttft_s": {
            "mean": summarize(ttft_values)["mean"],
            "p95": summarize(ttft_values)["p95"],
        },
        "tpot_s": {
            "mean": summarize(tpot_values)["mean"],
            "p95": summarize(tpot_values)["p95"],
        },
        "generated_tokens": summarize(generated_tokens_values),
        "generated_tokens_per_s": {
            "mean": summarize(tokens_per_s_values)["mean"],
            "p95": summarize(tokens_per_s_values)["p95"],
        },
        "generate_latency_s": summarize(generate_latencies),
    }


def profile_batch_throughput(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    generate_kwargs = build_generate_kwargs(args, tokenizer)
    generate_batch_latencies: List[float] = []
    e2e_batch_latencies: List[float] = []
    sample_latencies: List[float] = []
    total_samples = 0
    total_e2e_time = 0.0

    for batch_prompts in chunked(prompts, args.batch_size):
        batch_prompts = build_prompt_batch(
            tokenizer,
            batch_prompts,
            args.use_chat_template,
            args.system_prompt,
        )

        e2e_start = time.perf_counter()
        inputs = tokenize_batch(tokenizer, batch_prompts, args.max_input_length, device)
        sync_if_needed(device)
        generate_start = time.perf_counter()
        with torch.inference_mode():
            _ = model.generate(**inputs, **generate_kwargs)
        sync_if_needed(device)
        generate_latency = time.perf_counter() - generate_start
        e2e_latency = time.perf_counter() - e2e_start

        batch_size = len(batch_prompts)
        generate_batch_latencies.append(generate_latency)
        e2e_batch_latencies.append(e2e_latency)
        sample_latencies.append(generate_latency / batch_size)
        total_samples += batch_size
        total_e2e_time += e2e_latency

    samples_per_sec = (total_samples / total_e2e_time) if total_e2e_time > 0 else None
    return {
        "num_batches": len(generate_batch_latencies),
        "num_samples": total_samples,
        "generate_batch_latency_s": summarize(generate_batch_latencies),
        "e2e_batch_latency_s": summarize(e2e_batch_latencies),
        "generate_sample_latency_mean_s": summarize(sample_latencies)["mean"],
        "eval_samples_per_sec": samples_per_sec,
    }


def profile_static_cost(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    params_total = sum(param.numel() for param in model.parameters())
    params_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    report: Dict[str, Any] = {
        "params_total": int(params_total),
        "params_trainable": int(params_trainable),
        "forward_macs": None,
        "forward_flops": None,
        "backend": None,
        "error": None,
    }

    if not args.profile_flops:
        return report

    sample_prompt = build_prompt_batch(
        tokenizer,
        [prompt],
        args.use_chat_template,
        args.system_prompt,
    )
    sample_inputs = tokenize_batch(tokenizer, sample_prompt, args.max_input_length, device)
    flops_report = maybe_profile_flops(model, sample_inputs)
    report.update(flops_report)
    return report


def print_human_readable(report: Dict[str, Any]) -> None:
    static_cost = report["static_cost"]
    single_latency = report["single_sample_generation"]
    throughput = report["throughput"]
    memory = report["memory_peak"]

    print("=" * 80)
    print("1. Compute Cost & Parameters")
    print("=" * 80)
    print(f"Params (total):      {human_count(static_cost['params_total'])} ({static_cost['params_total']})")
    print(f"Params (trainable):  {human_count(static_cost['params_trainable'])} ({static_cost['params_trainable']})")
    print(f"Forward MACs:        {human_count(static_cost['forward_macs'])} ({static_cost['forward_macs']})")
    print(f"Forward FLOPs:       {human_count(static_cost['forward_flops'])} ({static_cost['forward_flops']})")
    if static_cost.get("backend"):
        print(f"Static analysis backend: {static_cost['backend']}")
    if static_cost.get("error"):
        print(f"Static analysis note:    {static_cost['error']}")

    print()
    print("=" * 80)
    print("2. Generation Latency (single-sample)")
    print("=" * 80)
    print(
        "TTFT (s):            "
        f"mean={single_latency['ttft_s']['mean']}  p95={single_latency['ttft_s']['p95']}"
    )
    print(
        "TPOT (s):            "
        f"mean={single_latency['tpot_s']['mean']}  p95={single_latency['tpot_s']['p95']}"
    )
    print(
        "Generated tok/s:     "
        f"mean={single_latency['generated_tokens_per_s']['mean']}  "
        f"p95={single_latency['generated_tokens_per_s']['p95']}"
    )
    gen_summary = single_latency["generate_latency_s"]
    print(
        "Generate latency(s): "
        f"mean={gen_summary['mean']}  std={gen_summary['std']}  "
        f"p50={gen_summary['p50']}  p90={gen_summary['p90']}  p95={gen_summary['p95']}"
    )

    print()
    print("=" * 80)
    print("3. Throughput & Batch Latency")
    print("=" * 80)
    gen_batch = throughput["generate_batch_latency_s"]
    e2e_batch = throughput["e2e_batch_latency_s"]
    print(
        "Generate batch latency(s): "
        f"mean={gen_batch['mean']}  p50={gen_batch['p50']}  p90={gen_batch['p90']}  p95={gen_batch['p95']}"
    )
    print(
        "E2E batch latency(s):      "
        f"mean={e2e_batch['mean']}  p50={e2e_batch['p50']}  p90={e2e_batch['p90']}  p95={e2e_batch['p95']}"
    )
    print(f"Generate sample latency mean(s): {throughput['generate_sample_latency_mean_s']}")
    print(f"Eval samples / sec:              {throughput['eval_samples_per_sec']}")

    print()
    print("=" * 80)
    print("4. VRAM Memory Peak")
    print("=" * 80)
    print(f"CUDA Max Memory Allocated (MB): {memory['cuda_max_memory_allocated_mb']}")
    print(f"CUDA Max Memory Reserved  (MB): {memory['cuda_max_memory_reserved_mb']}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer_path = args.tokenizer_path or args.model_path
    prompts = load_prompts(args.prompt, args.prompt_file, args.text_key, args.limit_samples)
    if not prompts:
        raise ValueError("No prompts available for profiling.")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    print(f"Loading model from: {args.model_path}")
    print(f"Using device={device} dtype={dtype}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.to(device)
    model.eval()

    warmup_model(model, tokenizer, prompts, args, device)

    reset_peak_memory(device)

    static_cost = profile_static_cost(model, tokenizer, prompts[0], args, device)
    single_sample_prompts = prompts[: max(1, min(args.single_latency_samples, len(prompts)))]
    single_sample_generation = profile_single_sample_latency(
        model, tokenizer, single_sample_prompts, args, device
    )
    throughput = profile_batch_throughput(model, tokenizer, prompts, args, device)
    memory_peak = collect_peak_memory(device)

    report = {
        "model_path": args.model_path,
        "tokenizer_path": tokenizer_path,
        "device": str(device),
        "dtype": str(dtype),
        "num_prompts": len(prompts),
        "batch_size": args.batch_size,
        "max_input_length": args.max_input_length,
        "max_new_tokens": args.max_new_tokens,
        "static_cost": static_cost,
        "single_sample_generation": single_sample_generation,
        "throughput": throughput,
        "memory_peak": memory_peak,
    }

    print_human_readable(report)
    print()
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nSaved report to {args.output_json}")


if __name__ == "__main__":
    main()
