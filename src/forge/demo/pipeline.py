"""FORGE Demo — End-to-end showcase for VC presentations.

Runs the full pipeline in demo mode: load → forward → distill → compress → benchmark.
Produces a rich terminal output with all metrics.

Usage:
    forge demo --device cuda --steps 100
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from forge.config import StudentConfig, apply_student_variant
from forge.provenance import build_provenance


def _save_demo_checkpoint(
    path: Path,
    *,
    student: torch.nn.Module,
    config: StudentConfig,
    results: dict,
    dataset: object,
    model_dir: str,
) -> dict[str, str]:
    """Save a demo checkpoint with provenance from the actual runtime inputs."""
    provenance = build_provenance(
        student=student,
        config=config,
        dataset=dataset,
        model_dir=model_dir,
    )
    results["provenance"] = provenance
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "config": vars(config),
            "results": results,
            "provenance": provenance,
        },
        path,
    )
    return provenance


def _demo_student_config(allow_mock: bool | None = None) -> StudentConfig:
    """Build the demo from the same canonical nano preset as the real pipeline."""
    config = StudentConfig()
    apply_student_variant(config, "nano")
    if allow_mock is not None:
        config.allow_mock = bool(config.allow_mock or allow_mock)
    return config


def run_demo(
    model_dir: str,
    device: str = "cuda",
    steps: int = 100,
    output_dir: str = "./outputs/demo",
    allow_mock: bool | None = None,
) -> dict:
    """Run the full FORGE demo pipeline.

    Returns a dict with all metrics for the VC report.
    """
    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "steps": steps,
    }

    out_path = Path(output_dir)

    # ── Step 1: Load Real Models ──
    print("\n🔧 Step 1/5: Loading FORGE-Nano (SigLIP2 + Qwen3-0.6B)...")
    from forge.student import FORGEStudent

    config = _demo_student_config(allow_mock)
    if not config.allow_mock:
        raise ValueError(
            "The legacy demo synthesizes teacher labels. Pass allow_mock=True "
            "or set FORGE_ALLOW_MOCK=1 to run this explicit mock workflow."
        )

    out_path.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    student = FORGEStudent(config, model_dir=model_dir)
    student = student.to(device)
    load_time = time.time() - t0

    results["model"] = {
        "variant": "nano",
        "total_params_M": round(student.total_params / 1e6, 1),
        "trainable_params_M": round(student.trainable_params / 1e6, 1),
        "frozen_params_M": round((student.total_params - student.trainable_params) / 1e6, 1),
        "size_bf16_GB": round(student.total_params * 2 / (1024**3), 2),
        "load_time_s": round(load_time, 1),
    }
    print(f"   ✓ {results['model']['total_params_M']}M params loaded in {load_time:.1f}s")

    # ── Step 2: Inference Benchmark ──
    print("\n⚡ Step 2/5: Benchmarking inference latency...")
    student.eval()
    images = torch.randn(1, 3, 384, 384, device=device)
    lang = torch.randint(0, 32000, (1, 32), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = student(images, language_ids=lang)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)

    latencies = []
    with torch.no_grad():
        for _ in range(30):
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            t0 = time.time()
            out = student(images, language_ids=lang)
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            latencies.append(time.time() - t0)

    avg_ms = sum(latencies) / len(latencies) * 1000
    p50 = sorted(latencies)[len(latencies) // 2] * 1000
    p99 = sorted(latencies)[int(len(latencies) * 0.99)] * 1000

    results["inference"] = {
        "latency_avg_ms": round(avg_ms, 1),
        "latency_p50_ms": round(p50, 1),
        "latency_p99_ms": round(p99, 1),
        "throughput_fps": round(1000 / avg_ms, 1),
    }
    if device.startswith("cuda"):
        results["inference"]["gpu_memory_GB"] = round(torch.cuda.memory_allocated(device) / 1e9, 2)

    print(f"   ✓ Latency: {avg_ms:.1f}ms avg | {1000 / avg_ms:.1f} fps")

    # ── Step 3: Knowledge Distillation ──
    print(f"\n📚 Step 3/5: Knowledge distillation ({steps} steps)...")
    import numpy as np

    from forge.data.label_writer import LabelWriter
    from forge.data.teacher_dataset import TeacherLabelDataset
    from forge.losses import ForgeDistillationLoss
    from forge.types import EpisodeData

    data_dir = Path("/tmp/forge_demo_labels")
    data_dir.mkdir(parents=True, exist_ok=True)
    writer = LabelWriter(
        str(data_dir),
        episodes_per_file=50,
        save_vision_features=False,
        labels_provenance="mock",
    )
    for i in range(100):
        episode_steps = 10
        ep = EpisodeData(
            episode_id=f"demo_{i}",
            task_id=f"task_{i % 5}",
            language_instruction="pick up the block",
            timesteps=episode_steps,
            images=np.random.randint(0, 255, (episode_steps, 64, 64, 3), dtype=np.uint8),
            proprioception=np.random.randn(episode_steps, 7).astype(np.float32) * 0.1,
            teacher_action_logits=np.random.randn(episode_steps, 7).astype(np.float32) * 0.1,
            teacher_action_mean=np.random.randn(episode_steps, 7).astype(np.float32) * 0.1,
            teacher_action_std=np.abs(np.random.randn(episode_steps, 7).astype(np.float32) * 0.1) + 0.01,
            teacher_vision_features=None,
            confidence=np.random.rand(episode_steps, 7).astype(np.float32),
            ground_truth_actions=np.random.randn(episode_steps, 7).astype(np.float32) * 0.1,
            success=True,
        )
        writer.write_episode(ep)
    writer.finalize()

    dataset = TeacherLabelDataset(str(data_dir))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
    criterion = ForgeDistillationLoss(temperature=4.0, alpha_kd=0.4, alpha_task=0.3, alpha_feat=0.2, alpha_action=0.1)

    for param in student.parameters():
        param.requires_grad = False
    for param in student.bridge.parameters():
        param.requires_grad = True
    for param in student.action_head.parameters():
        param.requires_grad = True
    for name, param in student.language.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.01)
    student.train()

    losses_history = []
    data_iter = iter(dataloader)
    t_start = time.time()

    for step in range(steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        imgs = batch["image"].to(device)
        gt = batch["ground_truth_actions"].to(device)
        t_logits = batch["teacher_action_logits"].to(device)
        t_mean = batch["teacher_action_mean"].to(device)
        t_std = batch["teacher_action_std"].to(device)
        conf = batch["confidence"].to(device)

        out = student(imgs, gt_actions=gt)
        losses = criterion(
            student_actions=out["actions"],
            teacher_action_logits=t_logits,
            ground_truth_actions=gt,
            teacher_action_mean=t_mean,
            teacher_action_std=t_std,
            teacher_confidence=conf,
        )
        loss = losses["total"] + out.get("loss", 0)
        loss.backward()

        if (step + 1) % 2 == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()

        losses_history.append(loss.item())

    train_time = time.time() - t_start
    first_avg = sum(losses_history[:10]) / 10
    last_avg = sum(losses_history[-10:]) / 10

    results["training"] = {
        "steps": steps,
        "time_s": round(train_time, 1),
        "steps_per_sec": round(steps / train_time, 1),
        "loss_start": round(first_avg, 4),
        "loss_end": round(last_avg, 4),
        "loss_reduction_pct": round((1 - last_avg / max(first_avg, 1e-8)) * 100, 1),
    }
    print(f"   ✓ Loss: {first_avg:.4f} → {last_avg:.4f} ({results['training']['loss_reduction_pct']:.1f}% reduction)")

    # ── Step 4: Compression ──
    print("\n🗜️  Step 4/5: Pruning + INT4 quantization...")
    import copy

    from forge.config import PruningConfig
    from forge.prune import _find_transformer_layers, prune_layers
    from forge.quantize import quantize_model

    student_cpu = copy.deepcopy(student).cpu()
    layers = _find_transformer_layers(student_cpu)
    n_before = len(layers)

    if n_before >= 8:
        scores = {i: 1.0 - abs(i - n_before / 2) / (n_before / 2) for i in range(n_before)}
        target = max(n_before * 2 // 3, 8)
        pruned, removed = prune_layers(
            student_cpu, scores, PruningConfig(target_layers=target, keep_first_n=2, keep_last_n=2)
        )
        n_after = len(_find_transformer_layers(pruned))
    else:
        pruned = student_cpu
        removed = []
        n_after = n_before

    q4 = quantize_model(pruned, uniform_bits=4)
    q4_params = sum(p.numel() for p in q4.parameters())
    q4_size_mb = q4_params * 0.5 / 1e6
    orig_size_mb = student.total_params * 2 / 1e6

    results["compression"] = {
        "layers_before": n_before,
        "layers_after": n_after,
        "layers_removed": len(removed),
        "original_size_MB": round(orig_size_mb, 1),
        "int4_size_MB": round(q4_size_mb, 1),
        "compression_ratio": round(orig_size_mb / q4_size_mb, 1),
    }
    print(
        f"   ✓ {n_before} → {n_after} layers | "
        f"{orig_size_mb:.0f}MB → {q4_size_mb:.0f}MB "
        f"({orig_size_mb / q4_size_mb:.1f}x)"
    )

    # ── Step 5: ONNX Export ──
    print("\n📦 Step 5/5: ONNX export...")
    from forge.export.onnx_export import export_onnx

    try:
        onnx_path = export_onnx(pruned, out_path / "forge_nano.onnx", image_size=384, max_seq_len=32)
        onnx_size = onnx_path.stat().st_size / 1e6
        results["export"] = {"onnx_path": str(onnx_path), "onnx_size_MB": round(onnx_size, 1), "status": "success"}
        print(f"   ✓ ONNX: {onnx_size:.1f} MB")
    except Exception as e:
        results["export"] = {"status": "failed", "error": str(e)}
        print(f"   ✗ ONNX failed: {e}")

    # ── Save checkpoint ──
    ckpt_path = out_path / "forge_nano_demo.pt"
    _save_demo_checkpoint(
        ckpt_path,
        student=student,
        config=config,
        results=results,
        dataset=dataset,
        model_dir=model_dir,
    )

    # ── Save results ──
    results_path = out_path / "demo_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Cleanup
    del student, student_cpu, pruned, q4
    gc.collect()
    if device.startswith("cuda"):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    return results
