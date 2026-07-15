"""Benchmark 07: Cross-Embodiment Transfer — action mapping speed, accuracy, all strategies."""

from __future__ import annotations

import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch

from forge.benchmark.artifacts import write_json_artifact
from forge.benchmark.suites.real_data import data_provenance, load_real_dataset, real_batch
from forge.benchmark.suites.runtime import results_dir

RESULTS_DIR = results_dir()

MODEL_DIR = Path(os.environ.get("FORGE_MODEL_DIR", "./models"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ProfileConfig(TypedDict):
    """Typed constructor inputs for one benchmark embodiment."""

    action_dim: int
    joint_names: list[str]
    joint_min: list[float]
    joint_max: list[float]
    has_gripper: bool


# Robot profiles
PROFILES: dict[str, ProfileConfig] = {
    "franka": {
        "action_dim": 7,
        "joint_names": ["j1", "j2", "j3", "j4", "j5", "j6", "j7"],
        "joint_min": [-2.9, -1.8, -2.9, -3.1, -2.9, -0.02, -2.9],
        "joint_max": [2.9, 1.8, 2.9, -0.07, 2.9, 3.75, 2.9],
        "has_gripper": True,
    },
    "ur5e": {
        "action_dim": 6,
        "joint_names": ["shoulder", "upper_arm", "forearm", "wrist1", "wrist2", "wrist3"],
        "joint_min": [-6.28] * 6,
        "joint_max": [6.28] * 6,
        "has_gripper": False,
    },
    "aloha": {
        "action_dim": 14,
        "joint_names": [f"left_j{i}" for i in range(7)] + [f"right_j{i}" for i in range(7)],
        "joint_min": [-3.14] * 14,
        "joint_max": [3.14] * 14,
        "has_gripper": True,
    },
    "xarm": {
        "action_dim": 6,
        "joint_names": ["j1", "j2", "j3", "j4", "j5", "j6"],
        "joint_min": [-6.28] * 6,
        "joint_max": [6.28] * 6,
        "has_gripper": False,
    },
}

N_SAMPLES = 1000
N_TIMING = 10000


def main():
    from forge.cross_embodiment import EmbodimentProfile, EmbodimentTransfer, TransferConfig

    if DEVICE != "cuda":
        print("SKIP: No CUDA device")
        sys.exit(0)

    profiles = {}
    for name, cfg in PROFILES.items():
        profiles[name] = EmbodimentProfile(name=name, **cfg)

    from forge.config import ForgeConfig
    from forge.student import FORGEStudent

    dataset = load_real_dataset(MODEL_DIR, max_samples=32)
    print("Generating actions from FORGEStudent on real observations...")
    student = FORGEStudent(ForgeConfig.default().student, model_dir=str(MODEL_DIR)).to(DEVICE)
    student.eval()
    action_chunks = []
    batch_size = 8
    with torch.no_grad():
        for index in range(0, N_SAMPLES, batch_size):
            bs = min(batch_size, N_SAMPLES - index)
            images, _ = real_batch(dataset, bs, DEVICE, start=index)
            action_chunks.append(student(images)["actions"].cpu())
    franka_actions = torch.cat(action_chunks, dim=0).numpy()
    del student
    torch.cuda.empty_cache()
    print(f"  Generated {franka_actions.shape[0]} actions, shape={franka_actions.shape}")
    action_source = "real_observations_model_inference"

    source_actions = {"franka": franka_actions}
    for name in profiles:
        if name == "franka":
            continue
        mapper = EmbodimentTransfer(profiles["franka"], profiles[name], TransferConfig(mapping_strategy="linear"))
        source_actions[name] = mapper.map_actions(franka_actions)

    transfer_results = {}
    pairs = [
        ("franka", "ur5e"),
        ("franka", "aloha"),
        ("ur5e", "franka"),
        ("aloha", "franka"),
        ("franka", "xarm"),
        ("xarm", "aloha"),
    ]

    for src_name, tgt_name in pairs:
        src_actions = source_actions[src_name]

        pair_results = {}
        for strategy in ["linear", "joint_name", "learned"]:
            print(
                f"\n{src_name}({profiles[src_name].action_dim}D) → "
                f"{tgt_name}({profiles[tgt_name].action_dim}D) [{strategy}]"
            )

            config = TransferConfig(mapping_strategy=strategy)
            transfer = EmbodimentTransfer(profiles[src_name], profiles[tgt_name], config)
            info = transfer.info()

            # A joint-name transfer is meaningful only when the two profiles expose
            # at least one semantic correspondence.  The production mapper fails
            # closed for an empty mapping; record non-applicability here instead of
            # benchmarking a dangerous all-zero action tensor as a success.
            if strategy == "joint_name" and info.get("joint_mapping_complete") is not True:
                pair_results[strategy] = {
                    "strategy": strategy,
                    "source": src_name,
                    "target": tgt_name,
                    "source_dim": profiles[src_name].action_dim,
                    "target_dim": profiles[tgt_name].action_dim,
                    "dim_change": profiles[tgt_name].action_dim - profiles[src_name].action_dim,
                    "applicable": False,
                    "reason_code": "incomplete_semantic_joint_correspondence",
                    "matched_joints": len(info.get("joint_mapping", {})),
                    "unmatched_target_joints": info.get("unmatched_target_joints", []),
                }
                print("  Not applicable: incomplete semantic joint-name correspondence")
                continue

            learned_training = None
            if strategy == "learned":
                target_teacher = EmbodimentTransfer(
                    profiles[src_name],
                    profiles[tgt_name],
                    TransferConfig(mapping_strategy="linear"),
                ).map_actions(src_actions)
                learned_training = transfer.fit_learned_adapter(src_actions, target_teacher)
                learned_training["quality_passed"] = bool(
                    learned_training["loss_after"] < learned_training["loss_before"]
                )
                info = transfer.info()

            # Map actions
            t0 = time.perf_counter()
            mapped = transfer.map_actions(src_actions)
            map_time = (time.perf_counter() - t0) * 1000

            # Timing (many iterations for accurate measurement)
            single = src_actions[:1]
            times = []
            for _ in range(N_TIMING):
                t0 = time.perf_counter()
                _ = transfer.map_actions(single)
                times.append((time.perf_counter() - t0) * 1e6)  # microseconds

            ta = np.array(times)
            result = {
                "strategy": strategy,
                "source": src_name,
                "target": tgt_name,
                "source_dim": profiles[src_name].action_dim,
                "target_dim": profiles[tgt_name].action_dim,
                "dim_change": profiles[tgt_name].action_dim - profiles[src_name].action_dim,
                "batch_map_time_ms": round(map_time, 3),
                "batch_size": N_SAMPLES,
                "single_map_mean_us": round(float(ta.mean()), 2),
                "single_map_p50_us": round(float(np.percentile(ta, 50)), 2),
                "single_map_p99_us": round(float(np.percentile(ta, 99)), 2),
                "maps_per_sec": round(float(1e6 / ta.mean()), 0),
                "output_shape": list(mapped.shape),
                "output_range": [round(float(mapped.min()), 4), round(float(mapped.max()), 4)],
                "output_mean": round(float(mapped.mean()), 4),
                "output_std": round(float(mapped.std()), 4),
            }

            if "joint_mapping" in info:
                result["joint_mapping"] = info["joint_mapping"]
                result["n_matched_joints"] = len(info["joint_mapping"])
                result["applicable"] = True
            if "adapter_params" in info:
                result["adapter_params"] = info["adapter_params"]
            if learned_training is not None:
                result["adapter_training"] = learned_training
                result["target_provenance"] = "deterministic_linear_mapping_of_real_model_actions"

            pair_results[strategy] = result
            print(f"  Batch: {map_time:.1f}ms, Single: {ta.mean():.1f}us, {1e6 / ta.mean():.0f} maps/s")

        transfer_results[f"{src_name}_to_{tgt_name}"] = pair_results

    joint_name_results = [pair["joint_name"] for pair in transfer_results.values()]
    learned_results = [pair["learned"] for pair in transfer_results.values()]
    strategy_coverage = {
        "joint_name": {
            "applicable_pairs": sum(result.get("applicable") is True for result in joint_name_results),
            "not_applicable_pairs": sum(result.get("applicable") is False for result in joint_name_results),
            "required_applicable_pairs": 1,
        },
        "learned": {
            "trained_pairs": sum(
                result.get("adapter_training", {}).get("quality_passed") is True for result in learned_results
            ),
            "required_trained_pairs": len(learned_results),
        },
    }
    strategy_coverage["joint_name"]["coverage_passed"] = (
        strategy_coverage["joint_name"]["applicable_pairs"]
        >= strategy_coverage["joint_name"]["required_applicable_pairs"]
    )
    strategy_coverage["learned"]["coverage_passed"] = (
        strategy_coverage["learned"]["trained_pairs"] >= strategy_coverage["learned"]["required_trained_pairs"]
    )

    results = {
        "benchmark": "cross_embodiment",
        "timestamp": datetime.now(UTC).isoformat(),
        "device": DEVICE,
        "action_source": action_source,
        "data_provenance": data_provenance(dataset),
        "n_samples": N_SAMPLES,
        "n_timing_iterations": N_TIMING,
        "profiles": {
            name: {"action_dim": p.action_dim, "n_joints": len(p.joint_names), "has_gripper": p.has_gripper}
            for name, p in profiles.items()
        },
        "strategy_coverage": strategy_coverage,
        "transfers": transfer_results,
    }

    out_path = RESULTS_DIR / "bench_07_cross_embodiment.json"
    write_json_artifact(out_path, results)

    print(f"\nResults saved to {out_path}")
    print("BENCH 07: DONE")


if __name__ == "__main__":
    main()
