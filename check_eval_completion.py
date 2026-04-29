#!/usr/bin/env python3
"""Check whether an eval run (from molmo_spaces/evaluation/eval_main.py) is complete.

Compares the trajectories produced under an eval output directory against the
benchmark it was run against, and reports:
  - expected vs actual house count
  - expected vs actual episode count
  - last-frame and any-frame success rates over the episodes that did run
  - which houses are missing entirely, and which houses are short on episodes

Usage:
    python check_eval_completion.py <eval_output_dir>
    python check_eval_completion.py <eval_output_dir> --benchmark_dir <path>
    python check_eval_completion.py <eval_output_dir> --show-missing-episodes
    python check_eval_completion.py <eval_output_dir> --delete-partial-houses

After --delete-partial-houses, re-run the eval against the same output directory
by setting WANDB_RUN_NAME to the eval dir's basename, e.g.:

    WANDB_RUN_NAME=pnp-v1_20260428_103426 ./run_molmospaces_evals.sh

The runner skips any house whose trajectories h5 still exists, so only the
deleted (partial) houses get re-rolled.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def find_benchmark_dir_from_log(eval_dir: Path) -> Path | None:
    """Parse benchmark_path out of running_log.log written by eval_main.py."""
    log_path = eval_dir / "running_log.log"
    if not log_path.exists():
        return None
    pattern = re.compile(r"'benchmark_path':\s*PosixPath\('([^']+)'\)")
    with open(log_path, errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return Path(m.group(1))
    return None


def load_expected_episodes_per_house(benchmark_dir: Path) -> dict[int, int]:
    """Return {house_index: expected_episode_count} from a benchmark directory.

    Supports the single-file benchmark.json format and the legacy
    house_*/episode_*.json layout.
    """
    benchmark_file = benchmark_dir / "benchmark.json"
    if benchmark_file.exists():
        with open(benchmark_file) as f:
            episodes = json.load(f)
        return dict(Counter(int(ep["house_index"]) for ep in episodes))

    counts: dict[int, int] = {}
    for house_dir in sorted(benchmark_dir.glob("house_*")):
        if not house_dir.is_dir():
            continue
        try:
            house_id = int(house_dir.name.removeprefix("house_"))
        except ValueError:
            continue
        n = len(list(house_dir.glob("episode_*.json")))
        if n:
            counts[house_id] = n
    if not counts:
        raise FileNotFoundError(
            f"Could not find benchmark.json or house_*/episode_*.json under {benchmark_dir}"
        )
    return counts


def collect_actual_trajectories(eval_dir: Path) -> dict[int, list[tuple[Path, str, np.ndarray | None]]]:
    """For each house dir under eval_dir, return list of (h5_path, traj_key, success_array).

    Handles multiple trajectory h5 files per house (e.g. batch_1_of_2, batch_2_of_2)
    by concatenating the traj_* groups across all of them.
    """
    actual: dict[int, list[tuple[Path, str, np.ndarray | None]]] = {}
    for house_dir in sorted(eval_dir.glob("house_*")):
        if not house_dir.is_dir():
            continue
        try:
            house_id = int(house_dir.name.removeprefix("house_"))
        except ValueError:
            continue
        trajs: list[tuple[Path, str, np.ndarray | None]] = []
        for h5_path in sorted(house_dir.glob("trajectories*.h5")):
            try:
                with h5py.File(h5_path, "r") as f:
                    for key in sorted(f.keys()):
                        if not key.startswith("traj_"):
                            continue
                        group = f[key]
                        success = (
                            np.asarray(group["success"]) if "success" in group else None
                        )
                        trajs.append((h5_path, key, success))
            except OSError as e:
                print(f"Warning: could not open {h5_path}: {e}")
        actual[house_id] = trajs
    return actual


def main():
    parser = argparse.ArgumentParser(
        description="Check completion of an eval run against its benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("eval_dir", type=str, help="Path to eval output directory.")
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        default=None,
        help="Override benchmark dir. Default: parse from running_log.log in eval_dir.",
    )
    parser.add_argument(
        "--show-missing-episodes",
        action="store_true",
        help="Also list every (house, episode_idx) that is missing or short.",
    )
    parser.add_argument(
        "--max-missing-houses",
        type=int,
        default=20,
        help="Cap the number of missing houses listed in the summary.",
    )
    parser.add_argument(
        "--delete-partial-houses",
        action="store_true",
        help=(
            "Delete house_<id>/ directories that ran fewer episodes than expected. "
            "After running, eval_main.py / run_molmospaces_evals.sh can be re-run "
            "(with the same WANDB_RUN_NAME) to re-roll the deleted houses; the "
            "runner skips houses whose trajectories h5 still exists."
        ),
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir).resolve()
    if not eval_dir.is_dir():
        raise NotADirectoryError(f"Eval dir is not a directory: {eval_dir}")

    if args.benchmark_dir is not None:
        benchmark_dir = Path(args.benchmark_dir).resolve()
    else:
        bd = find_benchmark_dir_from_log(eval_dir)
        if bd is None:
            raise FileNotFoundError(
                f"Could not find benchmark_path in {eval_dir}/running_log.log. "
                "Pass --benchmark_dir explicitly."
            )
        # Log paths may point to absolute locations on the original machine;
        # if that path doesn't exist, try mapping into the local repo's assets/.
        if not bd.exists():
            repo_assets = Path(__file__).resolve().parent / "assets" / "benchmarks"
            for marker in ("molmospaces-bench-v1", "molmospaces-bench-v2"):
                if marker in bd.parts:
                    idx = bd.parts.index(marker)
                    candidate = repo_assets / Path(*bd.parts[idx:])
                    if candidate.exists():
                        bd = candidate
                        break
        benchmark_dir = bd

    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark dir does not exist: {benchmark_dir}")

    expected = load_expected_episodes_per_house(benchmark_dir)
    actual = collect_actual_trajectories(eval_dir)

    expected_houses = set(expected)
    actual_houses = {h for h, trajs in actual.items() if trajs}
    expected_episode_total = sum(expected.values())
    actual_episode_total = sum(len(t) for t in actual.values())

    missing_houses = sorted(expected_houses - actual_houses)
    extra_houses = sorted(actual_houses - expected_houses)

    short_houses: list[tuple[int, int, int]] = []  # (house, actual, expected)
    for h, exp_n in expected.items():
        got_n = len(actual.get(h, []))
        # "Short" = partially populated. Houses with 0 trajs are reported as
        # "Missing" instead, so we don't double-count them here.
        if 0 < got_n < exp_n:
            short_houses.append((h, got_n, exp_n))
    short_houses.sort()

    last_frame_successes = 0
    any_frame_successes = 0
    counted_episodes = 0
    for trajs in actual.values():
        for _, _, success in trajs:
            if success is None or len(success) == 0:
                continue
            counted_episodes += 1
            if bool(success[-1]):
                last_frame_successes += 1
            if bool(np.any(success)):
                any_frame_successes += 1

    print(f"Eval dir:      {eval_dir}")
    print(f"Benchmark dir: {benchmark_dir}")
    print()
    print("=== Coverage ===")
    print(
        f"Houses:   {len(actual_houses):>5} / {len(expected_houses):>5} "
        f"({len(actual_houses) / max(len(expected_houses), 1):.2%})"
    )
    print(
        f"Episodes: {actual_episode_total:>5} / {expected_episode_total:>5} "
        f"({actual_episode_total / max(expected_episode_total, 1):.2%})"
    )

    print()
    print("=== Success (over completed episodes) ===")
    if counted_episodes == 0:
        print("No trajectories with a 'success' array found.")
    else:
        print(
            f"Last-frame success: {last_frame_successes} / {counted_episodes} = "
            f"{last_frame_successes / counted_episodes:.2%}"
        )
        print(
            f"Any-frame success:  {any_frame_successes} / {counted_episodes} = "
            f"{any_frame_successes / counted_episodes:.2%}"
        )

    is_complete = (
        not missing_houses
        and not short_houses
        and actual_episode_total == expected_episode_total
    )
    print()
    print("=== Status ===")
    if is_complete:
        print("BENCHMARK COMPLETE: every expected (house, episode) is present.")
    else:
        print("BENCHMARK INCOMPLETE.")

        if missing_houses:
            shown = missing_houses[: args.max_missing_houses]
            extra_msg = (
                f" (+{len(missing_houses) - len(shown)} more)"
                if len(missing_houses) > len(shown)
                else ""
            )
            print(f"  Missing houses ({len(missing_houses)}): {shown}{extra_msg}")

        if short_houses:
            shown = short_houses[: args.max_missing_houses]
            extra_msg = (
                f" (+{len(short_houses) - len(shown)} more)"
                if len(short_houses) > len(shown)
                else ""
            )
            short_str = ", ".join(f"house_{h}({got}/{exp})" for h, got, exp in shown)
            print(f"  Short houses   ({len(short_houses)}): {short_str}{extra_msg}")

        if args.show_missing_episodes:
            print()
            print("  Missing episodes (house, episode_idx):")
            for h, got_n, exp_n in short_houses:
                for ep_idx in range(got_n, exp_n):
                    print(f"    house_{h}, episode_{ep_idx}")
            for h in missing_houses:
                for ep_idx in range(expected[h]):
                    print(f"    house_{h}, episode_{ep_idx}")

    if extra_houses:
        print()
        print(
            f"Note: {len(extra_houses)} houses present in eval dir but NOT in benchmark "
            f"(first 10): {extra_houses[:10]}"
        )

    if args.delete_partial_houses:
        print()
        print("=== Deleting partial houses ===")
        if not short_houses:
            print("No partial houses to delete.")
        else:
            for h, got_n, exp_n in short_houses:
                house_dir = eval_dir / f"house_{h}"
                if house_dir.exists():
                    shutil.rmtree(house_dir)
                    print(f"  Deleted {house_dir} (had {got_n}/{exp_n} episodes)")
                else:
                    print(f"  Skipped {house_dir} (does not exist)")
            print(
                f"Deleted {len(short_houses)} partial house dir(s). Re-run with:"
            )
            print(f"  WANDB_RUN_NAME={eval_dir.name} ./run_molmospaces_evals.sh")


if __name__ == "__main__":
    main()
