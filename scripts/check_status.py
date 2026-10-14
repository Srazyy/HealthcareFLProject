#!/usr/bin/env python3
"""
Live status checker for FL Healthcare Benchmark sweeps.
Prints process state, completed progress, active configuration, and output file stats.

Usage:
    python scripts/check_status.py
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_process_info():
    try:
        res = subprocess.run(
            ["ps", "-eo", "pid,stat,%cpu,%mem,etime,command"],
            capture_output=True,
            text=True,
            check=True,
        )
        main_proc: dict[str, Any] | None = None
        worker_count = 0
        total_cpu = 0.0
        total_mem = 0.0

        for line in res.stdout.splitlines():
            if "sweep.py" in line and "grep" not in line and "check_status" not in line:
                parts = line.strip().split(None, 5)
                if len(parts) >= 6:
                    main_proc = {
                        "pid": parts[0],
                        "stat": parts[1],
                        "cpu": parts[2],
                        "mem": parts[3],
                        "etime": parts[4],
                        "cmd": parts[5],
                        "total_cpu": 0.0,
                        "total_mem": 0.0,
                        "workers": 0,
                    }
                    try:
                        total_cpu += float(parts[2])
                        total_mem += float(parts[3])
                    except ValueError:
                        pass
            elif "ClientAppActor" in line or "raylet" in line:
                parts = line.strip().split(None, 5)
                if len(parts) >= 6:
                    try:
                        total_cpu += float(parts[2])
                        total_mem += float(parts[3])
                        worker_count += 1
                    except ValueError:
                        pass

        if main_proc:
            main_proc["total_cpu"] = total_cpu
            main_proc["total_mem"] = total_mem
            main_proc["workers"] = worker_count
            return main_proc
    except Exception:
        pass
    return None


def check_caffeinate_status(pid: str | None = None) -> bool:
    try:
        res = subprocess.run(
            ["pgrep", "-fla", "caffeinate"],
            capture_output=True,
            text=True,
        )
        for line in res.stdout.splitlines():
            if pid and str(pid) in line:
                return True
            if "caffeinate" in line and ("-dim" in line or "-s" in line):
                return True
    except Exception:
        pass
    return False


def get_log_progress():
    log_path = REPO_ROOT / "results" / "sweep_run.log"
    saved_models_dir = REPO_ROOT / "saved_models"

    saved_models = []
    if saved_models_dir.exists():
        saved_models = sorted([
            d.name for d in saved_models_dir.iterdir()
            if d.is_dir() and (d / "adapter_config.json").exists()
        ])

    if not log_path.exists():
        return None, None, [], saved_models

    content = log_path.read_text(errors="ignore")
    lines = [line.strip() for line in content.splitlines() if line.strip()]

    # Explicit progress markers if present
    progress_matches = re.findall(r"\[(\d+)/(\d+)\]\s+(.+)", content)
    latest_progress = progress_matches[-1] if progress_matches else None

    # Ground truth: count physical saved models on disk
    completed_count = len(saved_models)
    current_config = min(48, completed_count + 1)

    # Look for [ROUND X]
    round_matches = re.findall(r"\[ROUND\s+(\d+)\]", content)
    latest_round = round_matches[-1] if round_matches else None

    # Last 5 meaningful lines
    tail_lines = lines[-5:] if len(lines) >= 5 else lines
    return (current_config, completed_count, 48, latest_progress), latest_round, tail_lines, saved_models


def main():
    print("=" * 70)
    print("  HEALTHCARE FL BENCHMARK — SWEEP STATUS MONITOR")
    print("=" * 70)

    # 1. Process Status
    proc = get_process_info()
    if proc:
        stat_desc = "RUNNING (Active)" if "R" in proc["stat"] or "S" in proc["stat"] or "U" in proc["stat"] else f"STOPPED/PAUSED ({proc['stat']})"
        print(f"  ● Orchestrator Process: PID {proc['pid']}  [{stat_desc}]")
        print(f"    Elapsed Uptime:       {proc['etime']}")
        is_caff = check_caffeinate_status(proc["pid"])
        caff_str = "ACTIVE (Sleep Blocked via Caffeinate)" if is_caff else "INACTIVE (System May Sleep)"
        print(f"    Sleep Prevention:     {caff_str}")
        print(f"    Hardware Compute:     Apple Silicon MPS (Metal GPU Acceleration)")
        print(f"    Sim Cluster Workers:  {proc.get('workers', 0)} Ray Actors Active")
        print(f"    Aggregate CPU Usage:  {proc.get('total_cpu', 0.0):.1f}% (GPU offloaded) | Cluster RAM: {proc.get('total_mem', 0.0):.1f}%")
        print(f"    Active Command:       {proc['cmd']}")
        if "T" in proc["stat"]:
            print("\n  ⚠️  WARNING: Process is SUSPENDED (STAT=T).")
            print("     Run 'kill -9 " + proc["pid"] + "' and restart with '< /dev/null'.")
    else:
        print("  ○ Background Process:  NO SWEEP CURRENTLY RUNNING (Idle)")

    print("-" * 70)

    # 2. Progress from Log & Saved Adapters
    progress_info, latest_round, tail_lines, saved_models = get_log_progress()
    if progress_info:
        curr, completed_count, total, explicit = progress_info
        percent = (completed_count / total) * 100
        print(f"  ● Config Grid Progress: Running Config [{curr}/{total}] ({completed_count} Finished, {percent:.1f}% Completed)")
        if explicit:
            _, _, details = explicit
            print(f"  ● Active Details:      {details}")
        if latest_round:
            print(f"  ● Federation Round:     Round {latest_round} of 5 in progress for Config [{curr}]")
        if saved_models:
            print(f"  ● Saved LoRA Adapters:  {len(saved_models)} saved to disk:")
            for m in saved_models[-3:]:
                print(f"      ✔ saved_models/{m}")
    else:
        if latest_round:
            print(f"  ● Federation Round:     Round {latest_round} of 5 in progress")

    print("-" * 70)

    # 3. Output Results File Check
    csv_path = REPO_ROOT / "results" / "sweep_results.csv"
    if csv_path.exists():
        line_count = sum(1 for _ in open(csv_path)) - 1
        size_kb = csv_path.stat().st_size / 1024
        print(f"  ● Results Ledger:      results/sweep_results.csv ({max(0, line_count)} rows recorded, {size_kb:.1f} KB)")
    else:
        print("  ● Results Ledger:      results/sweep_results.csv (Not yet created)")

    # 4. Dashboard
    print(f"  ● Comparison Dashboard: http://localhost:8501/dashboard/index.html")

    print("-" * 70)
    print("  Recent Log Output (tail):")
    for l in tail_lines:
        print(f"    {l[:100]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
