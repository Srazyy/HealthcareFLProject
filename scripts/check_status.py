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


def get_log_progress():
    log_path = REPO_ROOT / "results" / "sweep_run.log"
    if not log_path.exists():
        return None, None, []

    content = log_path.read_text(errors="ignore")
    lines = [line.strip() for line in content.splitlines() if line.strip()]

    # Look for [X/Y] progress markers
    progress_matches = re.findall(r"\[(\d+)/(\d+)\]\s+(.+)", content)
    latest_progress = progress_matches[-1] if progress_matches else None

    # Look for [ROUND X]
    round_matches = re.findall(r"\[ROUND\s+(\d+)\]", content)
    latest_round = round_matches[-1] if round_matches else None

    # Last 5 meaningful lines
    tail_lines = lines[-5:] if len(lines) >= 5 else lines
    return latest_progress, latest_round, tail_lines


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
        print(f"    Sim Cluster Workers:  {proc.get('workers', 0)} Ray Actors Active")
        print(f"    Aggregate CPU Usage:  {proc.get('total_cpu', 0.0):.1f}%   |   Cluster RAM: {proc.get('total_mem', 0.0):.1f}%")
        print(f"    Active Command:       {proc['cmd']}")
        if "T" in proc["stat"]:
            print("\n  ⚠️  WARNING: Process is SUSPENDED (STAT=T).")
            print("     Run 'kill -9 " + proc["pid"] + "' and restart with '< /dev/null'.")
    else:
        print("  ○ Background Process:  NO SWEEP CURRENTLY RUNNING (Idle)")

    print("-" * 70)

    # 2. Progress from Log
    latest_progress, latest_round, tail_lines = get_log_progress()
    if latest_progress:
        curr, total, details = latest_progress
        percent = (int(curr) / int(total)) * 100 if int(total) > 0 else 0
        print(f"  ● Config Grid Progress: [{curr}/{total}] ({percent:.1f}% Completed)")
        print(f"  ● Active Configuration: {details}")
    else:
        print("  ● Config Grid Progress: [1/48] Initializing Grid Run (Round " + (latest_round or "1") + "/5)")

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
