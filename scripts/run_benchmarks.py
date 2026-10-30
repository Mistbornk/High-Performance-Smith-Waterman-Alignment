#!/usr/bin/env python3
"""Sweep sw-bench over backends x thread counts x datasets and collect one CSV.

Output directory (new, never overwritten):
  results.csv    one row per run (sw-bench columns + dataset)
  metadata.json  git revision, host CPU/GPU, build configuration, exact commands
  logs/          stderr of every run

Unavailable backends (sw-bench exit 77) are recorded as skipped, not as failures.
Score checksums are compared across backends per dataset and mode; a mismatch
means the backends disagree on at least one optimal score and exits non-zero.
"""
import argparse
import csv
import datetime
import io
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKIP = 77
# Device backends ignore the thread sweep and run once with threads=1.
DEVICE_BACKENDS = {"cuda"}


def capture(cmd, cwd=None):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cpu_quota():
    """Effective CPU limit from the cgroup (containers), or None when unlimited."""
    try:
        quota, period = pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:
        quota = int(pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        return None if quota <= 0 else quota / period
    except (OSError, ValueError):
        return None


def host_metadata(build):
    cpu = None
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    cache = {}
    try:
        for line in (build / "CMakeCache.txt").read_text().splitlines():
            for key in ("CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS_RELEASE",
                        "SW_ENABLE_STRIPED"):
                if line.startswith(key + ":"):
                    cache[key] = line.split("=", 1)[1]
    except OSError:
        pass
    gpu = None
    if shutil.which("nvidia-smi"):
        gpu = capture(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    return {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": capture(["git", "rev-parse", "HEAD"], cwd=ROOT),
        "git_dirty": bool(capture(["git", "status", "--porcelain"], cwd=ROOT)),
        "hostname": platform.node(),
        "cpu": cpu,
        "logical_cpus": len(os.sched_getaffinity(0)),
        # A container quota caps parallel speedup below logical_cpus.
        "cgroup_cpu_quota": cpu_quota(),
        "load_average": os.getloadavg(),
        "gpu": gpu,
        "build": cache,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", type=pathlib.Path, default=ROOT / "build" / "sw-bench")
    parser.add_argument("--dataset", type=pathlib.Path, action="append", required=True,
                        help="directory with reference.fa and query.fa (repeatable)")
    parser.add_argument("--out", type=pathlib.Path, required=True, help="new results directory")
    parser.add_argument("--backends", default="striped,scalar,cuda")
    parser.add_argument("--threads", default="1", help="comma-separated CPU thread counts")
    parser.add_argument("--pairing", default="zip", choices=["zip", "all_vs_all"])
    parser.add_argument("--scoring", default="1,-4,-6,-1")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()

    if not args.bench.is_file():
        parser.error(f"{args.bench} not found; build with -DSW_BUILD_BENCHMARKS=ON")
    backends = [b for b in args.backends.split(",") if b]
    threads = sorted({int(t) for t in args.threads.split(",") if t})
    if not threads or min(threads) < 1:
        parser.error("--threads must list positive integers")
    for dataset in args.dataset:
        for name in ("reference.fa", "query.fa"):
            if not (dataset / name).is_file():
                parser.error(f"missing {dataset / name}")

    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "logs").mkdir()
    metadata = host_metadata(args.bench.parent)
    quota = metadata["cgroup_cpu_quota"]
    if quota and max(threads) > quota:
        print(f"warning: cgroup CPU quota is {quota:g} CPUs; runs above that cannot scale",
              file=sys.stderr)
    metadata["commands"] = []
    rows, skipped, failed = [], [], []
    for dataset in args.dataset:
        for backend in backends:
            for t in ([1] if backend in DEVICE_BACKENDS else threads):
                label = f"{dataset.name}/{backend}/t{t}"
                cmd = [str(args.bench), "--backend", backend, "--threads", str(t),
                       "--reference", str(dataset / "reference.fa"), "--query", str(dataset / "query.fa"),
                       "--pairing", args.pairing, "--scoring", args.scoring,
                       "--warmup", str(args.warmup), "--repetitions", str(args.repetitions),
                       "--label", label]
                if args.score_only:
                    cmd.append("--score-only")
                metadata["commands"].append(cmd)
                print(f"[run] {label}", file=sys.stderr, flush=True)
                proc = subprocess.run(cmd, capture_output=True, text=True)
                log = args.out / "logs" / (label.replace("/", "_") + ".log")
                log.write_text(proc.stderr)
                if proc.returncode == SKIP:
                    skipped.append(label)
                    print(f"  skipped: {proc.stderr.strip()}", file=sys.stderr)
                    break  # remaining thread counts would be skipped too
                if proc.returncode != 0:
                    failed.append(label)
                    print(f"  FAILED ({proc.returncode}): {proc.stderr.strip()}", file=sys.stderr)
                    continue
                row = next(csv.DictReader(io.StringIO(proc.stdout)))
                row["dataset"] = dataset.name
                rows.append(row)
                print(f"  {float(row['median_s']):.4f} s  {float(row['gcups_median']):.3f} GCUPS",
                      file=sys.stderr)

    metadata["skipped"], metadata["failed"] = skipped, failed
    mismatches = []
    for dataset in args.dataset:
        sums = {(r["backend"], r["threads"]): r["score_checksum"] for r in rows if r["dataset"] == dataset.name}
        if len(set(sums.values())) > 1:
            mismatches.append({"dataset": dataset.name, "checksums": {f"{b}/t{t}": c for (b, t), c in sums.items()}})
    metadata["score_mismatches"] = mismatches
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if rows:
        fields = ["dataset"] + [k for k in rows[0] if k != "dataset"]
        with (args.out / "results.csv").open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(f"{len(rows)} runs, {len(skipped)} skipped, {len(failed)} failed -> {args.out}", file=sys.stderr)
    for m in mismatches:
        print(f"SCORE MISMATCH in {m['dataset']}: {m['checksums']}", file=sys.stderr)
    return 1 if failed or mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
