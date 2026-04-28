"""Diagnostic: does sustained MKL-BLAS load exhibit thermal throttling?

v7 brief Diagnostic 1. Tests whether per-iteration wall-clock on a fixed
CPU-heavy workload grows monotonically (consistent with thermal throttling
or sustained-load clock decay), replicating the v6 LSTM observation of
~1.7x per-epoch slowdown from epoch 4 onward.

If iteration-N time is >=1.5x iteration-1 time, H3 is confirmed.
If <1.2x, H3 is ruled out and the v6 slowdown must have a different cause.

Usage:
    python scripts/diag_thermal_cpu.py
"""
import time

import numpy as np

# N chosen to give ~5-10 s per iteration -- similar CPU profile to one
# LSTM epoch's dominant matmul workload. If tuning is needed per the
# Tuning Notes in the v7 brief:
#   - If iter 1 < 2 s, bump N up (e.g. to 10000)
#   - If iter 1 > 20 s, drop N down (e.g. to 6000)
N = 8000
ITERATIONS = 10
KILL_SWITCH_SECONDS = 600  # 10 min total wall-clock cap


def workload():
    a = np.random.randn(N, N).astype(np.float32)
    b = np.random.randn(N, N).astype(np.float32)
    return (a @ b).sum()


def main():
    print(f"CPU thermal-throttling diagnostic: {N}x{N} float32 matmul "
          f"x {ITERATIONS} iterations")
    print(f"{'Iter':>4} {'Time (s)':>10} {'Ratio vs iter 1':>18}")
    print("-" * 36)

    t0 = time.time()
    times = []
    for i in range(ITERATIONS):
        it_start = time.time()
        _ = workload()
        it_time = time.time() - it_start
        times.append(it_time)
        ratio = it_time / times[0]
        print(f"{i+1:>4} {it_time:>10.2f} {ratio:>17.2f}x", flush=True)

        if time.time() - t0 > KILL_SWITCH_SECONDS:
            print(f"\nKILL SWITCH: wall-clock exceeded {KILL_SWITCH_SECONDS}s, aborting")
            break

    total = time.time() - t0
    max_ratio = max(times) / times[0]
    last_ratio = times[-1] / times[0]

    print(f"\nTotal wall-clock: {total:.1f} s over {len(times)} iterations")
    print(f"Iter 1 time:   {times[0]:.2f} s")
    print(f"Iter {len(times)} time:  {times[-1]:.2f} s  (ratio {last_ratio:.2f}x)")
    print(f"Max ratio:     {max_ratio:.2f}x")

    if max_ratio >= 1.5:
        verdict = "H3 CONFIRMED -- thermal / sustained-load slowdown detected"
    elif max_ratio >= 1.2:
        verdict = "H3 PARTIAL -- some slowdown, but not matching v6 1.7x pattern"
    else:
        verdict = "H3 RULED OUT -- per-iteration time stable"
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
