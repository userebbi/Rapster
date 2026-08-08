#python3 run_grid.py --npoints 5 --ncpu 4 --outdir /path/output

import argparse
import itertools
import subprocess
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

M_AVG = 0.586

def logspace(lo, hi, n):
    return np.logspace(np.log10(lo), np.log10(hi), n)

def build_grid(n):
    Z_vals   = logspace(1e-4, 1e-2,   n)
    Mcl_vals = logspace(1e4,  3e7,    n)
    r_vals   = logspace(0.1,  10,     n)
    R_vals   = logspace(1,    100000, n)
    return list(itertools.product(Z_vals, Mcl_vals, r_vals, R_vals))

"""
def build_grid(n):
    Z_vals   = logspace(1e-2, 1e-2,   n)
    Mcl_vals = logspace(1e5,  1e6,    n)
    r_vals   = logspace(1,  1,     n)
    R_vals   = logspace(8000,    8000, n)
    return list(itertools.product(Z_vals, Mcl_vals, r_vals, R_vals))
"""

def run_sim(args):
    Z, Mcl, r, R_pc, base_outdir, dry_run = args
    N         = int(round(Mcl / M_AVG))
    n_central = 0.53 * N / r**3
    tag       = f"Z{Z:.2e}_Mcl{Mcl:.2e}_r{r:.3f}_R{R_pc:.1f}"
    outdir    = os.path.join(base_outdir, "output_grid")
    sim_outdir = os.path.join(outdir, tag)
    cmd = [
        "python3", "-m", "rapster.run_cluster",
        "-Z",   f"{Z}",
        "-N",   f"{N}",
        "-r",   f"{r}",
        "-n",   f"{n_central:.4e}",
        "-R",   f"{R_pc}",
        "-P",   "0",
        "-Hi",  "0",
        "-BOi", "0",
        "-RF",  sim_outdir,
        "-MF",  f"mergers_{tag}",
        "-EF",  f"evolution_{tag}",
    ]
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return tag, 0
    os.makedirs(sim_outdir, exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    log_path = os.path.join(sim_outdir, f"log_{tag}.txt")
    with open(log_path, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)
    return tag, result.returncode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npoints", type=int, default=5)
    parser.add_argument("--ncpu",    type=int, default=4)
    parser.add_argument("--outdir",  type=str, default=".")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    grid  = build_grid(args.npoints)
    total = len(grid)
    print(f"Grid: {args.npoints} points/axis -> {total} simulations total")
    print(f"Parallelism: {args.ncpu} CPUs\n")
    jobs = [(Z, Mcl, r, R, args.outdir, args.dry_run) for Z, Mcl, r, R in grid]
    done   = 0
    failed = []
    with ProcessPoolExecutor(max_workers=args.ncpu) as pool:
        futures = {pool.submit(run_sim, j): j for j in jobs}
        for fut in as_completed(futures):
            tag, code = fut.result()
            done += 1
            status = "OK" if code == 0 else f"FAILED (code {code})"
            print(f"[{done}/{total}] {tag}  {status}")
            if code != 0:
                failed.append(tag)
    print(f"\nDone. {done - len(failed)}/{total} succeeded.")
    if failed:
        print(f"Failed runs ({len(failed)}):")
        for f in failed:
            print("  ", f)

if __name__ == "__main__":
    main()
