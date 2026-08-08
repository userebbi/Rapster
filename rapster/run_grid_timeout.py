# nohup python3 run_grid_timeout.py --npoints 5 --ncpu 16 --outdir /path/to/outdir > nohup_grid.log 2>&1 &
import argparse
import itertools
import subprocess
import numpy as np
import os
from scipy.integrate import quad
from scipy.optimize import brentq
from concurrent.futures import ProcessPoolExecutor, as_completed

M_AVG = 0.586
HUBBLE_TIME_MYR = 13800.0
SIM_TIMEOUT_SEC = 900  # 15 minutes

H0_GYR = 67.7 / 977.8

def _lookback_gyr(z):
    integrand = lambda zp: 1.0 / ((1+zp) * np.sqrt(0.307*(1+zp)**3 + 0.693))
    return (1/H0_GYR) * quad(integrand, 0, z)[0]

def lookback_to_redshift(t_gyr):
    return brentq(lambda z: _lookback_gyr(z) - t_gyr, 1e-4, 30)

def logspace(lo, hi, n):
    return np.logspace(np.log10(lo), np.log10(hi), n)

def build_grid(n, n_z):
    Z_vals   = logspace(1e-4, 1e-2,   n)
    Mcl_vals = logspace(1e4,  3e7,    n)
    r_vals   = logspace(0.1,  10,     n)
    R_vals   = logspace(1,    100000, n)
    t_form_vals = np.linspace(11.0, 13.5, n_z)
    z_form_vals = np.array([lookback_to_redshift(t) for t in t_form_vals])
    return list(itertools.product(Z_vals, Mcl_vals, r_vals, R_vals, z_form_vals))

"""
def build_grid(n, n_z):
    Z_vals   = logspace(1e-2, 1e-2, n)
    Mcl_vals = logspace(1e5,  1e6,  n)
    r_vals   = logspace(1,    1,    n)
    R_vals   = logspace(8000, 8000, n)
    t_form_vals = np.linspace(11.0, 13.5, n_z)
    z_form_vals = np.array([lookback_to_redshift(t) for t in t_form_vals])
    return list(itertools.product(Z_vals, Mcl_vals, r_vals, R_vals, z_form_vals))
"""

def run_sim(args):
    Z, Mcl, r, R_pc, z_form, base_outdir, dry_run, rapster_dir = args
    N         = int(round(Mcl / M_AVG))
    n_central = 0.53 * N / r**3
    tag       = f"Z{Z:.2e}_Mcl{Mcl:.2e}_r{r:.3f}_R{R_pc:.1f}_zf{z_form:.3f}"
    outdir    = os.path.join(base_outdir, "output_grid_tris")
    sim_outdir = os.path.join(outdir, tag)
    cmd = [
        "python3", "-m", "rapster.run_cluster",
        "-Z",   f"{Z}",
        "-N",   f"{N}",
        "-r",   f"{r}",
        "-n",   f"{n_central:.4e}",
        "-R",   f"{R_pc}",
        "-z",   f"{z_form:.6f}",
        "-tM",  f"{HUBBLE_TIME_MYR}",
        "-P",   "0",
        "-Hi",  "0",
        "-BOi", "0",
        "-RF",  sim_outdir,
        "-MF",  f"mergers_{tag}",
        "-EF",  f"evolution_{tag}",
    ]
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return tag, Z, Mcl, r, R_pc, z_form, 0
    os.makedirs(sim_outdir, exist_ok=True)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=SIM_TIMEOUT_SEC, cwd=rapster_dir)
        stdout = result.stdout
        stderr = result.stderr
        returncode = result.returncode
    except subprocess.TimeoutExpired:
        stdout = "SIMULATION_TIMEOUT: exceeded 15 minutes, skipped.\n"
        stderr = ""
        returncode = -1
    log_path = os.path.join(sim_outdir, f"log_{tag}.txt")
    with open(log_path, "w") as f:
        f.write(stdout)
        if stderr:
            f.write("\n--- STDERR ---\n")
            f.write(stderr)
    return tag, Z, Mcl, r, R_pc, z_form, returncode


def merge_outputs(results, base_outdir):
    outdir = os.path.join(base_outdir, "output_grid_tris")
    os.makedirs(outdir, exist_ok=True)
    mergers_out   = os.path.join(outdir, "all_mergers.txt")
    evolution_out = os.path.join(outdir, "all_evolution.txt")

    merger_rows    = []
    evolution_rows = []
    sim_id = 0

    for tag, Z, Mcl, r, R_pc, z_form, code in sorted(results, key=lambda x: x[0]):
        if code != 0:
            continue
        sim_id += 1
        sim_outdir = os.path.join(outdir, tag)

        mf = os.path.join(sim_outdir, f"mergers_{tag}.txt")
        if os.path.exists(mf):
            with open(mf) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    merger_rows.append(f"{sim_id} {Z:.6e} {Mcl:.6e} {r:.6e} {R_pc:.6e} {z_form:.6f} {line}")

        ef = os.path.join(sim_outdir, f"evolution_{tag}.txt")
        if os.path.exists(ef):
            with open(ef) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    evolution_rows.append(f"{sim_id} {Z:.6e} {Mcl:.6e} {r:.6e} {R_pc:.6e} {z_form:.6f} {line}")

    with open(mergers_out, 'w') as f:
        f.write("# ID Z_grid Mcl_grid r_grid R_grid z_form_grid [original mergers columns...]\n")
        for row in merger_rows:
            f.write(row + "\n")

    with open(evolution_out, 'w') as f:
        f.write("# ID Z_grid Mcl_grid r_grid R_grid z_form_grid [original evolution columns...]\n")
        for row in evolution_rows:
            f.write(row + "\n")

    print(f"\nMerged files written:")
    print(f"  {mergers_out}  ({len(merger_rows)} rows)")
    print(f"  {evolution_out}  ({len(evolution_rows)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npoints",   type=int, default=5, help="Points per axis for Z, Mcl, r, R")
    parser.add_argument("--nz",        type=int, default=3, help="Points for z_form (linear in lookback time 11-13.5 Gyr)")
    parser.add_argument("--ncpu",      type=int, default=4)
    parser.add_argument("--outdir",    type=str, default=".", help="Base output directory")
    parser.add_argument("--rapsterdir",type=str, default=None, help="Working directory for rapster (default: parent of outdir)")
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args()

    # working directory for subprocess: where rapster/ package lives
    rapster_dir = args.rapsterdir if args.rapsterdir else os.path.dirname(os.path.abspath(args.outdir))

    grid  = build_grid(args.npoints, args.nz)
    total = len(grid)
    print(f"Grid: {args.npoints} points/axis x {args.nz} z_form points -> {total} simulations total")
    print(f"Parallelism: {args.ncpu} CPUs, timeout: {SIM_TIMEOUT_SEC//60} min/sim")
    print(f"Rapster dir: {rapster_dir}\n")

    jobs = [(Z, Mcl, r, R, z_form, args.outdir, args.dry_run, rapster_dir)
            for Z, Mcl, r, R, z_form in grid]
    results = []
    done      = 0
    failed    = []
    timed_out = []

    with ProcessPoolExecutor(max_workers=args.ncpu) as pool:
        futures = {pool.submit(run_sim, j): j for j in jobs}
        for fut in as_completed(futures):
            tag, Z, Mcl, r, R_pc, z_form, code = fut.result()
            results.append((tag, Z, Mcl, r, R_pc, z_form, code))
            done += 1
            if code == 0:
                status = "OK"
            elif code == -1:
                status = "TIMEOUT (>15 min)"
                timed_out.append(tag)
            else:
                status = f"FAILED (code {code})"
                failed.append(tag)
            print(f"[{done}/{total}] {tag}  {status}")

    print(f"\nDone. {done - len(failed) - len(timed_out)}/{total} succeeded.")
    if timed_out:
        print(f"Timed out ({len(timed_out)}):")
        for t in timed_out:
            print("  ", t)
    if failed:
        print(f"Failed ({len(failed)}):")
        for f in failed:
            print("  ", f)

    if not args.dry_run:
        merge_outputs(results, args.outdir)


if __name__ == "__main__":
    main()
