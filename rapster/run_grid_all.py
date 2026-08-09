"""
Parallel grid runner for Rapster with cosmological coupling.

Strategy:
  - N_base sets of physical initial conditions (Z, Mcl, r, R, z_form) are drawn
    via Latin Hypercube Sampling (space-filling, far better coverage than a
    coarse Cartesian product for the same number of runs).
  - Each physical set is simulated at EVERY value of k (fixed list), so the
    effect of the cosmological coupling can be isolated at fixed cluster.
  - A single global seed is used for every run (full reproducibility).

Total runs = N_base * len(K_VALUES).

Resume model (important):
  Rapster has no mid-run checkpoint/restart. A run is either COMPLETE or must be
  re-run from scratch. A run counts as COMPLETE iff its Rapster log.txt contains
  "END OF SIMULATION". With --resume, only COMPLETE runs are skipped; anything
  incomplete (never started, killed by timeout, crashed) is re-run.

  When a run times out or exits without completing, its directory is renamed to
  "<tag>.TIMEOUT" / "<tag>.FAILED" so the partial output is preserved for
  inspection but is never mistaken for a completed run.

Typical workflow:
  # first pass: short timeout, most runs finish, heavy ones get flagged .TIMEOUT
  python3 run_grid_all.py --nbase 70 --ncpu 30 --outdir /path/base

  # inspect which ones didn't finish
  ls /path/base/output_grid/ | grep TIMEOUT

  # second pass: re-run only the unfinished ones with a longer timeout
  python3 run_grid_all.py --nbase 70 --ncpu 30 --outdir /path/base \
          --resume --timeout 43200
"""
import argparse
import os
import shutil
import subprocess
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import qmc
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
M_AVG = 0.586
HUBBLE_TIME_MYR = 13800.0  # 1 Hubble time in Myr

# Marker written by Rapster to log.txt on normal completion.
DONE_MARKER = "END OF SIMULATION"

# Cosmological coupling values to scan (fixed list)
K_VALUES = [-3, -2, -1, 0, 1, 2, 3]

# Physical parameter ranges for LHS (min, max, log?); log=True -> sampled in log space
PARAM_RANGES = {
    "Z":      (1e-4, 1e-2, True),    # metallicity
    "Mcl":    (1e4,  3e7,  True),    # cluster mass [Msun]
    "r":      (0.1,  10.0, True),    # half-mass radius [pc]
    "R":      (1.0,  1e5,  True),    # galactocentric radius [pc]
    "t_form": (11.0, 13.5, False),   # formation lookback time [Gyr] -> converted to z_form
}

# ----------------------------------------------------------------------------
# Cosmology (flat LCDM, Planck18): lookback time <-> redshift
# ----------------------------------------------------------------------------
H0_GYR = 67.7 / 977.8  # H0 in 1/Gyr


def _lookback_gyr(z):
    integrand = lambda zp: 1.0 / ((1 + zp) * np.sqrt(0.307 * (1 + zp) ** 3 + 0.693))
    return (1 / H0_GYR) * quad(integrand, 0, z)[0]


def lookback_to_redshift(t_gyr):
    """Convert lookback time in Gyr to redshift (Planck18 flat LCDM)."""
    return brentq(lambda z: _lookback_gyr(z) - t_gyr, 1e-4, 30)


# ----------------------------------------------------------------------------
# Grid construction via Latin Hypercube Sampling
# ----------------------------------------------------------------------------
def build_physical_sets(n_base, lhs_seed):
    """Draw n_base physical initial-condition sets via LHS.
    Returns a list of dicts: {Z, Mcl, r, R, z_form}."""
    keys = list(PARAM_RANGES.keys())
    sampler = qmc.LatinHypercube(d=len(keys), seed=lhs_seed)
    unit = sampler.random(n=n_base)  # shape (n_base, d), each column in [0,1)

    sets = []
    for row in unit:
        vals = {}
        for j, key in enumerate(keys):
            lo, hi, is_log = PARAM_RANGES[key]
            u = row[j]
            if is_log:
                vals[key] = 10 ** (np.log10(lo) + u * (np.log10(hi) - np.log10(lo)))
            else:
                vals[key] = lo + u * (hi - lo)
        z_form = lookback_to_redshift(vals.pop("t_form"))
        vals["z_form"] = z_form
        sets.append(vals)
    return sets


def build_jobs(physical_sets):
    """Cartesian product of physical sets x K_VALUES -> full job list."""
    jobs = []
    for phys in physical_sets:
        for k in K_VALUES:
            job = dict(phys)
            job["k"] = k
            jobs.append(job)
    return jobs


def make_tag(job):
    """Deterministic identifier including k (needed by make_summary.py)."""
    return (f"k{job['k']:+.1f}_"
            f"Z{job['Z']:.2e}_Mcl{job['Mcl']:.2e}_"
            f"r{job['r']:.3f}_R{job['R']:.1f}_zf{job['z_form']:.3f}")


def is_complete(sim_outdir):
    """A run is complete iff its Rapster log.txt contains the DONE marker."""
    log_path = os.path.join(sim_outdir, "log.txt")
    if not os.path.exists(log_path):
        return False
    try:
        with open(log_path) as f:
            return DONE_MARKER in f.read()
    except OSError:
        return False


# ----------------------------------------------------------------------------
# Single simulation
# ----------------------------------------------------------------------------
def run_sim(args):
    job, base_outdir, seed, timeout_s, dry_run, resume = args

    Z, Mcl, r, R_pc = job["Z"], job["Mcl"], job["r"], job["R"]
    z_form, k = job["z_form"], job["k"]

    N = int(round(Mcl / M_AVG))
    n_central = 0.53 * N / r ** 3
    tag = make_tag(job)

    outdir = os.path.join(base_outdir, "output_grid")
    sim_outdir = os.path.join(outdir, tag)

    # --- resume: skip only if the run genuinely completed ---
    if resume and is_complete(sim_outdir):
        return tag, job, 0, "SKIPPED"

    cmd = [
        "python3", "-m", "rapster.run_cluster",
        "-Z",   f"{Z}",
        "-N",   f"{N}",
        "-r",   f"{r}",
        "-n",   f"{n_central:.4e}",
        "-R",   f"{R_pc}",
        "-z",   f"{z_form:.6f}",
        "-k",   f"{k}",              # cosmological coupling
        "-S",   f"{seed}",           # global fixed seed
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
        return tag, job, 0, "DRYRUN"

    # clean up any stale flagged dir from a previous attempt of this tag
    for suffix in (".TIMEOUT", ".FAILED"):
        stale = sim_outdir + suffix
        if os.path.isdir(stale):
            shutil.rmtree(stale, ignore_errors=True)

    os.makedirs(sim_outdir, exist_ok=True)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        code = result.returncode
        stdout, stderr = result.stdout, result.stderr
        status = "OK" if code == 0 else f"FAILED(code {code})"
    except subprocess.TimeoutExpired as e:
        code = -1
        stdout = e.stdout or ""
        stderr = (e.stderr or "") + f"\n--- TIMEOUT after {timeout_s}s ---\n"
        status = "TIMEOUT"

    # write the subprocess stdout/stderr (Rapster's own log.txt is separate)
    proc_log = os.path.join(sim_outdir, f"proc_{tag}.log")
    with open(proc_log, "w") as f:
        f.write(stdout if isinstance(stdout, str) else "")
        if stderr:
            f.write("\n--- STDERR ---\n")
            f.write(stderr if isinstance(stderr, str) else "")

    # Decide final status by Rapster's own completion marker, not just exit code.
    if status == "TIMEOUT":
        flagged = sim_outdir + ".TIMEOUT"
        shutil.rmtree(flagged, ignore_errors=True)
        os.rename(sim_outdir, flagged)
        return tag, job, code, "TIMEOUT"

    if not is_complete(sim_outdir):
        # process returned but Rapster did not reach END OF SIMULATION
        flagged = sim_outdir + ".FAILED"
        shutil.rmtree(flagged, ignore_errors=True)
        os.rename(sim_outdir, flagged)
        return tag, job, code if code != 0 else -2, "FAILED"

    return tag, job, 0, "OK"


# ----------------------------------------------------------------------------
# Merge per-run outputs into cumulative files (k included as a grid column)
# ----------------------------------------------------------------------------
def merge_outputs(results, base_outdir):
    outdir = os.path.join(base_outdir, "output_grid")
    mergers_out = os.path.join(outdir, "all_mergers.txt")
    evolution_out = os.path.join(outdir, "all_evolution.txt")

    grid_cols = "# ID k_cosmo Z_grid Mcl_grid r_grid R_grid z_form_grid [original columns...]\n"

    def prefix(sim_id, job):
        return (f"{sim_id} {job['k']:+.1f} {job['Z']:.6e} {job['Mcl']:.6e} "
                f"{job['r']:.6e} {job['R']:.6e} {job['z_form']:.6f}")

    n_mer = n_evo = 0
    with open(mergers_out, "w") as fm, open(evolution_out, "w") as fe:
        fm.write(grid_cols)
        fe.write(grid_cols)

        sim_id = 0
        for tag, job, code, status in sorted(results, key=lambda x: x[0]):
            if status not in ("OK", "SKIPPED"):
                continue
            sim_outdir = os.path.join(outdir, tag)
            if not is_complete(sim_outdir):
                continue
            sim_id += 1
            pfx = prefix(sim_id, job)

            mf = os.path.join(sim_outdir, f"mergers_{tag}.txt")
            if os.path.exists(mf):
                with open(mf) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        fm.write(f"{pfx} {line}\n")
                        n_mer += 1

            ef = os.path.join(sim_outdir, f"evolution_{tag}.txt")
            if os.path.exists(ef):
                with open(ef) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        fe.write(f"{pfx} {line}\n")
                        n_evo += 1

    print("\nMerged files written:")
    print(f"  {mergers_out}  ({n_mer} rows)")
    print(f"  {evolution_out}  ({n_evo} rows)")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nbase", type=int, default=70,
                        help="Number of physical initial-condition sets (LHS). "
                             "Total runs = nbase * %d." % len(K_VALUES))
    parser.add_argument("--ncpu", type=int, default=30)
    parser.add_argument("--outdir", type=str, default=".",
                        help="Base dir; outputs go to <outdir>/output_grid/")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Global fixed Rapster seed (reproducibility)")
    parser.add_argument("--lhs-seed", type=int, default=0,
                        help="Seed for the LHS sampler (which physical sets are drawn). "
                             "Keep FIXED across passes so --resume matches the same grid.")
    parser.add_argument("--timeout", type=float, default=900,
                        help="Per-run timeout in seconds (default 900 = 15 min)")
    parser.add_argument("--max-runs", type=int, default=2000,
                        help="Safety cap: refuse to launch more runs than this")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs whose log.txt shows completion; re-run the rest")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the interactive confirmation prompt")
    args = parser.parse_args()

    physical_sets = build_physical_sets(args.nbase, args.lhs_seed)
    jobs = build_jobs(physical_sets)
    total = len(jobs)

    print(f"LHS physical sets : {args.nbase}")
    print(f"k values          : {K_VALUES}")
    print(f"Total runs        : {total}  ({args.nbase} x {len(K_VALUES)})")
    print(f"Parallelism       : {args.ncpu} CPUs")
    print(f"Global seed       : {args.seed}   LHS seed: {args.lhs_seed}")
    print(f"Per-run timeout   : {args.timeout:.0f} s")
    print(f"Resume            : {args.resume}")
    print(f"Output base       : {os.path.abspath(args.outdir)}/output_grid/")

    if total > args.max_runs:
        raise SystemExit(
            f"\nABORT: {total} runs exceeds --max-runs={args.max_runs}. "
            f"Raise --max-runs deliberately if this is intended.")

    if not args.dry_run and not args.yes:
        reply = input(f"\nLaunch {total} runs on {args.ncpu} CPUs? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            raise SystemExit("Aborted by user.")

    job_args = [(job, args.outdir, args.seed, args.timeout, args.dry_run, args.resume)
                for job in jobs]

    results = []
    done = 0
    failed, timed_out, skipped = [], [], []

    with ProcessPoolExecutor(max_workers=args.ncpu) as pool:
        futures = {pool.submit(run_sim, ja): ja for ja in job_args}
        for fut in as_completed(futures):
            tag, job, code, status = fut.result()
            results.append((tag, job, code, status))
            done += 1
            print(f"[{done}/{total}] {tag}  {status}")
            if status == "TIMEOUT":
                timed_out.append(tag)
            elif status == "SKIPPED":
                skipped.append(tag)
            elif status not in ("OK", "DRYRUN"):
                failed.append(tag)

    n_ok = sum(1 for _, _, _, s in results if s == "OK")
    print(f"\nDone. {n_ok} OK, {len(skipped)} skipped, "
          f"{len(failed)} failed, {len(timed_out)} timed out  (of {total}).")
    if failed:
        print(f"Failed ({len(failed)}):");  [print("  ", t) for t in failed]
    if timed_out:
        print(f"Timed out ({len(timed_out)}) -> re-run with --resume --timeout <larger>:")
        [print("  ", t) for t in timed_out]

    if not args.dry_run:
        merge_outputs(results, args.outdir)


if __name__ == "__main__":
    main()