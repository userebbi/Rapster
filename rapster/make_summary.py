"""
Reads all per-simulation output files in output_grid/ and writes a summary file.

Usage:
    python3 make_summary.py --outdir /path/to/outdir [--k 0]
"""
import argparse
import os
import numpy as np

# --- column indices in evolution.txt (0-based, after stripping the # header) ---
# # seed t z dt m_avg Mcl rh R_gal v_gal t_rh t_rhBH n_star N_BH mBH_avg mBH_max ...
COL_T        = 1
COL_Z        = 2
COL_MAVG     = 4
COL_MCL      = 5
COL_RH       = 6
COL_RGAL     = 7
COL_NBH      = 13
COL_MBHAVG   = 14
COL_MBHMAX   = 15
COL_NME      = 26
COL_NBBH     = 27
COL_N3BB     = 35
COL_N2CAP    = 36
COL_NEXCH    = 40  # N_ex (BBH-BH exchanges)
COL_NTRIPLES = 58
COL_NZLK     = 59
COL_NTDEBHWD = 63
COL_NTDESTAR = 68

# --- column indices in mergers.txt (0-based) ---
# # seed ind channel a e m1 m2 chi1 chi2 g1 g2 ... Mcl0 rh0 Z zClForm Rgal0 Mcl rh Rgal
COL_M_CHANNEL = 2
COL_M_M1      = 5
COL_M_M2      = 6
COL_M_MREM    = 18
COL_M_ZMERGE  = 17


def parse_log(log_path):
    """Read log.txt and return exit reason."""
    reasons = ['BH SUBSYSTEM EVAPORATED', 'CLUSTER DISSOLVED',
               'CLUSTER REACHED GALAXY CENTER', 'REDSHIFT 0 REACHED',
               'END OF SIMULATION']
    if not os.path.exists(log_path):
        return 'UNKNOWN'
    with open(log_path) as f:
        content = f.read()
    for r in reasons:
        if r in content:
            return r
    return 'RUNNING/INCOMPLETE'


def last_row(filepath):
    """Return the last non-comment, non-empty line as a float array."""
    last = None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                last = line
    if last is None:
        return None
    return np.array(last.split(), dtype=float)


def merger_stats(filepath):
    """Return number of mergers per channel and max remnant mass."""
    if not os.path.exists(filepath):
        return {}
    channels = []
    mrem_max = 0.0
    z_merges = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            cols = line.split()
            try:
                channels.append(int(float(cols[COL_M_CHANNEL])))
                mrem_max = max(mrem_max, float(cols[COL_M_MREM]))
                z_merges.append(float(cols[COL_M_ZMERGE]))
            except Exception:
                continue
    stats = {
        'N_me_check': len(channels),
        'mrem_max': mrem_max,
        'z_merge_min': min(z_merges) if z_merges else float('nan'),
        'z_merge_max': max(z_merges) if z_merges else float('nan'),
        'N_channel_1': channels.count(1) + channels.count(-1),
        'N_channel_2': channels.count(2),
        'N_channel_3': channels.count(3) + channels.count(-3),
        'N_channel_4': channels.count(4),
        'N_channel_5': channels.count(5) + channels.count(-5),
        'N_channel_6': channels.count(6),
    }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='.')
    parser.add_argument('--k',     type=float, default=0.0, help='cosmological coupling index used')
    args = parser.parse_args()

    grid_dir = os.path.join(args.outdir, 'output_grid')
    summary_path = os.path.join(grid_dir, 'summary.txt')

    # collect all simulation subdirectories
    sims = sorted([d for d in os.listdir(grid_dir)
                   if os.path.isdir(os.path.join(grid_dir, d))])

    header = (
        "# ID k_cosmo Z_grid Mcl_grid r_grid R_grid z_form_grid "
        "exit_reason "
        "t_final z_final "
        "Mcl_final rh_final Rgal_final m_avg_final "
        "N_BH_final mBH_avg_final mBH_max_final "
        "N_me N_BBH N_3bb N_2cap N_ex N_triples N_ZLK N_tdeBHWD N_tdeBHstar "
        "mrem_max z_merge_min z_merge_max "
        "N_ch1 N_ch2 N_ch3 N_ch4 N_ch5 N_ch6\n"
    )

    rows = []
    sim_id = 0

    for sim in sims:
        sim_id += 1
        sim_dir = os.path.join(grid_dir, sim)

        # parse grid parameters from tag
        # format: Z{Z}_Mcl{Mcl}_r{r}_R{R}_zf{zf}
        try:
            parts = sim.split('_')
            Z_grid    = float(parts[0][1:])
            Mcl_grid  = float(parts[1][3:])
            r_grid    = float(parts[2][1:])
            R_grid    = float(parts[3][1:])
            zf_grid   = float(parts[4][2:])
        except Exception:
            print(f"Could not parse tag: {sim}, skipping.")
            continue

        log_path = os.path.join(sim_dir, 'log.txt')
        exit_reason = parse_log(log_path).replace(' ', '_')

        # find evolution and merger files
        evo_file = os.path.join(sim_dir, f"evolution_{sim}.txt")
        mer_file = os.path.join(sim_dir, f"mergers_{sim}.txt")

        evo = last_row(evo_file) if os.path.exists(evo_file) else None

        if evo is not None and len(evo) > COL_NTDESTAR:
            t_final    = evo[COL_T]
            z_final    = evo[COL_Z]
            Mcl_fin    = evo[COL_MCL]
            rh_fin     = evo[COL_RH]
            Rgal_fin   = evo[COL_RGAL]
            mavg_fin   = evo[COL_MAVG]
            N_BH_fin   = evo[COL_NBH]
            mBHavg_fin = evo[COL_MBHAVG]
            mBHmax_fin = evo[COL_MBHMAX]
            N_me       = evo[COL_NME]
            N_BBH      = evo[COL_NBBH]
            N_3bb      = evo[COL_N3BB]
            N_2cap     = evo[COL_N2CAP]
            N_ex       = evo[COL_NEXCH]
            N_tri      = evo[COL_NTRIPLES]
            N_ZLK      = evo[COL_NZLK]
            N_tdeWD    = evo[COL_NTDEBHWD]
            N_tdestar  = evo[COL_NTDESTAR]
        else:
            t_final = z_final = Mcl_fin = rh_fin = Rgal_fin = mavg_fin = float('nan')
            N_BH_fin = mBHavg_fin = mBHmax_fin = float('nan')
            N_me = N_BBH = N_3bb = N_2cap = N_ex = N_tri = N_ZLK = float('nan')
            N_tdeWD = N_tdestar = float('nan')

        mstats = merger_stats(mer_file)
        mrem_max    = mstats.get('mrem_max', float('nan'))
        z_merge_min = mstats.get('z_merge_min', float('nan'))
        z_merge_max = mstats.get('z_merge_max', float('nan'))
        N_ch1 = mstats.get('N_channel_1', 0)
        N_ch2 = mstats.get('N_channel_2', 0)
        N_ch3 = mstats.get('N_channel_3', 0)
        N_ch4 = mstats.get('N_channel_4', 0)
        N_ch5 = mstats.get('N_channel_5', 0)
        N_ch6 = mstats.get('N_channel_6', 0)

        row = (
            f"{sim_id} {args.k:.4f} {Z_grid:.6e} {Mcl_grid:.6e} {r_grid:.6e} {R_grid:.6e} {zf_grid:.6f} "
            f"{exit_reason} "
            f"{t_final:.4f} {z_final:.6f} "
            f"{Mcl_fin:.4e} {rh_fin:.4f} {Rgal_fin:.4e} {mavg_fin:.4f} "
            f"{N_BH_fin:.0f} {mBHavg_fin:.4f} {mBHmax_fin:.4f} "
            f"{N_me:.0f} {N_BBH:.0f} {N_3bb:.0f} {N_2cap:.0f} {N_ex:.0f} {N_tri:.0f} {N_ZLK:.0f} {N_tdeWD:.0f} {N_tdestar:.0f} "
            f"{mrem_max:.4f} {z_merge_min:.4f} {z_merge_max:.4f} "
            f"{N_ch1} {N_ch2} {N_ch3} {N_ch4} {N_ch5} {N_ch6}"
        )
        rows.append(row)

    with open(summary_path, 'w') as f:
        f.write(header)
        for row in rows:
            f.write(row + '\n')

    print(f"Summary written: {summary_path}  ({len(rows)} simulations)")


if __name__ == '__main__':
    main()
