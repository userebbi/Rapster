import numpy as np
import pandas as pd
import os
_dir = os.path.dirname(os.path.abspath(__file__))
bbh_df = pd.read_csv(os.path.join(_dir, 'bbh_df.csv'))

def extract_primordial_binaries(mBH, sBH, gBH, binaries, pbf, R_sun, G_Newton, 
                                mBH_avg_ini, vBH_ini, tBH_form, z_i_0):
    """
    Extracts a percentage (pbf) of single BHs and forms primordial binaries (Channel 11)
    using a distribution from an external CSV file.
    """
    
    if mBH.size >= 2:
        num_to_extract = int(pbf * mBH.size)
        num_prim_bi = min(num_to_extract // 2, mBH.size // 2)  # cap: non può superare mBH.size
        
        if num_prim_bi > 0:
            ind_prim = np.random.choice(mBH.size, size=num_prim_bi * 2, replace=False)
            valid_pairs = []

            # Upload file and convert in R_sun
            bbh_df = pd.read_csv('bbh_df.csv')
            sma_samples_rsun = bbh_df['SemiMajorAxis_AU'].values * 215.032 * R_sun

            for i in range(0, len(ind_prim), 2):
                idx1, idx2 = ind_prim[i], ind_prim[i+1]
                m1, m2 = mBH[idx1], mBH[idx2]
                s1, s2 = sBH[idx1], sBH[idx2]
                g1, g2 = gBH[idx1], gBH[idx2]

                aBHb_min = 2 * R_sun
                aBHb_max = G_Newton * (m1 + m2 + mBH_avg_ini) * m1 * m2 / \
                           ((m1 + m2) * mBH_avg_ini * 2 * vBH_ini**2)

                if aBHb_max <= aBHb_min:
                    continue

                # Sample from distribution and clip to physical limits
                a_prim_candidate = np.random.choice(sma_samples_rsun)
                a_prim = np.clip(a_prim_candidate, aBHb_min, aBHb_max)

                new_bin = [[np.random.randint(0, int(1e9)), 11, a_prim, 0.0,
                            m1, m2, s1, s2, g1, g2, 0, 0, tBH_form, z_i_0, 0]]
                binaries = np.append(binaries, new_bin, axis=0)
                valid_pairs.append(idx1)
                valid_pairs.append(idx2)

            if valid_pairs:
                mBH = np.delete(mBH, valid_pairs)
                sBH = np.delete(sBH, valid_pairs)
                gBH = np.delete(gBH, valid_pairs)
                n_prim = len(valid_pairs) // 2
            else:
                n_prim = 0
        else:
            n_prim = 0

    else:
        n_prim = 0  # fix: n_prim non era definito se mBH.size < 2

    return mBH, sBH, gBH, binaries, n_prim