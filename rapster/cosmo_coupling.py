import numpy as np

def apply_cosmological_coupling(t, tBH_form, z, z_i, z_i_0, k_cosmo, binaries, mBH, sBH, pairs, m_avg):    
    if t > tBH_form and z <= z_i_0:
        factor = ((1 + z_i) / (1 + z))**k_cosmo

        if binaries.shape[0] > 1:
            binaries[:, 4] *= factor
            binaries[:, 5] *= factor
            # exact BBH semi-major axis scaling (both components grow equally):
            binaries[:, 2] *= factor**(-3)
            binaries[:, 6] = np.minimum(1.0, binaries[:, 6] * (factor**(-3)))
            binaries[:, 7] = np.minimum(1.0, binaries[:, 7] * (factor**(-3)))

        if mBH.size > 0:
            mBH *= factor
            sBH = np.minimum(1.0, sBH * (factor**(-3)))

        if pairs.shape[0] > 1:
            # BH mass fraction of the pair, using CURRENT masses
            # (must be computed before pairs[:, 1] is updated below):
            m_BH_current = pairs[:, 1]
            x = m_BH_current / (m_BH_current + m_avg)

            # exact BH-star semi-major axis scaling (only the BH mass grows):
            pairs[:, 0] *= ((1 - x) + x * factor) / factor**2
            pairs[:, 1] *= factor
            pairs[:, 2] = np.minimum(1.0, pairs[:, 2] * (factor**(-3)))

        return binaries, mBH, sBH, pairs, z

    return binaries, mBH, sBH, pairs, z_i