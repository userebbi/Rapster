import numpy as np
def apply_cosmological_coupling(t, tBH_form, z, z_i, z_i_0, k_cosmo, binaries, mBH, sBH, pairs):
    """
    Applies cosmological coupling scaling to masses, semi-major axes, and spins.
    """
    if t > tBH_form and z <= z_i_0:
        factor = ((1 + z_i) / (1 + z))**k_cosmo
        
        if binaries.shape[0] > 1:
            binaries[:, 4] *= factor
            binaries[:, 5] *= factor
            binaries[:, 2] *= (4 - 3 * factor)
            binaries[:, 6] = np.minimum(1.0, binaries[:, 6] * (factor**(-3)))
            binaries[:, 7] = np.minimum(1.0, binaries[:, 7] * (factor**(-3)))

        if mBH.size > 0:
            mBH *= factor
            sBH = np.minimum(1.0, sBH * (factor**(-3)))

        if pairs.shape[0] > 1:
            pairs[:, 1] *= factor
            pairs[:, 0] *= (9/5 - (5/4) * factor)
            pairs[:, 2] = np.minimum(1.0, pairs[:, 2] * (factor**(-3)))

        # Alternativa per pairs (commentata):
        # if pairs.shape[0] > 1:
        #     m1 = pairs[:, 1]
        #     m2 = m_avg
        #     mu = m1 / (m1 + m2)
        #     pairs[:, 1] *= factor
        #     pairs[:, 0] *= 3 - mu - factor * (2 - mu)
        #     pairs[:, 2] = np.minimum(1.0, pairs[:, 2] * (factor**(-3)))

        return binaries, mBH, sBH, pairs, z

    return binaries, mBH, sBH, pairs, z_i