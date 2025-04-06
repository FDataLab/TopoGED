# Extracted functions from Kiarash's DOS code, necessary for my reward

import numpy as np
from numpy import linalg as LA

from dos_utils.rescale_matrix import rescale_matrix
from dos_utils.moment_comp import moments_cheb_dos
from dos_utils.moment_filter import filter_jackson
from dos_utils.plot_cheb import plot_chebhist


def get_dos(L, Nmoment=50, nZ=100, outname="none", npts=20, maxsize=-1, compute_range=False, adj=False):
    if (not adj):
        if (compute_range):
            L, _ = rescale_matrix(L)
        else:
            L, _ = rescale_matrix(L, range=[0, 2])  # the range is specified by normalized Laplacian

    n = L.shape[0]
    c = moments_cheb_dos(L, n, N=Nmoment, nZ=nZ)[0]
    c = filter_jackson(c)

    if (outname == "none"):
        xm, yy = plot_chebhist((c,), npts=(npts + 1), pflag=False)
    else:
        if (maxsize != -1):
            xm, yy = plot_chebhist((c,), pflag=True, npts=(npts + 1), outname=outname, maxsize=maxsize)
        else:
            xm, yy = plot_chebhist((c,), pflag=True, npts=(npts + 1), outname=outname)

    return yy


def cosine_similarity(u, v):
    cos_sim = abs(np.dot(u, v) / LA.norm(u) / LA.norm(v))
    return cos_sim
