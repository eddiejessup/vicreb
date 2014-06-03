'''
Script to simulate a two particle-species variant of the Vicsek model.
In this variant, each species can either align, anti-align or not align
with its neighbours, with a distinct rule for each 4 combinations of species
'A' and species 'B'.
'''

from __future__ import print_function
import argparse
import numpy as np
import scipy.stats
import utils
import os

def vicreb(alg, seed, dim, nA, nB, t_max, v, R_v, L, D_rot, out=None):
    n = nA + nB

    np.random.seed(seed)

    r = np.random.uniform(-L / 2.0, L / 2.0, [n, dim])
    u = utils.sphere_pick(n=n, d=dim)

    As = np.zeros([n], dtype=np.bool)
    As[:nA] = True
    Bs = np.logical_not(As)
    r0 = r.copy()
    wraps = np.zeros_like(r, dtype=np.int)

    if out is not None:
        np.savez(os.path.join(out, 'static.npz'), L=L, As=As, r0=r0, v=v)
    ums, ums_a, ums_b = [], [], []
    for t in range(t_max):
        abssep = np.abs(r[:, np.newaxis] - r[np.newaxis, :])
        seps = np.minimum(abssep, L - abssep)
        withins = utils.vector_mag_sq(seps) < R_v ** 2.0

        u_o = u.copy()
        u[...] = 0.0
        for i_n in range(n):
            Ac, Bc = alg[not As[i_n]]

            w_As = np.logical_and(withins[i_n], As)
            w_Bs = np.logical_and(withins[i_n], Bs)
            u[i_n] += Ac * np.sum(u_o[w_As], axis=0)
            u[i_n] += Bc * np.sum(u_o[w_Bs], axis=0)
            if np.all(u[i_n] == 0.0):
                u[i_n] = u_o[i_n]

        u = utils.vector_unit_nonull(u)
        u = utils.rot_diff(u, D_rot, 1.0)

        r += v * u

        wraps_cur = (r > L / 2.0).astype(np.int) - (r < -L / 2.0).astype(np.int)
        wraps += wraps_cur
        r -= wraps_cur * L

        if out is not None:
            np.savez(os.path.join(out, 'dyn_{:010d}'.format(t)), t=t, r=r, u=u, w=wraps)
        ums_a.append(utils.vector_mag(np.mean(u[As], axis=0)))
        ums_b.append(utils.vector_mag(np.mean(u[Bs], axis=0)))
        ums.append(utils.vector_mag(np.mean(u, axis=0)))
    return (np.mean(ums_a), scipy.stats.sem(ums_a),
            np.mean(ums_b), scipy.stats.sem(ums_b),
            np.mean(ums), scipy.stats.sem(ums),
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        description='Simulate Vicsek particles with variable alignment rules')
    parser.add_argument('alg', nargs=4,
                        help='Algorithm matrix, each entry of 4 one of [-1, 0, +1]')
    parser.add_argument('-o', '--out',
                        help='Output data directory')
    parser.add_argument('-s', '--seed', type=int,
                        help='Random number generator seed')
    parser.add_argument('-d', '--dim', type=int,
                        help='System dimensionality')
    parser.add_argument('-a', '--na', type=int,
                        help='Number of A particles')
    parser.add_argument('-b', '--nb', type=int,
                        help='Number of A particles')
    parser.add_argument('-t', type=int,
                        help='System rutime in iterations')
    parser.add_argument('-v', type=float,
                        help='Particle speed')
    parser.add_argument('-R', type=float,
                        help='Vicsek alignment radius')
    parser.add_argument('-L', type=float,
                        help='System period')
    parser.add_argument('-Dr', '--D_rot', type=float,
                        help='Rotational diffusion constant')
    args = parser.parse_args()

    alg = np.array([int(e) for e in args.alg]).reshape(2, 2)

    print(*vicreb(alg, args.seed, args.dim, args.na, args.nb, args.t, 
                args.v, args.R, args.L, args.D_rot, args.out))

    # for args.D_rot in np.arange(0.31, 1.01, 0.01):
    #     print(args.D_rot, *vicreb(alg, args.seed, args.dim, args.na, args.nb, args.t, 
    #         args.v, args.R, args.L, args.D_rot, args.out))
    # for f in np.arange(0.0, 1.01, 0.01):
    #     n = args.na + args.nb
    #     args.na = int(round(f * n))
    #     args.nb = n - args.na
    #     print(args.na, *vicreb(alg, args.seed, args.dim, args.na, args.nb, args.t, 
    #           args.v, args.R, args.L, args.D_rot, args.out))
    # for args.nb in range(100):
    #     print(args.nb, *vicreb(alg, args.seed, args.dim, args.na, args.nb, args.t, 
    #           args.v, args.R, args.L, args.D_rot, args.out))
