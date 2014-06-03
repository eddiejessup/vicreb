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

def vicrot(delta, n, seed, dim, t_max, v, R_v, L, D_rot, out=None):
    np.random.seed(seed)

    r = np.random.uniform(-L / 2.0, L / 2.0, [n, dim])
    u = utils.sphere_pick(n=n, d=dim)

    r0 = r.copy()
    wraps = np.zeros_like(r, dtype=np.int)

    if out is not None:
        np.savez(os.path.join(out, 'static.npz'), L=L, r0=r0, v=v)
    ums = []
    for t in range(t_max):
        abssep = np.abs(r[:, np.newaxis] - r[np.newaxis, :])
        seps = np.minimum(abssep, L - abssep)
        withins = utils.vector_mag_sq(seps) < R_v ** 2.0

        u_o = u.copy()
        u[...] = 0.0
        for i_n in range(n):
            u_net = np.sum(u_o[withins[i_n]], axis=0)
            u[i_n] = utils.rotate(u_net, delta)

        u = utils.vector_unit_nonull(u)
        u = utils.rot_diff(u, D_rot, 1.0)

        r += v * u

        wraps_cur = (r > L / 2.0).astype(np.int) - (r < -L / 2.0).astype(np.int)
        wraps += wraps_cur
        r -= wraps_cur * L

        if out is not None:
            np.savez(os.path.join(out, 'dyn_{:010d}'.format(t)), t=t, r=r, u=u, w=wraps)
        ums.append(utils.vector_mag(np.mean(u, axis=0)))
    return np.mean(ums), scipy.stats.sem(ums)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        description='Simulate Vicsek particles with rotating trajectory')
    parser.add_argument('delta', type=float,
                        help='Angle by which to rotate trajectory')
    parser.add_argument('-n', type=int,
                        help='Number of particles')
    parser.add_argument('-o', '--out',
                        help='Output data directory')
    parser.add_argument('-s', '--seed', type=int,
                        help='Random number generator seed')
    parser.add_argument('-d', '--dim', type=int,
                        help='System dimensionality')
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

    print(*vicrot(args.delta, args.n, args.seed, args.dim, args.t, 
                args.v, args.R, args.L, args.D_rot, args.out))
