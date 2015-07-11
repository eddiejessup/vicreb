'''
Script to simulate a two particle-species variant of the Vicsek model.
In this variant, each species can either align, anti-align or not align
with its neighbours, with a distinct rule for each 4 combinations of species
'A' and species 'B'.
'''


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

        wraps_cur = ((r > L / 2.0).astype(np.int) -
                     (r < -L / 2.0).astype(np.int))
        wraps += wraps_cur
        r -= wraps_cur * L

        if out is not None:
            np.savez(os.path.join(out, 'dyn_{:010d}'.format(t)),
                     t=t, r=r, u=u, w=wraps)
        ums_a.append(utils.vector_mag(np.mean(u[As], axis=0)))
        ums_b.append(utils.vector_mag(np.mean(u[Bs], axis=0)))
        ums.append(utils.vector_mag(np.mean(u, axis=0)))
    return (np.mean(ums_a), scipy.stats.sem(ums_a),
            np.mean(ums_b), scipy.stats.sem(ums_b),
            np.mean(ums), scipy.stats.sem(ums),
            )


def vicreb_2D(delta_A_A, delta_A_B, delta_B_A, delta_B_B,
              seed, n, f_A, i_max, v, R_v, L, eta):
    n_A = int(round(f_A * n))

    np.random.seed(seed)

    r = np.random.uniform(-L / 2.0, L / 2.0, size=(n, 2))
    theta = np.random.uniform(-np.pi, np.pi, size=n)
    u = np.zeros_like(r)

    As = np.zeros([n], dtype=np.bool)
    As[:n_A] = True
    Bs = np.logical_not(As)

    for i in range(i_max):
        abssep = np.abs(r[:, np.newaxis] - r[np.newaxis, :])
        seps = np.minimum(abssep, L - abssep)
        within = utils.vector_mag_sq(seps) < R_v ** 2.0

        theta_old = theta.copy()
        for i_n in range(n):
            if As[i_n]:
                delta_A = delta_A_A
                delta_B = delta_A_B
            else:
                delta_A = delta_B_A
                delta_B = delta_B_B

            As_within = np.logical_and(within[i_n], As)
            Bs_within = np.logical_and(within[i_n], Bs)

            As_theta = np.sum(theta_old[As_within], axis=0)
            Bs_theta = np.sum(theta_old[Bs_within], axis=0)

            theta[i_n] += np.sign(As_theta) * delta_A
            theta[i_n] += np.sign(Bs_theta) * delta_B

        theta += np.random.normal(loc=0.0, scale=eta)

        u[:, 0] = np.cos(theta)
        u[:, 1] = np.sin(theta)

        r += v * u
        r[r > L / 2.0] -= L
        r[r < -L / 2.0] += L

        print(r[0], theta[0])


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

        wraps_cur = ((r > L / 2.0).astype(np.int) -
                     (r < -L / 2.0).astype(np.int))
        wraps += wraps_cur
        r -= wraps_cur * L

        if out is not None:
            np.savez(os.path.join(out, 'dyn_{:010d}'.format(t)),
                     t=t, r=r, u=u, w=wraps)
        ums.append(utils.vector_mag(np.mean(u, axis=0)))
    return np.mean(ums), scipy.stats.sem(ums)


def vicreb_rot(deltas, seed, dim, nA, nB, t_max, v, R_v, L, D_rot, out=None):
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
    ums, ums_A, ums_B = [], [], []
    for t in range(t_max):
        abssep = np.abs(r[:, np.newaxis] - r[np.newaxis, :])
        seps = np.minimum(abssep, L - abssep)
        withins = utils.vector_mag_sq(seps) < R_v ** 2.0

        u_o = u.copy()
        u[...] = 0.0
        for i_n in range(n):
            delta_A, delta_B = deltas[not As[i_n]]

            w_As = np.logical_and(withins[i_n], As)
            w_Bs = np.logical_and(withins[i_n], Bs)

            u_net_A = np.sum(u_o[w_As], axis=0)
            u_net_B = np.sum(u_o[w_Bs], axis=0)
            u[i_n] += utils.rotate(u_net_A, delta_A)[0]
            u[i_n] += utils.rotate(u_net_B, delta_B)[0]

            if np.all(u[i_n] == 0.0):
                u[i_n] = u_o[i_n]

        u = utils.vector_unit_nonull(u)
        u = utils.rot_diff(u, D_rot, 1.0)

        r += v * u

        wraps_cur = ((r > L / 2.0).astype(np.int) -
                     (r < -L / 2.0).astype(np.int))
        wraps += wraps_cur
        r -= wraps_cur * L

        if out is not None:
            np.savez(os.path.join(out, 'dyn_{:010d}'.format(t)),
                     t=t, r=r, u=u, w=wraps)
        ums_A.append(utils.vector_mag(np.mean(u[As], axis=0)))
        ums_B.append(utils.vector_mag(np.mean(u[Bs], axis=0)))
        ums.append(utils.vector_mag(np.mean(u, axis=0)))
    return (np.mean(ums_A), scipy.stats.sem(ums_A),
            np.mean(ums_B), scipy.stats.sem(ums_B),
            np.mean(ums), scipy.stats.sem(ums),
            )
