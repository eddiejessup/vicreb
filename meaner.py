import numpy as np
import scipy.stats


def meaner(fnames):
    xs, ys, ys_err = [], [], []
    first = True
    for fname in args.fnames:
        x, y, y_err = np.loadtxt(fname, unpack=True)

        if first:
            x_use = x.copy()
            first = False
        if not (x.shape == x_use.shape and np.allclose(x, x_use)):
            print('{} invalid, skipping'.format(fname))
            continue

        ys.append(y)
        ys_err.append(y_err)
        xs.append(x)

    for i in range(len(xs) - 1):
        assert np.allclose(xs[i], xs[i + 1])
    x_mean = np.mean(xs, axis=0)
    y_mean = np.mean(ys, axis=0)
    y_mean_err = scipy.stats.sem(ys, axis=0)
    # y_mean_err = np.sqrt(np.sum(np.square(ys_err), axis=0))
    for x, y, y_err in zip(x_mean, y_mean, y_mean_err):
        print(x, y, y_err)
