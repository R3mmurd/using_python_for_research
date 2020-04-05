import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

from sklearn import datasets


def distance(p1, p2):
    return np.sqrt(np.sum(np.power(p1 - p2, 2)))


def get_nearest_neighbors(p, ps, k=5):
    ds = np.zeros(ps.shape[0])

    for i in range(len(ds)):
        ds[i] = distance(p, ps[i])

    ind = np.argsort(ds)
    return ind[:k]


def majority(s):
    mode, count = ss.mstats.mode(s)
    return mode


def knn_predict(p, ps, outcomes, k=5):
    ind = get_nearest_neighbors(p, ps, k)
    return majority(outcomes[ind])


def generate_synthetic_data(n=50, num_classes=2):

    points = np.array(ss.norm(0, 1).rvs((n, 2)))
    outcomes = np.array(np.repeat(0, n))

    for i in range(1, num_classes):
        points = np.concatenate((points, ss.norm(i, 1).rvs((n, 2))), axis=0)
        outcomes = np.concatenate((outcomes, np.repeat(i, n)))
    return points, outcomes


def make_prediction_grid(predictors, outcomes, limits, h, k):
    """
    Classify each point on the prediction grid.
    """
    x_min, x_max, y_min, y_max = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys)

    prediction_grid = np.zeros(xx.shape, dtype=int)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = np.array([x, y])
            prediction_grid[j, i] = knn_predict(p, predictors, outcomes, k)

    return xx, yy, prediction_grid


def plot_prediction_grid (xx, yy, prediction_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap(
        ["hotpink", "lightskyblue", "yellowgreen"]
    )
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap=background_colormap, alpha=0.5)
    plt.scatter(
        predictors[:,0], predictors [:,1], c=outcomes,
        cmap=observation_colormap, s = 50
    )
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    plt.savefig(filename)


iris = datasets.load_iris()

predictors = iris.data[:, 0:2]
outcomes = iris.target

plt.plot(predictors[outcomes==0][:, 0], predictors[outcomes==0][:, 1], 'ro')
plt.plot(predictors[outcomes==1][:, 0], predictors[outcomes==0][:, 1], 'go')
plt.plot(predictors[outcomes==2][:, 0], predictors[outcomes==0][:, 1], 'bo')
plt.savefig('iris.pdf')
