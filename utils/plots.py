from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


class CustomPlots(object):

    def __init__(self):
        pass

    def plot_decision_regions_1(self, X, y, classifier,
                                test_idx=None, resolution=0.02):

        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # Plot decision surface
        x1_min = X[:, 0].min() - 1
        x1_max = X[:, 0].max() - 1
        x2_min = X[:, 1].min() - 1
        x2_max = X[:, 1].max() - 1

        (xx1, xx2) = np.meshgrid(
            np.arange(x1_min, x1_max, resolution),
            np.arange(x2_min, x2_max, resolution)
        )

        print(np.array(([xx1.ravel(), xx2.ravel()])))

        z = classifier.predict(
            np.array(([xx1.ravel(), xx2.ravel()])).T
        )
        print(z)
        z = z.reshape(xx1.shape)
        print(z)

        plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # plt.show()
        # plt.close()

        # Plot class samples
        for index, class_label in enumerate(np.unique(y)):
            plt.scatter(
                x=X[y == class_label, 0],
                y=X[y == class_label, 1],
                alpha=0.8,
                c=cmap(index),
                marker=markers[index],
                label=class_label
            )

        # plt.show()
