import matplotlib.pyplot as plt
import numpy as np


class GridDataVisual:
    """
    Creates a 2d meshgrid that serves as testing data for a 2 variable
    classification model and provides functionality to plot decision boundary
    using contour plot.
    """
    def __init__(self, df, x, y, target):
        """
        :param df: pandas dataframe
        :param x: first or x variable
        :param y: second or y variable
        :param target: binary value, in this case {-1,1}
        """
        self.df = df
        self.x = x
        self.y = y
        self.pos = df[target.values > 0]
        self.neg = df[target.values < 0]
        self.x_grid = None
        self.y_grid = None
        self.xy_grid = self.meshgrid_2d_test_data()

    def meshgrid_2d_test_data(self):
        n_grid_points = 50
        x_min, x_max = self.df[self.x].min(), self.df[self.x].max()
        y_min, y_max = self.df[self.y].min(), self.df[self.y].max()
        self.x_grid, self.y_grid = np.meshgrid(
            np.linspace(x_min, x_max, n_grid_points),
            np.linspace(y_min, y_max, n_grid_points)
        )
        xy_grid = np.stack([self.x_grid.reshape(-1, 1).flatten(),
                            self.y_grid.reshape(-1, 1).flatten()], axis=1)
        return xy_grid

    def plot_contour(self, yhat, title, image_name= None):
        z = yhat.reshape(self.x_grid.shape[0], self.y_grid.shape[0])
        # fig, ax = plt.subplots(constrained_layout=True)
        # CS = ax2.contourf(X, Y, Z, 10, cmap=plt.cm.bone, origin=origin)
        fig, ax = plt.subplots(1, 1, figsize=(10, 9))
        cp = ax.contourf(self.x_grid, self.y_grid, z, cmap=plt.cm.bone)
        # ax.contourf(cp, colors='k')
        fig.colorbar(cp)
        ax.set_title(title, fontsize=10)
        plt.plot(self.pos[self.x], self.pos[self.y], 'r.', label='Class +1')
        plt.plot(self.neg[self.x], self.neg[self.y], 'g.', label='Class -1')
        plt.legend()
        # plt.show()
        if image_name:
            plt.savefig(image_name)

    @staticmethod
    def plt_show():
        plt.show()
