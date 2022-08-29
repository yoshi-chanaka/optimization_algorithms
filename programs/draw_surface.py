# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

def draw(function, ax, xlim=[-10, 10], ylim=[-10, 10], zlim='auto'):

    x = np.arange(xlim[0], xlim[1], 1)
    y = np.arange(ylim[0], ylim[1], 1)
    x, y = np.meshgrid(x, y)
    z = function(x, y)
    # ax.plot_wireframe(X, Y, Z, color='blue', linewidth=0.5)
    ax.plot_surface(
        x, y, z,
        rstride=1, cstride=1,
        cmap=plt.cm.coolwarm, linewidth=0.1, alpha=0.7
    )

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    if zlim == 'auto':
        pass
    else:
        ax.set_zlim(zlim[0], zlim[1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.view_init(40, 40)

    return ax


if __name__ == "__main__":

    from functions import function_saddle

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax = draw(function=function_saddle, ax=ax)
    save_path = '../figures/saddle_function.jpg'
    plt.savefig(save_path)
    plt.show()
