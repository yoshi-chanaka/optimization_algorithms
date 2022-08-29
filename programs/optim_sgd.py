# -*- coding: utf-8 -*-

import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation

from functions import function_saddle, function_convex
from draw_surface import draw

if __name__ == "__main__":

    func = function_convex
    num_iters = 150
    init_point = [-10., -10.]  # 初期値

    x = torch.tensor(init_point[0], requires_grad=True)
    y = torch.tensor(init_point[1], requires_grad=True)
    params = [x, y]

    optimizer = optim.SGD(params, lr=0.01)

    x_list, y_list, z_list = \
        [x.item()], [y.item()], [func(x, y).item()]

    for i in range(num_iters):
        optimizer.zero_grad()
        outputs = func(x, y)
        x_list.append(x.item())
        y_list.append(y.item())
        z_list.append(outputs.item())
        outputs.backward()
        optimizer.step()

    lower_cl, upper_cl = - 10 ** 50, 10 ** 50
    x_list, y_list, z_list = \
        np.clip(np.array(x_list), lower_cl, upper_cl), \
        np.clip(np.array(y_list), lower_cl, upper_cl), \
        np.clip(np.array(z_list), lower_cl, upper_cl)

    print('x\tz\n================')
    for x_, z_ in zip(x_list[-30:], z_list[-30:]):
        print('{:.6f}\t{:.6f}'.format(x_, z_))

    # 画像
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax = draw(func, ax)
    ax.plot(x_list, y_list, z_list, color='r', lw=3, label='SGD')
    plt.legend()
    save_path = '../figures/sgd.jpg'
    plt.savefig(save_path)
    plt.show()

    # 動画
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    theta = np.arange(1, len(x_list))

    def update(t):
        global x_list, y_list, z_list

        ax.cla() # ax をクリア
        xlim, ylim = [-10, 10], [-10, 10]
        X = np.arange(xlim[0], xlim[1], 1)
        Y = np.arange(ylim[0], ylim[1], 1)
        X, Y = np.meshgrid(X, Y)
        Z = func(X, Y)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0.1, alpha=0.7)
        ax.plot(x_list[:t], y_list[:t], z_list[:t], color='r', lw=5, label='SGD')
        ax.scatter([float(x_list[t])], [float(y_list[t])], [float(z_list[t])], s=100, c='r', edgecolors='k')
        ax.scatter(x_list[:t], y_list[:t], z_list[:t], c='r', s=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(40, 20)
        plt.legend(fontsize='large')

    anim = FuncAnimation(fig, update, frames=theta, interval=50)
    save_path = "../figures/sgd.gif"
    writergif = animation.PillowWriter(fps=30)
    anim.save(save_path, writer=writergif)
    plt.close()


"""
x       z
================
-0.885379       1.567791
-0.867671       1.505706
-0.850318       1.446080
-0.833311       1.388816
-0.816645       1.333818
-0.800312       1.280999
-0.784306       1.230272
-0.768620       1.181553
-0.753247       1.134763
-0.738182       1.089827
-0.723419       1.046670
-0.708950       1.005221
-0.694771       0.965415
-0.680876       0.927184
-0.667259       0.890468
-0.653913       0.855205
-0.640835       0.821339
-0.628018       0.788814
-0.615458       0.757577
-0.603149       0.727577
-0.591086       0.698765
-0.579264       0.671094
-0.567679       0.644519
-0.556325       0.618996
-0.545199       0.594483
-0.534295       0.570942
-0.523609       0.548333
-0.513137       0.526619
-0.502874       0.505765
-0.492817       0.485736
"""
