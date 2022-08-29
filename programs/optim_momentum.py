# -*- coding: utf-8 -*-

import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation

from functions import function_saddle, function_convex

if __name__ == "__main__":

    func = function_saddle
    function_const = 10.
    num_iters = 150
    init_point = [-18., 10**(-8)]  # 初期値

    algorithms = ['SGD', 'Momentum']
    optimizers = {}

    x = torch.tensor(init_point[0], requires_grad=True)
    y = torch.tensor(init_point[1], requires_grad=True)
    x_list, y_list, z_list = {}, {}, {}
    for alg in algorithms:
        x_list[alg], y_list[alg], z_list[alg] = \
            [x.item()], [y.item()], [func(x, y, const=function_const).item()]

    lower_cl, upper_cl = - 10 ** 50, 10 ** 50
    for alg in algorithms:
        x = torch.tensor(init_point[0], requires_grad=True)
        y = torch.tensor(init_point[1], requires_grad=True)
        params = [x, y] # 初期値

        optimizers['SGD'] = optim.SGD(params, lr=0.1)
        optimizers['Momentum'] = optim.SGD(params, lr=0.1, momentum=0.9)
        optimizer = optimizers[alg]
        for i in range(num_iters):
            optimizer.zero_grad()
            outputs = func(x, y, const=function_const)
            x_list[alg].append(x.item())
            y_list[alg].append(y.item())
            z_list[alg].append(outputs.item())
            outputs.backward()
            optimizer.step()

        x_list[alg], y_list[alg], z_list[alg] = \
            np.clip(np.array(x_list[alg]), lower_cl, upper_cl), \
            np.clip(np.array(y_list[alg]), lower_cl, upper_cl), \
            np.clip(np.array(z_list[alg]), lower_cl, upper_cl)

    # 画像
    colors_dict = {'SGD' : 'r', 'Momentum': 'orange'}
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    xlim, ylim = [-20, 20], [-20, 20]

    X = np.arange(xlim[0], xlim[1], 1)
    Y = np.arange(ylim[0], ylim[1], 1)
    X, Y = np.meshgrid(X, Y)
    Z = func(X, Y, const=function_const)
    zlim = [Z.min() - 1, Z.max() + 1]

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0.1, alpha=0.7)
    for alg in algorithms:
        # x, y, z = x_list[alg], y_list[alg], z_list[alg]
        ax.plot(x_list[alg], y_list[alg], z_list[alg], color=colors_dict[alg], lw=3, label=alg)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_zlim(zlim[0], zlim[1])
    plt.legend()
    save_path = '../figures/momentum.jpg'
    ax.view_init(40, 40)
    plt.savefig(save_path)
    plt.show()

    # 動画
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    theta = np.arange(1, len(x_list[list(x_list.keys())[0]]))

    def update(t):
        global x_list, y_list, z_list, xlim, ylim, zlim, algorithms, colors_dict

        ax.cla() # ax をクリア
        X = np.arange(xlim[0], xlim[1], 1)
        Y = np.arange(ylim[0], ylim[1], 1)
        X, Y = np.meshgrid(X, Y)
        Z = func(X, Y, const=function_const)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0.1, alpha=0.7)
        for alg in algorithms:
            ax.plot(x_list[alg][:t], y_list[alg][:t], z_list[alg][:t], color=colors_dict[alg], lw=4, label=alg)
            ax.scatter([float(x_list[alg][t])], [float(y_list[alg][t])], [float(z_list[alg][t])], s=100, c=colors_dict[alg], edgecolors='k')
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_zlim(zlim[0], zlim[1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(40, 40)
        plt.legend(fontsize='large')

    anim = FuncAnimation(fig, update, frames=theta, interval=50)
    save_path = "../figures/momentum.gif"
    writergif = animation.PillowWriter(fps=30)
    anim.save(save_path, writer=writergif)
    plt.close()
