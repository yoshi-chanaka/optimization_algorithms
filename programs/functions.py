# -*- coding: utf-8 -*-

import numpy as np

def function_convex(x, y, const=1.):
    """
    凸関数
    """
    return const * (x ** 2) + (y ** 2)


def function_saddle(x, y, const=1.):
    """
    鞍型
    """
    return const * (x ** 2) - (y ** 2)
