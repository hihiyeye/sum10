__all__ = ['move_directions',
           'is_pos_valid',
           'pos_dist',
           'rect_area',
           'get_zero_pos']

import numpy as np

move_directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def is_pos_valid(pos, board_shape):
    return 0 <= pos[0] < board_shape[0] and 0 <= pos[1] < board_shape[1]


def pos_dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def rect_area(a, b):
    return abs(a[0] - b[0]) * abs(a[1] - b[1])


def get_zero_pos(board):
    return list(zip(*np.where(board == 0)))
