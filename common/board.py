__all__ = ['board_to_key',
           'empty_board_key',
           'print_board',
           'addends_of_10',
           'get_addends_random',
           'estimate_groups_nums',
           'tune_groups_nums']

import numpy as np
import numpy.random as random


def board_to_key(board: np.ndarray):
    return board.tobytes()


empty_board_key = b''


def print_board(board):
    print('{} x {}:'.format(*board.shape))
    for line in board:
        print(line)


def _pre_calc_addends():
    two_num_addends: list[list[tuple]]
    two_num_addends = [[] for _ in range(11)]
    for x in range(1, 10):
        for y in range(x, 10):
            if x + y > 10:
                break
            two_num_addends[x + y].append((x, y))

    _addends_of_10 = {2: two_num_addends[10],
                      3: [],
                      4: []}

    # 3个数, 4+4+4 > 10
    for i in range(1, 4):
        for a, b in two_num_addends[10 - i]:
            if i > a:
                continue
            _addends_of_10[3].append((i, a, b))

    # 4个数, 3*4 > 10
    for i in range(1, 3):
        for a, b in two_num_addends[i]:
            for x, y in two_num_addends[10 - i]:
                if b > x:
                    break
                _addends_of_10[4].append((a, b, x, y))

    # 5个数，只记录以下几种组合
    _addends_of_10[5] = [[1, 1, 2, 2, 4], [1, 2, 2, 2, 3], [2, 2, 2, 2, 2]]
    return _addends_of_10


addends_of_10 = _pre_calc_addends()


def get_addends_random(addends_num):
    if not 2 <= addends_num <= 5:
        raise ValueError('{} out of range [2, 5].'.format(addends_num))
    selected_index = random.choice(range(len(addends_of_10[addends_num])))
    addends = list(addends_of_10[addends_num][selected_index])
    random.shuffle(addends)
    return addends


_default_groups_percent = [0, 0, 0.50, 0.30, 0.15, 0.05]


def estimate_groups_nums(height: int, width: int, groups_percent=None):
    groups_nums = [0] * 6
    groups_percent = groups_percent or _default_groups_percent

    total_grid_num = height * width
    used_grid_num = 0
    for i in range(2, 6):
        groups_nums[i] = int(total_grid_num * groups_percent[i] / i)
        used_grid_num += i * groups_nums[i]

    remaining_grid_num = total_grid_num - used_grid_num
    if remaining_grid_num > 0:
        groups_nums[2] += remaining_grid_num // 2
        if remaining_grid_num & 1 == 1:
            groups_nums[2] -= 1
            groups_nums[3] += 1

    return groups_nums


def tune_groups_nums(height: int, width: int, groups_nums):
    if groups_nums[2] >= 3:
        if (groups_nums[3] == 0 and random.random() < 0.3) or random.random() < 0.2:
            groups_nums[3] += 2
            groups_nums[2] -= 3

    if groups_nums[2] >= 2:
        if (groups_nums[4] == 0 and random.random() < 0.2) or random.random() < 0.1:
            groups_nums[4] += 1
            groups_nums[2] -= 2
        elif groups_nums[4] > 0 and random.random() < 0.05:
            groups_nums[4] -= 1
            groups_nums[2] += 2

    if groups_nums[5] == 0 and height * width == 5 and random.random() < 0.15:
        groups_nums[5] = 1
        groups_nums[2] = 0
        groups_nums[3] = 0
    elif groups_nums[5] > 0 and random.random() < 0.05:
        groups_nums[5] -= 1
        groups_nums[2] += 1
        groups_nums[3] += 1
