import functools
import itertools

import numpy as np
import numpy.random as random

from common.board import *
from common.utils import *


def generate_board(height, width):
    bg = BoardGenerator(height, width)
    return bg.generate()


class BoardGenerator:
    def __init__(self, height, width):
        self.height, self.width = height, width

    def _next_pos(self, pos, board=None):
        x, y = pos
        next_pos = []
        check_board = board is not None
        if 0 <= x < self.height - 1 and 0 <= y < self.width and check_board and board[x + 1, y] == 0:
            next_pos.append((x + 1, y))
        if 0 <= x < self.height and 0 <= y < self.width - 1 and check_board and board[x, y + 1] == 0:
            next_pos.append((x, y + 1))
        if 1 <= x < self.height and 0 <= y < self.width and check_board and board[x - 1, y] == 0:
            next_pos.append((x - 1, y))
        if 0 <= x < self.height and 1 <= y < self.width and check_board and board[x, y - 1] == 0:
            next_pos.append((x, y - 1))
        return next_pos

    def _expand_path(self, board, pos, num: int, path: list):
        path.append(pos)
        if num <= 1:
            return True

        next_pos = [p for p in self._next_pos(pos, board=board) if p not in path]
        random.shuffle(next_pos)
        for next_p in next_pos:
            if self._expand_path(board, next_p, num - 1, path):
                return True
        path.pop()
        return False

    def generate(self):
        _board = np.zeros(shape=(self.height, self.width), dtype=np.int8)
        _groups_nums = tuple(estimate_groups_nums(self.height, self.width))

        valid_pos = list(itertools.product(range(self.height), range(self.width)))
        random.shuffle(valid_pos)

        for i, group_num in enumerate(_groups_nums[2:][::-1]):
            addends_num = len(_groups_nums) - 1 - i
            has_connected_path = True
            for _ in range(group_num):
                path = []
                # 1.先找能完整安排addends_num个数的path，必须连续
                if has_connected_path:
                    pos = valid_pos.pop()
                    while _board[pos] != 0 and valid_pos:
                        pos = valid_pos.pop()

                    if _board[pos] == 0 and not self._expand_path(_board, pos, addends_num, path):
                        has_connected_path = False

                # 2.如果都不行，先随机找到一个空位，再找其最近的空位，保证不会多个path之间互相锁死
                # 不能和上面的if合并，这里需要处理if中has_connected_path变为false之后的逻辑
                if not has_connected_path:
                    unused_pos = list(zip(*np.where(_board == 0)))
                    random.shuffle(unused_pos)

                    path = [unused_pos.pop()]
                    for _ in range(addends_num - 1):
                        min_dis_unused_pos = min(unused_pos,
                                                 key=functools.partial(pos_dist, path[-1]))
                        path.append(min_dis_unused_pos)
                        unused_pos.remove(min_dis_unused_pos)

                selected_index = random.choice(range(len(addends_of_10[addends_num])))
                addends = list(addends_of_10[addends_num][selected_index])
                random.shuffle(addends)
                for pos, addend in zip(path, addends):
                    if _board[pos] != 0:
                        raise RuntimeError('pos {} is occupied.'.format(pos))
                    _board[pos] = addend
        if not _board.all():
            raise RuntimeError('board has value 0.')
        return _board


if __name__ == '__main__':
    for _ in range(100000):
        print(_)
        print_board(generate_board(10, 11))
        print('------------')
