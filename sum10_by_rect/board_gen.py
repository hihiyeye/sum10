import random
import time

import numpy as np

from common.board import *
from common.utils import *


def generate_board(height, width):
    bg = BoardGenerator(height, width)
    return bg.generate()


class _RectNode:
    def __init__(self, no: int, nodes_mapping: list):
        if len(nodes_mapping) != no:
            raise ValueError()

        self.no = no
        self._nodes_mapping = nodes_mapping
        self._nodes_mapping.append(self)

        self.next = set()
        self.pre = set()

    def add_next(self, next_no):
        self.next.update(next_no)
        for _no in next_no:
            self._nodes_mapping[_no].pre.add(_no)

    def is_node_in_cycle(self, node, path: list):
        if node in path:
            return True

        for _next in node.next:
            if self.is_node_in_cycle(_next, path + [node]):
                return True
        return False


_RectGraph = _RectNode


class BoardGenerator:
    def __init__(self, height, width):
        self.height, self.width = height, width
        self._groups_nums = estimate_groups_nums(self.height, self.width)

        self._rect_no = 0
        self._rect_graph = None
        self._nodes_mapping = []

        self._board = None
        self._board_tags = None

    def _init_board(self):
        tune_groups_nums(self.height, self.width, self._groups_nums)

        self._nodes_mapping = []
        self._rect_no = 0
        self._rect_graph = _RectGraph(0, self._nodes_mapping)

        self._board = np.zeros(shape=(self.height, self.width), dtype=np.int8)
        self._board_tags = np.empty(shape=(self.height, self.width), dtype=object)
        for i in range(self.height):
            for j in range(self.width):
                self._board_tags[i, j] = []

    def generate(self, try_times=0, time_limit_ms=0):
        time_beg = time.time() * 1000
        loop = 0
        while True:
            if 0 < time_limit_ms < time.time() * 1000 - time_beg or 0 < try_times <= loop:
                break
            board = self.generate_once()
            if board is not None:
                return board
        return None

    def generate_once(self, return_invalid_board=False):
        self._init_board()
        if not self._generate_rect_with_fit_shapes(5, self._groups_nums[5], [(1, 5), (5, 1)]) \
                or not self._generate_rect_with_fit_shapes(4, self._groups_nums[4], [(1, 4), (4, 1), (2, 2)]) \
                or not self._generate_rect(3, self._groups_nums[3],
                                           fit_shapes=[(1, 3), (3, 1)], unfit_shapes=[(2, 2), (1, 4), (4, 1)]) \
                or not self._generate_rect(2, self._groups_nums[2],
                                           fit_shapes=[(1, 2), (2, 1)], unfit_shapes=[(1, 3), (3, 1)]):
            return self._board.copy() if return_invalid_board else None
        return self._board.copy()

    def _generate_rect_with_fit_shapes(self, addends_num, group_num, shapes):
        valid_pos = get_zero_pos(self._board)
        random.shuffle(valid_pos)

        for i, pos in enumerate(valid_pos):
            if group_num <= 0:
                break
            if self._board[pos] != 0:
                continue

            rects_to_be_selected = self._get_rects(pos, shapes)
            if not rects_to_be_selected:
                continue
            rect = random.choice(rects_to_be_selected)
            self._fill_rect(rect, get_addends_random(addends_num))
            group_num -= 1

        return group_num == 0

    def _generate_rect(self, addends_num, group_num, fit_shapes, unfit_shapes, fit_prob=0.75):
        valid_pos = get_zero_pos(self._board)
        random.shuffle(valid_pos)
        for i, pos in enumerate(valid_pos):
            if group_num <= 0:
                break
            if self._board[pos] != 0:
                continue

            rects_fit = self._get_rects(pos, fit_shapes) if fit_shapes else None
            rects_unfit = self._get_rects(pos, unfit_shapes, valid_num=addends_num) if unfit_shapes else None
            if not rects_fit and not rects_unfit:
                continue

            if not rects_fit:
                choose_from_fit = False
            elif not rects_unfit:
                choose_from_fit = True
            else:
                choose_from_fit = True if random.random() < fit_prob else False

            if choose_from_fit:
                rect = random.choice(rects_fit)
                self._fill_rect(rect, get_addends_random(addends_num))
            else:
                random.shuffle(rects_unfit)
                find_rect_flag = False
                for rect in rects_unfit:
                    if self._verify_rect(rect):
                        self._fill_rect(rect, get_addends_random(addends_num), fit=False)
                        find_rect_flag = True
                        break
                if not find_rect_flag:
                    if rects_fit:
                        rect = random.choice(rects_fit)
                        self._fill_rect(rect, get_addends_random(3))
                    else:
                        continue

            group_num -= 1

        if group_num == 0:
            return True

        valid_pos = get_zero_pos(self._board)
        random.shuffle(valid_pos)
        while 0 < group_num <= len(valid_pos):
            find_rect_flag = False
            for rect_board_pos in self._expand_rect(valid_pos[:], addends_num, []):
                rect = self._get_min_rect(rect_board_pos)
                if self._verify_rect(rect):
                    find_rect_flag = True
                    # self._fill_rect(rect, get_addends_random(addends_num), fit=False)
                    for pos, addend in zip(rect_board_pos, get_addends_random(addends_num)):
                        self._board[pos] = addend
                        valid_pos.remove(pos)
                    break

            if not find_rect_flag:
                return False
            group_num -= 1
        return group_num == 0

    @staticmethod
    def _get_min_rect(rect_pos):
        rect_pos_len = len(rect_pos)
        if rect_pos_len == 1:
            return rect_pos[0], rect_pos[0]
        elif rect_pos_len > 1:
            x_list, y_list = list(zip(*rect_pos))
            return (min(x_list), min(y_list)), (max(x_list), max(y_list))
        else:
            raise ValueError(f'rect_pos len is {rect_pos_len}.')

    @staticmethod
    def _remeasure_rect(rect, pos):
        return (min(rect[0][0], pos[0]), min(rect[0][1], pos[1])), \
               (max(rect[1][0], pos[0]), max(rect[1][1], pos[1]))

    def _expand_rect(self, valid_pos, required_pos_num: int, rect_pos: list):
        if required_pos_num == 0:
            yield rect_pos.copy()
            return
        if len(valid_pos) < required_pos_num:
            return

        if not rect_pos:
            for i in range(len(valid_pos)):
                valid_pos[i], valid_pos[-1] = valid_pos[-1], valid_pos[i]
                rect_pos.append(valid_pos[-1])
                yield from self._expand_rect(valid_pos[:-1], required_pos_num - 1, rect_pos)
                rect_pos.pop()

        else:
            sorted_valid_pos = sorted(valid_pos,
                                      key=lambda x: sum([rect_area(x, p) for p in rect_pos]))
            top_k_pos = sorted_valid_pos[:4]
            for i in range(len(top_k_pos)):
                pos = sorted_valid_pos.pop(-1 - i)
                rect_pos.append(pos)
                yield from self._expand_rect(sorted_valid_pos, required_pos_num - 1, rect_pos)
                rect_pos.pop()

    def _verify_rect(self, rect):
        # pre_rect_list = [abs(_no) for _no in rect_board.flatten() if _no < 0]
        # TODO
        return True

    def _fill_rect(self, rect, addends, fit=True):
        src, dst = rect
        if fit:
            addends = np.array(addends).reshape(dst[0] - src[0] + 1, dst[1] - src[1] + 1)
            self._board[src[0]:dst[0] + 1, src[1]:dst[1] + 1] = addends
        else:
            rect_pos = np.where(self._board[src[0]:dst[0] + 1, src[1]:dst[1] + 1] == 0)
            if len(rect_pos) == 1:
                # 取出来的rect是board的一行
                rect_pos = list(zip(rect_pos[0], [0] * len(rect_pos)))
            else:
                rect_pos = list(zip(*rect_pos))

            rect_pos_len, addends_len = len(rect_pos), len(addends)
            if rect_pos_len < addends_len:
                raise ValueError(f'addends num is more than needed. rect has {rect_pos_len} pos, '
                                 f'but addends num is {addends_len}.')
            if rect_pos_len > addends_len:
                random.shuffle(rect_pos)
                rect_pos = rect_pos[:addends_len]

            board_pos = tuple(((x + src[0], y + src[1]) for x, y in rect_pos))
            for pos, addend in zip(board_pos, addends):
                self._board_tags[pos].append(self._rect_no)
                self._board[pos] = addend

    def _get_rects(self, pos, shapes, valid_num=-1):
        rects = []
        x, y = pos
        for shape in shapes:
            h, w = shape
            # 记a，b，c，d分别为以(x, y)为原点的第1，2，3，4象限的矩形
            # h = 1时，有a = d, b = c
            # w = 1时，有a = b, c = d
            # 所以h>1或者w>1必须是对角的2个矩形，才能不会漏算，即ac,bd
            if h > 1:
                rects.append(((x - (h - 1), y - (w - 1)), (x, y)))  # 左上
                rects.append(((x, y), (x + h - 1, y + w - 1)))  # 右下
            if w > 1:
                rects.append(((x - (h - 1), y), (x, y + w - 1)))  # 右上
                rects.append(((x, y - (w - 1)), (x + h - 1, y)))  # 左下

        board_shape = self._board.shape
        filtered_rects = []
        for rect in rects:
            src, dst = rect
            if not is_pos_valid(src, board_shape) or not is_pos_valid(dst, board_shape):
                continue
            rect_board = self._board[src[0]:dst[0] + 1, src[1]:dst[1] + 1]
            if valid_num < 0:
                if not rect_board.any():
                    filtered_rects.append(rect)
            elif len(np.where(rect_board == 0)[0]) >= valid_num:
                filtered_rects.append(rect)
        return filtered_rects


if __name__ == '__main__':
    random.seed(0)
    for _ in range(100):
        print(_)
        bg = BoardGenerator(11, 13)
        _board = bg.generate_once(return_invalid_board=True)
        print_board(_board)
        # print_board(generate_board(10, 11))
        print('------------')
