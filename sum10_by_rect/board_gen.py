import itertools
import time

import numpy as np
import numpy.random as random

from common.board import *
from common.utils import *


def generate_board(height, width):
    bg = BoardGenerator(height, width)
    return bg.generate()


class RectNode:
    def __init__(self, node_no: int, nodes_mapping: list):
        if len(nodes_mapping) != node_no:
            raise ValueError()

        self.node_no = node_no
        self._nodes_mapping = nodes_mapping
        self._nodes_mapping.append(self)
        self.parents = []

    def is_in_cycle(self):
        return self._is_node_in_cycle(self.node_no, [])

    def _is_node_in_cycle(self, node_no, path: list):
        # it only check the cycle with node in it.
        # maybe can use BFS instead.
        if node_no in path:
            return True

        new_path = path + [node_no]
        for parent_no in self._nodes_mapping[node_no].parents:
            if self._is_node_in_cycle(parent_no, new_path):
                return True
        return False


RectGraph = RectNode


def remove_rect_node_from_mapping(node_no, children_no, nodes_mapping):
    nodes_mapping.pop(node_no)
    # remove parent in rect_node's children.
    for _no in children_no:
        nodes_mapping[_no].parents.pop(node_no)


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
        """preparation before generate board."""
        tune_groups_nums(self.height, self.width, self._groups_nums)

        self._nodes_mapping = []
        self._rect_no = 0
        self._rect_graph = RectGraph(0, self._nodes_mapping)

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
        if not self._generate_rects_of_5_addends() \
                or not self._generate_rects_of_4_addends() \
                or not self._generate_rects_of_3_addends() \
                or not self._generate_rects_of_2_addends():
            return self._board.copy() if return_invalid_board else None
        return self._board.copy()

    def _generate_rects_of_5_addends(self):
        remaining_group_num = self._generate_rects(
            5, self._groups_nums[5], fit_shapes=[(1, 5), (5, 1)])
        return remaining_group_num == 0

    def _generate_rects_of_4_addends(self):
        remaining_group_num = self._generate_rects(
            4, self._groups_nums[4], fit_shapes=[(1, 4), (4, 1), (2, 2)])
        return remaining_group_num == 0

    def _generate_rects_of_3_addends(self):
        remaining_group_num = self._generate_rects(
            3, self._groups_nums[3], fit_shapes=[(1, 3), (3, 1)], unfit_shapes=[(1, 4), (4, 1), (2, 2)])
        return remaining_group_num == 0 or self._generate_rects_with_separate_pos(3, remaining_group_num)

    def _generate_rects_of_2_addends(self):
        remaining_group_num = self._generate_rects(
            2, self._groups_nums[2], fit_shapes=[(1, 2), (2, 1)], unfit_shapes=[(1, 3), (3, 1)])
        return remaining_group_num == 0 or self._generate_rects_with_separate_pos(2, remaining_group_num)

    @staticmethod
    def _get_smallest_rect(rect_pos):
        """Get the smallest rect contains rect_pos."""
        rect_pos_len = len(rect_pos)
        if rect_pos_len == 1:
            return rect_pos[0], rect_pos[0]
        elif rect_pos_len > 1:
            x_list, y_list = list(zip(*rect_pos))
            return (min(x_list), min(y_list)), (max(x_list), max(y_list))
        else:
            raise ValueError(f'rect_pos len is {rect_pos_len}.')

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

    def _verify_unfit_rect(self, rect, rect_pos):
        """
        Only verify rect in unfit shape.

        In other words, rect in fit shape don't need to be verified.

        :param rect: tuple, (src_pos, dst_pos).
        :param rect_pos: a list of pos that will be occupied in rect.
        :return: bool, True if rect has no dead lock when removing rects, otherwise False..
        """
        rect_no = self._rect_no + 1
        rect_node = RectNode(rect_no, self._nodes_mapping)
        children_no = []
        for pos in rect_pos:
            if self._board[pos] > 0:
                # 1. occupied pos means we must remove it before.
                rect_node.parents.append(self._board_tags[pos])
            else:
                # 2. otherwise we must remove this rect before.
                children_no += self._board_tags[pos]

        for child_no in children_no:
            self._nodes_mapping[child_no].parents.append(rect_no)

        if not rect_node.is_in_cycle():
            self._rect_no = rect_no
            return True
        else:
            # '-1' means the rect is the last one.
            remove_rect_node_from_mapping(-1, children_no, self._nodes_mapping)
            return False

    def _fill_rect(self, rect, addends, rect_pos=None):
        """
        Fill addends into rect in board.

        :param rect: tuple, (src_pos, dst_pos). where we fill addends into.
        :param addends: list of addends. addends num is equal to rect area or rect_pos size.
        :param rect_pos: default is `None`, all rect will be filled with addends.
                        Otherwise only pos in rect_pos will be filled with addends.
        :return: None.
        """
        if rect_pos is None:
            src, dst = rect
            rect_shape = (dst[0] - src[0] + 1, dst[1] - src[1] + 1)
            addends_arr = np.array(addends).reshape(rect_shape)
            self._board[src[0]:dst[0] + 1, src[1]:dst[1] + 1] = addends_arr
        else:
            rect_pos_len, addends_len = len(rect_pos), len(addends)
            if rect_pos_len != addends_len:
                raise ValueError(f'rect has {rect_pos_len} pos, '
                                 f'but addends num is {addends_len}.')

            board_pos = self._get_board_pos_in_rect(rect)
            addend_iter = iter(addends)
            for pos in board_pos:
                if self._board[pos] != 0:
                    continue
                if pos in rect_pos:
                    self._board[pos] = next(addend_iter)
                    self._board_tags[pos] = self._rect_no
                else:
                    self._board_tags[pos].append(self._rect_no)

    def _get_rect_board(self, *args):
        """
        Get the part of board in rect.

        :param args: 1 arg: rect, or 2 args: src, dst.
        :return: np.array, a board slice.
        """
        if len(args) == 1:
            src, dst = args[0]
        else:
            src, dst = args[0], args[1]
        return self._board[src[0]:dst[0] + 1, src[1]:dst[1] + 1]

    @staticmethod
    def _get_board_pos_in_rect(*args):
        """
        Get all pos in rect under board reference system.

        :param args: 1 arg: rect, or 2 args: src, dst.
        :return: list, in which a elem is a pos, like: (x,y).
        """
        if len(args) == 1:
            src, dst = args[0]
        else:
            src, dst = args[0], args[1]

        xx, yy = np.meshgrid(np.arange(src[0], dst[0] + 1), np.arange(src[1], dst[1] + 1))
        return list(zip(xx.ravel(), yy.ravel()))

    def _fit_rects_iter(self, src_pos, shapes: list):
        random.shuffle(shapes)
        for shape in shapes:
            dst_pos = (src_pos[0] + shape[0] - 1, src_pos[1] + shape[1] - 1)
            if is_pos_valid(dst_pos, self._board.shape) \
                    and not self._get_rect_board(src_pos, dst_pos).any():
                yield src_pos, dst_pos
        return

    def _unfit_rects_iter(self, src_pos, shapes: list, required_zero_num=-1):
        random.shuffle(shapes)
        for shape in shapes:
            dst_pos = (src_pos[0] + shape[0] - 1, src_pos[1] + shape[1] - 1)
            if not is_pos_valid(dst_pos, self._board.shape):
                continue
            rect_board = self._get_rect_board(src_pos, dst_pos)
            if required_zero_num < 0 or required_zero_num <= len(np.where(rect_board == 0)[0]):
                yield src_pos, dst_pos
        return

    def _get_rect_with_fit_shape(self, src_pos, shapes, fit=True):
        try:
            if fit:
                return next(self._fit_rects_iter(src_pos, shapes))
            else:
                return next(self._unfit_rects_iter(src_pos, shapes))
        except StopIteration:
            return None

    def _generate_rects_with_fit_shape(self, addends_num, group_num, shapes):
        valid_pos = get_zero_pos(self._board)
        random.shuffle(valid_pos)
        for i, pos in enumerate(valid_pos):
            if group_num <= 0:
                break
            if self._board[pos] != 0:
                continue

            rect = self._get_rect_with_fit_shape(pos, shapes)
            self._fill_rect(rect, get_addends_random(addends_num))
            group_num -= 1

        return group_num == 0

    def _generate_rects(self, addends_num, group_num, fit_shapes, unfit_shapes=None, fit_prob=0.75):
        valid_pos = get_zero_pos(self._board)
        random.shuffle(valid_pos)
        # 1. find rects in fit shape
        for i, pos in enumerate(valid_pos):
            if group_num <= 0:
                break
            if self._board[pos] != 0:
                continue

            if unfit_shapes:
                find_rect_flag = False
                choose_from_unfit_shapes = random.random() < (1.0 - fit_prob)
                if choose_from_unfit_shapes:
                    # loop for all unfit shapes rects
                    for unfit_rect in self._unfit_rects_iter(pos, unfit_shapes, addends_num):
                        valid_pos_in_rect = \
                            [p for p in self._get_board_pos_in_rect(unfit_rect) if self._board[p] == 0]
                        random.shuffle(valid_pos_in_rect)
                        # loop for all pos in selected unfit rect
                        for rect_pos in itertools.combinations(valid_pos_in_rect, addends_num):
                            # avoid rect_pos to consist a fit shape rect.
                            # such as: [a, b, c] use [a, b], [a, b] is a fit shape rect.
                            if self._get_smallest_rect(rect_pos) != unfit_rect:
                                continue
                            if self._verify_unfit_rect(unfit_rect, rect_pos):
                                self._fill_rect(unfit_rect, get_addends_random(addends_num), rect_pos)
                                find_rect_flag = True
                                break
                        if find_rect_flag:
                            break
                if not find_rect_flag:
                    rect = self._get_rect_with_fit_shape(pos, fit_shapes)
                    if not rect:
                        continue
                    self._fill_rect(rect, get_addends_random(addends_num))
            else:
                rect = self._get_rect_with_fit_shape(pos, fit_shapes)
                if not rect:
                    continue
                self._fill_rect(rect, get_addends_random(addends_num))

            group_num -= 1
        return group_num

    def _generate_rects_with_separate_pos(self, addends_num, group_num):
        """
        generate_rects_with_separate_pos is the last step for generating addends_num rect,
        so it returns a bool value.

        :param addends_num: int, num of addends.
        :param group_num: int, num of addends groups.
        :return: bool, if generating group_num rect return True, otherwise return False.
        """
        valid_pos = get_zero_pos(self._board)
        random.shuffle(valid_pos)

        while 0 < group_num <= (len(valid_pos) // addends_num):
            find_rect_flag = False
            for rect_pos in self._expand_rect(valid_pos, addends_num, []):
                rect = self._get_smallest_rect(rect_pos)
                if self._verify_unfit_rect(rect, rect_pos):
                    find_rect_flag = True
                    for pos, addend in zip(rect_pos, get_addends_random(addends_num)):
                        self._board[pos] = addend
                        valid_pos.remove(pos)
                    break

            if not find_rect_flag:
                return group_num == 0
            group_num -= 1
        return group_num == 0


if __name__ == '__main__':
    random.seed(234)
    for _ in range(100):
        print(_)
        # bg = BoardGenerator(11, 13)
        # _board = bg.generate_once(return_invalid_board=True)
        # print_board(_board)
        print_board(generate_board(10, 11))
        print('------------')
