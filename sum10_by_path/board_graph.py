__all__ = ['BoardGraph']

import itertools

import numpy as np

from common.utils import *


class UnionFind:
    """
    Union `0` grids, and record its `non-zero` neighboring grids.

    Any two neighboring grids can reach each other.
    """

    def __init__(self, board):
        self._parents = np.empty_like(board, dtype=object)
        self._neighbors = np.empty_like(board, dtype=object)
        self._roots = set(zip(*np.where(board == 0)))  # a root means a union.
        self._cnt = 0

        # init relations
        for pos in self._roots:
            self._parents[pos] = pos  # notice pos must be tuple.
            self._neighbors[pos] = set()
            self._cnt += 1

        height, width = board.shape
        for pos in self._roots.copy():
            next_pos = [(pos[0] + delta[0], pos[1] + delta[1]) for delta in move_directions]
            for i, j in next_pos:
                if not (0 <= i < height and 0 <= j < width):
                    continue

                neighbor_pos = (i, j)
                if board[neighbor_pos] != 0:
                    self._neighbors[pos].add(neighbor_pos)
                else:
                    self.union(pos, neighbor_pos)

    def __deepcopy__(self, memo=None):
        """
        It makes no sense, just practice. You can use copy.deepcopy in python standard lib instead.

        :return: a deep copy obj of self.
        """
        other = self.__new__(UnionFind)
        # elem in _parents is None or tuple.
        other._parents = self._parents.copy()
        # elem in _neighbor is None or set.
        other._neighbors = self._neighbors.copy()

        height, width = other._neighbors.shape
        # np.copy not do copy if elem is set, we must recopy it.
        for pos in itertools.product(range(height), range(width)):
            if other._neighbors[pos] is None:
                continue
            other._neighbors[pos] = other._neighbors[pos].copy()

        other._roots = self._roots.copy()
        other._cnt = self._cnt
        return other

    @property
    def union_cnt(self):
        return self._cnt

    @property
    def roots(self):
        return self._roots

    @property
    def neighbors(self):
        return self._neighbors

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return

        # remain root_x.
        self._parents[root_y] = root_x
        self._roots.discard(root_y)
        self._neighbors[root_x].update(self._neighbors[root_y])
        self._cnt -= 1

    def find(self, x):
        if self._parents[x] is None:
            return None

        if self._parents[x] != x:
            self._parents[x] = self.find(self._parents[x])
        return self._parents[x]

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)

    def update(self, board, path):
        for pos in path:
            self._parents[pos] = pos
            self._roots.add(pos)
            self._neighbors[pos] = set()

        height, width = board.shape
        for src_pos in path:
            for i, j in [(src_pos[0] + delta[0], src_pos[1] + delta[1]) for delta in move_directions]:
                if not (0 <= i < height and 0 <= j < width):
                    continue

                dst_pos = (i, j)
                if board[dst_pos] != 0 and dst_pos not in path:
                    self._neighbors[src_pos].add(dst_pos)
                else:
                    self.union(src_pos, dst_pos)

        # discard path in neighbors, must be after union.
        for root in self._roots:
            self._neighbors[root].difference_update(path)


class BoardGraph:
    def __init__(self, board: np.ndarray):
        # self.board = board.view()
        self._valid_pos = set(zip(*np.where(board != 0)))
        self._init_graph(board)
        self._uf = UnionFind(board)
        self._update_graph_by_uf()

    @property
    def graph(self):
        return self._graph

    @property
    def valid_pos(self):
        return self._valid_pos

    def copy_and_update(self, board: np.ndarray, path: list | tuple):
        if not path:
            return None

        # memo = {id(self._graph): self._graph}  # don't copy _graph, _graph will be reassigned below.
        # new_board_graph = deepcopy(self, memo)

        new_board_graph = self.__new__(BoardGraph)
        new_board_graph._valid_pos = self._valid_pos.copy()
        new_board_graph._uf = self._uf.__deepcopy__()

        new_board_graph._valid_pos.difference_update(path)

        new_board_graph._init_graph(board)
        new_board_graph._uf.update(board, path)
        new_board_graph._update_graph_by_uf()

        return new_board_graph

    def _init_graph(self, board):
        self._graph = np.full_like(board, np.nan, dtype=object)
        for pos in self._valid_pos:
            self._graph[pos] = set()

        height, width = board.shape
        # only search right and down.
        for src_pos in self._valid_pos:
            x, y = src_pos
            dst_pos = (x, y + 1)
            if y + 1 < width and board[dst_pos] != 0:
                self._graph[src_pos].add(dst_pos)
                self._graph[dst_pos].add(src_pos)

            dst_pos = (x + 1, y)
            if x + 1 < height and board[dst_pos] != 0:
                self._graph[src_pos].add(dst_pos)
                self._graph[dst_pos].add(src_pos)

    def _update_graph_by_uf(self):
        for root in self._uf.roots:
            for src_pos, dst_pos in itertools.combinations(self._uf.neighbors[root], 2):
                self._graph[src_pos].add(dst_pos)
                self._graph[dst_pos].add(src_pos)
