__all__ = ['SolutionGraph', 'SolutionNode', 'BoardSolver']

import numpy as np

from board_graph import BoardGraph
from common.board import *
from common.utils import *


class SolutionNode:
    def __init__(self, board_key, nodes_mapping):
        self.key = board_key
        nodes_mapping[self.key] = self

        self.parents = set()
        self.children = dict()  # {node: path}
        self.result = False

    def add_child(self, child, path):
        if child not in self.children:
            self.children[child] = path
            child.parents.add(self)
            if child.result and not self.result:
                self.result = True

    def add_parent(self, parent):
        self.parents.add(parent)


SolutionGraph = SolutionNode


class BoardSolverSimple:
    def __init__(self, sum_=10, quick_solve=False):
        self._sum = sum_
        self._quick_solve = quick_solve

        self._solution_nodes_mapping = {}
        self.solution_graph = SolutionGraph(empty_board_key, self._solution_nodes_mapping)

    def solve(self, board):
        board_key = board_to_key(board)
        self._solve_core(board, pre_board_key=empty_board_key,
                         cur_board_key=board_key, path=[])
        root = self._solution_nodes_mapping[board_key]
        return self._backtrack_result(root)

    def _solve_core(self, board, pre_board_key, cur_board_key, path: list):
        pre_node = self._solution_nodes_mapping[pre_board_key]
        if cur_board_key in self._solution_nodes_mapping:
            cur_node = self._solution_nodes_mapping[cur_board_key]
            pre_node.add_child(cur_node, path)
            cur_node.add_parent(pre_node)
            return True

        cur_node = SolutionNode(cur_board_key, self._solution_nodes_mapping)
        pre_node.add_child(cur_node, path)
        cur_node.add_parent(pre_node)
        if not board.any():
            cur_node.result = True
            return True

        height, width = board.shape
        res = False
        for x in range(height):
            for y in range(width):
                if board[x, y] == 0:
                    continue
                res |= self._search_path(board, visit_flags=np.zeros_like(board),
                                         src_pos=(x, y), num=0, path=[], board_key=cur_board_key)

                if self._quick_solve and res:
                    cur_node.result = True
                    return True

        cur_node.result = res
        return res

    def _search_path(self, board: np.ndarray, visit_flags, src_pos: tuple, num: int, path: list, board_key):
        if board[src_pos] != 0:
            path = path + [src_pos]
            if num + board[src_pos] > self._sum:
                return False
            elif num + board[src_pos] == self._sum:
                board = board.copy()
                board[tuple(zip(*path))] = 0
                return self._solve_core(board, pre_board_key=board_key,
                                        cur_board_key=board_to_key(board), path=path)
        # board[] == 0 or num + board[] < sum
        x, y = src_pos
        next_pos = [(x + delta[0], y + delta[1]) for delta in move_directions]
        next_pos = [x for x in next_pos if is_pos_valid(x, visit_flags.shape) and not visit_flags[x]]
        # next_pos = sorted([p for p in next_pos if is_pos_valid(p, board.shape) and not visit_flags[p]],
        #                   key=lambda a: min(board[a], 1) * 100)

        visit_flags[src_pos] = 1
        res = False
        for pos in next_pos:
            res |= self._search_path(board, visit_flags, pos, num + board[src_pos], path, board_key)

        visit_flags[src_pos] = 0
        return res

    def _backtrack_result(self, node: SolutionNode):
        if not node.children:
            return node.result

        res = False
        for child in node.children:
            res |= self._backtrack_result(child)
        node.result = res
        return res


class BoardSolverNormal(BoardSolverSimple):
    def __init__(self, sum_=10, quick_solve=False):
        super(BoardSolverNormal, self).__init__(sum_=sum_, quick_solve=quick_solve)

    def _solve_core(self, board, pre_board_key, cur_board_key, path: list):
        pass

    def _search_path(self, board: np.ndarray, board_graph: BoardGraph,
                     src_pos: tuple, num: int, path: list, board_key: bytes):
        pass


class BoardSolver:
    """
    Solver(board, sum_=10)

    A class to solve sum 10 problem.

    params:
        board: numpy.ndarray, between 0~9.
        sum_: int, default is 10.

    examples:
        board = np.array([[1,1,1], [2,2,2], [3,3,3]])

        s = Solver(board, sum_=6)
        res = s.solve()
    """

    def __init__(self, board, sum_=10):
        self.board = np.clip(np.array(board), 0, 9)
        self._sum = sum_
        self._nodes_mapping = {}
        self.solution_graph = SolutionGraph(empty_board_key, self._nodes_mapping)

    def solve(self, root_board_key=empty_board_key):
        board_graph = BoardGraph(self.board)
        return self._solve_core(self.board, board_graph,
                                pre_board_key=root_board_key, cur_board_key=board_to_key(self.board), path=[])

    def _solve_core(self, board: np.ndarray, board_graph: BoardGraph,
                    pre_board_key: bytes, cur_board_key: bytes, path: list):
        pre_node = self._nodes_mapping[pre_board_key]
        cur_node = SolutionNode(cur_board_key, self._nodes_mapping)

        res = (not board_graph.valid_pos)
        for pos in board_graph.valid_pos:
            res |= self._search_path(board, board_graph, src_pos=pos, num=0, path=[], board_key=cur_board_key)

        pre_node.add_child(cur_node, path)
        return res

    def _search_path(self, board: np.ndarray, board_graph: BoardGraph,
                     src_pos: tuple, num: int, path: list, board_key: bytes):
        if src_pos in path or num + board[src_pos] > self._sum:
            return False

        path = path + [src_pos]
        if num + board[src_pos] < self._sum:
            res = False
            for dst_pos in board_graph.graph[src_pos]:
                res |= self._search_path(board, board_graph, src_pos=dst_pos, num=num + board[src_pos], path=path,
                                         board_key=board_key)
            return res
        else:
            new_board = board.copy()
            new_board[tuple(zip(*path))] = 0  # modify board values in path.
            new_board_key = board_to_key(new_board)
            if new_board_key in self._nodes_mapping:
                self._nodes_mapping[board_key].add_child(self._nodes_mapping[new_board_key], path)
                return True

            if not self._verify_path(board, path, visit_flags=np.zeros_like(board), src_pos=path[0], index=0):
                return False

            new_board_graph = board_graph.copy_and_update(new_board, path)  # don't use deepcopy to get new board graph.
            return self._solve_core(new_board, new_board_graph,
                                    pre_board_key=board_key, cur_board_key=new_board_key, path=path)

    def _verify_path(self, board, path, visit_flags, src_pos, index):
        if not path:
            return False

        if index + 1 >= len(path):
            return True

        visit_flags[src_pos] = 1
        next_pos = [(src_pos[0] + delta[0], src_pos[1] + delta[1]) for delta in move_directions]
        next_pos = sorted([x for x in next_pos if is_pos_valid(x, visit_flags.shape) and not visit_flags[x]],
                          key=lambda pos: abs(path[index + 1][0] - pos[0]) + abs(path[index + 1][1] - pos[1]))

        reachable = False
        for dst_pos in next_pos:
            if reachable:
                break

            if dst_pos == path[index + 1]:
                reachable |= self._verify_path(board, path, visit_flags, dst_pos, index + 1)
            elif board[src_pos] == 0:
                reachable |= self._verify_path(board, path, visit_flags, dst_pos, index)
            else:
                reachable = False

        visit_flags[src_pos] = 0
        return reachable
