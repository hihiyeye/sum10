import sys

sys.path.append('..')

import time
import datetime
import cProfile
import numpy.random as random
from common import print_board
from board_solver import BoardSolverSimple
from board_gen import generate_board


def test(loop=1, seed=None):
    if seed is not None:
        random.seed(seed)
    for _ in range(int(loop)):
        run(4, 4)


def run(height, width):
    board = generate_board(height, width)
    # print_board(board)
    s = BoardSolverSimple(quick_solve=True)
    res = s.solve(board)

    # s = Solver(board)
    # res = s.solve()
    if not res:
        print('board data:')
        print_board(board)
        raise RuntimeError('board is solvable, but res is False.')


if __name__ == '__main__':
    cProfile.runctx('test(loop=1e2)', globals=globals(), locals=locals(), sort='time')

    time_beg = time.time()
    print(str(datetime.datetime.now()) + ' task begin...')
    for i in range(int(1e5)):
        run(6, 6)
        if time.time() - time_beg >= 5:
            print('{} loop for {} times.'.format(datetime.datetime.now(), i))
            time_beg = time.time()
    print(str(datetime.datetime.now()) + ' DONE!')
