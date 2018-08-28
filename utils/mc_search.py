import numpy as np
from utils.merge_game import *
from utils.logic import *
import itertools


# Originally from
# https://github.com/vpn1997/2048-Ai/blob/master/direct.py
# With modifications... Some logic seems off...
def direction(matrix):

    MAX_TILE = np.max(matrix)
    MAX_DEPTH = 3 if MAX_TILE < 1024 else 4 if MAX_TILE < 2048 else 5

    def search(matrix, depth, move=False):
        # if on leave node return the score or if the game is over
        l = game_state(matrix)

        if l == 'win' or l == 'lose':
            l = 1
        else:
            l = 0

        if depth == 0 or (move and l):
            return heuristic(matrix)

        alpha = heuristic(matrix)

        if move:
            for _, action in MERGE_FUNCTIONS.items():
                child = action(matrix)
                alpha = max(alpha, search(child, depth - 1))

        else:
            alpha = 0
            zeros = [(i, j) for i, j in itertools.product(range(4), range(4)) if matrix[i][j] == 0]
            for i, j in zeros:
                c1 = [[x for x in row] for row in matrix]
                c2 = [[x for x in row] for row in matrix]
                c1[i][j] = 2
                c2[i][j] = 4
                alpha += (.9 * search(c1, depth - 1, True) / len(zeros) +
                          .1 * search(c2, depth - 1, True) / len(zeros))
        return alpha

    def heuristic(matrix):
        def score(matrix):
            weight=[[pow(4,6),pow(4,5),pow(4,4),pow(4,3)],
                    [pow(4,5),pow(4,4),pow(4,3),pow(4,2)],
                    [pow(4,4),pow(4,3),pow(4,2),pow(4,1)],
                    [pow(4,3),pow(4,2),pow(4,1),pow(4,0)]]
            sco=0
            for i in range(0,4):
                for j in range(0,4):

                    sco=sco+int(weight[i][j])*int(matrix[i][j])
            return sco

        def penalty(matrix):

            pen=0
            li=[-1,1]
            for i in range(0,4):
                for j in range(0,4):
                    if (i - 1 >= 0):
                        pen += abs(matrix[i][j] - matrix[i - 1][j])
                    if (i + 1 < 4):
                        pen += abs(matrix[i][j] - matrix[i + 1][j])
                    if (j - 1 >= 0):
                        pen += abs(matrix[i][j] - matrix[i][j - 1])
                    if (j + 1 < 4):
                        pen += abs(matrix[i][j] - matrix[i][j + 1])

            pen2=0  #for not empty tiles

            # This is a stupid way of doing nonzeros...
            # Plus, I think it's the inverse...
            for i in range(0,4):
                for j in range(0,4):
                    if(matrix[i][j]):
                        pen2 += 1

            # This is not correct... Doesn't make sense...
            pen2 = 2*pen2 if MAX_TILE < 1024 else -80*pen2
            return pen-pen2

        return score(matrix)-penalty(matrix)

    results = []

    for direction, action in MERGE_FUNCTIONS.items():
        if matrix != action(matrix):
            result = direction, search(action(matrix), MAX_DEPTH)
            results.append(result)

    # This is quite arbitrary but to avoid error..
    if len(results) == 0:
        return 'left'
    return max(results, key = lambda x: x[1])[0]

