import numpy as np


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanOthelloPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        letters = "abcdefgh"
        valid = self.game.getValidMoves(board, 1)
        moves = []
        for i in range(len(valid)):
            if valid[i]:
                row, col = np.unravel_index(i, board.shape)
                moves.append(f"{letters[col]}{row+1}")
                # print("[", int(i/self.game.n), int(i%self.game.n), end="] ")
        print(", ".join(moves), end= ": ")
        while True:
            input_move = input()
            if input_move in moves:
                col = letters.find(input_move[0])
                row = int(input_move[1]) - 1
                a = np.ravel_multi_index((row, col), board.shape)
                break
            # input_a = input_move.split(" ")
            # if len(input_a) == 2:
            #     try:
            #         x,y = [int(i) for i in input_a]
            #         if ((0 <= x) and (x < self.game.n) and (0 <= y) and (y < self.game.n)) or \
            #                 ((x == self.game.n) and (y == 0)):
            #             a = self.game.n * x + y if x != -1 else self.game.n ** 2
            #             if valid[a]:
            #                 break
            #     except ValueError:
            #         # Input needs to be an integer
            #         'Invalid integer'
            else:
                print('Invalid move')
        return a


class GreedyOthelloPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]

import requests
class AZOPlayer():
    def __init__(self, game):
        self.game = game
        resp = requests.get("http://localhost:8080/start/AZO_big_5/o")
        js = resp.json()
        self.uuid = js['uuid']

    def xlate_pos(self, pos):
        lets = "abcdefgh"
        idx1 = lets.find(pos[0])
        if idx1 < 0:
            return self.game.n * self.game.n
        idx0 = int(pos[1]) - 1
        return idx0 * self.game.n + idx1


    def play(self, board):
        b = self.game.stringRepresentationReadable(board).lower().replace("-", ".")
        req = f"http://localhost:8080/get_action/{self.uuid}/{b}/o"
        resp = requests.get(req)
        js = resp.json()
        return self.xlate_pos(js['last_move'])
