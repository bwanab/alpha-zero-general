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
        if np.sum(valid) == 1 and valid[np.product(board.shape)] == 1:
            return np.product(board.shape)
        moves = []
        for i in range(len(valid)):
            if valid[i]:
                row, col = np.unravel_index(i, board.shape)
                moves.append(f"{letters[col]}{row+1}")
        print(", ".join(moves), end= ": ")
        while True:
            input_move = input()
            if input_move in moves:
                col = letters.find(input_move[0])
                row = int(input_move[1]) - 1
                a = np.ravel_multi_index((row, col), board.shape)
                break
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
