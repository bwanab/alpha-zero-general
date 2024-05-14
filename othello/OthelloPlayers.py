import numpy as np
from utils import *
from MCTS import MCTS
from othello.pytorch.NNet import NNetWrapper as NNet

class NNPlayer():
    def __init__(self, game, model_dir='./pretrained_models/othello/pytorch/', model_name='8x8_100checkpoints_best.pth.tar'):
        n1 = NNet(game)
        n1.load_checkpoint(model_dir,model_name)
        args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
        self.mcts1 = MCTS(game, n1, args1)

    def play(self, board):
        return np.argmax(self.mcts1.getActionProb(board, temp=0))
    
    def probs(self, board):
        return self.mcts1.getActionProb(board, temp=1.0)


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # a = np.random.randint(self.game.getActionSize())
        # valids = self.game.getValidMoves(board, 1)
        # while valids[a]!=1:
        #     a = np.random.randint(self.game.getActionSize())
        # return a
        moves = self.game.getAllLegalMoves(board, 1)
        if len(moves) == 0:
            return np.product(board.shape)
        return np.random.choice(moves)


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

from reversi_ai.reversi import GameHasEndedError
from reversi_ai.reversiai import ReversiAI

class RAIPlayer():
    def __init__(self, game):
        self.game = game
        self.rai_cell_map = {-1: 'w', 0: ' ', 1: 'b'}
        self.render_cell_map = {-1: 'x', 0: '.', 1: 'o'}

    def play(self, board):
        rai_board = self.build_rai_board(board)

        try:
            rai = ReversiAI()
            r_ai_player = self.rai_cell_map[1]
            coord = rai.get_next_move(rai_board, r_ai_player)
            if coord is None:
                return np.product(board.shape)
            x = coord.x
            y = coord.y
            rval = np.ravel_multi_index([x, y], board.shape)
        except GameHasEndedError:
            all_actions = self.game.getAllLegalMoves(board, 1)
            if len(all_actions) > 0:
                rval = np.random.choice(all_actions)
            else:
                rval = np.product(board.shape)
        return rval

    def build_rai_board(self, board):
        rai_board = []
        for x in range(8):
            rai_board.append([])
            for y in range(8):
                rai_board[x].append(self.rai_cell_map[board[x, y]])
        return rai_board
