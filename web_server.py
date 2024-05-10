# web_server.py

import uuid
import json

import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

from flask import Flask

app = Flask(__name__)
games = {}

response = {
    "uuid": "237a7e3e-0e6a-11ef-15ef-79a855cf1b09",
    "board": [
        "........",
        "........",
        "..ox.o..",
        "...xoo..",
        "...xoo..",
        "...x....",
        "........",
        "........"
    ],
    "player": "o",
    "available_actions": [
        "c2",
        "c4",
        "c5",
        "c6",
        "c7",
        "e3"
    ],
    "last_move": "d3"
}

player_to_num = {
    "x": -1,
    ".":  0,
    "o":  1
}

num_to_player = {
    -1: "x",
    +0: ".",
    +1: "o"
}

class StoredGame:
    def __init__(self, g, game_ai):
        self.g = g
        self.ai = game_ai

def response(uuid, board, player, actions, move):
    resp = {
        "uuid": uuid,
        "board": board,
        "player": player,
        "available_actions": actions,
        "last_move": move
    }
    return json.dumps(resp)

def get_position_desc(action, shape):
    letters = "abcdefgh"
    row, col = np.unravel_index(action, shape)
    return f"{letters[col]}{row+1}"

def get_moves(game, board):
    valid = game.getValidMoves(board, 1)
    if np.sum(valid) == 1 and valid[np.product(board.shape)] == 1:
        return ""
    moves = [get_position_desc(i, board.shape) for i in range(len(valid)) if valid[i]]
    return ", ".join(moves)

def build_x_board(b):
    return ["".join(s) for s in np.array([num_to_player[x] for x in b.reshape(np.prod(b.shape))]).reshape(b.shape).tolist()]

@app.route("/start/<model>/<player>")
def start(model, player):
    g = OthelloGame(8)
    net_player = NNPlayer(g, model_dir='./temp/', model_name='best.pth.tar').play
    u = str(uuid.uuid1())
    games[u] = StoredGame(g, net_player)
    b = g.getInitBoard()
    if player == "o":
        board = build_x_board(b)
        last_move = "xx"
    else:
        action = net_player(g.getCanonicalForm(b, 1))
        b, p = g.getNextState(b, 1, action)
        player = num_to_player[p]
        board = build_x_board(b)
        last_move = get_position_desc(action, b.shape)

    return response(u, board, player, get_moves(g, b), last_move)

@app.route("/get_action/<uuid>/<board>/<player>")
def get_action(uuid, board, player):
    b = np.array([player_to_num[x] for x in board]).reshape((8,8))
    p = player_to_num[player]
    stored_game = games[uuid]
    action = stored_game.ai(stored_game.g.getCanonicalForm(b, p))
    b, p = stored_game.g.getNextState(b, p, action)
    player = num_to_player[p]
    board = build_x_board(b)
    last_move = get_position_desc(action, b.shape)
    return response(uuid, board, player, get_moves(stored_game.g, b), last_move)

@app.route("/get_models")
def get_models():
    return ["nntp"]

# r = start("nnpt", "x")
# print(r)