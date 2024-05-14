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

response2 = {
    "uuid": "d1495e92-0fda-11ef-bf1a-3ab66b084fbf", 
    "board": [
        "........", 
        "........", 
        "........", 
        "...xo...", 
        "...ox...", 
        "........", 
        "........", 
        "........"
    ], 
    "player": "o", 
    "available_actions": "d3, c4, f5, e6", 
    "last_move": "xx"
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
        self.game = g
        self.ai = game_ai

def response(uuid, board, player, actions, move):
    resp = {
        "uuid": uuid,
        "board": board,
        "player": player,
        "available_actions": actions,
        "last_move": move
    }
    rval = json.dumps(resp)
    # print(rval)
    return rval

letters = "abcdefgh"

def get_position_desc(action, shape):
    row, col = np.unravel_index(action, shape)
    return f"{letters[col]}{row+1}"

def get_position(action_desc, shape):
    row = int(action_desc[1]) - 1
    col = letters.find(action_desc[0])
    return np.ravel_multi_index([row,col], shape)

def get_moves(game, board, player):
    valid = game.getValidMoves(board, player)
    if np.sum(valid) == 1 and valid[np.product(board.shape)] == 1:
        return []
    moves = [get_position_desc(i, board.shape) for i in range(len(valid)) if valid[i]]
    return moves

def get_probs(g, board, player):
    valid = g.game.getValidMoves(board, player)
    if np.sum(valid) == 1 and valid[np.product(board.shape)] == 1:
        return []
    probs = g.ai.probs(g.game.getCanonicalForm(board, player))
    moves = [(get_position_desc(i, board.shape), probs[i]) for i in range(len(valid)) if valid[i]]
    return moves


def build_x_board(b):
    return ["".join(s) for s in np.array([num_to_player[x] for x in b.reshape(np.prod(b.shape))]).reshape(b.shape).tolist()]

@app.route("/start/<model>/<player>")
def start(model, player):
    g = OthelloGame(8)
    net_player = NNPlayer(g, model_dir='./temp/', model_name='best.pth.tar')
    u = str(uuid.uuid1())
    games[u] = StoredGame(g, net_player)
    b = g.getInitBoard()
    if player == "o":
        board = build_x_board(b)
        last_move = "zz"
        p = 1
    else:
        action = net_player(g.getCanonicalForm(b, 1))
        b, p = g.getNextState(b, 1, action)
        player = num_to_player[p]
        board = build_x_board(b)
        last_move = get_position_desc(action, b.shape)

    return response(u, board, player, get_moves(g, b, p), last_move)

@app.route("/step_play/<uuid>/<board>/<player>/<action>")
def step_play(uuid, board, player, action):
    g = games[uuid]
    #
    # perform the external player's action
    external_player = player_to_num[player]
    b = np.array([player_to_num[x] for x in board]).reshape((8,8))
    next_action = get_position(action, (8,8))
    b, p = g.game.getNextState(b, external_player, next_action)
    #
    # do the internal player's response
    # repeat as long as p is not changed to external_player
    #
    n_valid_moves = 0   
    while n_valid_moves == 0:
        next_action = g.ai.play(g.game.getCanonicalForm(b, p))
        if next_action == 64:
            valid_moves = get_moves(g.game, b, -p)
            return response(uuid, build_x_board(b), num_to_player[-p], valid_moves, "zz")
        b, p = g.game.getNextState(b, p, next_action)
        valid_moves = get_moves(g.game, b, p)
        n_valid_moves = len(valid_moves)
        if n_valid_moves == 0:
            p = -p
            if get_moves(g.game, b, p) == []:   # game over
                return response(uuid, build_x_board(b), num_to_player[p], [], "zx")

    player = num_to_player[p]
    board = build_x_board(b)
    last_move = get_position_desc(next_action, b.shape)

    return response(uuid, board, player, valid_moves, last_move)

@app.route("/get_action/<uuid>/<board>/<player>")
def get_action(uuid, board, player):
    g = games[uuid]
    p = player_to_num[player]
    b = np.array([player_to_num[x] for x in board]).reshape((8,8))
    action = g.ai.play(g.game.getCanonicalForm(b, p))
    valid_moves = get_moves(g.game, b, p)
    probs = get_probs(g, b, p)
    print(probs)
    return response(uuid, board, player, valid_moves, get_position_desc(action, b.shape))

@app.route("/get_models")
def get_models():
    return ["nntp"]

#"""
r = start("nnpt", "o")
r = json.loads(r)
# step_play(r["uuid"], "".join(r["board"]), r["player"], "d3")
r = step_play(r["uuid"], ".xxxxxx.oxxoox.xoxoooxxxoooxoxxxoxoxoxxxoxxooxxxoxxxxxxx.xxxxx..", "o", "a1")
print(r)
#"""

