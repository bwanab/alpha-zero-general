import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = OthelloGame(8)

# all players
players = {
    "nnp": NNPlayer(g).play,
    "nnpt":NNPlayer(g, model_dir='./temp/', model_name='best.pth.tar').play,
    "nnpt0":NNPlayer(g, model_dir='./temp/', model_name='temp.pth.tar').play,
    "rp": RandomPlayer(g).play,
    "gp": GreedyOthelloPlayer(g).play,
    "hp": HumanOthelloPlayer(g).play,
    "rai": RAIPlayer(g).play,
    #"azop": AZOPlayer(g).play
}

import argparse
parser = argparse.ArgumentParser(
                prog = 'play',
                description = 'plays two othello players',
                epilog = 'Text at the bottom of help')

parser.add_argument("-g", "--games", default=10)
parser.add_argument("-1", "--player1", default="nnpt")
parser.add_argument("-2", "--player2", default="rai")
parser.add_argument("-v", "--verbose", action='store_true')
args = parser.parse_args()

arena = Arena.Arena(players[args.player1], players[args.player2], g, display=OthelloGame.display)

print(arena.playGames(int(args.games), verbose=args.verbose))
