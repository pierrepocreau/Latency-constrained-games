from LC_seesaw.seesaw import Seesaw
import networkx as nx
import numpy as np
from game import Game
import warnings

warnings.filterwarnings('ignore', message='Objective contains too many subexpressions')
warnings.filterwarnings('ignore', message='Constraint .* contains too many subexpressions')

num_players = 3
list_num_in = [2,2,2]
list_num_out = [2,2,2]

#Example paper:
correlators={"xyz": np.array([[[ 0.438,  0.61 ], [ 0.52,  -0.466]], [[ 0.58,  -0.502], [-0.724, -0.22 ]]])}

funcs_utility_player = lambda out_tuple, in_tuple: correlators["xyz"][in_tuple] * (-1)**(np.sum(out_tuple))
func_in_prior = lambda in_tuple: 1/8

game = Game(num_players, list_num_in, list_num_out, funcs_utility_player=[funcs_utility_player]*3, func_in_prior=func_in_prior)
classical_value = game.opt_classical()[0]
upperBound = game.compute_NPA(level=2, Nash=False, verbose=False, warmStart=False, solver="MOSEK")    

""" Algebraic """
algebraic = np.sum(np.abs(correlators["xyz"]))
print(f"Maximum algebraic value {algebraic/8}")

""" No communication """
print("\n Graph v1  v2  v3")
network = nx.Graph()
network.add_node(0)
network.add_node(1)
network.add_node(2)
seesaw = Seesaw(game, [2, 2, 2], 1, network)
q_value, strategy = seesaw.run_optimization_multiple_starts(num_random_starts=1, verbose=False)
q_upper = game.compute_NPA(level=2, Nash=False, verbose=False, warmStart=False, solver="MOSEK")    
c_value = game.opt_classical()[0]
print(f"classical value {c_value}, quantum lower bound {q_value}, NPA upper bound {q_upper}")

""" v1 - v2   v3 """
#Classical
print("\nGraph: v1 - v2  v3")
game_fwd = game.reduce_to_foward(0, 1)
c_value = game_fwd.opt_classical()[0]

# Quantum
network = nx.Graph()
network.add_edge(0, 1)
network.add_node(2)
seesaw = Seesaw(game, [2, 2, 2], 2, network)
q_value, strat = seesaw.run_optimization_multiple_starts(num_random_starts=1, verbose=False)

"""Merge(v1  v2)  v3"""
#NPA upper bound
game_merge = game.merge(0, 1)
  
q_upper = game_merge.compute_NPA(level=2, Nash=False, verbose=False,
                                                       warmStart=False, solver="MOSEK")
print(f"classical value: {c_value}, quantum value: {q_value}, NPA upper bound (merged) {q_upper}")

""" v1  v2 - v3 """
#Classical
print("\nGraph: v1  v2 - v3")
game_fwd = game.reduce_to_foward(1, 2)
value = game_fwd.opt_classical()[0]

# Quantum
network = nx.Graph()
network.add_edge(1, 2)
network.add_node(0)
seesaw = Seesaw(game, [2, 2, 2], 2, network)
q_value, strat = seesaw.run_optimization_multiple_starts(num_random_starts=1, verbose=False)

"""v1  Merge(v2  v3)"""
#NPA upper bound
game_merge = game.merge(1, 2)
  
q_upper = game_merge.compute_NPA(level=2, Nash=False, verbose=False,
                                                       warmStart=False, solver="MOSEK")
print(f"classical value: {c_value}, quantum value: {q_value}, NPA upper bound (merged) {q_upper}")

""" v2  v1 - v3 """
#Classical
print("\nGraph: v2  v1 - v3")
game_fwd = game.reduce_to_foward(0, 2)
value = game_fwd.opt_classical()[0]

# Quantum
network = nx.Graph()
network.add_edge(0, 2)
network.add_node(1)
seesaw = Seesaw(game, [2, 2, 2], 2, network)
q_value, strat = seesaw.run_optimization_multiple_starts(num_random_starts=1, verbose=False)

"""v2  Merge(v1  v3)"""
#NPA upper bound
game_merge = game.merge(0, 2)
  
q_upper = game_merge.compute_NPA(level=2, Nash=False, verbose=False,
                                                       warmStart=False, solver="MOSEK")
print(f"classical value: {c_value}, quantum value: {q_value}, NPA upper bound (merged) {q_upper}")
