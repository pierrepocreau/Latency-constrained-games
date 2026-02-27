import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LC_seesaw.seesaw import Seesaw
from game import Game
import networkx as nx
import numpy as np
import warnings
from itertools import product
import pickle

warnings.filterwarnings('ignore', message='Objective contains too many subexpressions')
warnings.filterwarnings('ignore', message='Constraint .* contains too many subexpressions')

def get_tilted_CHSH_score(a, b, x, y, t):
    """
    Create a bipartite game as a convex combination of CHSH and a ``very'' lazy guess your neighbors' input.
    """
    chsh_1 = (a ^ b) == (x * y)
    VLGYNI = 1
    if x == 1:
        VLGYNI = VLGYNI * (a == y)

    return t * chsh_1 + (1 - t) * VLGYNI

def run_communication_analysis(num_players, list_num_in, list_num_out, correlators, 
                               scale, 
                               num_random_starts_no_comm=20, 
                               num_random_starts_one_round=20, 
                               num_random_starts_merge=20,
                               num_random_starts_line=2):

    def funcs_utility_player(out_tuple, in_tuple):
        xor = np.maximum(correlators["xyz"][in_tuple] * (-1)**(np.sum(out_tuple)), 0)
        
        tilted = get_tilted_CHSH_score(out_tuple[0], out_tuple[2], in_tuple[0], in_tuple[2], 0.9)
        return (1 - scale) * xor + scale * tilted

    func_in_prior = lambda in_tuple: 1.0 / np.prod(list_num_in)

    # Algebraic Max
    algebraic_mixing = 0
    for q in product(*[range(n) for n in list_num_in]):
        max_q = -np.inf
        for a in product(*[range(n) for n in list_num_out]):
            max_q = max(max_q, funcs_utility_player(a, q))
        algebraic_mixing += max_q * func_in_prior(q)
    print(f"Algebraic maximum: {algebraic_mixing}")
    
    game = Game(num_players, list_num_in, list_num_out, funcs_utility_player=[funcs_utility_player]*3, func_in_prior=func_in_prior)
    
    # 1. No Communication
    net_no_comm = nx.Graph(); net_no_comm.add_nodes_from([0,1,2])
    seesaw = Seesaw(game, [2, 2, 2], 1, net_no_comm)
    no_comm_q, no_comm_sol = seesaw.run_optimization_multiple_starts(num_random_starts=num_random_starts_no_comm, verbose=False)
    no_comm_npa = game.compute_NPA(level=1, Nash=False, verbose=False, solver="MOSEK")
    no_comm_c = game.opt_classical()[0]
    network = nx.Graph(); network.add_node(0); network.add_node(1), network.add_node(2)
    g_sig = game.g_signaling(network)
    print(f"No Comm      | C: {no_comm_c:.5f}, Q: {no_comm_q:.5f}, NPA: {no_comm_npa:.5f}, G-signalling: {g_sig}")

    # 2. Forwarding v1 <-> v2  v3
    game_fwd = game.reduce_to_foward(0, 1)
    net_fwd = nx.Graph(); net_fwd.add_nodes_from([0,1,2])
    seesaw = Seesaw(game_fwd, [2, 2, 2], 1, net_fwd)
    fwd_q, fwd_sol = seesaw.run_optimization_multiple_starts(num_random_starts=num_random_starts_one_round, verbose=False)
    fwd_npa = game_fwd.compute_NPA(level=1, verbose=False)
    fwd_c_value = game_fwd.opt_classical()[0]
    print(f"Fwd v1-v2    | C: {fwd_c_value:0.5f}, Q: {fwd_q:.5f}, NPA: {fwd_npa:.5f}")

    # 3. Merging (v1 v2) v3
    game_merge = game.merge(0, 1)
    net_merge = nx.Graph(); net_merge.add_nodes_from([0,1])
    seesaw = Seesaw(game_merge, [2, 2], 1, net_merge)
    merge_q, merge_sol = seesaw.run_optimization_multiple_starts(num_random_starts=num_random_starts_merge, verbose=False)
    merge_npa = game_merge.compute_NPA(level=1, Nash=False, verbose=False, solver="MOSEK")
    
    net_sig_merge = nx.Graph(); net_sig_merge.add_edge(0, 1); net_sig_merge.add_node(2)
    g_sig_merge = game.g_signaling(net_sig_merge)
    print(f"Merge v1,v2  | C: {game_merge.opt_classical()[0]:.5f}, Q: {merge_q:.5f}, NPA: {merge_npa:.5f}, G-Sig: {g_sig_merge:.5f}")

    # 3.2 Merging v1 (v2 v3)
    game_merge_2 = game.merge(1, 2)
    seesaw = Seesaw(game_merge_2, [2, 2], 1, net_merge)
    merge_q_2, merge_sol_2 = seesaw.run_optimization_multiple_starts(num_random_starts=num_random_starts_merge, verbose=False)
    merge_npa_2 = game_merge_2.compute_NPA(level=1, Nash=False, verbose=False, solver="MOSEK")
    
    net_sig_merge_2 = nx.Graph(); net_sig_merge_2.add_edge(1, 2); net_sig_merge_2.add_node(0)
    g_sig_merge_2 = game.g_signaling(net_sig_merge_2)
    print(f"Merge v2,v3  | C: {game_merge_2.opt_classical()[0]:.5f}, Q: {merge_q_2:.5f}, NPA: {merge_npa_2:.5f}, G-Sig: {g_sig_merge_2:.5f}")

    # 3.2 Merging v2 (v3 v1)
    game_merge_3 = game.merge(0, 2)
    seesaw = Seesaw(game_merge_3, [2, 2], 1, net_merge)
    merge_q_3, merge_sol_3 = seesaw.run_optimization_multiple_starts(num_random_starts=num_random_starts_merge, verbose=False)
    merge_npa_3 = game_merge_3.compute_NPA(level=1, Nash=False, verbose=False, solver="MOSEK")
    
    net_sig_merge_3 = nx.Graph(); net_sig_merge_3.add_edge(0, 2); net_sig_merge_3.add_node(1)
    g_sig_merge_3 = game.g_signaling(net_sig_merge_3)
    print(f"Merge v1,v3  | C: {game_merge_3.opt_classical()[0]:.5f}, Q: {merge_q_3:.5f}, NPA: {merge_npa_3:.5f}, G-Sig: {g_sig_merge_3:.5f}")    

    # 4. Line Forwarding Strategy (v1 <=> v2 <=> v3)
    game_line_fwd = game.line_forward()
    net_line_fwd = nx.Graph(); net_line_fwd.add_nodes_from([0,1,2])
    seesaw = Seesaw(game_line_fwd, [2, 2, 2], 0, net_line_fwd)
    line_fwd_q, line_fwd_sol = seesaw.run_optimization_multiple_starts(num_random_starts=num_random_starts_line, verbose=False)
    line_fwd_npa = game_line_fwd.compute_NPA(level=1, Nash=False, verbose=False, solver="MOSEK")
    c_value = game_line_fwd.opt_classical()[0]       
    print(f"Line Fwd     | C: {c_value:0.5f}, Q: {line_fwd_q:.5f}, NPA: {line_fwd_npa:.5f}")

    # 5. Line Communication (v1 - v2 - v3)
    net_line = nx.Graph(); net_line.add_edges_from([(0,1), (1,2)])
    seesaw = Seesaw(game, [2, 2, 2], 2, net_line)
    line_comm_q, line_comm_sol = seesaw.run_optimization_multiple_starts(num_random_starts=num_random_starts_line, verbose=False)
    line_g_sig = game.g_signaling(net_line)
    print(f"Line Comm    | C: {c_value:.5f}, Q: {line_comm_q:.5f}, G-Sig: {line_g_sig:.5f}")


    # 6. Full Graph
    net_full = nx.Graph(); net_full.add_edges_from([(0,1), (1,2), (0,2)])
    full_g_sig = game.g_signaling(net_full)
    print(f"Full Graph   | G-Sig: {full_g_sig:.5f}")

    numerical_results = {
        'parameters': {'scale': scale},
        'algebraic': algebraic_mixing,
        'no_comm': {'c': no_comm_c, 'q': no_comm_q, 'npa': no_comm_npa, 'g_sig': g_sig},
        'fwd': {'c': fwd_c_value, 'q': fwd_q, 'npa': fwd_npa},
        'merge': {'q': merge_q, 'npa': merge_npa, 'g_sig': g_sig_merge},
        'merge_2': {'q': merge_q_2, 'npa': merge_npa_2, 'g_sig': g_sig_merge_2},
        'merge_3': {'q': merge_q_3, 'npa': merge_npa_3, 'g_sig': g_sig_merge_3},
        'line_fwd': {'c': c_value, 'q': line_fwd_q, 'npa': line_fwd_npa},
        'line_comm': {'c': c_value, 'q': line_comm_q, 'g_sig': line_g_sig},
        'full': {'g_sig': full_g_sig}
    }

    numerical_solutions = {
        'no_comm': no_comm_sol,
        'fwd': fwd_sol,
        'merge': merge_sol,
        'merge_2': merge_sol_2,
        'merge_3': merge_sol_3,
        'line_fwd': line_fwd_sol,
        'line_comm': line_comm_sol
    }
    
    return numerical_results, numerical_solutions

if __name__ == "__main__":
    corr_tensor = np.array([[[ 0.438,  0.61 ], [ 0.52,  -0.466]], 
                            [[ 0.58,  -0.502], [-0.724, -0.22 ]]])
      
    results = run_communication_analysis(
        num_players=3,
        list_num_in=[2, 2, 2],
        list_num_out=[2, 2, 2],
        correlators={"xyz": corr_tensor},
        scale=0.8,
        num_random_starts_no_comm=10,
        num_random_starts_one_round=10,
        num_random_starts_merge=10,
        num_random_starts_line=10
    )

    with open('LC_paper/results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("Results saved successfully to results.pkl")

"""
Without max
random correlators
scale = 0.8, t=0.9
Algebraic maximum: 0.9015000000000001
No Comm      | C: 0.66860, Q: 0.72289, NPA: 0.72289, G-signalling: 0.8445000000002617
Fwd v1-v2    | C: 0.66860, Q: 0.72653, NPA: 0.72653
Merge v1,v2  | C: 0.66860, Q: 0.72653, NPA: 0.72653, G-Sig: 0.84720
Merge v2,v3  | C: 0.66860, Q: 0.72497, NPA: 0.72497, G-Sig: 0.84450
Merge v1,v3  | C: 0.86450, Q: 0.86450, NPA: 0.86450, G-Sig: 0.86450
Line Fwd     | C: 0.70150, Q: 0.74691, NPA: 0.74691
Line Gen     | C: 0.70150, Q: 0.74950, G-Sig: 0.88150
Full Graph   | G-Sig: 0.90150
"""

"""
With max
Algebraic maximum: 0.9015000000000001
No Comm      | C: 0.68505, Q: 0.74884, NPA: 0.74884, G-signalling: 0.8630000000001546
Fwd v1-v2    | C: 0.68505, Q: 0.75022, NPA: 0.75022
Merge v1,v2  | C: 0.68505, Q: 0.75022, NPA: 0.75022, G-Sig: 0.86435
Merge v2,v3  | C: 0.68505, Q: 0.74934, NPA: 0.74934, G-Sig: 0.86300
Merge v1,v3  | C: 0.88300, Q: 0.88300, NPA: 0.88300, G-Sig: 0.88300
Line Fwd     | C: 0.70150, Q: 0.76136, NPA: 0.76135
Line Comm    | C: 0.70150, Q: 0.76277, G-Sig: 0.88150
Full Graph   | G-Sig: 0.90150
"""