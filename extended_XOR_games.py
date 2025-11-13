from LC_seesaw.seesaw import Seesaw
import networkx as nx
import numpy as np
import pandas as pd 
import dill
from game import Game

def binatodeci(binary):
    '''
    Convert a binary list to decimal. [1, 0, 0] -> 4
    '''
    return sum(val*(2**idx) for idx, val in enumerate(reversed(binary)))

def function_from_tt(tt, x, y):
    '''
    Create a function from a truth table.
    '''
    dict_tt = {}
    inc = 0
    for i in range(x):
        for j in range(y):
            inc += 1
            dict_tt[(i,j)] = tt[inc]
    return dict_tt

def extended_XOR_game(f, nb_x, nb_y, nb_z, dim_state, dim_message):
    """
    XOR game for a function f, with Alice and Bob's output that must match.  
    """
    network = nx.Graph()
    network.add_edge(0, 1)
    network.add_node(2)

    # Payout function
    xor = lambda out_tuple, in_tuple: int(f[(in_tuple[1], in_tuple[2])] == (out_tuple[1] ^ out_tuple[2]))
    is_eq = lambda out_tuple, in_tuple: int(out_tuple[0] == out_tuple[1])
    extended_xor = lambda out_tuple, in_tuple: xor(out_tuple, in_tuple) * is_eq(out_tuple, in_tuple)

    # Uniform distribution over inputs
    func_in_prior = lambda in_tuple: 1/(nb_x*nb_y*nb_z)

    extended_xor_game = Game(3, [nb_x, nb_y, nb_z], [2, 2, 2], 3 * [extended_xor], func_in_prior)

    seesaw = Seesaw(extended_xor_game, dim_state, dim_message, network)

    qsw, strategy = seesaw.run_optimization(warm_start=None, verbose=False)

    return qsw, strategy

if __name__ == "__main__":

    results = []
    seen_tt = []
    for i in range(50):
        nb_x = 1
        nb_y = 3
        nb_z = 3
        tt = np.random.randint(2, size=nb_y*nb_z+1)
        while binatodeci(tt) in results:
            tt = np.random.randint(2, size=nb_y*nb_z+1)

        seen_tt.append(binatodeci(tt))

        f = function_from_tt(tt, nb_y, nb_z)

        best_qsw, best_strategy = 0, None

        # Payout function
        xor = lambda out_tuple, in_tuple: int(f[(in_tuple[1], in_tuple[2])] == (out_tuple[1] ^ out_tuple[2]))
        is_eq = lambda out_tuple, in_tuple: int(out_tuple[0] == out_tuple[1])
        extended_xor = lambda out_tuple, in_tuple: xor(out_tuple, in_tuple) * is_eq(out_tuple, in_tuple)

        # NPA upper bound for foward strategies, party 0 and party 1 communicate their inputs.
        XORgame = Game(3, [nb_x*nb_y, nb_x*nb_y, nb_z], [2,2,2], [extended_xor]*3, lambda in_tuple: int(in_tuple[0] == in_tuple[1])/(nb_x*nb_y*nb_z))
        upperBound = XORgame.compute_NPA(level=2, Nash=False, verbose=False, warmStart=False, solver="MOSEK")

        c_value = XORgame.opt_classical()[0]
        classical_one_way = XORgame.reduce_to_foward(0, 1)
        c_one_way_value = classical_one_way.opt_classical()[0]

        qsw = 0
        trial = 0
        while trial <= 30 or qsw <= upperBound - 0.1:                        
            qsw, strategy = extended_XOR_game(f, nb_x, nb_y, nb_z, [1, 2, 2], 2)
            if qsw > best_qsw:
                best_qsw = qsw
                best_strategy = strategy
            trial += 1
            print(qsw)

        with open(f'./LC_seesaw/data/ExtendedXOR/functionID_{binatodeci(tt)}.dill', "wb") as f:
            dill.dump(best_strategy, f)

        print(f"Iteration {i} Function: {tt}, classical_value {c_value}, classical value with one-way {c_one_way_value}, Seesaw: {best_qsw}, upper-bound forwarding: {upperBound}, diff: {best_qsw - upperBound}, id: {binatodeci(tt)}")
        results.append({
            'function': binatodeci(tt),
            'c': '{:0.3e}'.format(c_value),
            'c_oneway': '{:0.3e}'.format(c_one_way_value),
            'upper_bound': '{:0.3e}'.format(upperBound),
            'seesaw': '{:0.3e}'.format(best_qsw),
            'difference': '{:0.3e}'.format(best_qsw - upperBound),
        })

    gap = 0
    total_with_sep = 0
    for el in results:
        if float(el['difference']) > 1e-3:
            gap += 1
        if abs(float(el['c']) - float(el['c_oneway'])) >= 1e-3:
            total_with_sep += 1

    print(f"Number of separations found: {gap} out of {len(results)} or {total_with_sep} with classical/quantum sep")

    df = pd.DataFrame(results).sort_values('function')
    #df.to_csv('extended_xor_n3m3_qubitcomm_2.csv', index=False)

    latex_table = df.to_latex(index=False, 
                           float_format="%.3e",
                           column_format='cccccc',
                           escape=False)
    print(latex_table)