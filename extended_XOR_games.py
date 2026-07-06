import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
            dict_tt[(i,j)] = tt[inc]
            inc += 1
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
        tt = np.random.randint(2, size=nb_y*nb_z)
        while binatodeci(tt) in seen_tt:
            tt = np.random.randint(2, size=nb_y*nb_z)

        seen_tt.append(binatodeci(tt))

        f = function_from_tt(tt, nb_y, nb_z)
        print(f"\n=== Iteration {i}/50 | function {tt} (id {binatodeci(tt)}) ===", flush=True)
        t_iter = time.time()

        best_qsw, best_strategy = 0, None

        # Payout function
        xor = lambda out_tuple, in_tuple: int(f[(in_tuple[1], in_tuple[2])] == (out_tuple[1] ^ out_tuple[2]))
        is_eq = lambda out_tuple, in_tuple: int(out_tuple[0] == out_tuple[1])
        extended_xor = lambda out_tuple, in_tuple: xor(out_tuple, in_tuple) * is_eq(out_tuple, in_tuple)

        # NPA upper bound for foward strategies, party 0 and party 1 communicate their inputs.
        XORgame = Game(3, [nb_x*nb_y, nb_x*nb_y, nb_z], [2,2,2], [extended_xor]*3, lambda in_tuple: int(in_tuple[0] == in_tuple[1])/(nb_x*nb_y*nb_z))
        t0 = time.time()
        upperBound = XORgame.compute_NPA(level=2, Nash=False, verbose=False, warmStart=False, solver="MOSEK")
        print(f"  [NPA level 2]      {time.time()-t0:6.2f}s  -> upperBound={upperBound:.4f}", flush=True)

        t0 = time.time()
        c_value = XORgame.opt_classical()[0]
        print(f"  [classical]        {time.time()-t0:6.2f}s  -> c={c_value:.4f}", flush=True)
        t0 = time.time()
        c_one_way_value = XORgame.opt_classical_forward(0, 1)[0]
        print(f"  [classical 1-way]  {time.time()-t0:6.2f}s  -> c_oneway={c_one_way_value:.4f}", flush=True)

        qsw = 0
        trial = 0
        MAX_TRIALS = 200
        t_seesaw = time.time()
        while trial <= 30 or (best_qsw <= upperBound - 0.1 and trial <= MAX_TRIALS):
            t0 = time.time()
            qsw, strategy = extended_XOR_game(f, nb_x, nb_y, nb_z, [1, 2, 2], 2)
            if qsw > best_qsw:
                best_qsw = qsw
                best_strategy = strategy
            trial += 1
            print(f"  [seesaw trial {trial:3d}] {time.time()-t0:6.2f}s  qsw={qsw:.4f}  best={best_qsw:.4f}  (target>{upperBound-0.1:.4f})", flush=True)
        print(f"  [seesaw total]     {time.time()-t_seesaw:6.2f}s over {trial} trials", flush=True)
        print(f"  === iteration {i} done in {time.time()-t_iter:6.2f}s ===", flush=True)

        with open(f'./LC_seesaw/data/ExtendedXOR/functionID_{binatodeci(tt)}.dill', "wb") as fh:
            dill.dump(best_strategy, fh)

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