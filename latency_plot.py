from NPA.NPAgame import NPAgame
from NetworkSeesaw.seesaw import Seesaw
import networkx as nx
import numpy as np
import dill
from cdnp import cdnp
import warnings
from itertools import product
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', message='Objective contains too many subexpressions')
warnings.filterwarnings('ignore', message='Constraint .* contains too many subexpressions')

""" Results used in the paper
lambda = 0.25
seed=5
Algebraic maximum of correlators 0.5075, of perturbations 0.8599999999999999, mixing of the two 0.595625

No communication, classical value 0.38650000000000007, quantum lower bound 0.40051798604431205, NPA upper bound 0.4005179562750757

One round v1 - v2  v3
c_value 0.40384374999999995, q_value 0.4177013561947153.

Merging v1 and v2: (v1  v2)  v3
c_value 0.40384374999999995, q_value 0.42896489239113256, NPA upper bound 0.4289644146477007.
"""

def print_perturbation(perturbation):
    """
    Print the perturbation as a LaTeX table.
    """
    questions = sorted(set(key[1] for key in perturbation.keys()))

    latex_output = []
    latex_output.append(r"\begin{table}[h]")
    latex_output.append(r"\centering")
    latex_output.append(r"\caption{Perturbation values $\varepsilon_{a_1a_2a_3}^{x_1x_2x_3}$ for each output $(a_1,a_2,a_3)$ and input $(x_1,x_2,x_3)$ configuration.}")
    latex_output.append(r"\begin{tabular}{c|cccccccc}")
    latex_output.append(r"\toprule")
    latex_output.append(r"$x \backslash a$ & 000 & 001 & 010 & 011 & 100 & 101 & 110 & 111 \\")
    latex_output.append(r"\midrule")

    for q in questions:
        q_str = ''.join(map(str, q))
        values = []
        for a in sorted(set(key[0] for key in perturbation.keys())):
            if (a, q) in perturbation:
                values.append(f"{perturbation[(a, q)]:.3f}")

        latex_output.append(f"{q_str} & " + " & ".join(values) + r" \\")

    latex_output.append(r"\bottomrule")
    latex_output.append(r"\end{tabular}")
    latex_output.append(r"\end{table}")

    # Print to console
    print('\n'.join(latex_output))

    # Save to file
    with open('perturbation_table.tex', 'w') as f:
        f.write('\n'.join(latex_output))

    print("\n\nLaTeX code saved to 'perturbation_table.tex'")

def plot_communication_advantage(results, save_path='communication_advantage.pdf'):
    """
    Plot communication advantage across different latency constraints with broken y-axis.
    
    Parameters
    ----------
    results : dict
        Dictionary returned by run_communication_analysis
    save_path : str
        Path to save the figure
    """
    import matplotlib.patheffects as patheffects
    
    # Extract values
    no_comm_quantum_ub = results['no_comm']['quantum_ub']
    merge_quantum_ub = results['merge']['quantum_ub']
    trivial_bound = results['algebraic']['mixing']
    
    classical_vals = np.array([results['no_comm']['classical'], 
                               results['one_round']['classical'], 
                               results['merge']['classical'], 
                               results['algebraic']['mixing']])
    quantum_vals = np.array([results['no_comm']['quantum_lb'], 
                            max(results['fwd']['quantum'], results['one_round']['quantum']), 
                            results['merge']['quantum'], 
                            results['algebraic']['mixing']])
    
    # Latency boundaries
    d_c = 1.0
    boundaries = np.array([0, d_c, 2*d_c, 3*d_c])
    x_extended = 4.0

    color_classical = '#082a54'
    color_quantum = "#e02b35"        
    color_npa_0 = '#f0c571'          
    color_npa_2 = '#59a89c'       
    color_algebraic = '#a559aa'

    # Setup figure with broken y-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                    gridspec_kw={'height_ratios': [1, 2.5]})
    fig.subplots_adjust(hspace=0.05)

    # Function to plot on both axes
    def plot_step_function(ax, include_legend=False):
        for i in range(len(classical_vals)):
            t_start = boundaries[i]
            t_end = boundaries[i+1] if i < len(boundaries)-1 else x_extended
            
            # Classical
            ax.hlines(classical_vals[i], t_start, t_end, colors=color_classical, linewidth=3,
                      label=r'Classical value $\omega_c$' if i == 0 and include_legend else '')
            
            if i == 0 or classical_vals[i] != classical_vals[i-1]:
                ax.plot(t_start, classical_vals[i], 'o', markersize=9, color=color_classical, zorder=5)
            
            if i < len(classical_vals) - 1 and classical_vals[i] != classical_vals[i+1]:
                ax.plot(t_end, classical_vals[i], 'o', markersize=9, color=color_classical, 
                        markerfacecolor='white', markeredgewidth=2.5, zorder=5)
            
            # Quantum lower bound
            ax.hlines(quantum_vals[i], t_start, t_end, colors=color_quantum, linewidth=3, linestyle='--',
                      label=r'Lower bound on $\omega_q$ (see-saw)' if i == 0 and include_legend else '')
            
            if i == 0 or quantum_vals[i] != quantum_vals[i-1]:
                ax.plot(t_start, quantum_vals[i], 'o', markersize=9, color=color_quantum, zorder=5)
            
            if i < len(quantum_vals) - 1 and quantum_vals[i] != quantum_vals[i+1]:
                ax.plot(t_end, quantum_vals[i], 'o', markersize=9, color=color_quantum, 
                        markerfacecolor='white', markeredgewidth=2.5, zorder=5)
        
        # NPA and algebraic bounds with ticked effect
        line1, = ax.plot([0, d_c], [no_comm_quantum_ub, no_comm_quantum_ub], 
                         color=color_npa_0, linewidth=2.5, linestyle='-',
                         label=r'Upper bound on $\omega_q$, $t \in [0, \frac{d}{c})$ (NPA)' if include_legend else '', 
                         alpha=0.8, zorder=1, solid_capstyle='round')
        line1.set_path_effects([patheffects.withTickedStroke(spacing=25, angle=90, length=0.5)])
        
        line2, = ax.plot([d_c, 3*d_c], [merge_quantum_ub, merge_quantum_ub], 
                         color=color_npa_2, linewidth=2.5, linestyle='-',
                         label=r"Upper bound on $\omega_q$, $t \in [\frac{d}{c}, \frac{d'}{c})$ (NPA)" if include_legend else '', 
                         alpha=0.8, zorder=1, solid_capstyle='round')
        line2.set_path_effects([patheffects.withTickedStroke(spacing=25, angle=90, length=0.5)])
        
        line3, = ax.plot([3*d_c, x_extended], [trivial_bound, trivial_bound], 
                         color=color_algebraic, linewidth=2.5, linestyle='-',
                         label=r'Maximum algebraic value $\omega_a$' if include_legend else '', 
                         alpha=0.8, zorder=1, solid_capstyle='round')
        line3.set_path_effects([patheffects.withTickedStroke(spacing=25, angle=90, length=0.5)])
        
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

    # Plot on both axes
    plot_step_function(ax1, include_legend=True)
    plot_step_function(ax2, include_legend=False)

    # Set y-limits for broken axis
    ax1.set_ylim(0.57, 0.605)
    ax2.set_ylim(0.38, 0.445)

    # Hide spines between panels
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Tick configuration
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False, top=False)
    ax2.xaxis.tick_bottom()

    # Add break marks
    d = .5
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    # Thicker remaining spines
    ax1.spines['left'].set_linewidth(1.2)
    ax2.spines['left'].set_linewidth(1.2)
    ax2.spines['bottom'].set_linewidth(1.2)

    # Labels
    ax2.set_xlabel('Latency constraint $t$', fontsize=19)
    ax2.set_ylabel('Game value', fontsize=19)
    ax2.yaxis.set_label_coords(-0.08, 0.75)

    # X-axis ticks and labels
    ax2.set_xticks([0, d_c, 2*d_c, 3*d_c])
    ax2.set_xticklabels(['$0$', r'$\frac{d}{c}$', r'$\frac{2d}{c}$', r"$\frac{d'}{c}$"], fontsize=18)
    ax2.tick_params(axis='both', labelsize=17)
    ax1.tick_params(axis='y', labelsize=17)
    ax1.set_yticks([0.58, 0.59, 0.60])

    # Set x-limits
    ax1.set_xlim([-0.1, x_extended])
    ax2.set_xlim([-0.1, x_extended])

    # Legend in top left of upper panel
    ax1.legend(loc='upper left', fontsize=18, framealpha=0.98, 
              edgecolor='gray', fancybox=False, shadow=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=3600, bbox_inches='tight')
    plt.show()

def run_communication_analysis(num_players, list_num_in, list_num_out, correlators, 
                                scale, seed, num_random_starts_no_comm=150, 
                                num_random_starts_one_round=150, 
                                num_random_starts_merge=150):
    """
    Analyze quantum communication advantage across different latency constraints.
    
    Parameters
    ----------
    num_players : int
        Number of players in the game
    list_num_in : list of int
        Number of inputs per player
    list_num_out : list of int
        Number of outputs per player
    correlators : dict
        Correlator dictionary (e.g., {"xyz": np.array(...)})
    scale : float
        Mixing parameter between correlators and perturbations
    seed : int
        Random seed for reproducibility
    num_random_starts_* : int
        Number of random starts for seesaw optimization
        
    Returns
    -------
    dict
        Contains all computed values, strategies, and parameters
    """
    random_gen = np.random.default_rng(seed=seed)
    
    # Generate perturbations
    perturbation = {}
    algebraic_pert = 0
    for q in product(*[list(range(list_num_in[i])) for i in range(num_players)]):
        max_q = 0
        for a in product(*[list(range(list_num_out[i])) for i in range(num_players)]):
            perturbation[a, q] = np.round(random_gen.random(), 3)
            if np.abs(perturbation[a, q]) > max_q:
                max_q = np.abs(perturbation[a, q])
        algebraic_pert += max_q

    funcs_utility_player = lambda out_tuple, in_tuple: (1-scale) * correlators["xyz"][in_tuple] * (-1)**(np.sum(out_tuple)) + scale * perturbation[out_tuple, in_tuple]
    func_in_prior = lambda in_tuple: 1/8

    # Initialize games
    game = NPAgame(num_players, list_num_in, list_num_out, funcs_utility_player=[funcs_utility_player]*3, func_in_prior=func_in_prior)
    cdnpGame = cdnp(num_players, list_num_in, list_num_out, func_utility=funcs_utility_player, func_in_prior=func_in_prior)
    
    # Algebraic bounds
    algebraic = np.sum(np.abs(correlators["xyz"]))
    alg_correlators = algebraic / 8
    alg_perturbations = algebraic_pert / 8
    alg_mixing = ((1 - scale) * algebraic + scale * algebraic_pert) / 8
    
    # No communication scenario
    network = nx.Graph()
    network.add_node(0)
    network.add_node(1)
    network.add_node(2)
    seesaw = Seesaw(game, [2, 2, 2], 1, network)
    no_comm_quantum_lb, no_comm_strategy = seesaw.run_optimization_multiple_starts(
        num_random_starts=num_random_starts_no_comm, verbose=False)
    no_comm_quantum_ub = game.optimize(level=3, Nash=False, verbose=False, 
                                       warmStart=False, solver="MOSEK")
    no_comm_classical = cdnpGame.opt_classical()[0]

    # Quantum forwarding v1 -> v2  v3
    game_fwd_01 = game.reduce_to_foward(0,1)
    network = nx.Graph()
    network.add_node(0)
    network.add_node(1)
    network.add_node(2)
    seesaw = Seesaw(game_fwd_01, [2, 2, 2], 1, network)
    fwd_quantum, fwd_quantum_strategy = seesaw.run_optimization_multiple_starts(
        num_random_starts=num_random_starts_one_round, verbose=False)
    fwd_quantum_up = game_fwd_01.optimize(level=2, verbose=False)

    # One round: v1 -> v2  v3
    cdnpGame_fwd_01 = cdnpGame.reduce_to_foward(0, 1)
    one_round_classical = cdnpGame_fwd_01.opt_classical()[0]
    
    network = nx.Graph()
    network.add_edge(0, 1)
    network.add_node(2)
    seesaw = Seesaw(game, [2, 2, 2], 2, network)
    one_round_quantum, one_round_strategy = seesaw.run_optimization_multiple_starts(
        num_random_starts=num_random_starts_one_round, verbose=False)
    
    # Merging v1 and v2: (v1  v2) v3
    game_merge_01 = game.merge(0, 1)
    
    network_merged = nx.Graph()
    network_merged.add_node(0)
    network_merged.add_node(1)
    seesaw = Seesaw(game_merge_01, [2, 2], 1, network_merged)
    merge_quantum, merge_strategy = seesaw.run_optimization_multiple_starts(
        num_random_starts=num_random_starts_merge, verbose=False)
    merge_quantum_ub = game_merge_01.optimize(level=2, Nash=False, verbose=False,
                                              warmStart=False, solver="MOSEK")
    
    return {
        'parameters': {
            'scale': scale,
            'seed': seed,
            'num_players': num_players,
            'list_num_in': list_num_in,
            'list_num_out': list_num_out
        },
        'algebraic': {
            'correlators': alg_correlators,
            'perturbations': alg_perturbations,
            'mixing': alg_mixing
        },
        'no_comm': {
            'classical': no_comm_classical,
            'quantum_lb': no_comm_quantum_lb,
            'quantum_ub': no_comm_quantum_ub,
            'strategy': no_comm_strategy
        },
        'fwd' : {
            'quantum': fwd_quantum,
            'quantum_ub': fwd_quantum_up,
            'strategy': fwd_quantum_strategy
        },
        'one_round': {
            'classical': one_round_classical,
            'quantum': one_round_quantum,
            'strategy': one_round_strategy
        },
        'merge': {
            'classical': one_round_classical, # same value
            'quantum': merge_quantum,
            'quantum_ub': merge_quantum_ub,
            'strategy': merge_strategy
        }
    }

if __name__ == "__main__":
    #Random generator with seed
    results = run_communication_analysis(
        num_players=3,
        list_num_in=[2, 2, 2],
        list_num_out=[2, 2, 2],
        correlators={"xyz": np.array([[[ 0.438,  0.61 ], [ 0.52,  -0.466]], 
                                   [[ 0.58,  -0.502], [-0.724, -0.22 ]]])},
        scale=0.25,
        seed=5,
    )
    # Plot results
    plot_communication_advantage(results, "test.pdf")