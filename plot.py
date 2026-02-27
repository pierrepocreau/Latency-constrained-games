import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from LC_seesaw.quantumStrategy import QuantumStrategy
from matplotlib.ticker import MaxNLocator


def plot_communication_advantage(file_path, save_path='communication_advantage.pdf'):
    """
    Plot communication advantage on a large single plot with a narrower zoomed inset 
    positioned deeply in the bottom-right corner.
    """

    # 1. Load Data
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            results, _ = pickle.load(f)

        print(results)
        # Extract Values from file
        no_comm_quantum_ub = results['no_comm']['npa']
        g_sig_line = results['line_comm']['g_sig']
        trivial_bound = results['algebraic']

        classical_vals = np.array([
            results['no_comm']['c'], 
            results['line_fwd']['c'], 
            results['algebraic']
        ])
        
        quantum_vals = np.array([
            results['no_comm']['q'], 
            results['line_comm']['q'], 
            results['algebraic']
        ])
    else:
        print(f"Warning: '{file_path}' not found. Please ensure the file exists.")
        return

    # Latency boundaries
    d_c = 1.0
    x_extended = 3.0
    boundaries = np.array([0, d_c, 2*d_c, 3*d_c])

    # 2. Styling
    color_classical = '#082a54'
    color_quantum = "#e02b35"        
    color_npa_0 = '#f0c571'           
    color_npa_2 = '#59a89c'   
    color_npa_3 = "#0400ff"        
    color_algebraic = '#a559aa'

    # 3. Setup Figure (Large size)
    fig, ax = plt.subplots(figsize=(14, 12))

    # --- Helper Function to Plot Data ---
    def draw_plots(target_ax, include_labels=False, marker_scale=1.0):
        ms_dot = 11 * marker_scale # Larger dots
        lw_line = 4                # Thicker lines
        
        for i in range(len(classical_vals)):
            t_start = boundaries[i]
            t_end = boundaries[i+1] if i < len(boundaries)-1 else x_extended
            
            # Classical
            target_ax.hlines(classical_vals[i], t_start, t_end, colors=color_classical, linewidth=lw_line,
                      label=r'Classical value $\omega_c$' if i == 0 and include_labels else '')
            
            if i == 0 or classical_vals[i] != classical_vals[i-1]:
                target_ax.plot(t_start, classical_vals[i], 'o', markersize=ms_dot, 
                               color=color_classical, zorder=5)
            
            if i < len(classical_vals) - 1 and classical_vals[i] != classical_vals[i+1]:
                target_ax.plot(t_end, classical_vals[i], 'o', markersize=ms_dot, 
                        color=color_classical, markerfacecolor='white', markeredgewidth=2.5, zorder=5)
            
            # Quantum
            target_ax.hlines(quantum_vals[i], t_start, t_end, colors=color_quantum, linewidth=lw_line, linestyle='--',
                      label=r'Lower bound on $\omega_q$ (see-saw)' if i == 0 and include_labels else '')
            
            if i == 0 or quantum_vals[i] != quantum_vals[i-1]:
                target_ax.plot(t_start, quantum_vals[i], 'o', markersize=ms_dot, 
                               color=color_quantum, zorder=5)
            
            if i < len(quantum_vals) - 1 and quantum_vals[i] != quantum_vals[i+1]:
                target_ax.plot(t_end, quantum_vals[i], 'o', markersize=ms_dot, 
                        color=color_quantum, markerfacecolor='white', markeredgewidth=2.5, zorder=5)
        
        # Bounds
        l1, = target_ax.plot([0, d_c], [no_comm_quantum_ub, no_comm_quantum_ub], 
                         color=color_npa_0, linewidth=2.5, linestyle='-',
                         label=r'Upper bound on $\omega_q$ (NPA)' if include_labels else '', 
                         alpha=0.8, zorder=1, solid_capstyle='round')
        l1.set_path_effects([patheffects.withTickedStroke(spacing=25, angle=90, length=0.5)])
        

        l3, = target_ax.plot([d_c, 2*d_c], [g_sig_line, g_sig_line], 
                         color=color_npa_3, linewidth=2.5, linestyle='-',
                         label=r"Upper bound on $\omega_q$ (G-sig)" if include_labels else '', 
                         alpha=0.8, zorder=1, solid_capstyle='round')
        l3.set_path_effects([patheffects.withTickedStroke(spacing=25, angle=90, length=0.5)])        
        
        l4, = target_ax.plot([2*d_c, x_extended], [trivial_bound, trivial_bound], 
                         color=color_algebraic, linewidth=2.5, linestyle='-',
                         label=r'Max algebraic value $\omega_a$' if include_labels else '', 
                         alpha=0.8, zorder=1, solid_capstyle='round')
        l4.set_path_effects([patheffects.withTickedStroke(spacing=25, angle=90, length=0.5)])
        
        target_ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

    # 4. Draw Main Plot
    draw_plots(ax, include_labels=True)

    # 6. Main Axes Configuration
    ax.set_ylim(0.6, 1.05) 
    ax.set_xlim([-0.1, x_extended])
    ax.set_xlabel('Latency constraint $t$', fontsize=23)
    ax.set_ylabel('Game value', fontsize=23)
    ax.set_xticks([0, d_c, 2*d_c])
    ax.set_xticklabels(['$0$', r'$\frac{d}{c}$', r'$\frac{2d}{c}$'], fontsize=22)
    ax.tick_params(axis='both', labelsize=21)
    
    # Legend
    ax.legend(loc='upper left', fontsize=20, framealpha=0.98, 
              edgecolor='gray', fancybox=False, ncol=2)

    plt.tight_layout()
    print(f"Saving plot to {save_path}...")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    if os.path.exists('LC_paper/results.pkl'):
        plot_communication_advantage('LC_paper/results.pkl')
    else:
        print("Error: 'results.pkl' not found.")