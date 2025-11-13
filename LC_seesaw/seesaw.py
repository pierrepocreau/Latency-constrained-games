import numpy as np
import cvxpy as cp
from typing import Any, Optional, Dict, Tuple
import itertools
from copy import deepcopy
import networkx as nx
import sys, os
from quantumStrategy import QuantumStrategy, link_product
from QCFO import QCFO
from cvxpy import Variable, Expression
import dill
import pathlib
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from NPA.NPAgame import NPAgame

class Seesaw:
    """
    Seesaw optimization for quantum social welfare maximization.
    
    Implements alternating optimization between quantum states and party strategies:
    1. Fix party strategies, optimize shared quantum state
    2. Fix shared state, optimize each party's strategy individually
    3. Repeat until convergence

    Note that only real (not complex) strategies are implemented for efficiency.
    """
    
    def __init__(self, game: NPAgame, dim_state: list[int], dim_message: int, network: nx.Graph):
        """
        Initialize seesaw optimizer.
        
        Args:
            game: NPAgame object defining payoffs and input/output distributions
            dim_state: List of shared state dimensions for each party
            dim_message: Dimension of quantum messages exchanged between parties
            network: Communication network topology
        """
        self.game = game
        self.dim_state = dim_state
        self.total_dim_state = np.prod(dim_state)
        self.dim_message = dim_message
        self.network = network

        # State optimization problem can be cachedl
        self._state_variable : Optional[Variable] = None
        self._state_constraints : Optional[list[Expression]] = None
        self._state_parameters: Dict[Tuple[Any, Any], cp.Parameter] = {}
        self._state_problem: Optional[Tuple] = None

        self._strategy : Optional[QuantumStrategy] = None  

    ### State Optimization Part ###

    def create_state(self):
        """
        Create cvxpy Variable for shared quantum state optimization.
        """
        self._state_variable = cp.Variable((self.total_dim_state, self.total_dim_state), PSD=True)
        self._state_constraints = [cp.trace(self._state_variable) == 1]
        
    def optimize_state(self, strategy: QuantumStrategy) -> Tuple[float, np.ndarray]:
        """
        Optimize quantum state ρ for fixed party strategies.
        
        Args:
            strategy: QuantumStrategy with numerical QCFO instruments for all parties

        Returns:
            tuple: (optimal_qsw_value, optimal_rho_matrix)
        """
        if self._state_variable is None:
            self.create_state()

        qsw_expr = self._build_state_qsw(strategy, state_var=self._state_variable)
        prob = cp.Problem(cp.Maximize(qsw_expr), self._state_constraints)
        prob.solve(solver='MOSEK', verbose=False)

        if prob.status != cp.OPTIMAL:
            raise RuntimeError(f"State optimization failed with status: {prob.status}")

        optimal_rho = self._state_variable.value
        return prob.value, optimal_rho
    
    def _build_state_qsw(self, strategy: QuantumStrategy, state_var: cp.Variable) -> cp.Expression:
        """
        Build quantum social welfare expression for state optimization.
        
        Args:
            strategy: QuantumStrategy with fixed numeric QCFO elements
            state_var: cvxpy Variable representing shared quantum state
            
        Returns:
            cp.Expression: Linear expression in state_var to maximize
        """
        connection_matrix, connection_space = strategy.connection_process() # May be None if no communication
        state_matrix, _ = state_var, [strategy.space_indices[k, -1, -1] for k in range(self.game.nbPlayers)]
        global_strategy_matrix, _ = None, None

        # Iterate over all possible input/output combinations
        for question in self.game.questions():
            question_prob = self.game.qDistrib(question)
        
            for answer in itertools.product(*[range(self.game.list_num_out[k]) for k in range(self.game.nbPlayers)]):
                if not self.game.validAnswer(answer, question):
                    continue
                
                # Perform the link product of all parties' strategies
                strategy_matrix, strategy_spaces = deepcopy(connection_matrix), deepcopy(connection_space)                
                
                for k in range(self.game.nbPlayers):
                    strategy_matrix_B = strategy.party_strategies[(k, question[k])].elements[answer[k]]
                    strategy_spaces_B = strategy.global_indexes[k]
                    if strategy_matrix is not None:
                        strategy_matrix, strategy_spaces = link_product(strategy_matrix, strategy_matrix_B, strategy_spaces, strategy_spaces_B, strategy.global_dims)
                    else:
                        strategy_matrix, strategy_spaces = strategy_matrix_B, strategy_spaces_B

                payout = self.game.answerPayoutWin(answer, question)

                if global_strategy_matrix is None:
                    global_strategy_matrix = question_prob * payout * strategy_matrix
                else:
                    global_strategy_matrix += question_prob * payout * strategy_matrix

        return cp.trace(global_strategy_matrix @ state_matrix)
    
    ### Party Strategy Optimization Part ###

    def optimize_party_strategy(self, party: int, strategy: QuantumStrategy, is_diag:bool=False) -> Tuple[float, Dict[int, QCFO]]:
        """
        Optimize strategy for a single party while keeping others fixed.
        
        Args:
            party: Index of party whose strategy to optimize
            strategy: Current numerical strategy composed of state and parties strategies
            
        Returns:
            tuple: (optimal_qsw_value, dict_of_optimized_QCFOs)
        """

        #Push this into strategy.
        # Build optimization variables for this party's strategy
        party_dims = strategy._build_party_dimensions(party)
        party_spaces = strategy._build_party_spaces(party)
        
        symbolic_QCFOs = {}
        all_constraints = []
        
        # Create QCFO optimization variables for each input
        for input_val in range(self.game.list_num_in[party]):
            qcfo = QCFO(
                is_numeric=False,  # Creates cvxpy Variables
                nb_elements=self.game.list_num_out[party],
                spaces=party_spaces,
                dimensions=party_dims,
                is_real=True,
                is_diag=is_diag
            )
            symbolic_QCFOs[input_val] = qcfo
            all_constraints.extend(qcfo.constraints)

        qsw_expr = self._build_parties_qsw(strategy, party_QCFOs=symbolic_QCFOs, party_idx=party)
        problem = cp.Problem(cp.Maximize(qsw_expr), all_constraints)
        problem.solve(solver='MOSEK', verbose=False)
        
        if problem.status != cp.OPTIMAL:
            raise RuntimeError(f"Strategy optimization for party {party} failed: {problem.status}")

        # Convert cvxpy Variables back to numeric QCFOs
        numeric_qcfos = {input_val: qcfo.to_numeric() for input_val, qcfo in symbolic_QCFOs.items()}
        
        return problem.value, numeric_qcfos
    
    def _build_parties_qsw(self, strategy: QuantumStrategy, party_QCFOs: Dict[int, QCFO], party_idx: int) -> cp.Expression:
        """
        Build QSW expression for party strategy optimization.
        
        Args:
            strategy: Current QuantumStrategy with fixed strategies for other parties
            party_QCFOs: Dict mapping input_val to QCFO with cvxpy Variables
            party_idx: Index of party being optimized
            
        Returns:
            cp.Expression: Objective function to maximize (currently very large/slow)
        """
        connection_matrix, connection_space = strategy.connection_process()
        state_matrix, state_spaces = strategy.shared_state, [strategy.space_indices[k, -1, -1] for k in range(self.game.nbPlayers)]
        objective = 0

        for question in self.game.questions():
            q_prob = self.game.qDistrib(question)
            
            for answer in itertools.product(*[range(self.game.list_num_out[k]) for k in range(self.game.nbPlayers)]):
                if not self.game.validAnswer(answer, question):
                    continue
                
                strategy_matrix, strategy_spaces = None, None
                
                for k in range(self.game.nbPlayers):
                    if k != party_idx:
                        strategy_matrix_B = strategy.party_strategies[(k, question[k])].elements[answer[k]]
                        strategy_spaces_B = strategy.global_indexes[k]

                        if strategy_matrix is None:
                            strategy_matrix = strategy_matrix_B
                            strategy_spaces = strategy_spaces_B
                        else:
                            strategy_matrix, strategy_spaces = link_product(strategy_matrix, strategy_matrix_B, strategy_spaces, strategy_spaces_B, strategy.global_dims)
                
                payout = self.game.answerPayoutWin(answer, question)
                
                if connection_matrix is not None:
                    strategy_matrix, strategy_spaces = link_product(strategy_matrix, connection_matrix, strategy_spaces, connection_space, strategy.global_dims)

                strategy_matrix, strategy_spaces = link_product(strategy_matrix, state_matrix, strategy_spaces, state_spaces, strategy.global_dims)
                objective += payout * q_prob * cp.trace(strategy_matrix @ party_QCFOs[question[party_idx]].elements[answer[party_idx]])

        return objective
    
    def run_optimization_multiple_starts(self, max_iterations: int = 20, 
                                        num_random_starts: int = 5, tolerance: float = 1e-5, 
                                        verbose: bool = False, is_diag: bool=False) -> Tuple[float, Any]:
        best_qsw = -np.inf
        best_strategy = None
        
        for _ in range(num_random_starts):
            qsw, strategy = self.run_optimization(
                max_iterations=max_iterations,
                warm_start=None,
                tolerance=tolerance,
                verbose=verbose,
                is_diag=is_diag
            )
            if qsw > best_qsw:
                best_qsw = qsw
                best_strategy = strategy
        
        return best_qsw, best_strategy    
        
    def run_optimization(self, max_iterations: int = 20, warm_start = Optional[QuantumStrategy], tolerance: float = 1e-5, verbose: bool = True, is_diag: bool= False) -> Tuple[float, QuantumStrategy]:
        """
        Run the main seesaw optimization algorithm.
        
        Alternates between optimizing shared quantum state and individual party strategies
        until convergence or maximum iterations reached.
            
        Returns:
            tuple: (final_qsw_value, optimized_OneStepStrategy)
        """
        if warm_start is not None:
            #assert warm_start.message_dim == self.dim_message
            #assert warm_start.shared_state_dim == self.dim_state
            strategy = deepcopy(warm_start)
            
        else:
            # Initialize strategy if not provided
            strategy = QuantumStrategy(
                nb_parties=self.game.nbPlayers,
                nb_input=self.game.list_num_in,
                nb_output=self.game.list_num_out,
                message_dim=self.dim_message,
                shared_state_dim=self.dim_state,
                connections=self.network
            )

        current_qsw = strategy.compute_qsw(self.game)

        # Main seesaw loop
        for iteration in range(max_iterations):
            prev_qsw = current_qsw


            qsw_state, optimal_state = self.optimize_state(strategy)
            strategy.update_shared_state(optimal_state)
            current_qsw = qsw_state
            if verbose:
                print(f"Iteration {iteration + 1}: State opt QSW = {current_qsw:.6f}")
                                

            # Optimize each party's strategy individually
            for party in range(self.game.nbPlayers):
                qsw_party, optimized_qcfos = self.optimize_party_strategy(party, strategy, is_diag)

                # Update strategy with optimized QCFOs
                for input_val, qcfo in optimized_qcfos.items():
                    strategy.update_strategy(party, input_val, qcfo)
                
                current_qsw = qsw_party

                if verbose:
                    print(f"  Party {party}: QSW = {current_qsw:.6f}")

            if abs(current_qsw - 0.75) < 1e-6:
                print("Warning: stuck at classical value, retrying...")
                break

            # Check convergence
            if abs(current_qsw - prev_qsw) < tolerance:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

            if verbose:
                print(f"iteration {iteration}, does it stop ?")
            #noise_level = (max_iterations - iteration)/max_iterations * 0.2
            #strategy.add_noise(noise_level)            
        
        return current_qsw, strategy

if __name__ == "__main__":

    def chsh_simulation(trial=10, saving=False):
        """Simple CHSH optimization """

        chsh_term = lambda out_tuple, in_tuple: int(in_tuple[1] & in_tuple[2] == (out_tuple[1] ^ out_tuple[2]))
        chsh_game = NPAgame(3, [2, 2, 2], [2, 2, 2], 3 * [chsh_term], lambda in_tuple: 1/8)

        network = nx.Graph()
        network.add_edge(0, 1)
        network.add_node(2)    
        seesaw_chsh = Seesaw(chsh_game, [1, 2, 2], 2, network)

        nb_sucess = 0
        saved = False
        for i in range(trial):
            print(f"\n=== Restart {i+1}/{trial} ===")
            qsw, strategy = seesaw_chsh.run_optimization(warm_start=None, verbose=True)
            if qsw > 0.76:
                if not saved and saving:
                    saved = True
                    print("Saving best strategy...")
                    with open("./NetworkSeesaw/data/CHSH.dill", "wb") as f:
                        dill.dump(strategy, f)
                nb_sucess += 1
        print("Better than classical rate:", nb_sucess/trial)

    chsh_simulation(10,True)

    def extended_chsh_simulation(trial=10, saving=False):
        """CHSH optimization with output of 0 and 1 matching"""

        # Load initial CHSH strategy
        print("Loading CHSH strategy...")
        with open("./NetworkSeesaw/data/CHSH.dill", "rb") as f:
            chsh_strategy = dill.load(f)
        print("Loaded.")

        is_chsh = lambda out_tuple, in_tuple: int(in_tuple[1] & in_tuple[2] == (out_tuple[1] ^ out_tuple[2]))
        is_eq = lambda out_tuple, in_tuple: int(out_tuple[1] == out_tuple[0])

        extended_chsh = lambda out_tuple, in_tuple: is_chsh(out_tuple, in_tuple) * is_eq(out_tuple, in_tuple)
    
        network = nx.Graph()
        network.add_edge(0, 1)
        network.add_node(2)    

        extended_chsh_game = NPAgame(3, [2, 2, 2], [2, 2, 2], 3 * [extended_chsh], lambda in_tuple: 1/8)
        seesaw = Seesaw(extended_chsh_game, [1, 2, 2], 2, network)
        
        nb_sucess = 0
        saved = False
        nb_start = trial
        for i in range(nb_start):
            print(f"\n=== Restart {i+1}/{nb_start} ===")

            #Add a bit of noise to not be deterministic
            initial_strategy = deepcopy(chsh_strategy)
            initial_strategy.add_noise(1e-5)

            qsw, strategy = seesaw.run_optimization(warm_start=initial_strategy, verbose=True)
            if qsw > 0.75:
                if saving and not saved:
                    print("Saving best strategy...")
                    with open("./NetworkSeesaw/data/Extended_CHSH.dill", "wb") as f:
                        dill.dump(strategy, f)
                    saved = True
                nb_sucess += 1
        print("Better than classical rate:", nb_sucess/trial)
        
    extended_chsh_simulation(saving=True)        

    def partial_distributed_chsh(trial=10, saving=False):
        """CHSH optimization with input distributed between party 0 and 1"""

        distributed_chsh = lambda out_tuple, in_tuple: int((in_tuple[1] ^ in_tuple[0]) & in_tuple[2] == (out_tuple[1] ^ out_tuple[2]))
        is_eq = lambda out_tuple, in_tuple: int(out_tuple[1] == out_tuple[0])
    
        network = nx.Graph()
        network.add_edge(0, 1)
        network.add_node(2)    

        distributed_chsh_game = NPAgame(3, [2, 2, 2], [2, 2, 2], 3 * [distributed_chsh], lambda in_tuple: 1/8)
        seesaw = Seesaw(distributed_chsh_game, [4, 4, 2], 4, network)
        
        nb_sucess = 0
        saved = False
        nb_start = trial
        for i in range(nb_start):
            print(f"\n=== Restart {i+1}/{nb_start} ===")
            qsw, strategy = seesaw.run_optimization(warm_start=None, verbose=True)

            if qsw > 0.75:
                if saving and not saved:
                    print("Saving best strategy...")
                    with open("./NetworkSeesaw/data/distributed_CHSH_s442_m4.dill", "wb") as f:
                        dill.dump(strategy, f)
                    saved = True
                nb_sucess += 1
        print("Better than classical rate:", nb_sucess/trial)    

    partial_distributed_chsh(saving=True)

    #Here does not work well, we need a better technic. It always converge to classical solutions.
    def fully_distributed_chsh(trial=10, saving=False):
        """CHSH optimization with input distributed between party 0 and 1 and aggrement in output of 0 and 1"""


        # Load initial strategy
        print("Loading strategy...")
        with open("./NetworkSeesaw/data/distributed_CHSH_s442_m4.dill", "rb") as f:
            distributed_chsh_strategy = dill.load(f)
        print("Loaded.")
        
        distributed_chsh = lambda out_tuple, in_tuple: int((in_tuple[1] ^ in_tuple[0]) & in_tuple[2] == (out_tuple[1] ^ out_tuple[2]))
        is_eq = lambda out_tuple, in_tuple: int(out_tuple[1] == out_tuple[0])
        fully_distributed_chsh = lambda out_tuple, in_tuple: distributed_chsh(out_tuple, in_tuple) * is_eq(out_tuple, in_tuple)
    
        network = nx.Graph()
        network.add_edge(0, 1)
        network.add_node(2)    

        fully_distributed_chsh_game = NPAgame(3, [2, 2, 2], [2, 2, 2], 3 * [fully_distributed_chsh], lambda in_tuple: 1/8)
        seesaw = Seesaw(fully_distributed_chsh_game, [4, 4, 2], 3, network)
        
        nb_sucess = 0
        saved = False
        nb_start = trial
        for i in range(nb_start):
            print(f"\n=== Restart {i+1}/{nb_start} ===")

            #Add a bit of noise to not be deterministic
            initial_strategy = deepcopy(distributed_chsh_strategy)
            noise = np.random.uniform(0,0.8)
            initial_strategy.add_noise(noise)
            qsw, strategy = seesaw.run_optimization(warm_start=initial_strategy, verbose=True)

            if qsw > 0.75:
                if saving and not saved:
                    print("Saving best strategy...")
                    with open("fully_distributed_CHSH_s442_m3.dill", "wb") as f:
                        dill.dump(strategy, f)
                    saved = True
                nb_sucess += 1
        print("Better than classical rate:", nb_sucess/trial)      

    #fully_distributed_chsh(trial=100, saving=True)