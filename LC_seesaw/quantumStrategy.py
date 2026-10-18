import networkx as nx
import numpy as np
from typing import Optional, Dict, Tuple
import itertools

from QCFO import QCFO
from QCFO_utils import pure_CJ
import sys, os
import string
from copy import deepcopy
from toqito.rand import random_density_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game import Game

def link_product(A, B, sys_A, sys_B, dims):
    """
    Real-valued link product via tensor contraction.

    A, B        : numpy arrays (operators)
    sys_A, sys_B: lists of subsystem indices that A, B act on
    dims        : list of global subsystem dimensions
    """

    # -- Identify subsystem sets --
    shared = sorted(list(set(sys_A) & set(sys_B)))  # systems to trace out
    active = sorted(list(set(sys_A) | set(sys_B)))  # all relevant systems
    remaining = [s for s in active if s not in shared]  # systems that remain

    # -- Convenience maps for reshaping --
    dims_A = [dims[i] for i in sys_A]
    dims_B = [dims[i] for i in sys_B]

    # reshape A and B into tensors with 2 indices per subsystem
    A_t = A.reshape([*dims_A, *dims_A])
    B_t = B.reshape([*dims_B, *dims_B])

    # Assign index labels (for einsum)
    letters = list(string.ascii_letters)
    label_pairs = {s: (letters.pop(0), letters.pop(0)) for s in active}

    # build index label strings
    A_labels = "".join(label_pairs[s][0] for s in sys_A) + "".join(label_pairs[s][1] for s in sys_A)
    B_labels = "".join(label_pairs[s][0] for s in sys_B) + "".join(label_pairs[s][1] for s in sys_B)

    # output labels: keep only remaining (non-shared) systems
    out_labels = "".join(label_pairs[s][0] for s in remaining) + "".join(label_pairs[s][1] for s in remaining)

    einsum_str = f"{A_labels},{B_labels}->{out_labels}"

    # contract and reshape to matrix
    result_t = np.einsum(einsum_str, A_t, B_t)
    out_dims = [dims[s] for s in remaining]
    result = result_t.reshape(int(np.prod(out_dims)), int(np.prod(out_dims)))

    return result, remaining

class QuantumStrategy:
    """
    Quantum strategy with message passing between connected parties.
    
    Implements a quantum protocol where parties share entangled states, receive
    classical inputs, exchange quantum messages with connected neighbors, and 
    produce classical outputs. The strategy uses QCFO (Quantum Circuit with Fixed Order)
    instruments to model the local quantum operations at each party.
    
    The protocol structure:
    1. Parties share an entangled quantum state
    2. Each party receives a classical input
    3. Parties perform quantum operations (QCFOs) that can:
       - Send quantum messages to connected neighbors  
       - Receive quantum messages from connected neighbors
       - Produce classical outputs
    4. All operations happen in a single communication round
    
    Args:
        nb_parties: Number of parties in the protocol
        nb_input: List of input alphabet sizes for each party
        nb_output: List of output alphabet sizes for each party  
        message_dim: Dimension of quantum messages exchanged
        shared_state_dim: List of shared state dimensions for each party
        connections: NetworkX graph defining communication topology
    """
    
    def __init__(self, nb_parties: int, nb_input: list[int], nb_output: list[int], 
                 message_dim: int, shared_state_dim: list[int], connections: nx.Graph):
        
        self.nb_parties = nb_parties
        self.nb_input = nb_input
        self.nb_output = nb_output
        self.message_dim = message_dim
        self.shared_state_dim = shared_state_dim
        self.connections = connections
        
        # Initialize random state
        total_dim = np.prod(self.shared_state_dim)
        self.shared_state = random_density_matrix(total_dim, is_real=True)
        
        # Storage for QCFO strategies
        self.party_strategies: Dict[Tuple[int, int], QCFO] = {}
        self.space_indices: Dict[Tuple[int, int, int], int] = {}
        self.global_indexes: Dict[int, list[int]] = {}
        self.global_dims: list[int] = []
        
        self._build_global_structure()
        self._generate_all_strategies()


    # Book keeping methods for structure
    
    def _build_global_structure(self):
        """
        Build the global quantum system structure encompassing all parties.
        
        Creates a unified indexing system for all quantum systems across parties:
        - Each party has systems for: shared state, output messages, input messages
        - Global ordering allows tensor products and permutations between parties
        - Space indices track where each logical system appears in global structure
        
        The global structure concatenates all party structures:
        Party 0 systems | Party 1 systems | ... | Party N systems
        
        For each party with k neighbors:
        [shared_state, out_msg_1, trivial_1, ..., out_msg_k, trivial_k, 
         trivial_1, in_msg_1, ..., trivial_k, in_msg_k, output_space]
        """
        offset = 0
        
        for party in range(self.nb_parties):
            indexes = []
            neighbors = list(self.connections.neighbors(party))
            nb_neighbors = len(neighbors)
            
            # Build dimensions for this party
            party_dims = self._build_party_dimensions(party)
            self.global_dims.extend(party_dims)
            
            # Store space indices for this party
            # Shared state space (each party has one)
            self.space_indices[(party, -1, -1)] = offset  
            indexes.append(offset)
            
            pointer = 1

            if nb_neighbors >= 1:
                # Output message spaces (party -> neighbor)
                for idx, neighbor in enumerate(neighbors):
                    self.space_indices[(party, party, neighbor)] = offset + pointer
                    indexes.append(offset + pointer)
                    pointer += 2

                pointer += 1
                # Input message spaces (neighbor -> party)  
                for idx, neighbor in enumerate(neighbors):
                    self.space_indices[(party, neighbor, party)] = offset + pointer
                    indexes.append(offset + pointer)
                    pointer += 2
            else:
                pointer += 1

            indexes.append(offset + pointer - 1)
            self.global_indexes[party] = indexes

            # Move offset to start of next party's systems
            offset += pointer


    def _build_party_dimensions(self, party: int) -> list[int]:
        """
        Build dimension structure for a single party's quantum systems.
        
        Each party's local Hilbert space includes:
        1. Space for shared state system (entangled with other parties)
        2. Output message systems (one per neighbor, with trivial ancilla)
        3. Input message systems (one per neighbor, with trivial ancilla)  
        4. Final trivial output system
            
        Returns:
            List of dimensions: [shared_state, output_messages, input_messages, trivial]
        """
        nb_neighbors = self.connections.degree(party)
        dims = [self.shared_state_dim[party]]  # Shared state space
        dims.extend([self.message_dim, 1] * nb_neighbors)  # Output messages with trivial ancillas
        dims.extend([1, self.message_dim] * nb_neighbors)  # Input messages with trivial ancillas
        dims.append(1)  # Final trivial output
        return dims
    
    def _build_party_spaces(self, party: int) -> list[list[int]]:
        """
        Build space structure for a party's QCFO instrument.
        
        Defines the causal ordering of quantum systems for QCFO constraints.
        The space structure determines which systems can influence which others:
        
        1. Shared state space (input, no causal dependencies)
        2. Output message spaces (party can write to these)
        3. Input message spaces (party can read from these) 
        4. Final output space (classical output production)
        
        Each space is a list of system indices that are grouped together
        for the purposes of the QCFO causal structure.
        
        Args:
            party: Index of party to build space structure for
            
        Returns:
            List of spaces, where each space is a list of system indices
        """
        neighbors = list(self.connections.neighbors(party))
        nb_neighbors = len(neighbors)
        spaces = []
        
        # Shared state space (input to protocol)
        spaces.append([0])
        
        # Output message spaces (party -> neighbor)
        for idx in range(nb_neighbors):
            spaces.append([2 * idx + 1, 2 * idx + 2])
        
        # Input message spaces (neighbor -> party)
        for idx in range(nb_neighbors):
            j = 2 * nb_neighbors + 2 * idx + 1
            spaces.append([j, j + 1])
        
        # Trivial output space (classical output)
        spaces.append([4 * nb_neighbors + 1])
        
        return spaces
    
    # Strategy generation and connector methods
    def _generate_all_strategies(self):
        """
        Generate random QCFO strategies for all parties.
        """
        for party in range(self.nb_parties):
            party_dims = self._build_party_dimensions(party)
            spaces = self._build_party_spaces(party)

            for input_val in range(self.nb_input[party]):
                # Create a single QCFO for all outputs given this input
                qcfo = QCFO(
                    is_numeric=True,
                    nb_elements=self.nb_output[party],  # Number of classical outputs
                    spaces=spaces,
                    dimensions=party_dims,
                    elements=None,  # Will be randomly generated
                    is_real=True   # Real matrices for efficiency
                )
                self.party_strategies[(party, input_val)] = qcfo


    def _create_identity_channel(self, sender: int, receiver: int):
        """
        Create quantum identity channel connecting sender to receiver.
        
        Args:
            sender: Index of party sending the message
            receiver: Index of party receiving the message
            
        Note:
            Uses Choi-Jamiołkowski representation for quantum channels
        """
        id_vec = pure_CJ(np.eye(self.message_dim))
        id_choi = np.outer(id_vec, id_vec.conj())

        out_space_idx = self.space_indices[(sender, sender, receiver)]
        in_space_idx = self.space_indices[(receiver, sender, receiver)]
        return id_choi, [out_space_idx, in_space_idx]
    
    def connection_process(self):
        """
        Get or compute the connection process matrix.
        
        The connection process implements all quantum communication channels
        between connected parties simultaneously. It's the tensor product of
        identity channels for all edges in the communication graph.

        Returns:
            numpy.ndarray: Global connection process matrix
        """ 
        #This could be cached
        connection_matrix = None
        connection_spaces = []
        for party in range(self.nb_parties):
            for neighbor in self.connections.neighbors(party):
                channel, spaces = self._create_identity_channel(party, neighbor)
                if connection_matrix is None:
                    connection_matrix = channel
                    connection_spaces = spaces
                else:
                    connection_matrix, connection_spaces = link_product(connection_matrix, channel, connection_spaces, spaces, self.global_dims)
        return connection_matrix, connection_spaces
    
    # Update method for individual party strategies and shared state
    
    def update_strategy(self, party: int, input_val: int, qcfo: QCFO):
        """
        Update strategy for a specific party and input.
        
        Args:
            party: Index of party whose strategy to update
            input_val: Input value for which to update the strategy
            qcfo: New QCFO instrument to use
            
        """
        if qcfo.nb_elements != self.nb_output[party]:
            raise ValueError(f"Strategy must have {self.nb_output[party]} outputs")
        
        expected_dims = self._build_party_dimensions(party)
        if qcfo.dimensions != expected_dims:
            raise ValueError("Strategy dimensions don't match party structure")
        
        self.party_strategies[(party, input_val)] = qcfo
    
    def update_shared_state(self, state: np.ndarray):
        """
        Update the shared quantum state between parties.s.
        
        Args:
            state: New quantum state as density matrix
        """
        expected_dim = np.prod(self.shared_state_dim)
        if state.shape != (expected_dim, expected_dim):
            raise ValueError(f"State must be {expected_dim}x{expected_dim}")
        
        self.shared_state = state

    # Probability and QSW computation.

    def probability(self, outputs: list[int], inputs: list[int]) -> float:
        """
        Compute probability P(outputs|inputs) for the quantum strategy.
        """ 
        matrix, spaces = self.connection_process() # May be None, [] if no communication.
        for p in range(self.nb_parties):
            qcfo = self.party_strategies[(p, inputs[p])]
            strategy_matrix = qcfo.elements[outputs[p]]
            strategy_spaces = self.global_indexes[p]
            if matrix is not None:
                matrix, spaces = link_product(matrix, strategy_matrix, spaces, strategy_spaces, self.global_dims)
            else:
                matrix, spaces = strategy_matrix, strategy_spaces
        
        state_spaces = [self.space_indices[(p, -1, -1)] for p in range(self.nb_parties)]
        matrix, spaces = link_product(matrix, self.shared_state, spaces, state_spaces, self.global_dims)
        return matrix[0][0].real        
    
    def compute_qsw(self, game: Game) -> float:
        """
        Compute quantum social welfare (expected payoff) for a given game.
        
        Args:
            game: Game object defining the cooperative task
            
        Returns:
            float: Expected total payoff (quantum social welfare)
        """
        if game.nbPlayers != self.nb_parties:
            raise ValueError("Game must have same number of players as strategy")
        
        qsw = 0.0
        # Iterate over all possible input profiles
        for question in game.questions():
            prob_q = game.qDistrib(question)
            
            # Iterate over all possible output profiles
            for answer in itertools.product(*[range(game.list_num_out[k]) 
                                            for k in range(game.nbPlayers)]):
                if game.validAnswer(answer, question):
                    payout = game.answerPayoutWin(answer, question)
                    strategy_prob = self.probability(list(answer), list(question))
                    qsw += prob_q * payout * strategy_prob
                    
        return qsw
    
    def add_noise(self, noise_level: float):
        """
        Add white noise to all party strategies.
        
        Args:
            noise_level: Amount of white noise to add (0 = none, 1 = full noise)
        """

        # Add noise to each party's QCFO elements
        for k in range(self.nb_parties):
            for x in range(self.nb_input[k]):
                qcfo = self.party_strategies[(k, x)]
                noisy_elements = []
                noise = QCFO(
                    is_numeric=True,
                    nb_elements=self.nb_output[k],
                    spaces= deepcopy(qcfo.spaces),
                    dimensions= deepcopy(qcfo.dimensions),
                    elements=None,  # Will be randomly generated
                    is_real=True
                )

                for a in range(qcfo.nb_elements):
                    elem = qcfo.elements[a]
                    noisy_elem = (1 - noise_level) * elem + noise_level * noise.elements[a]
                    noisy_elements.append(noisy_elem)
                qcfo.elements = noisy_elements
                self.party_strategies[(k, x)] = qcfo

        # Add noise to shared state
        self.shared_state = (1 - noise_level) * self.shared_state + noise_level * random_density_matrix(self.shared_state.shape[0], is_real=True)


if __name__ == "__main__":
    # Example: Two-party connected graph for CHSH-type games
    graph = nx.Graph()
    graph.add_edge(0, 1)
    
    strat = QuantumStrategy(
        nb_parties=2, 
        nb_input=[2,2],   # Binary inputs for each party
        nb_output=[2,2],  # Binary outputs for each party
        message_dim=2,    # Qubit messages
        shared_state_dim=[4,4],  # 4-dimensional shared systems
        connections=graph
    )

    # Check normalization (probabilities sum to 1)
    total_prob = sum(strat.probability([a0, a1], [1, 1]) 
                    for a0 in range(2) for a1 in range(2))
    print(f"Total probability: {total_prob:.6f}")
        
    # CHSH game example
    func_utility = lambda out_tuple, in_tuple: int(in_tuple[0] & in_tuple[1] == out_tuple[0] ^ out_tuple[1])
    func_in_prior = lambda in_tuple: 1/4
    
    chsh_game = Game(2, [2, 2], [2, 2], [func_utility] * 2, func_in_prior)
    qsw = strat.compute_qsw(chsh_game)
    print(f"QSW CHSH: {qsw:.6f}")