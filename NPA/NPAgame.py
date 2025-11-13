from NPA.hierarchy import Hierarchy
import itertools as it
from NPA.operator import Operator
from typing import Optional, Callable
import numpy as np

class NPAgame:
    def __init__(self, num_players: int, list_num_in: list[int], list_num_out: list[int], funcs_utility_player:Optional[list[Callable]]=None, func_in_prior:Optional[Callable] = None, correlators:Optional[dict] = None, perturbation:Optional[dict] = None, merged:Optional[bool]=None, input_sharing:Optional[tuple]=None, scale: Optional[float] = None):
        self.nbPlayers = num_players
        self.list_num_in = list_num_in
        self.list_num_out = list_num_out
        self.func_in_prior = func_in_prior        
        self.is_correlator_mode = correlators is not None
        self.perturbation = perturbation
        self.is_merged = merged
        self.input_sharing = input_sharing
        self.scale = scale

        if func_in_prior is None:
            self.func_in_prior = lambda in_tuple: 1/np.prod(self.list_num_in)
        else:
            self.func_in_prior = func_in_prior
        
        if self.is_correlator_mode:
            self.correlators = correlators
        else:
            self.funcs_utility_player = funcs_utility_player
        
    def qDistrib(self, question):
        return self.func_in_prior(question)
    
    def questions(self):
        # Iterates over all possible questions
        for q in it.product(*[list(range(self.list_num_in[i])) for i in range(self.nbPlayers)]):
            if self.func_in_prior(q) != 0:
                yield q
    
    def validAnswer(self, answer, question):
        # A (answer, question) pair is valid if the utility function of at least one player is non-zero

        if not self.is_correlator_mode:
            for i in range(self.nbPlayers):
                if self.funcs_utility_player[i](answer, question) != 0:
                    return True
        else:
            return True
    
    def validAnswerIt(self, question):
        for answer in it.product(*[list(range(self.list_num_out[i])) for i in range(self.nbPlayers)]):
            if self.validAnswer(answer, question):
                yield answer

    def involvedPlayers(self,question):
        # This function may be used to define games where some players are not involved depending on the question
        return [i for i in range(self.nbPlayers)]
    
    def wrongAnswerIt(self, question):
        for answer in it.product(self.list_num_out, repeat=self.nbplayers):
            if not self.validAnswer(answer, question):
                yield answer
    
    def playerPayoutWin(self, answer, question, playerId):
        # Payout for a specific player
        if not self.is_correlator_mode:
            return self.funcs_utility_player[playerId](answer, question)
        else:
            self.answerPayoutWin(answer, question)

    def notPlayerPayoutWin(self, answer, question, playerId):
        # For the implementation of the Nash equilibrium constraints
        answer[playerId] = 1 - answer[playerId]        
        return self.playerPayoutWin(answer, question, playerId=playerId)
    
    def answerPayoutWin(self, answer, question):
        # Mean payout of all player for an answer        
        return sum([self.funcs_utility_player[i](answer, question) for i in range(self.nbPlayers)]) / self.nbPlayers 
    
    def merge(self, i, j):
        """ Merge parties i and j and return the corresponding bipartite game"""

        # helpers
        def decode_input(input):
            (x, y) = input
            #(x_i, x_j)
            return (x % n_i, x // n_i, y)

        def decode_output(output):
            (a,b) = output
            #(a_i, a_j)
            return (a % m_i, a // m_i, b)            
                    
        k = 3 - i - j
        n_i = self.list_num_in[i]
        n_j = self.list_num_in[j]
        n_k = self.list_num_in[k]

        m_i = self.list_num_out[i]
        m_j = self.list_num_out[j]
        m_k = self.list_num_out[k]

        def utility_func(out_tuple, in_tuple):
            out_ = [0, 0, 0]
            in_ = [0, 0, 0]
            dec_out_tuple = decode_output(out_tuple)
            dec_in_tuple = decode_input(in_tuple)

            out_[i] = dec_out_tuple[0]
            out_[j] = dec_out_tuple[1]
            out_[k] = dec_out_tuple[-1]

            in_[i] = dec_in_tuple[0]
            in_[j] = dec_in_tuple[1]
            in_[k] = dec_in_tuple[-1]
            
            return self.funcs_utility_player[0](tuple(out_), tuple(in_))
    
        func_in_prior = lambda out_tuple: 1/8

        return NPAgame(
            num_players=2,
            list_num_in=[n_i * n_j, n_k],
            list_num_out=[m_i * m_j, m_k],
            funcs_utility_player=[utility_func]*2,
            func_in_prior=func_in_prior
        )
    
    def reduce_to_foward(self, i, j):
        """
        Forwarding strategies, obtained by enlarging input spaces of agents i,j to n_i * n_j,
        and constraining them to always receive the same joint input w = (x_i, x_j).
        """

        # helpers
        def decode_input(input):
            (x, y) = input
            #(x_i, x_j)
            return (x % n_i, x // n_i, y) 
                
        if i > j:
            i, j = j, i
        
        k = 3 - i - j
        n_i = self.list_num_in[i]
        n_j = self.list_num_in[j]
        n_k = self.list_num_in[k]
        n_w = n_i * n_j
        
        # Build new_list_num_in respecting original positions
        new_list_num_in = [0, 0, 0]
        new_list_num_in[i] = n_w
        new_list_num_in[j] = n_w
        new_list_num_in[k] = n_k

        def utility_func(out_tuple, in_tuple):
            in_ = [0, 0, 0]
            in_tuple_decoded = decode_input((in_tuple[i], in_tuple[k]))
            in_[i] = in_tuple_decoded[0]
            in_[j] = in_tuple_decoded[1]
            in_[k] = in_tuple_decoded[2]
            return self.funcs_utility_player[0](out_tuple, in_tuple_decoded)

        def func_in_prior(in_tuple):
            return (in_tuple[i] == in_tuple[j]) / (n_i * n_j * n_k)
            
        return NPAgame(
            num_players=3,
            list_num_in=new_list_num_in,
            list_num_out=self.list_num_out,
            funcs_utility_player=[utility_func]*3,
            func_in_prior=func_in_prior
        )            
    
    def optimize(self, level=1, getVariable=False, Nash=False, verbose=False, warmStart=False, solver="MOSEK"):
        '''
        Optimize the game using the specified solver.
        '''

        # Set up the optimization problem        
        prob = Hierarchy(self, self.create_operators(), level)

        if Nash:
            assert(all(self.list_num_in) == 2), "Nash equilibrium constraints are only implemented for binary questions."
            assert(all(self.list_num_out) == 2), "Nash equilibrium constraints are only implemented for binary answers."
            prob.setNashEqConstraints()

        qsw = prob.optimize(verbose, warmStart, solver)
            
        if getVariable:
            return qsw, prob.X
        return qsw
    
    def get_variables(self):
        return 
    
    def create_operators(self):
        '''
        Create all measurement operators for each player.
        Note that the identity operator is included for each player, and therefore, each POVM is defined with one less element.
        '''
        operators = []
        for p in range(self.nbPlayers):
            player_ops = [Operator.identity()]
            for q in range(self.list_num_in[p]):
                for a in range(self.list_num_out[p] - 1):
                    player_ops.append(Operator(p, q, a))
            operators.append(player_ops)
        return operators    
