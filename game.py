from NPA.hierarchy import Hierarchy
import itertools as it
from NPA.operator import Operator
from typing import Optional, Callable
import networkx as nx
import numpy as np
import cvxpy as cv
import copy

class Game:
    def __init__(self, num_players: int, list_num_in: list[int], list_num_out: list[int], funcs_utility_player:Optional[list[Callable]]=None, func_in_prior:Optional[Callable] = None, correlators:Optional[dict] = None, perturbation:Optional[dict] = None, merged:Optional[bool]=None, input_sharing:Optional[tuple]=None, scale: Optional[float] = None):
        self.nbPlayers = num_players
        self.list_num_in = list_num_in
        self.list_num_out = list_num_out
        self.func_in_prior = func_in_prior        
        self.perturbation = perturbation
        self.is_merged = merged
        self.input_sharing = input_sharing
        self.scale = scale

        if func_in_prior is None:
            self.func_in_prior = lambda in_tuple: 1/np.prod(self.list_num_in)
        else:
            self.func_in_prior = func_in_prior
        
        self.funcs_utility_player = funcs_utility_player

        self.list_in = list(it.product(*[range(n) for n in list_num_in]))
        self.list_out = list(it.product(*[range(n) for n in list_num_out]))
        self.list_det_strategies = []        
        
    def qDistrib(self, question):
        return self.func_in_prior(question)
    
    def questions(self):
        # Iterates over all possible questions
        for q in it.product(*[list(range(self.list_num_in[i])) for i in range(self.nbPlayers)]):
            if self.func_in_prior(q) != 0:
                yield q
    
    def validAnswer(self, answer, question):
        # A (answer, question) pair is valid if the utility function of at least one player is non-zero

        for i in range(self.nbPlayers):
            if self.funcs_utility_player[i](answer, question) != 0:
                return True
    
    def validAnswerIt(self, question):
        for answer in it.product(*[list(range(self.list_num_out[i])) for i in range(self.nbPlayers)]):
            if self.validAnswer(answer, question):
                yield answer
    
    def AnswerIt(self):
        for answer in it.product(*[list(range(self.list_num_out[i])) for i in range(self.nbPlayers)]):            
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
        return self.funcs_utility_player[playerId](answer, question)

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
    
        func_in_prior = lambda in_tuple: 1/8

        return Game(
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
            
        return Game(
            num_players=3,
            list_num_in=new_list_num_in,
            list_num_out=self.list_num_out,
            funcs_utility_player=[utility_func]*3,
            func_in_prior=func_in_prior
        )   

    def line_forward(self):
        """
        Forwarding strategies for v_0 <-> v_1 <-> v_2, binary inputs.
        Player 1's input is (y, yx, yz) with yx forwarded from 0, yz forwarded from 2.
        """

        def decode_2(x):
            return (x % 2, x // 2)

        def decode_3(x):
            return (x % 2, (x // 2) % 2, x // 4)

        new_list_num_in = [4, 8, 4]

        def utility_func(out_tuple, in_tuple):
            x, xy = decode_2(in_tuple[0])
            y, yx, yz = decode_3(in_tuple[1])
            z, zy = decode_2(in_tuple[2])
            return self.funcs_utility_player[0](out_tuple, (x, y, z))

        def func_in_prior(in_tuple):
            x, xy = decode_2(in_tuple[0])
            y, yx, yz = decode_3(in_tuple[1])
            z, zy = decode_2(in_tuple[2])
            return ((xy == y) & (yx == x) & (yz == z) & (zy == y)) / 8

        return Game(
            num_players=3,
            list_num_in=new_list_num_in,
            list_num_out=self.list_num_out,
            funcs_utility_player=[utility_func] * 3,
            func_in_prior=func_in_prior
        )
    
    def compute_NPA(self, level=1, getVariable=False, Nash=False, verbose=False, warmStart=False, solver="MOSEK"):
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
    
    """ Computation G-signaling value """

    def g_signaling(self, graph: nx.Graph):
        """ Compute the G-signaling value for a given graph """
        probs = cv.Variable((np.prod(self.list_num_out), np.prod(self.list_num_in)))
        
        constraints = []
        dict_access = {}
        for i, q in enumerate(self.questions()):                    
            sum_answers = 0
            for j, a in enumerate(self.AnswerIt()):        
                dict_access[(tuple(a), tuple(q))] = probs[j, i]
                sum_answers += probs[j, i]

                # Constraints for valid probabilities
                constraints.append(probs[j, i] >= 0)
                constraints.append(probs[j, i] <= 1)
            constraints.append(sum_answers == 1)
        
        for i in range(self.nbPlayers):
            out_neighbors = list(graph.neighbors(i)) + [i]
            non_out_neighbors = [k for k in range(self.nbPlayers) if k not in out_neighbors]
            
            if len(non_out_neighbors) == 0:
                continue 

            # For each fixed question to all parties except i
            for q_others_vals in it.product(*[range(self.list_num_in[k]) for k in range(self.nbPlayers) if k != i]):
                question_base = [None] * self.nbPlayers
                idx = 0
                for k in range(self.nbPlayers):
                    if k != i:
                        question_base[k] = q_others_vals[idx]
                        idx += 1
                
                # Build all questions varying only s_i
                questions_i = []
                for q_i in range(self.list_num_in[i]):
                    q_ = question_base[:]
                    q_[i] = q_i
                    questions_i.append(tuple(q_))
                
                # Filter to only questions with non-zero prior
                valid_questions = [(k, q) for k, q in enumerate(questions_i) if self.func_in_prior(q) > 0]
                
                if len(valid_questions) < 2:
                    continue
                
                # For each fixed output config of non-out-neighbors
                for a_non_out in it.product(*[range(self.list_num_out[k]) for k in non_out_neighbors]):
                    values = {}
                    # Sum over outputs of out-neighbors
                    for a_out in it.product(*[range(self.list_num_out[k]) for k in out_neighbors]):
                        answer = [None] * self.nbPlayers
                        
                        for idx, k in enumerate(non_out_neighbors):
                            answer[k] = a_non_out[idx]
                        
                        for idx, k in enumerate(out_neighbors):
                            answer[k] = a_out[idx]
                        
                        answer_tuple = tuple(answer)
                        
                        # Accumulate for each valid question
                        for k, q in valid_questions:
                            if k not in values:
                                values[k] = 0
                            values[k] += dict_access[(answer_tuple, q)]
                    
                    # All marginals must be equal
                    first_key = valid_questions[0][0]
                    for k, q in valid_questions[1:]:
                        constraints.append(values[k] == values[first_key])
        
        obj = 0
        for question in self.questions():
            for answer in self.validAnswerIt(question):
                obj += self.func_in_prior(question) * self.answerPayoutWin(answer, question) * dict_access[(tuple(answer), tuple(question))]
        
        problem = cv.Problem(cv.Maximize(obj), constraints)
        problem.solve(solver='MOSEK', verbose=False)
                
        return obj.value
    
    
    """ Computation of best classical solution """

    def deterministic(self, arr_map):
        for i in range(self.nbPlayers):
            map = arr_map[i]
            assert len(map) == self.list_num_in[i]
        ave_utility = 0
        for in_tuple in self.list_in:
            out_tuple = tuple(arr_map[i][in_tuple[i]] for i in range(self.nbPlayers))
            ave_utility += self.func_in_prior(in_tuple=in_tuple) * self.answerPayoutWin(out_tuple, in_tuple)
        return ave_utility
    
    def gen_det_strategies(self):
        for i in range(self.nbPlayers):
            if i == 0:
                product = it.product(range(self.list_num_out[i]), repeat=self.list_num_in[i])
                product = list(product)
            elif i == 1:
                product = ((p,) + (item,) for p in product for item in it.product(
                    range(self.list_num_out[i]), repeat=self.list_num_in[i]))
                product = list(product)
            else:
                product = (p + (item,) for p in product for item in it.product(
                    range(self.list_num_out[i]), repeat=self.list_num_in[i]))
                product = list(product)
        self.list_det_strategies = list(product)

    def opt_classical(self):
        opt = np.iinfo(np.int64).min
        if self.list_det_strategies == []:
            self.gen_det_strategies()
        for arr_map in self.list_det_strategies:
            ave_utility = self.deterministic(arr_map)
            if ave_utility >= opt:
                opt_arr_map = arr_map
                opt = ave_utility
        return opt, opt_arr_map