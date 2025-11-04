import numpy as np
import itertools as it

class cdnp:
    def __init__(self, num_parties, list_num_in, list_num_out, func_utility=None, func_in_prior=None):
        self.num_parties = num_parties
        self.list_num_in = list_num_in
        self.list_num_out = list_num_out

        if func_in_prior is None:
            self.func_in_prior = lambda in_tuple: 1/np.prod(list_num_in)
        else:
            self.func_in_prior = func_in_prior

        self.func_utility = func_utility
        self.list_in = list(it.product(*[range(n) for n in list_num_in]))
        self.list_out = list(it.product(*[range(n) for n in list_num_out]))
        self.list_det_strategies = []

    def _utility(self, question, answer):
        return self.func_utility(in_tuple=question, out_tuple=answer)

    def reduce_to_foward(self, i, j):
        """
        Forwarding strategies, obtained by enlarging input spaces of agents i,j to n_i * n_j,
        and constraining them to always receive the same joint input w = (x_i, x_j).
        """
        def decode_input(input):
            (x, y) = input
            return (x % n_i, x // n_i, y) 
                
        if i > j:
            i, j = j, i
        
        k = 3 - i - j
        n_i = self.list_num_in[i]
        n_j = self.list_num_in[j]
        n_k = self.list_num_in[k]
        n_w = n_i * n_j
        
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
            return self.func_utility(out_tuple, in_tuple_decoded)

        def func_in_prior(in_tuple):
            return (in_tuple[i] == in_tuple[j]) / (n_i * n_j * n_k)
            
        return cdnp(
            num_parties=3,
            list_num_in=new_list_num_in,
            list_num_out=self.list_num_out,
            func_utility=utility_func,
            func_in_prior=func_in_prior
        )

    def deterministic(self, arr_map):
        for i in range(self.num_parties):
            map = arr_map[i]
            assert len(map) == self.list_num_in[i]
        ave_utility = 0
        for in_tuple in self.list_in:
            out_tuple = tuple(arr_map[i][in_tuple[i]] for i in range(self.num_parties))
            ave_utility += self.func_in_prior(in_tuple=in_tuple) * self._utility(in_tuple, out_tuple)
        return ave_utility
    
    def gen_det_strategies(self):
        for i in range(self.num_parties):
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