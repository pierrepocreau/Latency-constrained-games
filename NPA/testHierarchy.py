
from itertools import product
import unittest
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from NPA.NPAgame import NPAgame
from NPA.hierarchy import Hierarchy
from NPA.Operator import Operator
from NPA.canonicalOp import *
class Test(unittest.TestCase):

    def testCanonicalForm(self):
        operatorsP1 = [Operator.identity(), Operator(1, 0, 0), Operator(1, 1, 0)]
        operatorsP2 = [Operator.identity(), Operator(2, 0, 0), Operator(2, 1, 0)]
        operatorsP3 = [Operator.identity(), Operator(3, 0, 0), Operator(3, 1, 0)]
        
        P3 = [operatorsP1, operatorsP2, operatorsP3]
        S = [Monomial(s) for s in product(*P3)]

        proba = Monomial(S[3].canonicalRep + S[10].canonicalRep)
        proba3 = Monomial(S[8].canonicalRep + S[10].canonicalRep)

        #Test cannonic form and projection
        self.assertEqual(proba, Monomial([Operator(1, 0, 0), Operator(2, 0, 0), Operator(3, 0, 0), Operator.identity(), Operator.identity(), Operator.identity()]))
        self.assertEqual(proba3, Monomial([Operator(1, 0, 0), Operator(2, 1, 0), Operator(3, 0, 0), Operator(3, 1, 0), Operator.identity(), Operator.identity()]))

        #Test symetric
        self.assertEqual(Monomial(S[13].canonicalRep + S[24].canonicalRep), Monomial(S[24].canonicalRep + S[13].canonicalRep))
        self.assertNotEqual(Monomial(S[2].canonicalRep + S[10].canonicalRep), proba)

        #Test Id
        self.assertEqual(Monomial(S[10].canonicalRep + S[10].canonicalRep), Monomial(S[0].canonicalRep + S[10].canonicalRep))

    def testGamePOVMs(self):
        operatorsP1 = [Operator.identity(), Operator(0, 0, 0), Operator(0, 1, 0)]
        operatorsP2 = [Operator.identity(), Operator(1, 0, 0), Operator(1, 1, 0)]
        operatorsP3 = [Operator.identity(), Operator(2, 0, 0), Operator(2, 1, 0)]
        P3 = [operatorsP1, operatorsP2, operatorsP3]

        list_num_in = [2,2,2]
        list_num_out = [2,2,2]
        func_utility = lambda out_tuple, in_tuple: int(in_tuple[0] | in_tuple[1] | in_tuple[2] == out_tuple[0] ^ out_tuple[1] ^ out_tuple[2])
        func_in_prior = lambda in_tuple: 1/4 if in_tuple[0] ^ in_tuple[1] ^ in_tuple[2] == 0 else 0
        num_players = 3

        NPAghz = NPAgame(num_players, list_num_in, list_num_out, [func_utility]*num_players, func_in_prior)
        Sgame = NPAghz.create_operators()
        self.assertEqual(Sgame, P3)     

    def testMatrixCreation(self):
        list_num_in = [2,2,2]
        list_num_out = [2,2,2]
        func_utility = lambda out_tuple, in_tuple: int(in_tuple[0] | in_tuple[1] | in_tuple[2] == out_tuple[0] ^ out_tuple[1] ^ out_tuple[2])
        func_in_prior = lambda in_tuple: 1/4 if in_tuple[0] ^ in_tuple[1] ^ in_tuple[2] == 0 else 0
        num_players = 3

        NPAghz = NPAgame(num_players, list_num_in, list_num_out, [func_utility]*num_players, func_in_prior)
        operators = NPAghz.create_operators()
        sdp = Hierarchy(NPAghz, operators)
        matrix = sdp.projectorConstraints()

        self.assertListEqual(list(matrix[0,:]), list(range(27)))
        self.assertListEqual(list(matrix.diagonal()), list(range(27)))

    def testGenVec(self):
        # The monomialList should be set as a class, such that when testing for equality,  we test for the equality of each operator.
        
        list_num_in = [2,2,2]
        list_num_out = [2,2,2]
        func_utility = lambda out_tuple, in_tuple: int(in_tuple[0] | in_tuple[1] | in_tuple[2] == out_tuple[0] ^ out_tuple[1] ^ out_tuple[2])
        func_in_prior = lambda in_tuple: 1/4 if in_tuple[0] ^ in_tuple[1] ^ in_tuple[2] == 0 else 0
        num_players = 3

        NPAghz = NPAgame(num_players, list_num_in, list_num_out, [func_utility]*num_players, func_in_prior)
        operators = NPAghz.create_operators()
        sdp = Hierarchy(NPAghz, operators)

        encodingVec = sdp.genVec("000", "111")
        correct = [0] * 27
        correct[sdp.monomialList.index(Monomial([Operator(0, 1, 0), Operator(1, 1, 0), Operator(2, 1, 0)]))] = 1
        self.assertListEqual(encodingVec, correct)

        encodingVec = sdp.genVec("000", "010")
        correct = [0] * 27
        correct[sdp.monomialList.index(Monomial([Operator(0, 0, 0), Operator(1, 1, 0), Operator(2, 0, 0)]))] = 1
        self.assertListEqual(encodingVec, correct)

        encodingVec = sdp.genVec("100", "010")
        correct = [0] * 27
        # P(100 | 010) = P(I00|010) - P(000|010)
        correct[sdp.monomialList.index(Monomial([Operator.identity(), Operator(1, 1, 0), Operator(2, 0, 0)]))] = 1
        correct[sdp.monomialList.index(Monomial([Operator(0, 0, 0), Operator(1, 1, 0), Operator(2, 0, 0)]))] = -1
        self.assertListEqual(encodingVec, correct)

        encodingVec = sdp.genVec("110", "010")
        correct = [0] * 27
        #P(110|010) = P(II0|010) - P(OOO|O1O) - P(010|010) - P(100|010) = ...
        correct[sdp.monomialList.index(Monomial([Operator.identity(), Operator.identity(), Operator(2, 0, 0)]))] = 1
        correct[sdp.monomialList.index(Monomial([Operator(0, 0, 0), Operator(1, 1, 0), Operator(2, 0, 0)]))] += -1
        correct[sdp.monomialList.index(Monomial([Operator.identity(), Operator(1, 1, 0), Operator(2, 0, 0)]))] += -1
        correct[sdp.monomialList.index(Monomial([Operator(0, 0, 0), Operator(1, 1, 0), Operator(2, 0, 0)]))] += 1
        correct[sdp.monomialList.index(Monomial([Operator(0, 0, 0), Operator.identity(), Operator(2, 0, 0)]))] += -1
        correct[sdp.monomialList.index(Monomial([Operator(0, 0, 0), Operator(1, 1, 0), Operator(2, 0, 0)]))] += 1

        self.assertListEqual(encodingVec, correct)

    def testCHSH(self):
        num_players = 2
        list_num_in = [2,2]
        list_num_out = [2,2]
        func_utility = lambda out_tuple, in_tuple: int(in_tuple[0] & in_tuple[1] == out_tuple[0] ^ out_tuple[1])
        func_in_prior = lambda in_tuple: 1/4

        NPAchsh = NPAgame(num_players, list_num_in, list_num_out, [func_utility]*num_players, func_in_prior)
        upperBound = NPAchsh.optimize(level=2, Nash=False, verbose=True, warmStart=False, solver="SCS")
        self.assertAlmostEqual(upperBound, np.cos(np.pi/8)**2, delta=1e-5)

    def testCHSH_Modif(self):
        num_players = 2
        list_num_in = [2,2]
        list_num_out = [2,2]
        func_utility = lambda out_tuple, in_tuple: int(in_tuple[0] & in_tuple[1] == out_tuple[0] ^ out_tuple[1])
        dict_in_prior = {(0,0): 0.1, (0,1): 0.2, (1,0): 0.3, (1,1): 0.4}
        func_in_prior = lambda in_tuple: dict_in_prior[in_tuple] if in_tuple in dict_in_prior else 0

        NPAchsh = NPAgame(num_players, list_num_in, list_num_out, [func_utility]*num_players, func_in_prior)
        upperBound = NPAchsh.optimize(level=2, Nash=False, verbose=True, warmStart=False, solver="SCS")
        self.assertAlmostEqual(upperBound, 0.9005, delta=1e-4)

    def testGHZ(self):
        num_players = 3
        list_num_in = [2,2,2]
        list_num_out = [2,2,2]
        func_utility = lambda out_tuple, in_tuple: int(in_tuple[0] | in_tuple[1] | in_tuple[2] == out_tuple[0] ^ out_tuple[1] ^ out_tuple[2])
        func_in_prior = lambda in_tuple: 1/4 if in_tuple[0] ^ in_tuple[1] ^ in_tuple[2] == 0 else 0
        NPAghz = NPAgame(num_players, list_num_in, list_num_out, [func_utility]*num_players, func_in_prior)
        upperBound = NPAghz.optimize(level=2, Nash=False, verbose=True, warmStart=False, solver="MOSEK")
        self.assertAlmostEqual(upperBound, 1, delta=1e-4)

    def testCanonicalForm_3_output(self):
        operatorsP1 = [Operator.identity(), Operator(0, 0, 0), Operator(0, 0, 1), Operator(0, 1, 0), Operator(0, 1, 1)]
        operatorsP2 = [Operator.identity(), Operator(1, 0, 0), Operator(1, 0, 1), Operator(1, 1, 0), Operator(1, 1, 1)]
        operatorsP3 = [Operator.identity(), Operator(2, 0, 0), Operator(2, 0, 1), Operator(2, 1, 0), Operator(2, 1, 1)]
        P3 = [operatorsP1, operatorsP2, operatorsP3]
        S = [Monomial(s) for s in product(*P3)]
        
        proba = Monomial(S[3].canonicalRep + S[93].canonicalRep)
        proba2 = Monomial(S[93].canonicalRep + S[31].canonicalRep)

        #Test cannonic form and projection
        self.assertEqual(proba, Monomial([Operator(0, 1, 0), Operator(1, 1, 0), Operator(2, 1, 0), Operator.identity(), Operator.identity(), Operator.identity()]))
        self.assertEqual(proba2, Monomial([Operator(0, 0, 0), Operator(0, 1, 0), Operator(1, 0, 0), Operator(1, 1, 0), Operator(2, 0, 0), Operator(2, 1, 0)]))

        #Test symetric
        self.assertEqual(Monomial(S[13].canonicalRep + S[24].canonicalRep), Monomial(S[24].canonicalRep + S[13].canonicalRep))
        self.assertNotEqual(Monomial(S[2].canonicalRep + S[93].canonicalRep), proba)

        #Test Id
        self.assertEqual(Monomial(S[10].canonicalRep + S[10].canonicalRep), Monomial(S[0].canonicalRep + S[10].canonicalRep))        

    def testGamePOVMs_3_output(self):
        operatorsP1 = [Operator.identity(), Operator(0, 0, 0), Operator(0, 0, 1), Operator(0, 1, 0), Operator(0, 1, 1)]
        operatorsP2 = [Operator.identity(), Operator(1, 0, 0), Operator(1, 0, 1), Operator(1, 1, 0), Operator(1, 1, 1)]
        operatorsP3 = [Operator.identity(), Operator(2, 0, 0), Operator(2, 0, 1), Operator(2, 1, 0), Operator(2, 1, 1)]
        P3 = [operatorsP1, operatorsP2, operatorsP3]

        list_num_in = [2,2,2]
        list_num_out = [3,3,3]
        func_utility = lambda out_tuple, in_tuple: int(in_tuple[0] | in_tuple[1] | in_tuple[2] == out_tuple[0] ^ out_tuple[1] ^ out_tuple[2])
        func_in_prior = lambda in_tuple: 1/4 if in_tuple[0] ^ in_tuple[1] ^ in_tuple[2] == 0 else 0
        num_players = 3

        NPAghz = NPAgame(num_players, list_num_in, list_num_out, [func_utility]*num_players, func_in_prior)
        Sgame = NPAghz.create_operators()
        self.assertEqual(Sgame, P3)     

    def testMatrixCreation_3_output(self):
        list_num_in = [2,2,2]
        list_num_out = [3,3,3]
        func_utility = lambda out_tuple, in_tuple: int(in_tuple[0] | in_tuple[1] | in_tuple[2] == out_tuple[0] ^ out_tuple[1] ^ out_tuple[2])
        func_in_prior = lambda in_tuple: 1/4 if in_tuple[0] ^ in_tuple[1] ^ in_tuple[2] == 0 else 0
        num_players = 3

        NPAghz = NPAgame(num_players, list_num_in, list_num_out, [func_utility]*num_players, func_in_prior)
        operators = NPAghz.create_operators()
        sdp = Hierarchy(NPAghz, operators)
        matrix = sdp.projectorConstraints()

        self.assertListEqual(list(matrix[0,:]), list(range(125)))
        self.assertListEqual(list(matrix.diagonal()), list(range(125)))
    

    def testGenVec_3_input(self):
        # The monomialList should be set as a class, such that when testing for equality,  we test for the equality of each operator.
        
        list_num_in = [2,2,2]
        list_num_out = [3,3,3]
        func_utility = lambda out_tuple, in_tuple: int(in_tuple[0] | in_tuple[1] | in_tuple[2] == out_tuple[0] ^ out_tuple[1] ^ out_tuple[2])
        func_in_prior = lambda in_tuple: 1/4 if in_tuple[0] ^ in_tuple[1] ^ in_tuple[2] == 0 else 0
        num_players = 3

        NPAghz = NPAgame(num_players, list_num_in, list_num_out, [func_utility]*num_players, func_in_prior)
        operators = NPAghz.create_operators()
        sdp = Hierarchy(NPAghz, operators)

        encodingVec = sdp.genVec("000", "111")
        correct = [0] * 125
        correct[sdp.monomialList.index(Monomial([Operator(0, 1, 0), Operator(1, 1, 0), Operator(2, 1, 0)]))] = 1
        self.assertListEqual(encodingVec, correct)

        encodingVec = sdp.genVec("100", "010")
        correct = [0] * 125
        correct[sdp.monomialList.index(Monomial([Operator(0, 0, 1), Operator(1, 1, 0), Operator(2, 0, 0)]))] = 1
        self.assertListEqual(encodingVec, correct)

        encodingVec = sdp.genVec("200", "010")
        correct = [0] * 125
        # P(200 | 010) = P(I00|010) - P(000|010) - P(100|010)
        correct[sdp.monomialList.index(Monomial([Operator.identity(), Operator(1, 1, 0), Operator(2, 0, 0)]))] = 1
        correct[sdp.monomialList.index(Monomial([Operator(0, 0, 0), Operator(1, 1, 0), Operator(2, 0, 0)]))] = -1
        correct[sdp.monomialList.index(Monomial([Operator(0, 0, 1), Operator(1, 1, 0), Operator(2, 0, 0)]))] = -1
        self.assertListEqual(encodingVec, correct)

        encodingVec = sdp.genVec("220", "010")
        correct = [0] * 125
        #P(220|010) = P(I20|010) - P(020|010) - P(120|010) 
        # = (P(II0|010) - P(I10|010) - P(I00|010)) - (P(0I0|010) - P(010|010) - P(000|010)) - (P(1I0|010) - P(110|010) - P(100|010))
        correct[sdp.monomialList.index(Monomial([Operator.identity(), Operator.identity(), Operator(2, 0, 0)]))] += 1
        correct[sdp.monomialList.index(Monomial([Operator.identity(), Operator(1, 1, 1), Operator(2, 0, 0)]))] += -1
        correct[sdp.monomialList.index(Monomial([Operator.identity(), Operator(1, 1, 0), Operator(2, 0, 0)]))] += -1

        correct[sdp.monomialList.index(Monomial([Operator(0, 0, 0), Operator.identity(), Operator(2, 0, 0)]))] += -1
        correct[sdp.monomialList.index(Monomial([Operator(0, 0, 0), Operator(1, 1, 1), Operator(2, 0, 0)]))] += 1
        correct[sdp.monomialList.index(Monomial([Operator(0, 0, 0), Operator(1, 1, 0), Operator(2, 0, 0)]))] += 1

        correct[sdp.monomialList.index(Monomial([Operator(0, 0, 1), Operator.identity(), Operator(2, 0, 0)]))] += -1
        correct[sdp.monomialList.index(Monomial([Operator(0, 0, 1), Operator(1, 1, 1), Operator(2, 0, 0)]))] += 1
        correct[sdp.monomialList.index(Monomial([Operator(0, 0, 1), Operator(1, 1, 0), Operator(2, 0, 0)]))] += 1
        self.assertListEqual(encodingVec, correct)        

    def test_isLast(self):
        operator0 = Operator(0, 0, 0)
        operator2 = Operator(0, 0, 2)

        self.assertFalse(operator0.is_last(3))
        self.assertTrue(operator2.is_last(3))


    def testCGLMP(self):
        num_players = 2
        list_num_in = [2,2]
        d = 3
        list_num_out = [d,d]

        def func_utility(out_tuple, in_tuple):
            a = out_tuple[0]
            b = out_tuple[1]  
            x = in_tuple[0]   
            y = in_tuple[1]   
            utility = 0

            for k in range(d // 2):
                weight = 1 - (2 * k) / (d - 1)
                                
                if x == 0 and y == 0:  # A1 vs B1
                    if (a - b) % d == k:  # A1 = B1 + k
                        utility += weight
                elif x == 1 and y == 0:  # A2 vs B1
                    if (b - a) % d == (k + 1) % d:  # B1 = A2 + k + 1
                        utility += weight
                elif x == 1 and y == 1:  # A2 vs B2
                    if (a - b) % d == k:  # A2 = B2 + k
                        utility += weight
                elif x == 0 and y == 1:  # A1 vs B2
                    if (b - a) % d == k:  # B2 = A1 + k
                        utility += weight
                
                if x == 0 and y == 0:  # A1 vs B1
                    if (a - b) % d == (-k - 1) % d:  # A1 = B1 - k - 1
                        utility -= weight
                elif x == 1 and y == 0:  # A2 vs B1
                    if (b - a) % d == (-k) % d:  # B1 = A2 - k
                        utility -= weight
                elif x == 1 and y == 1:  # A2 vs B2
                    if (a - b) % d == (-k - 1) % d:  # A2 = B2 - k - 1
                        utility -= weight
                elif x == 0 and y == 1:  # A1 vs B2
                    if (b - a) % d == (-k - 1) % d:  # B2 = A1 - k - 1
                        utility -= weight
            
            return utility
    
                
        func_in_prior = lambda in_tuple: 1

        NPAchsh = NPAgame(num_players, list_num_in, list_num_out, [func_utility]*num_players, func_in_prior)
        upperBound, X = NPAchsh.optimize(level=3, getVariable=True, Nash=False, verbose=True, warmStart=False, solver="MOSEK")
        self.assertAlmostEqual(upperBound, 2.9149, delta=1e-4)

    def testCHSHCorrelators(self):
        num_players = 2
        list_num_in = [2,2]
        list_num_out = [2,2]
        correlators = {"xy": np.array([[1, 1], [1, -1]])}

        NPAchsh = NPAgame(num_players, list_num_in, list_num_out, correlators=correlators)
        upperBound = NPAchsh.optimize(level=2, Nash=False, verbose=True, warmStart=False, solver="SCS")
        self.assertAlmostEqual(upperBound, 2*np.sqrt(2), delta=1e-5)

    def testGHZCorrelators(self):
        num_players = 3
        list_num_in = [2,2,2]
        list_num_out = [2,2,2]
        correlators={"xyz": np.array([[[1, 0], 
                          [0, -1]],
                         [[0, -1], 
                          [-1, 0]]])
        }

        NPAghz = NPAgame(num_players, list_num_in, list_num_out, correlators=correlators)
        upperBound = NPAghz.optimize(level=2, Nash=False, verbose=True, warmStart=False, solver="MOSEK")
        self.assertAlmostEqual(upperBound, 4, delta=1e-5)

if __name__ == "__main__":
    unittest.main()
