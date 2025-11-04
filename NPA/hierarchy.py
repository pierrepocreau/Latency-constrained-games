import cvxpy as cp
import numpy as np

from NPA.canonicalOp import Monomial
import itertools
from NPA.Operator import Operator

def reduce_monomial_list(monomialList):
    monomialList = list(filter(lambda x: x.isNull() == False, monomialList))
    monomialList = list(set(monomialList))  # Remove duplicates
    monomialList.sort()    
    return monomialList

def all_compatible(ops):
    '''
    Check if the monomial does not contain twice the same measurement for a same party.
    '''
    seen_measurements = {}
    for op in ops:
        if op.player not in seen_measurements:
            seen_measurements[op.player] = set()
        if op.question in seen_measurements[op.player]:
            return False  # same measurement appears twice
        seen_measurements[op.player].add(op.question)
    return True

class Hierarchy:
    """
    Class for the implementation of the modified NPA's hierarchy.
    """

    def __init__(self, game, operatorsPlayers, level = 1.5, otherMonomials=None):

        self.game = game
        self.level = level

        self.operatorsPlayers = operatorsPlayers
        self.monomialList = [Monomial(s) for s in itertools.product(*operatorsPlayers)] # 1 + AB + BC + ABC...

        self.generate_monomials_up_to_level(level)

        if otherMonomials != None:
            assert(type(otherMonomials) == list)
            size = max(map(lambda mon: len(mon), otherMonomials))
            self.monomialList += otherMonomials
            self.monomialList = reduce_monomial_list(self.monomialList, monomialSize=size)

        self.n = len(self.monomialList)
        self.variableDict = {}

        # Constraints and SDP variables.
        self.constraints = []
        self.X = cp.bmat(self.init_variables())

        self.constraints += [self.X >> 0] #SDP
        self.constraints += [self.X[0][0] == 1] #Normalization

        # Objectif function and cvxpy problem.
        self.objectifFunc = self.objectifFunctions()
        self.prob = cp.Problem(cp.Maximize(cp.sum(self.X[0] @ cp.bmat(self.objectifFunc))), self.constraints)

    def generate_monomials_up_to_level(self, level):
        """
        Automatically generate all monomials up to the given NPA level.
        """ 
        new_monomials = [Monomial(s) for s in itertools.product(*self.operatorsPlayers)] # 1 + AB + BC + ABC...

        # Loop over all monomial lengths from 1 up to 'level'
        for l in range(1, int(level) + 1):
            for player_combination in itertools.combinations_with_replacement(range(self.game.nbPlayers), l):
                # For this combination of players (e.g., [0, 0, 1] means A, A, B)
                operator_lists = [self.operatorsPlayers[p] for p in player_combination]
                for op_tuple in itertools.product(*operator_lists):
                    monome = list(op_tuple)

                    # Pad to match total number of players with identities
                    n = len(monome)
                    for _ in range(self.game.nbPlayers - n):
                        monome.append(Operator.identity())

                    if all_compatible(monome):
                        new_monomials.append(Monomial(monome))

        # Add and reduce duplicates
        self.monomialList = reduce_monomial_list(new_monomials)        

    def updateProb(self):
        """
        Function which update the cvxpy problem.
        """
        self.prob = cp.Problem(cp.Maximize(cp.sum(self.X[0] @ cp.bmat(self.objectifFunc))), self.constraints)

    def projectorConstraints(self):
        '''
        Create the matrix filled with the canonical representation of each element of the moment matrix.
        '''
        matrix = np.zeros((self.n, self.n))
        variableId = 0

        for i, Si in enumerate(self.monomialList):
            for j, Sj in enumerate(self.monomialList):
                var = Monomial(Si.canonicalRep + Sj.canonicalRep)

                if var.isNull():
                    matrix[i][j] = -1
                    continue
                
                if var not in self.variableDict:
                    # If no other element as the same canonical representation has *var*, a new SDP variable will be created.
                    self.variableDict[var] = variableId
                    variableId += 1

                matrix[i][j] = self.variableDict[var]
        return matrix

    def init_variables(self):
        """
        Initialise the cvxpy variables.
        """
        matrix = self.projectorConstraints()
        variablesDict = {}
        variable = [[None for i in range(self.n)] for j in range(self.n)]

        for line in range(self.n):
            for column in range(self.n):

                varId = matrix[line][column]
                if varId == -1:
                    variable[line][column] = cp.Constant(0)
                    continue

                if varId not in variablesDict:
                    # One variable for each canonical element of the matrix of moments.
                    variablesDict[varId] = cp.Variable()

                variable[line][column] = variablesDict[varId]

        return variable

    def genVec(self, answer, question):
        '''
        Generate the encoding vector to get the probability of the answer given the question,
        from the probabilities present in the matrix of moments.
        '''
        # 
        assert(len(answer) == len(question) == self.game.nbPlayers)
        vec = [0] * len(self.monomialList)
        monome = Monomial([Operator(p, question[p], answer[p]) for p in range(self.game.nbPlayers)])

        def recursiveFunc(monome, coef):

            #The operator is in the matrix
            if monome in self.monomialList:
                vec[self.monomialList.index(monome)] += coef

            #Else, one of the element is encoded as Id - sum(elements[:a-1])
            else:
                #We find such element
                last_id = next(id_last for id_last, op in enumerate(monome.canonicalRep) if op.is_last(self.game.list_num_out[op.player]))
                opId = monome.canonicalRep.copy()
                opId[last_id] = Operator.identity()
                recursiveFunc(Monomial(opId), coef)
                for a in range(self.game.list_num_out[monome.canonicalRep[last_id].player]-1):
                    op2 = monome.canonicalRep.copy()
                    op2[last_id] = Operator(monome.canonicalRep[last_id].player, monome.canonicalRep[last_id].question, a)
                    recursiveFunc(Monomial(op2), -coef)

        recursiveFunc(monome, 1)
        return vec

    def genVecWelfareWin(self, answer, question):
        """
        Mean payout of all player.
        """
        coef = self.game.answerPayoutWin(answer, question)
        return list(map(lambda x: x * coef, self.genVec(answer, question)))

    def objectifFunctions(self):
        """
        The objectif function is the social welfare.
        """
        objectifFunctionPayout = []

        for question in self.game.questions():
            for validAnswer in self.game.validAnswerIt(question):
                objectifFunctionPayout.append(np.array(self.genVecWelfareWin(validAnswer, question)) * self.game.qDistrib(question))

        objectifFunction = np.array(objectifFunctionPayout).transpose()
        return objectifFunction

    def optimize(self, verbose, warmStart, solver):
        """
        Optimize on a given solver.
        """
        assert(solver == "SCS" or solver == "MOSEK")
        if solver == "SCS":
            self.prob.solve(solver=cp.SCS, verbose=verbose, warm_start=warmStart)
        else:
            self.prob.solve(solver=cp.MOSEK, verbose=verbose, warm_start=warmStart)
        
        return self.prob.value

