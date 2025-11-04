from itertools import groupby
from NPA.Operator import Operator

class Monomial:
    def __init__(self, operators):
        self.monome = operators
        self.adjoint = list(reversed(self.monome))
        self.n = len(self.monome)
        self.canonicalRep = self.canonical()

    def isNull(self):
        for op1 in self.monome:
            for op2 in self.monome:
                if op1.player == op2.player and op1.question == op2.question and op1.answer != op2.answer:
                    return True
        return False
    
    def canonical(self):
        return min(self.simplify(self.monome), self.simplify(self.adjoint))
    
    def simplify(self, monome):
        operators = filter(lambda op: not op.is_identity, monome)  # filter out identity
        operators = sorted(operators, key=lambda op: (op.player))  # Sort without commuting operators of a same player
    
        canonical = []

        for op, g in groupby(operators):
            canonical.append(op)

        # Fill the end with 0 (Id operators)
        while len(canonical) < self.n:
            canonical.append(Operator.identity())

        return canonical
        
    def __eq__(self, other):
        if not isinstance(other, Monomial):
            return False

        return other.canonicalRep == self.canonicalRep
    
    def __lt__(self, other):
        return self.canonicalRep < other.canonicalRep
        
    def __hash__(self):
        return tuple(self.canonicalRep).__hash__()
    
    def __repr__(self):
        return self.canonicalRep.__repr__()