from NPA.operator import Operator

class Monomial:
    def __init__(self, operators):
        self.monome = operators
        self.adjoint = list(reversed(self.monome))
        self.n = len(self.monome)
        self.canonicalRep = self.canonical()

    def _reduce(self, monome):
        """
        Reduce a word to its canonical operator sequence: operators of different
        parties commute, so we sort by player (stable, to keep the within-party
        order), then collapse adjacent identical projectors (E E = E). No reverse
        identification happens here.
        """
        operators = sorted([op for op in monome if not op.is_identity], key=lambda op: op.player)
        reduced = []
        for op in operators:
            if reduced and reduced[-1] == op:
                continue  # idempotent: E E = E
            reduced.append(op)
        return reduced

    def isNull(self):
        # A word is null only when two *adjacent* same-party operators (after the stable
        # sort-by-player) share a question but differ in answer: E^q_a E^q_a' = 0.
        # Non-adjacent repeats (e.g. E^q0 E^q1 E^q0) are legitimate.
        ops = sorted([op for op in self.monome if not op.is_identity], key=lambda op: op.player)
        for op1, op2 in zip(ops, ops[1:]):
            if op1.player == op2.player and op1.question == op2.question and op1.answer != op2.answer:
                return True
        return False

    def canonical(self):
        """
        Row/column identity used for set membership (==, hash, sorting).

        This reduces the word but does NOT identify a word with its reverse: A0 A1
        and A1 A0 are *distinct* monomials (they are distinct operators), so both
        appear as separate rows of the moment matrix. Hermiticity (a word equals its
        reverse in expectation) is a statement about matrix *entries*, handled by
        hermitian_key(), not about which rows exist.
        """
        reduced = self._reduce(self.monome)
        return reduced + [Operator.identity()] * (self.n - len(reduced))

    def hermitian_key(self):
        """
        Cell identity used for labeling moment-matrix entries (which entries share an
        SDP variable). A word and its reverse have equal expectation in a real moment
        matrix (<S> = <S^dagger>), so we identify them here via min(word, reverse).
        Returns a hashable, padding-independent tuple.
        """
        a = self._reduce(self.monome)
        b = self._reduce(self.adjoint)
        size = max(len(a), len(b))
        pad = lambda l: tuple(l + [Operator.identity()] * (size - len(l)))
        return min(pad(a), pad(b))

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
