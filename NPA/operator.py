class Operator:
    """Represents measurement operator for a player"""
    def __init__(self, player=None, question=None, answer=None, is_identity=False):
        self.player = player
        self.question = question
        self.answer = answer
        self.is_identity = is_identity
        
    @classmethod
    def identity(cls):
        """Creates an identity operator for a given player"""
        return cls(is_identity=True)
    
    def is_last(self, nbOutput):
        return int(self.answer) == int(nbOutput) - 1
    
    def __eq__(self, other):
        if isinstance(other, Operator):
            if self.is_identity and other.is_identity:
                return True
            elif self.is_identity or other.is_identity:
                return False
            return (int(self.player) == int(other.player) and 
                    int(self.question) == int(other.question) and 
                    int(self.answer) == int(other.answer))
        return False    
    
    def __hash__(self):
        return hash((self.player, self.question, self.answer, self.is_identity))
    
    def __repr__(self):
        if self.is_identity:
            return f"Id"
        else:
            return f"P{self.player}^{self.question}_{self.answer}"
        
    def __lt__(self, other):
        if isinstance(other, Operator):
            if self.is_identity and other.is_identity:
                return False
            elif self.is_identity or other.is_identity:
                return self.is_identity
            return (self.player, self.question, self.answer) < (other.player, other.question, other.answer)
        return NotImplemented
