"""
Definitions taken from Rules of inference in formal logic.
Particularly from rules of replacement.
"""

from typing import Callable, Generic, TypeVar

S = TypeVar("S", bound=set)

# Remove later if unnecessary class
class BinaryOperator(Generic[S]):
    """
    A binary operator on a set of elements S.
    """
    def __init__(self, s: S, op: Callable[[S, S], S]):
        """
        :param a: A value
        :param b: A value
        :param op: A binary operator
        """
        self.s = s
        self.op = op

    def __call__(self, a: S, b: S) -> S:
        return self.op(a, b)

    def __str__(self):
        return "{}: S x S -> S".format(self.op.__name__)

    def __str__(self, a: S, b: S) -> str:
        return "({} {}) |-> {}".format(a, b, self.op(a, b))

    def __repr__(self):
        return self.__str__()

    def __repr__(self, a: S, b: S) -> str:
        return self.__str__(a, b)

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b and self.op == other.op

    def __hash__(self):
        return hash(self.__str__())

class Associative(Generic[S]):
    