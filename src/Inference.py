"""
Definitions taken from rules of inference in formal logic.
Particularly from rules of replacement.
"""

from abc import ABC
from ast import Lambda
from typing import Callable, Generic, TypeVar

S = TypeVar("S", bound=set)

# Remove later if unnecessary class
class BinaryOperator(Generic[S], Lambda, ABC):
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

class Associative(Generic[S], Lambda, ABC):
    """
    An associative operator on a set of elements S.
    """
    def __init__(self, s: S, op: Callable[[S, S, S], S]):
        """
        :param a: A value
        :param b: A value
        :param op: An associative operator
        """
        self.s = s
        self.op = op

    def __call__(self, a: S, b: S, c: S) -> S:
        return self.op(a, b, c)

    @staticmethod
    def __call__() -> "Associative":
        """
        :param s: A set
        :param op: An associative operator
        :return: An associative operator on the set
        """
        return lambda x, y, z, opp: opp(opp(x, y), z) == opp(x, opp(y, z))

    def __str__(self):
        return "{}: S x S x S -> S".format(self.op.__name__)

    def __str__(self, a: S, b: S, c: S) -> str:
        return "({} {} {}) |-> {}".format(a, b, c, self.op(a, b, c))

    def __repr__(self):
        return self.__str__()

    def __repr__(self, a: S, b: S, c: S) -> str:
        return self.__str__(a, b, c)

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b and self.op == other.op

    def __hash__(self):
        return hash(self.__str__())

class Commutative(Generic[S], Lambda, ABC):
    """
    A commutative operator on a set of elements S.
    """
    def __init__(self, s: S, op: Callable[[S, S], S]):
        """
        :param a: A value
        :param b: A value
        :param op: A commutative operator
        """
        self.s = s
        self.op = op

    def __call__(self, a: S, b: S) -> S:
        return self.op(a, b)

    @staticmethod
    def __call__() -> Callable[[S, S], S]:
        """
        :param s: A set
        :param op: A commutative operator
        :return: A commutative operator on the set
        """
        return lambda x, y, op: op(x, y) == op(y, x)

    @staticmethod
    def __call__(op: Callable[[S, S], S]) -> Callable[[S, S], S]:
        """
        :param s: A set
        :param op: A commutative operator
        :return: A commutative operator on the set
        """
        return lambda x, y: op(x, y) == op(y, x)

    @staticmethod
    def __call__(x: S, op: Callable[[S, S], S]) -> Callable[[S, S], S]:
        """
        :param s: A set
        :param op: A commutative operator
        :return: A commutative operator on the set
        """
        return lambda y: op(x, y) == op(y, x)

    @staticmethod
    def __call__(y: S, op: Callable[[S, S], S]) -> Callable[[S, S], S]:
        """
        :param s: A set
        :param op: A commutative operator
        :return: A commutative operator on the set
        """
        return lambda x: op(x, y) == op(y, x)

    @staticmethod
    def __call__(x: S, y: S, op: Callable[[S, S], S]) -> Callable[[S, S], S]:
        """
        :param s: A set
        :param op: A commutative operator
        :return: A commutative operator on the set
        """
        return lambda: op(x, y) == op(y, x)

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

class Distributive(Generic[S], Lambda, ABC):
    """
    A distributive operator on a set of elements S.
    """
    def __init__(self, s: S, op0: Callable[[S, S], S], op1: Callable[[S, S], S]):
        """
        :param a: A value
        :param b: A value
        :param op: A distributive operator
        """
        self.s = s
        self.op0 = op0
        self.op1 = op1

    def __call__(self, a: S, b: S, c: S) -> S:
        return self.op(a, b, c)

    @staticmethod
    def __call__() -> "Distributive":
        """
        :param s: A set
        :param op: A distributive operator
        :return: A distributive operator on the set
        """
        return lambda x, y, z, op0, op1: op0(x, op1(y, z)) == op0(x, y) + op0(x, z)

    def __str__(self):
        return "{}: S x S x S -> S".format(self.op.__name__)

    def __str__(self, a: S, b: S, c: S) -> str:
        return "({} {} {}) |-> {}".format(a, b, c, self.op(a, b, c))

    def __repr__(self):
        return self.__str__()

    def __repr__(self, a: S, b: S, c: S) -> str:
        return self.__str__(a, b, c)

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b and self.op == other.op

    def __hash__(self):
        return hash(self.__str__())

class IdentityElement(Generic[S], Lambda, ABC):
    """
    An identity element on a set of elements S.
    """
    def __init__(self, s: S, op: Callable[[S], S]):
        """
        :param a: A value
        :param op: An identity element
        """
        self.s = s
        self.op = op

    def __call__(self, a: S) -> S:
        return self.op(a)

    @staticmethod
    def __call__(e: S) -> "IdentityElement":
        """
        :param s: A set
        :param op: An identity element
        :return: An identity element on the set
        """
        return lambda x, op: op(e, x) == op(x, e) == x

    def __str__(self):
        return "{}: S -> S".format(self.op.__name__)

    def __str__(self, a: S) -> str:
        return "({}) |-> {}".format(a, self.op(a))

    def __repr__(self):
        return self.__str__()

    def __repr__(self, a: S) -> str:
        return self.__str__(a)

    def __eq__(self, other):
        return self.a == other.a and self.op == other.op

    def __hash__(self):
        return hash(self.__str__())

class Inverse(Generic[S], Lambda, ABC):
    """
    An inverse operator on a set of elements S.
    """
    def __init__(self, s: S, op: Callable[[S], S]):
        """
        :param a: A value
        :param op: An inverse operator
        """
        self.s = s
        self.op = op

    def __call__(self, a: S) -> S:
        return self.op(a)

    @staticmethod
    def __call__(e: S) -> "Inverse":
        """
        :param s: A set
        :param op: An inverse operator
        :return: An inverse operator on the set
        """
        return lambda x, y, op: op(x, y) == op(y, x) == e

    def __str__(self):
        return "{}: S -> S".format(self.op.__name__)

    def __str__(self, a: S) -> str:
        return "({}) |-> {}".format(a, self.op(a))

    def __repr__(self):
        return self.__str__()

    def __repr__(self, a: S) -> str:
        return self.__str__(a)

    def __eq__(self, other):
        return self.a == other.a and self.op == other.op

    def __hash__(self):
        return hash(self.__str__())