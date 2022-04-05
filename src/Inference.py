"""
Definitions taken from rules of inference in formal logic.
Particularly from rules of replacement.
"""

from abc import ABC
from ast import Lambda
from typing import Callable, Generic, TypeVar

T = TypeVar("T")

# Remove later if unnecessary class
class BinaryOperator(Generic[T], Lambda, ABC):
    """
    A binary operator on a set of elements S.
    """
    def __init__(self, domain: set[T], op: Callable[[T, T], T], is_internal: bool = False):
        """
        :param domain: A set of elements
        :param op: A binary operator
        """
        if is_internal:
            for x, y in [(x, y) for x in domain for y in domain]:
                if self.op(x, y) not in domain:
                    raise ValueError("{} is not in the domain of {}".format(self.op(x, y), self))
        self.domain = domain
        self.op = op

    def __call__(self, a: T, b: T) -> T:
        return self.op(a, b)

    def __str__(self):
        return "{}: S x S -> S".format(self.op.__name__)

    def __str__(self, a: T, b: T) -> str:
        return "({} {}) |-> {}".format(a, b, self.op(a, b))

    def __repr__(self):
        return self.__str__()

    def __repr__(self, a: T, b: T) -> str:
        return self.__str__(a, b)

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b and self.op == other.op

    def __hash__(self):
        return hash(self.__str__())

class Associative(BinaryOperator[T], Generic[T], Lambda, ABC):
    """
    An associative operator satisfying the associative identity on a set of elements S.
    """
    def __init__(self, domain: set[T], op: Callable[[T, T], T]):
        """
        :param domain: Set of elements
        :param op: An associative operator
        """
        for x, y, z in [(x, y, z) for x in domain for y in domain for z in domain]: 
            if op(op(x, y), z) != op(x, op(y, z)):
                raise ValueError("Operation * does not satisfy the associative identity on a set {}".format(op, domain))
            
        super().__init__(domain, op)

    def __call__(self, a: T, b: T) -> T:
        return self.op(a, b)

    @staticmethod
    def __call__() -> "Associative":
        """
        :return: An associative identity on the set
        """
        return lambda x, y, z, opp: opp(opp(x, y), z) == opp(x, opp(y, z))

    def __str__(self):
        return "{}: S x S x S -> S".format(self.op.__name__)

    def __str__(self, a: T, b: T, c: T) -> str:
        return "({} {} {}) |-> {}".format(a, b, c, self.op(a, b, c))

    def __repr__(self):
        return self.__str__()

    def __repr__(self, a: T, b: T, c: T) -> str:
        return self.__str__(a, b, c)

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b and self.op == other.op

    def __hash__(self):
        return hash(self.__str__())

class Commutative(BinaryOperator[T], Generic[T], Lambda, ABC):
    """
    A commutative operator satisfying the commutative identity on a set of elements S.
    """
    def __init__(self, domain: set(T), op: Callable[[T, T], T]):
        """
        :param domain: Set of elements
        :param op: A commutative operator
        """
        self.domain = domain
        self.op = op

    def __call__(self, a: T, b: T) -> T:
        return self.op(a, b)

    @staticmethod
    def __call__() -> Callable[[T, T], T]:
        """
        :return: A commutative identity on the set
        """
        return lambda x, y, op: op(x, y) == op(y, x)

    @staticmethod
    def __call__(op: Callable[[T, T], T]) -> Callable[[T, T], T]:
        """
        :param op: A commutative operator
        :return: A commutative identity on the set
        """
        return lambda x, y: op(x, y) == op(y, x)

    @staticmethod
    def __call__(x: T, op: Callable[[T, T], T]) -> Callable[[T, T], T]:
        """
        :param x: An element of the set
        :param op: A commutative operator
        :return: A commutative identity on the set
        """
        return lambda y: op(x, y) == op(y, x)

    @staticmethod
    def __call__(y: T, op: Callable[[T, T], T]) -> Callable[[T, T], T]:
        """
        :param y: An element of the set
        :param op: A commutative operator
        :return: A commutative identity on the set
        """
        return lambda x: op(x, y) == op(y, x)

    @staticmethod
    def __call__(x: T, y: T, op: Callable[[T, T], T]) -> Callable[[T, T], T]:
        """
        :param x: An element of the set
        :param y: An element of the set
        :param op: A commutative operator
        :return: A commutative identity on the set
        """
        return lambda: op(x, y) == op(y, x)

    def __str__(self):
        return "{}: S x S -> S".format(self.op.__name__)

    def __str__(self, a: T, b: T) -> str:
        return "({} {}) ↦ {}".format(a, b, self.op(a, b))

    def __repr__(self):
        return self.__str__()

    def __repr__(self, a: T, b: T) -> str:
        return self.__str__(a, b)

    def __eq__(self, other):
        return self.domain == other.domain and self.op == other.op

    def __hash__(self):
        return hash(self.__str__())

class Distributive(BinaryOperator[T], Generic[T], Lambda, ABC):
    """
    A distributive operator satisfying the distributive identity on a set of elements S.
    """
    def __init__(self, domain: set(T), op0: Callable[[T, T], T], op1: Callable[[T, T], T]):
        """
        :param domain: Set of elements
        :param op0: A distributive operator
        :param op1: A distributive operator
        """
        self.domain = domain
        self.op0 = op0
        self.op1 = op1

    def __call__(self, a: T, b: T) -> list[T]:
        return [self.op0(a, b), self.op1(a, b)]

    @staticmethod
    def __call__() -> "Distributive":
        """
        :return: A distributive identity on the set
        """
        return lambda x, y, z, op0, op1: op0(x, op1(y, z)) == op0(x, y) + op0(x, z)

    def __str__(self):
        return "{}: S x S x S -> S".format(self.op.__name__)

    def __str__(self, a: T, b: T, c: T) -> str:
        return "({} {} {}) ↦ {}".format(a, b, c, self.op(a, b, c))

    def __repr__(self):
        return self.__str__()

    def __repr__(self, a: T, b: T, c: T) -> str:
        return self.__str__(a, b, c)

    def __eq__(self, other):
        return self.domain == other.domain and self.op0 == other.op0 and self.op1 == other.op1

    def __hash__(self):
        return hash(self.__str__())

class IdentityElement(BinaryOperator[T], Generic[T], Lambda, ABC):
    """
    An identity operator satisfying the identity element identity on a set of elements S.
    """
    def __init__(self, domain: set[T], op: Callable[[T, T], T]):
        """
        :param domain: Set of elements
        :param op: An identity operator
        """
        self.domain = domain
        self.op = op

    def __call__(self, a: T) -> T:
        return self.op(a)

    @staticmethod
    def __call__(e: T) -> "IdentityElement":
        """
        :param e: Identity element of the set
        :return: An identity element identity on the set
        """
        return lambda x, op: op(e, x) == op(x, e) == x

    def __str__(self):
        return "{}: S -> S".format(self.op.__name__)

    def __str__(self, a: T) -> str:
        return "({}) ↦ {}".format(a, self.op(a))

    def __repr__(self):
        return self.__str__()

    def __repr__(self, a: T) -> str:
        return self.__str__(a)

    def __eq__(self, other):
        return self.domain == other.domain and self.op == other.op

    def __hash__(self):
        return hash(self.__str__())

class Inverse(BinaryOperator[T] ,Generic[T], Lambda, ABC):
    """
    An inverse operator satisfying the inverse identity on a set of elements S.
    """
    def __init__(self, domain: set[T], op: Callable[[T, T], T]):
        """
        :param domain: A set of elements
        :param op: An inverse operator
        """
        self.domain = domain
        self.op = op

    def __call__(self, a: T, b: T) -> T:
        return self.op(a, b)

    @staticmethod
    def __call__(e: T) -> "Inverse":
        """
        :param e: Identity element of the set
        :param op: An inverse operator
        :return: An inverse identity on the set
        """
        return lambda x, y, op: op(x, y) == op(y, x) == e

    def __str__(self):
        return "{}: S -> S".format(self.op.__name__)

    def __str__(self, a: T) -> str:
        return "({}) ↦ {}".format(a, self.op(a))

    def __repr__(self):
        return self.__str__()

    def __repr__(self, a: T) -> str:
        return self.__str__(a)

    def __eq__(self, other):
        return self.a == other.a and self.op == other.op

    def __hash__(self):
        return hash(self.__str__())