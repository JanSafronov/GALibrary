from argparse import Action
from ast import Lambda, Set, Tuple
from dataclasses import Field
from enum import Enum
import inspect
from pyclbr import Function
import sys
from typing import Callable, List, TypeVar, Generic
from matplotlib import docstring
import numpy, matplotlib, math
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty

from soupsieve import Iterable
from FirstOrder import Identity
from Inference import BinaryOperator, Commutative, Inverse

F = TypeVar("F", bound=Field)
T = TypeVar("T")

symbols = {"*", "+", "-", "/", "^"}

class Vector:
    """
    A class for representing a vector.
    """
    def __init__(self, vector: List[T]):
        """
        :param vector: A list of elements
        """
        self.vector = vector
        self.dimension = len(vector)

    def __str__(self):
        return "Vector({})".format(self.vector)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.vector == other.vector

    def __hash__(self):
        return hash(self.__str__())

    def __contains__(self, element):
        return element in self.vector

    def __iter__(self):
        return iter(self.vector)

    def __len__(self):
        return len(self.vector)

    def __getitem__(self, index):
        return self.vector[index]

    def __setitem__(self, index, element):
        self.vector[index] = element

    def __delitem__(self, index):
        del self.vector[index]

    def __add__(self, other: "Vector") -> "Vector":
        """
        :param other: A vector
        :return: The sum of the two vectors
        """
        return Vector([a + b for a, b in zip(self.vector, other.vector)])

    def __sub__(self, other: "Vector") -> "Vector":
        """
        :param other: A vector
        :return: The difference of the two vectors
        """
        return Vector([a - b for a, b in zip(self.vector, other.vector)])

    def __mul__(self, other: "Vector") -> "Vector":
        """
        :param other: A vector
        :return: The dot product of the two vectors
        """
        return sum([a * b for a, b in zip(self.vector, other.vector)])

    def __rmul__(self, scalar: F) -> "Vector":
        """
        :param scalar: A scalar
        :return: The scalar product of the vector and the scalar
        """
        return Vector([scalar * a for a in self.vector])


    def __truediv__(self, other: "Vector") -> "Vector":
        """
        :param other: A vector
        :return: The quotient of the two vectors
        """
        return Vector([a / b for a, b in zip(self.vector, other.vector)])

class AlgebraicStructure(set[T], Generic[T]):
    """
    Algebraic structure is a set of elements
    with binary operations on it and identities that those operations satisfy.
    """
    def __init__(self, elements: set[T], operations: set[BinaryOperator], identities: set[Identity]):
        """
        :param elements: A set of elements
        :param operations: A set of binary operations
        :param identities: A set of identities
        """

        for identity in identities:
            for operation in operations:
                for (x, y) in [(x, y) for x in elements for y in elements]:
                    if not identity(x, y):
                        raise ValueError("Operation {} does not satisfy Identity {}".format(operation, identity))
        
        self.elements = elements
        self.operations = operations
        self.identities = identities

    def operate(self, a: T, b: T) -> T:
        """
        :param a: An element
        :param b: An element
        :return: The result of applying the binary operation on the two elements
        """
        return self.operations[0](a, b)

    @staticmethod
    def operate(a: set[T], b: set[T], operation: Callable[[T, T], T]) -> set[T]:
        """
        :param a: A set of elements
        :param b: A set of elements
        :param operation: A binary operation
        :return: The set of elements that satisfy the operation
        """
        #operation.
        return {operation(a_, b_) for a_ in a for b_ in b}

    def __or__(self, other: "AlgebraicStructure") -> "AlgebraicStructure":
        """
        :param other: An algebraic structure
        :return: The union of the two algebraic structures
        """
        return AlgebraicStructure(self.elements | other.elements, self.operations | other.operations, self.identities | other.identities)

    def union(self, *s: Iterable[_S]) -> set[_T | _S]:
        return super().union(*s)
    
    def __str__(self) -> str:
        return "A = {}\n\nfor all x, y ∈ A,   {}\n  {})".format(self.elements, " ".join(list(map(lambda op, s: str(inspect.signature(op)) + " ↦ " + " x " + s + " y; ", self.operations, symbols))), "\n".join(list(map(lambda id: id.__str__(), self.identities))))

class Monoid(AlgebraicStructure[T]):
    """
    A monoid is a set of elements
    with a binary operation on it and identities that those operations satisfy.
    """
    def __init__(self, elements: set[T], operation: Callable[[T, T], T]):
        """
        :param elements: A set of elements
        :param operation: A binary operation
        """
        super().__init__(elements, {operation}, 
        {lambda x, y, z: operation(operation(x, y), z) == operation(x, operation(y, z)),
        lambda x, e: operation(e, x) == operation(x, e) == e})

    def __str__(self) -> str:
        return "M = {}, (x, y) ↦ x * y\n there exists e ∈ M for all x, y, z ∈ M such that (x * y) * z = x * (y * z)\n and e * x = x * e = x".format(self.elements)

    def __repr__(self) -> str:
        return "Monoid(elements = {}, operations = {}, identities = {})".format(self.elements, self.operations, self.identities)

    def get_identities(self) -> set[Identity]:
        """
        :return: The set of identities defining the monoid
        """
        return self.identities

class Group(AlgebraicStructure[T], Generic[T]):
    """
    A group is a set of elements 
    with a binary operation on it and identities that this operation satisfies.
    """
    def __init__(self, elements: set[T], operation: Callable[[T, T], T]):
        """
        :param elements: A set of elements
        :param operation: A binary operation
        """
        super().__init__(elements, {operation}, 
        {lambda x, y, z: operation(operation(x, y), z) == operation(x, operation(y, z)), 
        lambda x, e, y: operation(x, e) == operation(e, x) == x and operation(x, y) == operation(y, x) == e})

    def __str__(self) -> str:
        return "G = {},  (x, y) ↦ x * y\n there exists e ∈ M for all x, y, z ∈ M such that (x * y) * z = x * (y * z)\n and e * x = x * e = x \n and x * y = y * x = e))".format(self.elements)

    def __repr__(self) -> str:
        return "Group(elements = {}, operations = {}, identities = {})".format(self.elements, self.operations, self.identities)

    def get_identities(self) -> set[Identity]:
        """
        :return: The set of identities defining the group
        """
        return self.identities

    def __eq__(self, other):
        return self.elements == other.elements and self.operations == other.operations and self.identities == other.identities

    def __hash__(self):
        return hash(self.__str__())

    def __contains__(self, element):
        return element in self.elements

    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, index):
        return self.elements[index]

    def __setitem__(self, index, element):
        self.elements[index] = element

    def __delitem__(self, index):
        del self.elements[index]

    def __add__(self, other):
        """
        :param other: A group
        :return: The sum of the two groups
        """
        if self.elements != other.elements:
            raise ValueError("The groups are not the same")
        if self.operations != other.operations:
            raise ValueError("The groups are not the same")

    def is_abelian(self) -> bool:
        """
        :return: True if the group is abelian, False otherwise
        """
        for x in self.elements:
            for y in self.elements:
                if (self.operations[0](x, y) != self.operations[0](y, x)):
                    return False
            

    def abelian(self) -> "Group":
        """
        :return: The abelian group of the group
        """
        if self.is_abelian():
            new = Group(self.elements, self.operations[0])
            new.identities.add(lambda x, y: self.operations[0](x, y) == self.operations[0](y, x))
            return new
        else:
            return Group(self.elements, self.operations[0])

class Ring(AlgebraicStructure[T], Generic[T]):
    """
    A ring is a set of elements 
    with two binary operations on it and identities that this operations satisfies.
    """
    def __init__(self, elements: set[T], operations: set[Callable[[T, T], T]]):
        """
        :param elements: A set of elements
        :param operations: A set of binary operations that are associative
        """
        super().__init__(elements, operations, 
        Group(elements, operations[0]).abelian().get_identities() | Monoid(elements, operations[1]).get_identities() |
        {lambda x, y: Commutative.__call__(self.operations[0](x, y), self.operations[1])})

    def __str__(self):
        return "R = {},  {}\n  {})".format(self.elements, self.operations[0], self.identities)

    def __repr__(self):
        return "Ring(elements = {}, operations = {}, identities = {})".format(self.elements, self.operations, self.identities)

    def __eq__(self, other):
        return self.elements == other.elements and self.operations == other.operations and self.identities == other.identities

    def __hash__(self):
        return hash(self.__str__())

    def __contains__(self, element):
        return element in self.elements

    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, index):
        return self.elements[index]

    def __setitem__(self, index, element):
        self.elements[index] = element

    def __delitem__(self, index):
        del self.elements[index]

    def __add__(self, other):
        """
        :param other: A ring
        :return: The sum of the two rings
        """

    def is_commutative(self) -> bool:
        """
        :return: True if the ring is commutative, False otherwise
        """
        for x in self.elements:
            for y in self.elements:
                if (self.operations[1](x, y) != self.operations[1](y, x)):
                    return False
            

    def commutative(self) -> "Ring":
        """
        :return: The commutative ring of the ring
        """
        if self.is_commutative():
            new = Ring(self.elements, self.operations)
            new.identities.add(lambda x, y: self.operations[1](x, y) == self.operations[1](y, x))
            return new
        else:
            return Group(self.elements, self.operations[0])
        

class Field(Ring[T], Generic[T]):
    """
    A field is a set of elements 
    with two binary operation on it and identities that this operations satisfies.
    Definitively it is a commutative ring with non zero invertible elements and 0 != 1.
    """
    def __init__(self, elements: set[T], operations: set[Callable[[T, T], T]]):
        """
        :param elements: A set of elements
        :param operations: A set of binary operations that are associative and commutative
        """
        super().__init__(elements, operations)
        self.identities.add(lambda x, y: self.operations[1](x, y) == self.operations[1](y, x))
        self.identities.add(Inverse.__call__(self.operations[1]))

    def __str__(self):
        return "F = {},  {}\n  {})".format(self.elements, self.operations[0], self.identities)

    def __repr__(self):
        return "Field(elements = {}, operations = {}, identities = {})".format(self.elements, self.operations, self.identities)

    def __eq__(self, other):
        return self.elements == other.elements and self.operations == other.operations and self.identities == other.identities

    def __hash__(self):
        return hash(self.__str__())

    def __contains__(self, element):
        return element in self.elements

    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, index):
        return self.elements[index]

    def __setitem__(self, index, element):
        self.elements[index] = element

    def __delitem__(self, index):
        del self.elements[index]

    def __add__(self, other):
        """
        :param other: A field
        :return: The sum of the two fields
        """
        if self.elements != other.elements:
            raise ValueError("The fields are not the same")
        if self.operations != other.operations:
            raise ValueError("The fields are not the same")

    def is_euclidean(self) -> bool:
        """
        :return: True if the field is euclidean, False otherwise
        """
        for x in self.elements:
            for y in self.elements:
                if (self.operations[1](x, y) != self.operations[1](y, x)):
                    return False

R = TypeVar("R")
G = TypeVar("G")

class Module(Generic[R, G]):
    """
    A Module (R-module) is a ring with an abelian group under addition
    with two binary operations on them and identities that this operations satisfies.
    """
    def __init__(self, ring: Ring[R], group: Group[G], operation: Callable[[R, G], G]):
        """
        :param elements: A set of elements
        :param operations: A set of binary operations that are associative
        """
        self.ring = ring
        self.group = group
        self.operation = operation
        self.identities = {
            lambda r, x, y: self.operation(r, group.operate(x, y)) == group.operate(self.operation(r, x), self.operation(r, y)),
            lambda r, s, x: self.operation(group.operate(r, s), x) == group.operate(self.operation(r, x), self.operation(s, x)),
            lambda r, s, x: self.operation(ring.operate(r, s), x) == self.operation(r, self.operation(s, x)),
            lambda x: self.operation(1, x) == x}
    def __str__(self):
        return "M = {},  {}\n  {})".format(self.elements, self.operations[0], self.identities)

    def __repr__(self):
        return "R_Module(elements = {}, operations = {}, identities = {})".format(self.elements, self.operations, self.identities)

    def __eq__(self, other):
        return self.elements == other.elements and self.operations == other.operations and self.identities == other.identities

    def __hash__(self):
        return hash(self.__str__())

    def __contains__(self, element):
        return element in self.elements

    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, index):
        return self.elements[index]

    def __setitem__(self, index, element):
        self.elements[index] = element

    def __delitem__(self, index):
        del self.elements[index]

    def __add__(self, other):
        """
        :param other: A module
        :return: The sum of the two modules
        """
        if self.elements != other.elements:
            raise ValueError("The modules are not the same")
        if self.operations != other.operations:
            raise ValueError("The modules are not the same")

class Metric(Generic[T], ABC):
    """A class for representing a metric. A metric is a function that takes two arguments and returns a their distance."""
    def __init__(self, x: T, y: T) -> None:
        self.x = x
        self.y = y

    @abstractmethod
    def __add__(self, other: "Metric") -> "Metric":
        pass

    def __sub__(self, other: "Metric") -> "Metric":
        pass

    def indiscernible(self) -> "Metric":
        """Axiom of the indiscernibility of identicals on a metric"""
        if self.x == self.y:
            return 0

    def symmetrical(self) -> "Metric":
        """Axiom of symmetry on a metric"""
        return Metric(self.y, self.x)

    def triangeInequality(self, z: T) -> float:
        """Axiom of triange inequality on a metric"""
        return Metric(self.x, self.y) + Metric(self.y, z) - Metric(self.x, z)

    @abstractmethod
    def __eq__(self, other: "Metric") -> bool:
        pass

    def __repr__(self) -> str:
        return f"d({self.x}, {self.y})"

    def __str__(self) -> str:
        return f"d({self.x}, {self.y})"

class MetricSpace(Generic[T], ABC):
    """A class for representing a metric space. A metric space is an ordered pair of a non empty set with a metric on that set"""
    def __init__(self, elements: set[T], metric: Metric[T]) -> None:
        self.elements = elements
        self.metric = metric

    @abstractmethod
    def __add__(self, other: "MetricSpace") -> "MetricSpace":
        pass

    @abstractmethod
    def __sub__(self, other: "MetricSpace") -> "MetricSpace":
        pass

    @abstractmethod
    def __eq__(self, other: "MetricSpace") -> bool:
        pass

    def __repr__(self) -> str:
        return f"({self.elements}, {self.metric})"

    def __str__(self) -> str:
        return f"({self.elements}, {self.metric})"

class Interval:
    """ 
    A class for representing an interval. An interval is a set of points that are lying between two points of the set"""
    def __init__(self, a: float, is_closed: bool, b: float, is_open: bool) -> None:
        self.a = a
        self.is_closed = is_closed
        self.b = b
        self.is_open = is_open

    def short() -> "Interval":
        return Interval(-0.0001, 0.0001)

    def __add__(self, other: "Interval") -> "Interval":
        return Interval(self.a + other.a, self.b + other.b)
    
    def __delete__(self, other: "Interval") -> "Interval":
        return Interval(self.a - other.b, self.b - other.a)

    def __eq__(self, other: "Interval") -> bool:
        return self.a == other.a & self.b == other.b

    def __str__(self) -> str:
        return "[" if self.is_closed else "(" + self.a + ", " + self.b + ")" if self.is_open else "]"

    def __str__(self) -> str:
        return "[" if self.is_closed else "(" + self.a + ", " + self.b + ")" if self.is_open else "]"

