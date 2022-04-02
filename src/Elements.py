from argparse import Action
from ast import Lambda, Set, Tuple
from dataclasses import Field
from pyclbr import Function
import sys
from typing import Callable, List, TypeVar, Generic
from matplotlib import docstring
import numpy, matplotlib, math
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from src.FirstOrder import Identity
from Inference import BinaryOperator

F = TypeVar("F", bound=Field)
T = TypeVar("T")

class Vector:
    """
    A class for representing a vector.
    """
    def __init__(self, vector: List[F]):
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
        :return: The product of the two vectors
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

class AlgebraicStructure(set, Generic[T]):
    """Algebraic structure is a set of elements
       with binary operations on it and identities that those operations satisfy.
    """
    def __init__(self, elements: set[T], operations: set[Callable[[T, T], T]], identities: set[Identity]):
        """
        :param elements: A set of elements
        :param operations: A set of binary operations
        :param identities: A set of identities
        """

        for element in elements:
            for operation in operations:
                for identity in identities:
                    if identity(operation(element, element)) != element:
                        raise ValueError("Identity {} does not satisfy operation {}".format(identity, operation))
        
        self.elements = elements
        self.operations = operations
        self.identities = identities

    @staticmethod
    def operate(a: set[T], b: set[T], operation: Callable[[T, T], T]) -> set[T]:
        """
        :param a: A set of elements
        :param b: A set of elements
        :param operation: A binary operation
        :return: The set of elements that satisfy the operation
        """
        return {operation(a_, b_) for a_ in a for b_ in b}

    def __str__(self):
        return "for all {},\n {}\n identities = {})".format(self.elements, self.operations, self.identities)

class Group(AlgebraicStructure[T], Generic[T]):
    """A group is a set of elements 
       with a single binary operation on it and identities that this operation satisfies.
    """
    def __init__(self, elements: set[T], operation: Callable[[T, T], T]):
        """
        :param elements: A set of elements
        :param operation: A binary operation
        """
        super().__init__(elements, {operation}, 
        {lambda x, y, z: operation(operation(x, y), z) == operation(x, operation(y, z)), 
        lambda x, y: operation(x, y) == operation(y, x) == y,
        lambda x, y: operation(x, y) == x,})

        # TODO: Add axioms

    def __str__(self):
        return "G = {},  {}\n  {})".format(self.elements, self.operations[0], self.identities)

    def __repr__(self):
        return "Group(elements = {}, operations = {}, identities = {})".format(self.elements, self.operations, self.identities)

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
        

class Field(AlgebraicStructure[T], Generic[T]):
    """A class for representing a field. A field is a set of elements with binary operations of addition and multiplication, and identities
    """
    def __init__(self, elements: set[T], operations: set[Callable[[T, T], T]], identities: set[Identity]):
        """
        :param elements: A list of elements in the field
        :param operations: A list of tuples of the form (a, b, c) where a, b, c are elements in the field and a*b = c
        :param identities: A list of elements in the field that are the identities of the operations
        """
        super().__init__(elements, operations, identities)

    def __str__(self):
        return "Field(elements = {}, operations = {}, identities = {})".format(self.elements, self.operations, self.identities)

    def __repr__(self):
        return self.__str__()

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
        if self.identities != other.identities:
            raise ValueError("The fields are not the same")
        return Field(self.elements, self.operations, self.identities)

    def __mul__(self, other):
        """
        :param other: A field
        :return: The product of the two fields
        """
        if self.elements != other.elements:
            raise ValueError("The fields are not the same")
        if self.operations != other.operations:
            raise ValueError("The fields are not the same")
        if self.identities != other.identities:
            raise ValueError("The fields are not the same")
        return Field(self.elements, self.operations, self.identities)

    def algebraically_closed(self):
        """
        :return: True if the field is algebraically closed, False otherwise
        """
        return self.operations == []

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