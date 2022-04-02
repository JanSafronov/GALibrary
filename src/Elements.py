from argparse import Action
from ast import Lambda, Set, Tuple
from dataclasses import Field
from pyclbr import Function
import sys
from typing import Callable, List, TypeVar, Generic
from matplotlib import docstring
import numpy, matplotlib, math
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from FirstOrder import Identity
from Algebraic import VectorSpace

F = TypeVar("F", bound=Field)
T = TypeVar("T")

class Vector:
    """A class for representing a vector in a vector space.
    """
    def __init__(self, vector_space: "VectorSpace", coordinates: List[F]):
        """
        :param vector_space: A vector space
        :param coordinates: A list of coordinates
        """
        self.vector_space = vector_space
        self.coordinates = coordinates
        self.dimension = len(coordinates)

    def __str__(self):
        return "Vector(vector_space = {}, coordinates = {})".format(self.vector_space, self.coordinates)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.vector_space == other.vector_space and self.coordinates == other.coordinates

    def __hash__(self):
        return hash(self.__str__())

    def __contains__(self, coordinate):
        return coordinate in self.coordinates

    def __iter__(self):
        return iter(self.coordinates)

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, index):
        return self.coordinates[index]

    def __setitem__(self, index, coordinate):
        self.coordinates[index] = coordinate

    def __delitem__(self, index):
        del self.coordinates[index]

    def __add__(self, other: "Vector"):
        """
        :param other: A vector
        :return: The sum of the two vectors
        """
        if self.vector_space != other.vector_space:
            raise ValueError("The vector spaces are not the same")
        return Vector(self.vector_space, [self.coordinates[i] + other.coordinates[i] for i in range(self.dimension)])

    def __sub__(self, other: "Vector"):
        """
        :param other: A vector
        :return: The difference of the two vectors
        """
        if self.vector_space != other.vector_space:
            raise ValueError("The vector spaces are not the same")
        return Vector(self.vector_space, [self.coordinates[i] - other.coordinates[i] for i in range(self.dimension)])

    def __mul__(self, other: "Vector"):
        """
        :param other: A vector
        :return: The dot product of the two vectors
        """
        if self.vector_space != other.vector_space:
            raise ValueError("The vector spaces are not the same")
        return sum([self.coordinates[i] * other.coordinates[i] for i in range(self.dimension)])

    def __rmul__(self, other: F):
        """
        :param other: A scalar
        :return: The product of the scalar and the vector
        """
        return Vector(self.vector_space, [other * self.coordinates[i] for i in range(self.dimension)])

def BinOp(a: object, b: object) -> object:
    pass

class AlgebraicStructure(Generic(T), set):
    """A class for representing an algebraic structure. Algebraic structure is a set of elements
       with binary operations on it and identities that those operations satisfy.
    """
    def __init__(self, elements: set[T], operations: set[Callable[[T, T], T]], identities: set[Identity]):
        """
        :param elements: A set of elements
        :param operations: A set of binary operations
        :param identities: A set of identities
        """
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
        return "AlgebraicStructure(elements = {}, operations = {}, identities = {})".format(self.elements, self.operations, self.identities)

class Group(AlgebraicStructure(T)):
    """A class for representing a group. A group is a set of elements with binary operations and identities
    """
    def __init__(self, elements: List[T], operations: List[Tuple[T, T, T]], identities: List[T]):
        """
        :param elements: A list of elements in the group
        :param operations: A list of tuples of the form (a, b, c) where a, b, c are elements in the group and a*b = c
        :param identities: A list of elements in the group that are the identities of the operations
        """
        super().__init__(elements, operations, identities)

    def __str__(self):
        return "Group(elements = {}, operations = {}, identities = {})".format(self.elements, self.operations, self.identities)

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
        :param other: A group
        :return: The sum of the two groups
        """
        if self.elements != other.elements:
            raise ValueError("The groups are not the same")
        if self.operations != other.operations:
            raise ValueError("The groups are not the same")
        if

class Field(AlgebraicStructure(T)):
    """A class for representing a field. A field is a set of elements with binary operations of addition and multiplication, and identities
    """
    def __init__(self, elements: List[T], operations: List[Tuple[T, T, T]], identities: List[T]):
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

class Metric(Generic(T), ABC):
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

class MetricSpace(ABC, Generic(T)):
    """A class for representing a metric space. A metric space is an ordered pair of a non empty set with a metric on that set"""
    def __init__(self, elements: Set[T], metric: Metric[T]) -> None:
        self.elements = set
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