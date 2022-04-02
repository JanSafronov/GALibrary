from ast import Set, Tuple
import sys
from typing import List, TypeVar, Generic
from matplotlib import docstring
import numpy, matplotlib, math
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty

T = TypeVar("T")

class AlgebraicStructure(Generic(T)):
    """A class for representing an algebraic structure. Algebraic structure is a set of elements with binary operations and identities
    """
    def __init__(self, elements: List[T], operations: List[Tuple[T, T, T]], identities: List[T]):
        """
        :param elements: A list of elements in the algebraic structure
        :param operations: A list of tuples of the form (a, b, c) where a, b, c are elements in the algebraic structure and a*b = c
        :param identities: A list of elements in the algebraic structure that are the identities of the operations
        """
        self.elements = elements
        self.operations = operations
        self.identities = identities

    def __str__(self):
        return "AlgebraicStructure(elements = {}, operations = {}, identities = {})".format(self.elements, self.operations, self.identities)

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
        :param other: An algebraic structure
        :return: The sum of the two algebraic structures
        """
        if self.elements != other.elements:
            raise ValueError("The algebraic structures are not the same")
        if self.operations != other.operations:
            raise ValueError("The algebraic structures are not the same")

class Field(AlgebraicStructure(T)):
    """A class for representing a field. A field is an algebraic structure with operations of addition and multiplication, the following identities:
       Associativity and commutativity of addition and multiplication, additive and multiplicative identity, additive inverses,
       multiplicative inverses, and distributivity of multiplication over addition."""
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