from src.Elements import Field, Vector
import src.Elements
import numpy as np
import matplotlib, math
from ast import Set, Tuple
import sys
from typing import List, TypeVar, Generic
from matplotlib import docstring
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty

F = TypeVar("F", bound=Field)
V = TypeVar("V", bound=Vector)	# Vector

class VectorSpace(Generic[F, V]):
    """A class for representing a vector space. Vector space is a set of vectors with a field and a basis
    """
    def __init__(self, field: F, basis: List[V]):
        """
        :param field: A field
        :param basis: A list of vectors in the vector space
        """
        self.field = field
        self.basis = basis
        self.dimension = len(basis)

    def __str__(self):
        return "VectorSpace(field = {}, basis = {})".format(self.field, self.basis)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.field == other.field and self.basis == other.basis

    def __hash__(self):
        return hash(self.__str__())

    def __contains__(self, vector):
        return vector in self.basis

    def __iter__(self):
        return iter(self.basis)

    def __len__(self):
        return len(self.basis)

    def __getitem__(self, index):
        return self.basis[index]

    def __setitem__(self, index, vector):
        self.basis[index] = vector

    def __delitem__(self, index):
        del self.basis[index]

    def __add__(self, other: "VectorSpace") -> "VectorSpace":
        """
        :param other: A vector space
        :return: The sum of the two vector spaces
        """
        if self.field != other.field:
            raise ValueError("The vector spaces are not in the same field")
        if self.dimension != other.dimension:
            raise ValueError("The vector spaces are not the same dimension")
        return VectorSpace(self.field, [self.basis[i] + other.basis[i] for i in range(self.dimension)])

    def __sub__(self, other: "VectorSpace") -> "VectorSpace":
        """
        :param other: A vector space
        :return: The difference of the two vector spaces
        """
        if self.field != other.field:
            raise ValueError("The vector spaces are not in the same field")
        if self.dimension != other.dimension:
            raise ValueError("The vector spaces are not the same dimension")
        return VectorSpace(self.field, [self.basis[i] - other.basis[i] for i in range(self.dimension)])

    def __mul__(self, other: "VectorSpace") -> F:
        """
        :param other: A vector space
        :return: The tensor product of the two vector spaces
        """
        if self.field != other.field:
            raise ValueError("The vector spaces are not in the same field")
        if self.dimension != other.dimension:
            raise ValueError("The vector spaces are not the same dimension")
        return sum(self.field, [self.basis[i] * other.basis[i] for i in range(self.dimension)])

    def __rmul__(self, scalar: F):
        """
        :param scalar: A scalar
        :return: The scalar product of the vector space with the scalar
        """
        if not isinstance(scalar, src.Elements.Scalar):
            raise ValueError("The argument is not a scalar")
        return VectorSpace(self.field, [self.basis[i] * scalar for i in range(self.dimension)])

class AffineSpace(Generic[V]):
    """A class for representing an affine space. An affine space is a vector space with a point
    """
    def __init__(self, vector_space: VectorSpace, point: V):
        """
        :param vector_space: A vector space
        :param point: A point in the affine space
        """
        self.vector_space = vector_space
        self.point = point

    def __str__(self):
        return "AffineSpace(vector_space = {}, point = {})".format(self.vector_space, self.point)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.vector_space == other.vector_space and self.point == other.point

    def __hash__(self):
        return hash(self.__str__())

    def __contains__(self, vector):
        return vector in self.vector_space

    def __iter__(self):
        return iter(self.vector_space)

    def __len__(self):
        return len(self.vector_space)

    def __getitem__(self, index):
        return self.vector_space[index]

    def __setitem__(self, index, vector):
        self.vector_space[index] = vector

    def __delitem__(self, index):
        del self.vector_space[index]

    def __add__(self, other: "AffineSpace") -> "AffineSpace":
        """
        :param other: An affine space
        :return: The sum of the two affine spaces
        """
        if self.vector_space != other.vector_space:
            raise ValueError("The affine spaces are not in the same vector space")
        return AffineSpace(self.vector_space, self.point + other.point)

    def __sub__(self, other: "AffineSpace") -> "AffineSpace":
        """
        :param other: An affine space
        :return: The difference of the two affine spaces
        """
        if self.vector_space != other.vector_space:
            raise ValueError("The affine spaces are not in the same vector space")
        return AffineSpace(self.vector_space, self.point - other.point)
