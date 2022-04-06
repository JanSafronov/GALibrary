from audioop import cross
from typing import Callable, Union
from elements import AlgebraicStructure, Group, Field, Module, Vector, Scalar
import elements
import numpy as np
import matplotlib, math
from ast import Set, Tuple
import sys
from typing import List, TypeVar, Generic
from matplotlib import docstring
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty

F = TypeVar("F", bound=Field)
V = TypeVar("V", bound=Vector)	# Vector
T = TypeVar("T")

class VectorSpace(Set, Generic[F, V]):
    """
    Vector space is an abelian group under addition and an F-module of vectors with a field F
    """
    def __init__(self, field: F, vectors: Union[Module[F, V], Group[V]]):
        """
        :param field: A field
        :param vectors: An abelian group or a module of vectors
        """

        self.field = field
        self.vectors = vectors

    def __init__(self, field: F, basis: list[V]):
        """
        :param field: A field
        :param basis: A list of vectors
        """
        self.field = field
        self.basis = basis
        self.vectors = set((lambda Λ: map(lambda v, λ: λ * v, basis, Λ))({x, y, z}) for x in field for y in field for z in field)

    def __str__(self):
        return "V = {} vector space over field F = {}".format(self.vectors, self.field)

    def __repr__(self):
        return "VectorSpace(field = {}, vectors = {})".format(self.field, self.vectors)

    def __eq__(self, other):
        return self.field == other.field and self.vectors == other.vectors

    def __hash__(self):
        return hash(self.__str__())

    def __contains__(self, vector):
        return vector in self.vectors

    def __iter__(self):
        return iter(self.vectors)

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, index):
        return self.vectors[index]

    def __setitem__(self, index, vector):
        self.vectors[index] = vector

    def __delitem__(self, index):
        del self.vectors[index]

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
        if not isinstance(scalar, Scalar):
            raise ValueError("The argument is not a scalar")
        return VectorSpace(self.field, [self.basis[i] * scalar for i in range(self.dimension)])

    def parametricEquation(self) -> Callable[[tuple[V, ...]], V]:
        """
        :return: A parametric equation for the vector space
        """
        print(self.basis)
        mcrossp = np.cross(self.basis[0], self.basis[1])
        print(mcrossp)
        dim = len(mcrossp)
        print(sum(mcrossp[:dim - 1] / mcrossp[dim - 1] * [2, 3]))
        return lambda Λ: sum((mcrossp[:dim - 1] / mcrossp[dim - 1])[i] * Λ[i] for i in range(dim - 1))

class AffineSpace():
    """
    An affine space is a vector space with a point
    """
    def __init__(self, vector_space: VectorSpace, point: Union[Vector, List[F]]):
        """
        :param vector_space: A vector space
        :param point: A point in the affine space
        """
        self.vector_space = vector_space
        self.point = point

    def __str__(self):
        return "V = {} affine space over field F = {}".format(self.vectors, self.field)

    def __repr__(self):
        return "AffineSpace(vector_space = {}, point = {})".format(self.vector_space, self.point)

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

    def parametricEquation(self) -> Callable[[tuple[V, ...]], V]:
        """
        :return: A parametric equation for the affine space
        """
        mcrossp = np.cross(self.vector_space.basis[0], self.vector_space.basis[1])
        dim = len(mcrossp)
        return lambda Λ: sum((mcrossp[:dim - 1] / mcrossp[dim - 1])[i] * (Λ[i] + self.point[i]) for i in range(dim - 1))

class AffineVariety(Generic[F, V]):
    """ 
    An affine variety is a set of solutions of a system of polynomial equations with coefficients in the field
    """
    def __init__(self, field: F, equations: set[Callable[[tuple[V, ...]], F]]):
        """
        :param field: A field
        :param equations: A list of equations
        """
        self.field = field
        self.equations = equations

    def __str__(self):
        return "A = {} affine variety with polynomial equations \n{} over field F = {}".format(self.vectors, self.equations, self.field)

    def __repr__(self):
        return "AffineVariety(field = {}, equations = {})".format(self.field, self.equations)

    def __eq__(self, other):
        return self.field == other.field and self.equations == other.equations

    def __hash__(self):
        return hash(self.__str__())

    def __contains__(self, affine_space: AffineSpace):
        return affine_space in self.equations

    def __iter__(self):
        return iter(self.equations)

    def __len__(self):
        return len(self.equations)

    def __getitem__(self, index):
        return self.equations[index]

    def __setitem__(self, index, affine_space):
        self.equations[index] = affine_space

    def __delitem__(self, index):
        del self.equations[index]

    def __add__(self, other: "AffineVariety") -> "AffineVariety":
        """
        :param other: An affine variety
        :return: The sum of the two affine varieties
        """
        if self.field != other.field:
            raise ValueError("The affine varieties are not in the same field")
        if self.dimension != other.dimension:
            raise ValueError("The affine varieties are not the same dimension")
        return AffineVariety(self.field, self.equations | other.equations)

    def __or__(self, other: "AffineVariety") -> "AffineVariety":
        """
        :param other: An affine variety
        :return: The union of the two affine varieties
        """
        if self.field != other.field:
            raise ValueError("The affine varieties are not in the same field")
        if self.dimension != other.dimension:
            raise ValueError("The affine varieties are not the same dimension")
        return AffineVariety(self.field, self.equations | other.equations)

    def __and__(self, other: "AffineVariety") -> "AffineVariety":
        """
        :param other: An affine variety
        :return: The intersection of the two affine varieties
        """
        if self.field != other.field:
            raise ValueError("The affine varieties are not in the same field")
        if self.dimension != other.dimension:
            raise ValueError("The affine varieties are not the same dimension")
        return AffineVariety(self.field, (lambda T: self.equations[i](T) * other.equations[i](T) for i in range(self.dimension)))

    def solutions(self) -> set[tuple[V, ...]]:
        """
        :param affine_space: An affine space
        :return: The solutions of the affine variety in the affine space
        """
        print(np.roots(self.equations))
        return set(filter(lambda T: self.equations[i](T) == self.field.zero for i in range(self.dimension)))

class RationalFunction(Generic[F, V]):
    """
    A rational function with coefficients in a field is a quotient of two polynomials 
    """
    def __init__(self, field: F, polynomial: Callable[[tuple[V, ...]], F]):
        """
        :param field: A field
        :param polynomial: A polynomial
        """
        self.field = field
        self.polynomial = polynomial

    def __str__(self):
        return "RationalFunction(field = {}, polynomial = {})".format(self.field, self.polynomial)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.field == other.field and self.polynomial == other.polynomial

    def __hash__(self):
        return hash(self.__str__())

    def __contains__(self, affine_variety: AffineVariety):
        return affine_variety in self.polynomial

    def __iter__(self):
        return iter(self.polynomial)

    def __len__(self):
        return len(self.polynomial)

    def __getitem__(self, index):
        return self.polynomial[index]

    def __setitem__(self, index, affine_variety):
        self.polynomial[index] = affine_variety

    def __delitem__(self, index):
        del self.polynomial[index]

    def __add__(self, other: "RationalFunction") -> "RationalFunction":
        """
        :param other: A rational function
        :return: The sum of the two rational functions
        """
        if self.field != other.field:
            raise ValueError("The rational functions are not in the same field")
        return RationalFunction(self.field, self.polynomial + other.polynomial)

    def __sub__(self, other: "RationalFunction") -> "RationalFunction":
        """
        :param other: A rational function 
        :return: The difference of the two rational functions
        """
        if self.field != other.field:
            raise ValueError("The rational functions are not in the same field")
        return RationalFunction(self.field, self.polynomial - other.polynomial)

class Monomial(Generic[F, V]):
    """
    A monomial is a rational function with a single variable
    """
    def __init__(self, field: F, variable: V):
        """
        :param field: A field
        :param variable: A variable
        """
        self.field = field
        self.variable = variable

    def __str__(self):
        return "Monomial(field = {}, variable = {})".format(self.field, self.variable)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.field == other.field and self.variable == other.variable

    def __hash__(self):
        return hash(self.__str__())

    def __contains__(self, affine_variety: AffineVariety):
        return affine_variety in self.variable

    def __iter__(self):
        return iter(self.variable)

    def __len__(self):
        return len(self.variable)

    def __getitem__(self, index):
        return self.variable[index]

    def __setitem__(self, index, affine_variety):
        self.variable[index] = affine_variety

    def __delitem__(self, index):
        del self.variable[index]

    def __add__(self, other: "Monomial") -> "Monomial":
        """
        :param other: A monomial
        :return: The sum of the two monomials
        """
        if self.field != other.field:
            raise ValueError("The monomials are not in the same field")
        return Monomial(self.field, self.variable + other.variable)

    def __sub__(self, other: "Monomial") -> "Monomial":
        """
        :param other: A monomial
        :return: The difference of the two monomials
        """
        if self.field != other.field:
            raise ValueError("The monomials are not in the same field ")
        return Monomial(self.field, self.variable - other.variable)

class Ideal(Generic[F, V]):
    """
    An ideal is a set of solutions of a field of a system of polynomial equations with coefficients in the field
    """
    def __init__(self, field: F, equations: List[Callable[[tuple[V, ...]], F]]):
        """
        :param field: A field
        :param equations: A list of equations
        """
        self.field = field
        self.equations = equations
        self.dimension = len(equations)

    def __str__(self):
        return "I = ⟨⟩(field = {}, equations = {})".format(self.field, self.equations)

    def __repr__(self):
        return "Ideal(field = {}, equations = {})".format(self.field, self.equations)

    def __eq__(self, other):
        return self.field == other.field and self.equations == other.equations

    def __hash__(self):
        return hash(self.__str__())

    def __contains__(self, rational_function: RationalFunction):
        return rational_function in self.equations

    def __iter__(self):
        return iter(self.equations)

    def __len__(self):
        return len(self.equations)

    def __getitem__(self, index):
        return self.equations[index]

    def __setitem__(self, index, rational_function):
        self.equations[index] = rational_function

    def __delitem__(self, index):
        del self.equations[index]

    def __add__(self, other: "Ideal") -> "Ideal":
        """
        :param other: An ideal
        :return: The sum of the two ideals
        """
        if self.field != other.field:
            raise ValueError("The ideals are not in the same field")
        if self.dimension != other.dimension:
            raise ValueError("The ideals are not the same dimension")
        return Ideal(self.field, [self.equations[i] + other.equations[i] for i in range(self.dimension)])

    def is_monomial(self) -> bool:
        """
        :return: True if the ideal is a monomial ideal
        """
        return all(self.equations[i].polynomial.is_monomial() for i in range(self.dimension))