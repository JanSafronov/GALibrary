from audioop import cross
from lib2to3.pgen2.pgen import generate_grammar
from typing import Callable, Iterable, Iterator, Union
from matplotlib import pyplot as plt

from sympy import Sum, gcd, lambdify
from inference import BinaryOperator, Distributive
from elements import AlgebraicStructure, Group, Field, Module, Vector
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
    def __init__(self, vectors: Module[F, V], basis: list[V]) -> None:
        """
        :param field: A field
        :param vectors: An abelian group or a module of vectors
        """

        self.field = vectors.ring
        self.vectors = vectors
        self.basis = basis

    def __str__(self):
        return "V = {} vector space over field F = {}".format(self.vectors, self.field)

    def __repr__(self):
        return "VectorSpace(field = {}, vectors = {})".format(self.field, self.vectors)

    def __eq__(self, other: "VectorSpace") -> bool:
        return self.field == other.field and self.vectors == other.vectors

    def __hash__(self) -> int:
        return hash(self.__str__())

    def __contains__(self, vector: Vector) -> bool:
        return vector in self.vectors
    
    def __contains__(self, vector: Vector) -> bool:
        return vector in self.vectors

    def __iter__(self) -> Iterator[V]:
        return iter(self.vectors)

    def __len__(self) -> int:
        return len(self.vectors)

    def __getitem__(self, index) -> V:
        return self.vectors[index]

    def __setitem__(self, index: int, vector: V) -> None:
        self.vectors[index] = vector

    def __delitem__(self, index: int) -> None:
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
        return VectorSpace(self.field, [self.vectors[i] + other.vectors[i] for i in range(self.dimension)])

    def __sub__(self, other: "VectorSpace") -> "VectorSpace":
        """
        :param other: A vector space
        :return: The difference of the two vector spaces
        """
        if self.field != other.field:
            raise ValueError("The vector spaces are not in the same field")
        if self.dimension != other.dimension:
            raise ValueError("The vector spaces are not the same dimension")
        return VectorSpace(self.field, [self.vectors[i] - other.vectors[i] for i in range(self.dimension)])

    def __mul__(self, other: "VectorSpace") -> F:
        """
        :param other: A vector space
        :return: The tensor product of the two vector spaces
        """
        if self.field != other.field:
            raise ValueError("The vector spaces are not in the same field")
        if self.dimension != other.dimension:
            raise ValueError("The vector spaces are not the same dimension")
        return sum(self.field, [self.vectors[i] * other.vectors[i] for i in range(self.dimension)])

    def __rmul__(self, scalar: F) -> "VectorSpace":
        """
        :param scalar: A scalar
        :return: The scalar product of the vector space with the scalar
        """
        if not isinstance(scalar, float):
            raise ValueError("The argument is not a scalar")
        return VectorSpace(self.field, [self.vectors[i] * scalar for i in range(self.dimension)])

    def parametricEquation(self) -> Callable[[tuple[V, ...]], V]:
        """
        :return: A parametric equation for the vector space
        """
        mcrossp = np.cross(self.basis[0].coordinates, self.basis[1].coordinates)
        dim = len(mcrossp)
        return lambda x: sum((mcrossp[:dim - 1] / mcrossp[dim - 1])[i] * x[i] for i in range(dim - 1))

class AffineSpace():
    """
    An affine space is a vector space with a point
    """
    def __init__(self, vector_space: VectorSpace, point: Union[Vector, List[F]]) -> None:
        """
        :param vector_space: A vector space
        :param point: A point in the affine space
        """
        self.vector_space = vector_space
        self.point = point

    def __str__(self) -> str:
        return "V = {} affine space over field F = {}".format(self.vectors, self.field)

    def __repr__(self) -> str:
        return "AffineSpace(vector_space = {}, point = {})".format(self.vector_space, self.point)

    def __eq__(self, other: "AffineSpace") -> bool:
        return self.vector_space == other.vector_space and self.point == other.point

    def __hash__(self) -> int:
        return hash(self.__str__())

    def __contains__(self, vector) -> bool:
        return vector in self.vectors

    def __contains__(self, vector: Vector) -> bool:
        return vector in self.vector_space

    def __iter__(self) -> Iterator[int]:
        return iter(self.vector_space)

    def __len__(self) -> int:
        return len(self.vector_space)

    def __getitem__(self, index: int) -> Vector:
        return self.vector_space[index]

    def __setitem__(self, index: int, vector: Vector) -> None:
        self.vector_space[index] = vector

    def __delitem__(self, index: int) -> None:
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
        mcrossp = np.cross(self.vector_space.basis[0].coordinates, self.vector_space.basis[1].coordinates)
        dim = len(mcrossp)
        return lambda Λ: sum((mcrossp[:dim - 1] / mcrossp[dim - 1])[i] * (Λ[i] + self.point[i]) for i in range(dim - 1))

class AffineVariety(Generic[F, V]):
    """
    An affine variety is a set of solutions of a system of polynomial equations with coefficients in the field
    """
    def __init__(self, field: F, equations: set[Callable[[tuple[V, ...]], F]]) -> None:
        """
        :param field: A field
        :param equations: A list of equations
        """
        self.field = field
        self.equations = equations

    def __str__(self) -> str:
        return "A = {} affine variety with polynomial equations \n{} over field F = {}".format(self.vectors, self.equations, self.field)

    def __repr__(self) -> str:
        return "AffineVariety(field = {}, equations = {})".format(self.field, self.equations)

    def __eq__(self, other: "AffineVariety") -> bool:
        return self.field == other.field and self.equations == other.equations

    def __hash__(self) -> int:
        return hash(self.__str__())

    def __contains__(self, affine_space: AffineSpace) -> bool:
        return affine_space in self.equations

    def __contains__(self, element: F) -> bool:
        return map(self.equations[i](element) == 0 for i in range(len(self.equations)))

    def __iter__(self) -> Iterator[AffineSpace]:
        return iter(self.equations)

    def __len__(self) -> int:
        return len(self.equations)

    def __getitem__(self, index: int) -> Callable[[tuple[V, ...]], F]:
        return self.equations[index]

    def __setitem__(self, index: int, equation: Callable[[tuple[V, ...]], F]) -> None:
        self.equations[index] = equation

    def __delitem__(self, index: int) -> None:
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

    def __del__(self) -> None:
        del self.field
        del self.equations

    def __or__(self, other: "AffineVariety") -> "AffineVariety":
        """
        :param other: An affine variety
        :return: The union of the two affine varieties
        """
        if self.field != other.field:
            raise ValueError("The affine varieties are not in the same field")
        if self.dimension != other.dimension:
            raise ValueError("The affine varieties are not the same dimension")
        return AffineVariety(self.field, (lambda T: self.equations[i](T) + other.equations[i](T) for i in range(self.dimension)))

    def __and__(self, other: "AffineVariety") -> "AffineVariety":
        """
        :param other: An affine variety
        :return: The intersection of the two affine varieties
        """
        if self.field != other.field:
            raise ValueError("The affine varieties are not in the same field")
        if self.dimension != other.dimension:
            raise ValueError("The affine varieties are not the same dimension")
        return AffineVariety(self.field, (lambda T: self.equations[i](T) + other.equations[i](T) for i in range(self.dimension)))

    def solutions(self) -> set[tuple[V, ...]]:
        """
        :param affine_space: An affine space
        :return: The solutions of the affine variety in the affine space
        """
        return set(filter(lambda T: self.equations[i](T) == self.field.zero for i in range(self.dimension)))

class RationalFunction(Generic[F, V]):
    """
    A rational function with coefficients in a field is a quotient of two polynomials 
    """
    def __init__(self, field: F, polynomial: Callable[[tuple[V, ...]], F]) -> None:
        """
        :param field: A field
        :param polynomial: A polynomial
        """
        self.field = field
        self.polynomial = polynomial

    def __str__(self) -> str:
        return "F = {}, f = {})".format(self.field, self.polynomial)

    def __repr__(self) -> str:
        return "RationalFunction(field = {}, polynomial = {})".format(self.field, self.polynomial)

    def __eq__(self, other: "RationalFunction") -> bool:
        return self.field == other.field and self.polynomial == other.polynomial

    def __hash__(self) -> int:
        return hash(self.__str__())

    def __contains__(self, affine_variety: AffineVariety) -> bool:
        return affine_variety in self.polynomial

    def __iter__(self) -> Iterator[AffineVariety]:
        return iter(self.polynomial)

    def __len__(self) -> int:
        return len(self.polynomial)

    def __delitem__(self, index: int) -> None:
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
    A monomial is a polynomial with a single term
    """

    def __init__(self, field: F, degrees: list[int], coefficient: F) -> None:
        """
        :param field: A field
        :param degrees: A tuple of degrees
        :param coefficient: A coefficient
        """
        self.field = field
        self.monomial = lambda X: math.prod(map(lambda x, i: x ** degrees[i], X, range(len(degrees)))) * coefficient

    def __str__(self) -> str:
        return "F = {}, f(X) = {})".format(self.field, self.monomial)

    def __repr__(self) -> str:
        return "Monomial(field = {}, monomial = {})".format(self.field, self.monomial)

    def __eq__(self, other: "Monomial") -> bool:
        return self.field == other.field and self.monomial == other.monomial

    def __hash__(self) -> int:
        return hash(self.__str__())

    def __contains__(self, affine_variety: AffineVariety) -> bool:
        return affine_variety in self.monomial

    def __add__(self, other: "Monomial") -> Callable[[tuple[V, ...]], F]:
        """
        :param other: A monomial
        :return: The sum of the two monomials
        """
        if self.field != other.field:
            raise ValueError("The monomials are not in the same field")
        return lambda X: self.monomial(X) + other.monomial(X)

    def __sub__(self, other: "Monomial") -> Callable[[tuple[V, ...]], F]:
        """
        :param other: A monomial
        :return: The difference of the two monomials
        """
        if self.field != other.field:
            raise ValueError("The monomials are not in the same field ")
        return lambda X: self.monomial(X) - other.monomial(X)

    def plot2d(self, x_range: tuple[float, float], y_range: tuple[float, float], resolution: int = 100) -> None:
        """
        :param x_range: The range of the x-axis
        :param y_range: The range of the y-axis
        :param resolution: The resolution of the plot
        :return: None
        """
        x_values = np.linspace(x_range[0], x_range[1], resolution)
        y_values = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x_values, y_values)
        Z = self.monomial([X, Y])
        plt.show()

    def plot3d(self, x_range: tuple[float, float], y_range: tuple[float, float], z_range: tuple[float, float], resolution: int = 100) -> None:
        """
        :param x_range: The range of the x-axis
        :param y_range: The range of the y-axis
        :param z_range: The range of the z-axis
        :param resolution: The resolution of the plot
        :return: None
        """
        x_values = np.linspace(x_range[0], x_range[1], resolution)
        y_values = np.linspace(y_range[0], y_range[1], resolution)
        z_values = np.linspace(z_range[0], z_range[1], resolution)
        X, Y, Z = np.meshgrid(x_values, y_values, z_values)
        Z = np.array([self.monomial((x, y, z)) for x, y, z in zip(np.ravel(X), np.ravel(Y), np.ravel(Z))])
        Z = Z.reshape(X.shape)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap=plt.cm.jet)
        plt.show()

class Ideal(Generic[F, V]):
    """
    An ideal of an affine variety is a set of polynomials in a polynomial ring over a field that vanish on the variety
    """
    def __init__(self, field: F, affine: AffineVariety[F, V]) -> None:
        """
        :param field: A field
        :param affine: An affine variety
        """
        if BinaryOperator(field, lambda x, y: x * y, False)(field, affine):
            raise ValueError("The ideal doesn't generate the affine variety")
        self.field = field
        self.affine = affine
        self.dimension = len(affine)

    def __str__(self) -> str:
        return "I = {f ∈ k[x_1, ..., x_n] | f(x) = 0 ∀x ∈ V}" + "\n k = {}, V = {}".format(self.field, self.affine)

    def __repr__(self) -> str:
        return "Ideal(field = {}, affine = {})".format(self.field, self.affine)

    def __eq__(self, other: "Ideal") -> bool:
        return self.field == other.field and self.affine == other.affine

    def __hash__(self) -> int:
        return hash(self.__str__())

    def __contains__(self, polynomial: Callable[[V], F]) -> bool:
        return map(polynomial(v) == 0 for v in self.affine)

    def __iter__(self) -> Iterator[Callable[[tuple[V, ...]], F]]:
        return iter(self.equations)

    def __len__(self) -> int:
        return len(self.equations)

    def __getitem__(self, index: int) -> Callable[[tuple[V, ...]], F]:
        return self.equations[index]

    def __setitem__(self, index: int, polynomial: Callable[[tuple[V, ...]], F]) -> None:
        self.affine[index] = polynomial

    def __delitem__(self, index: int) -> None:
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
        return Ideal(self.field, AffineVariety(self.field, [self.equations[i] + other.equations[i] for i in range(self.dimension)]))

    def __div__(self, other: "Ideal") -> "Ideal":
        """
        :param other: An ideal
        :return: The division of the two ideals
        """
        if self.field != other.field:
            raise ValueError("The ideals are not in the same field")
        if self.dimension != other.dimension:
            raise ValueError("The ideals are not the same dimension")
        return Ideal(self.field, self.affine - other.affine)

    def prime(self) -> bool:
        """
        :return: Whether the ideal is prime
        """
        return Ideal.generate(
        )

    def proper(self) -> bool:
        """
        :return: Whether the ideal is proper
        """
        return self != Ideal(self.field, self.field)

    def primary(self) -> bool:
        """
        :return: Whether the ideal is primary
        """
        pass

    def irreducable(self) -> bool:
        """
        :return: True if the ideal is irreducible, False otherwise
        """
        return Ideal(self.field, self.affine).is_prime()

    def is_monomial(self) -> bool:
        """
        :return: True if the ideal is a monomial ideal
        """
        return Ideal.generate(self.field, self.affine.equations) == self
    
    @staticmethod
    def generate(field: F, equations: set[Callable[[tuple[V, ...]], F]]) -> set[Callable[[tuple[V, ...]], F]]:
        """
        :return: A set of linear combinations of ideal's polynomials with polynomials from the affine space's polynomial ring
        """
        return Ideal(field, AffineVariety(field, equations))

    @staticmethod
    def reduce(polynomial: Callable[[tuple[V, ...]], F]) -> Callable[[tuple[V, ...]], F]:
        """
        Note that the algorithm may fail if the field doesn't contain the rational numbers.
        :param polynomial: A polynomial
        :return: The reduction of the polynomial
        """
        return lambda X: polynomial(X) / math.gcd(polynomial(X), np.gradient(polynomial(X)))

def zariski(affine: AffineSpace) -> Ideal[F, V]:
    """
    :param affine: An affine space
    """
    return AffineVariety(affine.field, Ideal(affine.field, affine).equations)