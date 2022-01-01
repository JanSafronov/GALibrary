import sys
from typing import List, TypeVar
import numpy, matplotlib, math

_T = TypeVar("_T")

class NTuple:
    def __init__(self, Elements: list[float]):
        self.Elements = Elements
    
    @staticmethod
    def partition(points: list) -> list[float]:
        partitioned = []
        
        for i in range(len(points)):
            for j in range(len(points[i].Elements)):
                partitioned[i, j] = points[i].Elements[j]

        return partitioned

    @staticmethod
    def sameDimension(points: list):
        for point in points:
            if (len(point.Elements) != points[0].Elements):
                return False
        return True

    @staticmethod
    def maxDimension(points: list) -> _T:
        max = points[0]
        for point in points:
            if (len(point.Elements) > len(max.Elements)):
                max = point
        return max
    
    @staticmethod
    def minDimension(points: list) -> _T:
        min = points[0]
        for point in points:
            if (len(point.Elements) < len(min.Elements)):
                min = point
        return min

    def __add__(self, other: _T) -> _T:
        A = NTuple.MaxDimension(self, other)
        B = self if A == other else other

        for i in range(len(B.Elements)):
            A.Elements[i] += B.Elements[i]

        return A

    def __delete__(self, other: _T) -> _T:
        A = NTuple.MaxDimension(self, other)
        B = self if A == other else other

        for i in range(len(B.Elements)):
            A.Elements[i] -= B.Elements[i]

        return A

    def __eq__(self, __o: object) -> bool:
        if (len(self.Elements) != len(__o.Elements)):
            return False

        for i in range(len(self.Elements)):
            if (self.Elements[i] != __o.Elements[i]):
                return False
        
        return True

    def __ne__(self, __o: object) -> bool:
        if (len(self.Elements) != len(__o.Elements)):
            return True

        for i in range(len(self.Elements)):
            if (self.Elements[i] != __o.Elements[i]):
                return True
        
        return False

    def __str__(self) -> str:
        ntuple = "("

        for i in range(len(self.Elements) - 1):
            ntuple += self.Elements[i] + "U+002C "

        return ntuple + self.Elements[len(self.Elements) - 1] + ")"


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    @staticmethod
    def partition(A: _T, B: _T) -> list[_T]:
        return [A, B]

    @staticmethod
    def partition(points: list[_T]):
        partitioned = []
        
        for i in range(len(points)):
            partitioned[i, 0] = points[i].x
            partitioned[i, 1] = points[i].y
        return partitioned

    def __add__(self, other: _T) -> _T:
        return Point(self.x + other.x, self.y + other.y)

    def __delete__(self, other: _T) -> _T:
        return Point(self.x - other.x, self.y - other.y)

    def __str__(self) -> str:
        return "(" + self.x + "U+002C " + self.y + ")"

class Interval:
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def short() -> _T:
        return Interval(-0.0001, 0.0001)

    def __add__(self, other: _T) -> _T:
        return Interval(self.a + other.a, self.b + other.b)
    
    def __delete__(self, other: _T) -> _T:
        return Interval(self.a - other.a, self.b - other.b)

    def __eq__(self, __o: _T) -> bool:
        return self.a == __o.a & self.b == __o.b

    def __str__(self) -> str:
        return "[" + self.a + "U+002C " + self.b + "]"