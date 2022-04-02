"""
Definitions taken from first order logic.
"""

from argparse import Action
from ast import Lambda, Set, Tuple
from dataclasses import Field
from pyclbr import Function
import sys
from typing import List, TypeVar, Generic
from matplotlib import docstring
import numpy, matplotlib, math
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from Algebraic import VectorSpace

V = TypeVar("V")

class Identity(Generic(V)):
    """
    Identity is a true universally quantified formula.
    """
    def __init__(self, formula: Lambda(), variables: List[V]):
        """
        :param formula: A lambda expression
        :param variables: A list of variables
        """
        self.formula = formula
        self.variables = variables

    def __str__(self):
        return "for all {} : {}".format(self.variables, self.formula)