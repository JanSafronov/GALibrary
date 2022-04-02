"""
Definitions taken from first order logic.
"""

from argparse import Action
from ast import Lambda, Set, Tuple
from dataclasses import Field
from pyclbr import Function
import sys
from typing import Callable, List, TypeVar, Generic
from matplotlib import docstring
import numpy, matplotlib, math
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty

V = TypeVar("V")

class Identity(Generic[V], Lambda, ABC):
    """
    Identity is a true universally quantified formula.
    """
    def __init__(self, variables: set[V], formula: Lambda):
        """
        :param variables: A set of variables
        :param formula: A lambda expression
        """
        self.variables = variables
        self.formula = formula

    def __call__(self, *args):
        return self.formula(*args)

    def __repr__(self):
        return "Identity({}, {})".format(self.variables, self.formula)

    def __str__(self):
        return "for all {} : {}".format(self.variables, self.formula)