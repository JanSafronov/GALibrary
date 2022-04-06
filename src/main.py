from elements import AlgebraicStructure, Monoid
from firstorder import Identity

A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
OP = {lambda x, y: x + y, lambda x, y: x * y, lambda x, y: x - y, lambda x, y: x / y}
ID = {Identity(A, lambda x, y: OP[0](x, y) == OP[0](y, x))}

#print(AlgebraicStructure({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {lambda x, y: x + y, lambda x, y: x - y, lambda x, y: x * y, lambda x, y: x / y}, {Identity({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, lambda x, y: x + y == y + x)}))
print(AlgebraicStructure(A, OP, ID))
print(Monoid({1, 2, 3, 4, 5}, lambda x, y: x * y))