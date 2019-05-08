#!/usr/bin/env python3

def prop_associative(x, y, z) :
    return (x * y) * z == x * (y * z)

def prop_distributive(x, y, z) :
    return x * (y + z) == (x * y) + (x * z)

def prop_identity(x) :
    return (1 * x) == x

def prop_zero(x) :
    return (0 * x) == 0


import QM as q
import numpy as np


# Test the bra and ket class
x = q.bra([1,3,5,2])    # <x|
y = q.ket([4,7,2,9])    # |y>
z = q.bra([8,4,1,0])    # <z|

# Test the associative property
# ( <x||y> ) <z| = <x| ( |y><z| )
assert(prop_associative(x, y, z))

# Test distributive property
# <x| ( |y> + |z*> ) = <x||y> + <x||z*>
assert(prop_distributive(x, y, z.T()))

# Test identity property
# 1 * <x| = <x|
assert(prop_identity(x))

# Test zero identity
# 0 * <x| = 0
assert(prop_zero(x))

# Test the operator class
A = q.operator(np.array([[1,3,5],[2,4,7],[2,9,8]]))
B = q.operator(np.array([[6,3,9],[0,7,2],[1,9,3]]))
C = q.operator(np.array([[3,6,2],[8,0,4],[6,8,2]]))

# Test the associative property
# (AB) C = A (B*C)
assert(prop_associative(A, B, C))

# Test distributive property
# A * (B + C) = AB + AC
assert(prop_distributive(A, B, C))

# Test identity property
# 1*A = A
assert(prop_identity(A))

# Test zero identity
# 0 * A = 0
assert(prop_zero(A))

print('All tests passed')