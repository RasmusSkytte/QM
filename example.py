#!/usr/bin/env python3

import QM as q
import numpy

# Create a bra and a ket using lists
v = q.bra([1,2,3,4])
u = q.ket([4,1,2,9])

# You can also create bra and ket using numpy arrays
w = q.bra(numpy.array([1,2,3,4]))

# Use print to see the new output
print('print() on a bra outputs:')
print(v)
print('')
print('print() on a ket outputs:')
print(u)
print('')

# Perform arithmitic using bra and ket
print('Testing arithmitic')
print('v*u = <v||u> =')
print(v*u)
print('')

print('u*v = |u><v| = ')
print(u*v)
print('')

print('v*v.T() = <v||v†> = ')
print(v*v.T())
print('')

# Numpy functions work directly with the objects
print('Testing numpy interaction')
print('numpy.shape(u)')
print(numpy.shape(u))
print('')

print('numpy.shape(v)')
print(numpy.shape(v))
print('')

print('numpy.linalg.norm(u)')
print(numpy.linalg.norm(u))
print('')
