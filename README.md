# Quantum Mechanics Library
A small python quantum mechanics library which simplifies the notation of numpy and tailors it for quantum mechanics teaching
# Installation
The library can be used by placing the <span>QM</span>.py alongside your code, or be installed on your system by typing the following into your terminal:
```bash
python setup.py install
```
# Quick start
```python
# Load library
import QM as q

# bra <v| = [1, 5, 2]
v = q.bra([1,5,2])

# bra |w> = [9, 0, 5]
w = q.ket([9,0,5])

# Operator M = [[7, 3, 0], [2, 7, 3], [8, 4, 5]]
M = q.operator([[7, 3, 0], [2, 7, 3], [8, 4, 5]])

# <v|w>
a = v * w

# <v|v*>
b = v * v.H()

# <v|M|w>
c = v * M * w
```
# Documentation
This library introduces three classes ```bra```, ```ket``` and ```operator```.

All of three objects are array-like:
```bra```, ```ket``` are vector-like, while ```operator``` is matrix-like.

Since the classes are array-like, they all work with numpy functions. (e.g. np.shape(), np.linalg.norm())

