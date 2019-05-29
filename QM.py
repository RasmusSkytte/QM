#!/usr/bin/env python3

# Use numpy as backend for storing data
import numpy as np
import numbers
import os

# Define the state class


class state(np.ndarray):

    # Class initializers
    def __new__(cls, input_array):
        return np.asarray(input_array).view(cls)

    def __init__(self, input_array):
        return None

    def __array_finalize__(self, obj):
        return None

    # Define the transpose operator
    @property
    def T(self):
        if isinstance(self, bra):
            # bra becomes a ket
            return ket(self)

        elif isinstance(self, ket):
            # ket becomes a bra
            return bra(self)

        elif isinstance(self, operator):
            # Operator stays operator
            return super().T

        else:
            raise NotImplementedError

    def transpose(self):
        return self.T

    # Define the hermitian transpose operator
    @property
    def H(self):
        if isinstance(self, bra):
            # bra becomes a ket
            return ket(np.conj(self))

        elif isinstance(self, ket):
            # ket becomes a bra
            return bra(np.conj(self))

        elif isinstance(self, operator):
            # Operator stays operator
            return operator(np.conj(self.T))

        else:
            raise NotImplementedError

    # Define conversion to probability
    def prob(self):
        # This function is only defined for a bra or a ket
        if isinstance(self, bra) or isinstance(self, ket):
            return np.real(np.multiply(self, np.conj(self)))
        else:
            raise NotImplementedError

    # Return a string representation of the data in the state
    def array_str(self):

        # Ensure truncated print output
        np.set_printoptions(precision=4, threshold=10)

        # Format based on type
        if isinstance(self, bra) or isinstance(self, operator):
            # Get horizontally formatted string
            return np.array_str(self)

        elif isinstance(self, ket):
            # Get vertically formatted string
            return np.array_str(np.reshape(self, (self.size, 1)))
        else:
            NotImplementedError

    # Define the addition operators
    def __add__(self, other):

        # Compare the types
        if type(self) == type(other):
            # If they have same type, addition can be made
            return super().__add__(other)

        elif isinstance(other, numbers.Number):
            # If it is just a number, just add each value
            return super().__add__(other)

        else:
            # If the have different type, addition is not defined
            raise Exception('Must have same type! Cannot add ket and bra')

    # Define the subtraction operator
    def __sub__(self, other):

        # Compare the types
        if type(self) == type(other):
            # If they have same type, subtraction can be made
            return super().__sub__(other)

        elif isinstance(other, numbers.Number):
            # If it is just a number, just subtract each value
            return super().__sub__(other)

        else:
            # If the have different type, subtraction is not defined
            raise Exception('Must have same type! Cannot subtract ket and bra')

    # Define the multiplication operator
    def __mul__(self, other):

        # The type of the objects determines how multiplication is done
        if isinstance(self, bra) and isinstance(other, ket):
            # Compute the inner product
            return np.dot(self, other)

        elif isinstance(self, ket) and isinstance(other, bra):
            # Compute the outer product
            return operator(np.outer(self, other))

        elif isinstance(self, bra) and isinstance(other, operator):
            # Compute the matrix multiplication
            return bra(np.dot(self, other))

        elif isinstance(self, operator) and isinstance(other, ket):
            # Compute the matrix multiplication
            return ket(np.dot(self, other))

        elif isinstance(self, operator) and isinstance(other, operator):
            # Compute the matrix multiplication
            return operator(np.dot(self, other))

        elif isinstance(other, numbers.Number):
            # Otherwise, just multiply each value
            return super().__mul__(other)

        # If the have same type, multiplication is not defined
        elif isinstance(self, bra) and isinstance(other, bra):
            raise Exception('Must have different type! Cannot evaluate <v|<v|')
        elif isinstance(self, ket) and isinstance(other, ket):
            raise Exception('Must have different type! Cannot evaluate |v>|v>')

        else:
            raise NotImplementedError

    # Define representation
    def __repr__(self):
        if isinstance(self, bra) or isinstance(self, ket):
            return self.__name__() + '(' + np.array_str(self) + ')'
        elif isinstance(self, operator):
            return self.__name__() + '(' + str.replace(np.array_str(self), '\n', '\n         ') + ')'
        elif isinstance(self, state):
            return np.array_str(self)
        else:
            raise NotImplementedError

# Define the bra class


class bra(state):

    # Class initializer
    def __new__(self, data):
        # Copy the parents information
        return super().__new__(bra, verify_data_format(data))

    # Define print string
    def __str__(self):
        # Format the output to show bra notation
        return '<v| = ' + self.array_str()

    # Define name
    def __name__(self):
        return 'bra'

# Define the ket class


class ket(state):

    # Class initializer
    def __new__(self, data):
        # Copy the parents information
        return super().__new__(ket, verify_data_format(data))

    # Define print string
    def __str__(self):
        # Format the output to show ket notation
        return '|v> = ' + str.replace(self.array_str(), '\n', '\n      ')

    # Define name
    def __name__(self):
        return 'ket'

# Define the operator class


class operator(state):

    def __new__(self, data):
        # Copy the parents information
        return super().__new__(operator, verify_data_format(data, dim='operator'))

    # Define eigenvalue function
    def eig(self):
        # Use numpy to compute eigenvalues and eigenvectors
        w, v = np.linalg.eig(self)

        # Sort eigenvalues and eigenvectors according to eigenvalue
        I = np.argsort(w)
        w = w[I]
        v = v[:, I]

        # Store the output as a list of kets's
        v = operator(v.T)

        return w, v

    # Define print string
    def __str__(self):
        # Format the output to show ket notation
        return u'O = ' + str.replace(np.array_str(self), '\n', '\n    ')

    # Define name
    def __name__(self):
        return 'operator'

    # Overwrite the index operators
    def __getitem__(self, index):
        if isinstance(index, tuple):
            return super().__getitem__(index)
        else:
            return bra(super().__getitem__(index))

# Define function to verify the data inputs


def verify_data_format(data, dim='1D'):

    # Convert lists to numpy arrays
    if isinstance(data, list):
        data = np.array(data)

    if isinstance(data, np.ndarray):  # In numpy array format, reshape to be shape (N,) or (N,N)

        # Get current shape
        shape = np.shape(data)

        # data is numpy 1-D type and already the right format
        if np.size(shape) == 1 and dim == '1D':
            return data

        # data is 2-D type
        elif np.size(shape) == 2:
            dx, dy = sorted(shape)

            # Process 1D data types
            if dim == '1D':

                # verify data is 1D
                if dx != 1:
                    raise Exception(
                        'Input must be 1-D. Are you trying to make an operator?')

                # Reshape the output to be of shape (N,)
                return np.reshape(data, (dy,))

            elif dim == 'operator':

                # Verify data is a square matrix
                if dx != dy:
                    raise Exception('Input must be a square matrix')

                # Reshape the output to be of shape (N,N)
                return data

            # Type not recognized
            else:
                raise NotImplementedError

        # Data type unknown
        else:
            raise NotImplementedError

# Define video writer function


def make_video(fmtstr, framerate=30):
    # fmtstr includes information of where the images are stored and how they are named
    # e.g. fmtstr = 'video/%3d.png'

    # Generate an output path
    outputpath = 'video/video.mp4'

    # Check that ffmpeg is installed
    try:
        os.system('ffmpeg -r %d -f image2 -i %s -vcodec libx264 -crf 25 -pix_fmt yuv420p %s' %
                  (framerate, fmtstr, outputpath))
    except:
        raise Exception('Could not find ffmpeg. Are you sure it is installed?')


# Define a few useful quantities
c = 299792458         # [m / s]   Speed of light
h = 6.62606896e-34    # [J]       Planck Constant
hbar = 1.054571628e-34   # [J]       Planck Constant / 2 pi (Diracs constant)
eV = 1.602176565e-19   # [J]       Electron Volt
m_electron = 9.10938356e-31    # [kg]      Mass of electron
pi = np.pi  # Pi
