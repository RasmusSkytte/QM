#!/usr/bin/env python3

# Use numpy as backend for storing data
import numpy as np
import numbers

# Define the state class
class state :

    # Class initializer
    def __init__(self, data) :

        # Verify that data has the right format
        if isinstance(data, list) :  # In list format, find the inner most list
            while isinstance(data[0], list) :
                data = data[0]

            # Store the data as a numpy array
            self.array = np.array(data)

        elif isinstance(data, np.ndarray) : # In numpy array format, reshape to be shape (1,N)
            # Get current shape
            shape = np.shape(data)

            # data is numpy 1-D type
            if np.size(shape) == 1 :
                # Reshape the output to be (1,N)
                self.array = data

            # data is 2-D type
            else :
                dx, dy = sorted(shape)

                # Verify input is 1-D
                if dx != 1 :
                    raise Exception('Input must be 1-D. Are you trying to make an operator?')

                # Reshape the output to be of shape (N,)
                self.array = np.reshape(data,(dy,))

        # Define array interface
        self.__array_interface__ = self.array.__array_interface__


    # Define the hermitian transpose operator
    def T(self) :
        if isinstance(self, bra) :
            return ket(np.conj(self.array))

        elif isinstance(self, ket) :
            return bra(np.conj(self.array))

        else :
            raise NotImplementedError


    # Define conversion to probability
    def prob(self) :
        return np.multiply(self.array,self.array)

    # Define conversion to numpy array
    def asarray(self) :
        return self.array


    # Return a string representation of the data in the state
    def array_str(self) :

        # Ensure truncated print output
        np.set_printoptions(threshold=10)

        # Format based on type
        if isinstance(self, bra) :
            # Get horizontally formatted string
            return np.array_str(self.array)

        elif isinstance(self, ket) :
            # Get vertically formatted string
            return np.array_str(np.reshape(self.array,(np.size(self.array),1)))
        else :
            NotImplementedError


    # Define the additiopn operators
    def __add__(self, other) :

        # Compare the types
        if type(self) == type(other) :
            # If they have same type, addiion can be made
            return self.__class__(self.array + other.array)

        else :
            # If the have different type, addition is not defined
            raise Exception('Must have same type! Cannot add ket and bra')


    # Define the subtraction operator
    def __sub__(self, other) :

        # Compare the types
        if type(self) == type(other) :
            # If they have same type, subtraction can be made
            return self.__class__(self.array - other.array)

        else :
            # If the have different type, subtraction is not defined
            raise Exception('Must have same type! Cannot subtract ket and bra')


    # Define the multiplication operator
    def __mul__(self, other) :

        if type(self) != type(other) :

            # If they have different type, multiplication can be made
            if isinstance(self, bra) and isinstance(other, ket):
                # Compute the inner product
                return np.dot(self.array, other.array)

            elif isinstance(self, ket) and isinstance(other, bra) :
                # Compute the outer product
                return operator(np.outer(self.array, other.array))

            elif isinstance(self, bra) and isinstance(other, operator) :
                # Compute the matrix multiplication
                return bra(np.dot(self.array, other.array))

            else :
                raise NotImplementedError
        else :
            # If the have same type, multiplication is not defined
            raise Exception('Must have different type! Cannot evaluate <v|<v| or |v>|v>')

    # Define the unary negation operator
    def __neg__(self) :
        return self.__class__(-self.array)

    # Define the index operators
    def __getitem__(self, index) :
        return self.array[index]

    def __setitem__(self, index, value) :
        self.array[index] = value

# Define the bra class
class bra(state) :

    # Class initializer
    def __init__(self, data):

        # Copy the parents information
        return super().__init__(data)

    # Define print string
    def __str__(self):

        # Format the output to show bra notation
        return '<v| = ' + self.array_str()


# Define the ket class
class ket(state) :

    # Class initializer
    def __init__(self, data):

        # Copy the parents information
        return super().__init__(data)

    # Define print string
    def __str__(self):

        # Format the output to show ket notation
        return '|v> = ' + str.replace(self.array_str(),'\n','\n      ')


# Define the operator class
class operator :

    def __init__(self, data):
        # Store the data
        self.array = data

        # Define array interface
        self.__array_interface__ = self.array.__array_interface__

    # Define eigenvalue function
    def eig(self) :
        # Use numpy to compute eigenvalues and eigenvectors
        w, v = np.linalg.eig(self.array)

        # Sort eigenvalues and eigenvectors according to eigenvalue
        I = np.argsort(w)
        w = w[I]
        v = v[:,I]

        # Store the output as a list of bra's
        v = [bra(u) for u in v.T]

        return w, v

    # Define print string
    def __str__(self):

        # Ensure truncated print output
        np.set_printoptions(threshold=10)

        # Format the output to show ket notation
        return u'O = ' + str.replace(np.array_str(self.array),'\n','\n    ')

    # Define conversion to numpy array
    def asnumpy(self) :
        return self.array


    # Return a string representation of the data in the state
    def array_str(self) :

        # Ensure truncated print output
        np.set_printoptions(threshold=10)

        return np.array_str(self.array)


    # Define the addition operators
    def __add__(self, other) :

        # Compare the types
        if isinstance(other, operator) :
            return operator(self.array + other.array)

        else :
            raise NotImplementedError


    # Define the subtraction operator
    def __sub__(self, other) :

        # Compare the types
        if isinstance(other, operator) :
            return operator(self.array - other.array)

        else :
            raise NotImplementedError


    # Define the multiplication operator
    def __mul__(self, other) :

        # Compare the types
        if isinstance(other, numbers.Number) :
            # Multiply each element with the scalar
            return operator(self.array * other)

        elif isinstance(other, ket) :
            # Compute the matrix multiplication
            return ket(np.dot(self.array, other.array))

        else :
            raise NotImplementedError

    # Define the true division operator
    def __truediv__(self, other) :

        # Compare the types
        if isinstance(other, numbers.Number) :
            return operator(self.array / other)

        else :
            raise NotImplementedError

    # Define the unary negation operator
    def __neg__(self) :
        return operator(-self.array)

    # Define the index operators
    def __getitem__(self, index) :
        return self.array[index]

    def __setitem__(self, index, value) :
        self.array[index] = value