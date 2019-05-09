#!/usr/bin/env python3

# Use numpy as backend for storing data
import numpy as np
import numbers

# Define the state class
class state :

    # Class initializer
    def __init__(self, data) :

        # Store the data
        self.array = data

        # Define array interface
        self.__array_interface__ = self.array.__array_interface__


    # Define the hermitian transpose operator
    def H(self) :
        if isinstance(self, bra) :
            # bra becomes a ket
            return ket(np.conj(self.array))

        elif isinstance(self, ket) :
            # ket becomes a bra
            return bra(np.conj(self.array))

        else :
            raise NotImplementedError


    # Define conversion to probability
    def prob(self) :
        # This function is only defined for a bra or a ket
        if isinstance(self, bra) or isinstance(self, ket) :
            return np.real(np.multiply(self.array,np.conj(self.array)))
        else :
            raise NotImplementedError

    # Define conversion to numpy array
    def asarray(self) :
        return self.array


    # Return a string representation of the data in the state
    def array_str(self) :

        # Ensure truncated print output
        np.set_printoptions(precision=4, threshold=10)

        # Format based on type
        if isinstance(self, bra) or isinstance(self, operator) :
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

        elif isinstance(other, numbers.Number) :
            # If it is just a number, just add each value
            return self.__class__(self.array + other)

        else :
            # If the have different type, addition is not defined
            raise Exception('Must have same type! Cannot add ket and bra')

    # Define the subtraction operator
    def __sub__(self, other) :

        # Compare the types
        if type(self) == type(other) :
            # If they have same type, subtraction can be made
            return self.__class__(self.array - other.array)

        elif isinstance(other, numbers.Number) :
            # If it is just a number, just subtract each value
            return self.__class__(self.array - other)

        else :
            # If the have different type, subtraction is not defined
            raise Exception('Must have same type! Cannot subtract ket and bra')


    # Define the multiplication operator
    def __mul__(self, other) :

        # The type of the objects determines how multiplication is done
        if isinstance(self, bra) and isinstance(other, ket) :
            # Compute the inner product
            return np.dot(self.array, other.array).item()

        elif isinstance(self, ket) and isinstance(other, bra) :
            # Compute the outer product
            return operator(np.outer(self.array, other.array))

        elif isinstance(self, bra) and isinstance(other, operator) :
            # Compute the matrix multiplication
            return bra(np.dot(self.array, other.array))

        elif isinstance(self, operator) and isinstance(other, ket) :
            # Compute the matrix multiplication
            return ket(np.dot(self.array, other.array))

        elif isinstance(self, operator) and isinstance(other, operator) :
            # Compute the matrix multiplication
            return operator(np.dot(self.array, other.array))

        elif isinstance(other, numbers.Number) :
            # Treat the zero case on its own
            if other.value == 0 :
                return 0

            # Otherwise, just multiply each value
            return self.__class__(self.array * other)

        # If the have same type, multiplication is not defined
        elif isinstance(self, bra) and isinstance(other, bra) :
            raise Exception('Must have different type! Cannot evaluate <v|<v|')
        elif isinstance(self, ket) and isinstance(other, ket) :
            raise Exception('Must have different type! Cannot evaluate |v>|v>')

        else :
            raise NotImplementedError

    # Define the true division operator
    def __truediv__(self, other) :

        # Compare the types
        if isinstance(other, numbers.Number) :
            return self.__class__(self.array / other)

        else :
            raise NotImplementedError


    # Define the unary negation operator
    def __neg__(self) :
        return self.__class__(-self.array)


    # Define the right hand operators (add, sub, and mul)
    def __radd__(self, other) :
        # Compare the types
        if isinstance(other, numbers.Number) :
            return self.__class__(self.array + other)
        else :
            raise NotImplementedError
    def __rsub__(self, other) :
        # Compare the types
        if isinstance(other, numbers.Number) :
            return self.__class__(self.array - other)
        else :
            raise NotImplementedError
    def __rmul__(self, other) :
        # Compare the types
        if isinstance(other, numbers.Number) :
            if other == 0 : # Check if the number is zero
                return 0

            else :  # Else multiply each element with the scalar
                return self.__class__(self.array * other)
        else :
            raise NotImplementedError

    # Define equal opeator
    def __eq__(self, other) :
        # Compare the types
        if isinstance(other, self.__class__) and np.all(self.array==other.array) :
            return True
        elif isinstance(other, numbers.Number) :
            return self.array == other
        else :
            raise NotImplementedError

    # Define less than operator
    def __lt__(self, other) :
        # Compare the types
        if isinstance(other, numbers.Number) :
            return self.array < other
        else :
            raise NotImplementedError

    # Define less than or equal operator
    def __le__(self, other) :
        # Compare the types
        if isinstance(other, numbers.Number) :
            return self.array <= other
        else :
            raise NotImplementedError

    # Define greater than operator
    def __gt__(self, other) :
        # Compare the types
        if isinstance(other, numbers.Number) :
            return self.array > other
        else :
            raise NotImplementedError

    # Define greater than or equal operator
    def __ge__(self, other) :
        # Compare the types
        if isinstance(other, numbers.Number) :
            return self.array >= other
        else :
            raise NotImplementedError

    # Define the index operators
    def __getitem__(self, index) :
        return self.array[index]

    def __setitem__(self, index, value) :
        self.array[index] = value


# Define the bra class
class bra(state) :

    # Class initializer
    def __init__(self, data) :

        # Copy the parents information
        return super().__init__(verify_data_format(data))

    # Define print string
    def __str__(self) :

        # Format the output to show bra notation
        return '<v| = ' + self.array_str()


# Define the ket class
class ket(state) :

    # Class initializer
    def __init__(self, data) :

        # Copy the parents information
        return super().__init__(verify_data_format(data))

    # Define print string
    def __str__(self) :

        # Format the output to show ket notation
        return '|v> = ' + str.replace(self.array_str(),'\n','\n      ')


# Define the operator class
class operator(state) :

    def __init__(self, data) :

        # Copy the parents information
        return super().__init__(verify_data_format(data, type='operator'))

    # Define eigenvalue function
    def eig(self) :
        # Use numpy to compute eigenvalues and eigenvectors
        w, v = np.linalg.eig(self.array)

        # Sort eigenvalues and eigenvectors according to eigenvalue
        I = np.argsort(w)
        w = w[I]
        v = v[:,I]

        # Store the output as a list of kets's
        v = [ket(u) for u in v.T]

        return w, v

    # Define print string
    def __str__(self) :

        # Format the output to show ket notation
        return u'O = ' + str.replace(np.array_str(self.array),'\n','\n    ')


# Define function to verify the data inputs
def verify_data_format(data, type='1D') :

    # Convert lists to numpy arrays
    if isinstance(data, list) :
        data = np.array(data)

    if isinstance(data, np.ndarray) : # In numpy array format, reshape to be shape (N,) or (N,N)
        # Get current shape
        shape = np.shape(data)

        # data is numpy 1-D type and already the right format
        if np.size(shape) == 1 :
            return data

        # data is 2-D type
        elif np.size(shape) == 2 :
            dx, dy = sorted(shape)

            # Process 1D data types
            if type == '1D' :

                # verify data is 1D
                if dx != 1 :
                    raise Exception('Input must be 1-D. Are you trying to make an operator?')

                # Reshape the output to be of shape (N,)
                return np.reshape(data,(dy,))

            elif type == 'operator' :

                # Verify data is a square matrix
                if dx != dy :
                    raise Exception('Input must be a square matrix')

                # Reshape the output to be of shape (N,N)
                return data

        # Data type unknown
        else :
            raise NotImplementedError
