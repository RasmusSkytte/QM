#!/usr/bin/env python3

# Use numpy as backend for storing data
import numpy as np

# Define the state class
class state :

    # Class initializer
    def __init__(self, data, foo='bra') :   # TODO: find a good name for 'foo'

        # Verify that data has the right format
        if isinstance(data, list) :  # In list format, find the inner most list
            while isinstance(data[0], list) :
                data = data[0]

            # Store the data as a numpy array (of complex type)
            self.array = np.array([data], dtype=complex)

        elif isinstance(data, np.ndarray) : # In numpy array format, reshape to be shape (1,N)
            # Get current shape
            shp = np.shape(data)

            # data is annoying numpy 1-D type
            if np.size(shp) == 1 :
                # Reshape the output to be (1,N)
                self.array = np.reshape(data,(1,np.size(data))).astype(complex)

            # data is 2-D type
            else :
                dx, dy = sorted(shp)

                # Verify input is 1-D
                if dx != 1 :
                    raise Exception('Input must be 1-D. Are you trying to make an operator?')

                # Reshape the output to be (1,N)
                self.array = np.reshape(data,(1,dy)).astype(complex)

        # Store the type of object
        self.foo  = foo

        # If state is 'ket', transpose the array
        if self.foo == 'ket':
            self.array = np.transpose(self.array)

        # Define array interface
        self.__array_interface__ = self.array.__array_interface__


    # Define the hermitian transpose operator
    def T(self) :
        if self.foo == 'bra' :
            return ket(np.conj(self.array))
        elif self.foo == 'ket' :
            return bra(np.conj(self.array))
        else :
            raise NotImplementedError


    # Define conversion to numpy array
    def asnumpy(self) :
        return self.array


    # Return a string representation of the data in the state
    def array_str(self) :

        # Ensure truncated print output
        np.set_printoptions(threshold=10)

        return np.array_str(self.array)


    # Define the additiopn operators
    def __add__(self, other) :

        # Compare the types
        if self.foo == other.foo :
            # If they have same type, addiion can be made
            return self.__class__((self.array + other.array)[0])
        else :
            # If the have different type, addition is not defined
            raise Exception('Must have same type! Cannot add ket and bra')


    # Define the subtraction operator
    def __sub__(self, other) :

        # Compare the types
        if self.foo == other.foo :
            # If they have same type, subtraction can be made
            return self.__class__((self.array - other.array)[0])
        else :
            # If the have different type, subtraction is not defined
            raise Exception('Must have same type! Cannot subtract ket and bra')


    # Define the multiplication operator
    def __mul__(self, other) :
        if self.foo != other.foo :
            # If they have different type, multiplication can be made
            if self.foo == 'bra' :
                # Compute the inner product
                return np.dot(self.array, other.array)[0][0]
            elif self.foo == 'ket' :
                # Compute the outer product
                return np.outer(self.array, other.array)
            else :
                raise NotImplementedError
        else :
            # If the have same type, multiplication is not defined
            raise Exception('Must have different type! Cannot evaluate <v|<v| or |v>|v>')

    # Define the index operators
    def __getitem__(self, index) :
        if self.foo == 'bra' :
            return self.array[0][index]
        elif self.foo == 'ket' :
            return self.array[index][0]
        else :
            raise NotImplementedError
    def __setitem__(self, index, value) :
        if self.foo == 'bra' :
            self.array[0][index] = value
        elif self.foo == 'ket' :
            self.array[index][0] = value
        else :
            raise NotImplementedError

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
        return super().__init__(data, foo='ket')

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

    # Define print string
    def __str__(self):

        # Ensure truncated print output
        np.set_printoptions(threshold=10)

        # Format the output to show ket notation
        return 'Ô = ' + str.replace(np.array_str(self.array),'\n','\n    ')

    # Define conversion to numpy array
    def asnumpy(self) :
        return self.array


    # Return a string representation of the data in the state
    def array_str(self) :

        # Ensure truncated print output
        np.set_printoptions(threshold=10)

        return np.array_str(self.array)


    # Define the additiopn operators
    def __add__(self, other) :
        raise NotImplementedError


    # Define the subtraction operator
    def __sub__(self, other) :
        raise NotImplementedError


    # Define the multiplication operator
    def __mul__(self, other) :
        raise NotImplementedError


    # Define the index operators
    def __getitem__(self, index) :
        return self.array[index]
    def __setitem__(self, index, value) :
        self.array[index] = value