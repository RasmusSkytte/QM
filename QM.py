#!/usr/bin/env python3

# Use numpy as backend for storing data
import numpy as np
import numbers
import os

# Define the state class
class state :

    # Class initializer
    def __init__(self, data) :

        # Store the data
        self.array = data

        # Define array interface
        self.__array_interface__ = self.array.__array_interface__

        # Implementing the numpy attributes
        self.data     = self.array.data       # Python buffer object pointing to the start of the array’s data.
        self.dtype    = self.array.dtype      # Data-type of the array’s elements.
        self.flags    = self.array.flags      # flags 	Information about the memory layout of the array.
        self.flat     = self.array.flat       # A 1-D iterator over the array.
        self.imag     = self.array.imag       # The imaginary part of the array.
        self.real     = self.array.real       # The real part of the array.
        self.size     = self.array.size       # Number of elements in the array.
        self.itemsize = self.array.itemsize   # Length of one array element in bytes.
        self.nbytes   = self.array.nbytes     # Total bytes consumed by the elements of the array.
        self.ndim     = self.array.ndim 	  # Number of array dimensions.
        self.shape    = self.array.shape 	  # Tuple of array dimensions.
        self.strides  = self.array.strides    # Tuple of bytes to step in each dimension when traversing an array.
        self.ctypes   = self.array.ctypes     # An object to simplify the interaction of the array with the ctypes module.
        self.base     = self.array.base 	  # Base object if memory is from some other object.


    # Define the transpose operator
    @property
    def T(self) :
        if isinstance(self, bra) :
            # bra becomes a ket
            return ket(self.array)

        elif isinstance(self, ket) :
            # ket becomes a bra
            return bra(self.array)

        elif isinstance(self, operator) :
            # Operator stays operator
            return operator(self.array.T)

        else :
            raise NotImplementedError

    # Define the hermitian transpose operator
    @property
    def H(self) :
        if isinstance(self, bra) :
            # bra becomes a ket
            return ket(np.conj(self.array))

        elif isinstance(self, ket) :
            # ket becomes a bra
            return bra(np.conj(self.array))

        elif isinstance(self, operator) :
            # Operator stays operator
            return operator(np.conj(self.array.T))

        else :
            raise NotImplementedError

    # Define conversion to probability
    def prob(self) :
        # This function is only defined for a bra or a ket
        if isinstance(self, bra) or isinstance(self, ket) :
            return np.real(np.multiply(self.array,np.conj(self.array)))
        else :
            raise NotImplementedError

    # Define the numpy methods (commented methods can be implemented in the future)
    def all(self, *args, **kwargs) :                    # 	Returns True if all elements evaluate to True.
        return self.array.all(*args, **kwargs)
    def any(self, *args, **kwargs) :                    # 	Returns True if any of the elements of a evaluate to True.
        return self.array.any(*args, **kwargs)
    def argmax(self, *args, **kwargs) :                 # 	Return indices of the maximum values along the given axis.
        return self.array.argmax(*args, **kwargs)
    def argmin(self, *args, **kwargs) :                 # 	Return indices of the minimum values along the given axis of a.
        return self.array.argmin(*args, **kwargs)
    # def argpartition(self, kth, *args) :              # 	Returns the indices that would partition this array.

    def argsort(self, *args, **kwargs) :                # 	Returns the indices that would sort this array.
        return self.array.argsort(*args, **kwargs)
    def astype(self, dtype, *args, **kwargs) :          # 	Copy of the array, cast to a specified type.
        return self.array.astype(dtype, *args, **kwargs)
    # def byteswap(self, *args) :                       # 	Swap the bytes of the array elements
    # def choose(self, choices, *args) :                # 	Use an index array to construct a new array from a set of choices.
    # def clip(self, *args) :                           # 	Return an array whose values are limited to [min, max].
    # def compress(self, condition, *args) :            # 	Return selected slices of this array along given axis.
    def conj(self) :                                    # 	Complex-conjugate all elements.
        return self.array.conj()
    def conjugate(self) :                               # 	Return the complex conjugate, element-wise.
        return self.array.conjugate()
    def cumprod(self, *args, **kwargs) :                # 	Return the cumulative product of the elements along the given axis.
        return self.array.cumprod(*args, **kwargs)
    def cumsum(self, *args, **kwargs) :                 # 	Return the cumulative sum of the elements along the given axis.
        return self.array.cumsum(*args, **kwargs)
    # def diagonal(self, *args) :                       # 	Return specified diagonals.
    # def dot(b[, out]) :                               # 	Dot product of two arrays.
    # def dump(file) :                                  # 	Dump a pickle of the array to the specified file.
    # def dumps() :                                     # 	Returns the pickle of the array as a string.
    # def fill(value) :                                 # 	Fill the array with a scalar value.
    # def flatten(self, *args) :                        # 	Return a copy of the array collapsed into one dimension.
    # def getfield(dtype[, offset]) :                   # 	Returns a field of the given array as a certain type.
    # def item(*args) :                                 # 	Copy an element of an array to a standard Python scalar and return it.
    # def itemset(*args) :                              # 	Insert scalar into an array (scalar is cast to array’s dtype, if possible)
    def max(self, *args, **kwargs) :                    # 	Return the maximum along a given axis.
        return self.array.max(*args, **kwargs)
    def mean(self, *args, **kwargs) :                   # 	Returns the average of the array elements along given axis.
        return self.array.mean(*args, **kwargs)
    def min(self, *args, **kwargs) :                    # 	Return the minimum along a given axis.
        return self.array.min(*args, **kwargs)
    # def newbyteorder(self, *args) :                   # 	Return the array with the same data viewed with a different byte order.
    # def nonzero() :                                   # 	Return the indices of the elements that are non-zero.
    # def partition(kth[, axis, kind, order]) :         # 	Rearranges the elements in the array in such a way that value of the element in kth position is in the position it would be in a sorted array.
    def prod(self, *args, **kwargs) :                   # 	Return the product of the array elements over the given axis
        return self.array.prod(*args, **kwargs)
    # def ptp(self, *args) :                            # 	Peak to peak (maximum - minimum) value along a given axis.
    # def put(indices, values[, mode]) :                # 	Set a.flat[n] = values[n] for all n in indices.
    # def ravel(self, *args) :                          # 	Return a flattened array.
    # def repeat(repeats[, axis]) :                     # 	Repeat elements of an array.
    # def reshape(shape[, order]) :                     # 	Returns an array containing the same data with a new shape.
    # def resize(new_shape[, refcheck]) :               # 	Change shape and size of array in-place.
    def round(self, *args, **kwargs) :                  # 	Return a with each element rounded to the given number of decimals.
        return self.array.round(*args, **kwargs)
    # def searchsorted(v[, side, sorter]) :             # 	Find indices where elements of v should be inserted in a to maintain order.
    # def setfield(val, dtype[, offset]) :              # 	Put a value into a specified place in a field defined by a data-type.
    # def setflags(self, *args) :                       # 	Set array flags WRITEABLE, ALIGNED, (WRITEBACKIFCOPY and UPDATEIFCOPY), respectively.
    def sort(self, *args, **kwargs) :                   # 	Sort an array, in-place.
        self.array.sort(*args, **kwargs)
    # def squeeze(self, *args) :                        # 	Remove single-dimensional entries from the shape of a.
    def std(self, *args, **kwargs) :                    # 	Returns the standard deviation of the array elements along given axis.
        return self.array.std(*args, **kwargs)          #   TODO: Implement the non-biased standard deviation as default
    def sum(self, *args, **kwargs) :                    # 	Return the sum of the array elements over the given axis.
        return self.array.sum(*args, **kwargs)
    # def swapaxes(axis1, axis2) :                      # 	Return a view of the array with axis1 and axis2 interchanged.
    # def take(indices[, axis, out, mode]) :            # 	Return an array formed from the elements of a at the given indices.
    # def tobytes(self, *args) :                        # 	Construct Python bytes containing the raw data bytes in the array.
    # def tofile(fid[, sep, format]) :                  # 	Write array to a file as text or binary (default).
    def tolist(self) :                                  # 	Return the array as a (possibly nested) list.
        return self.array.tolist()
    # def tostring(self, *args) :                       # 	Construct Python bytes containing the raw data bytes in the array.
    def trace(self, *args, **kwargs) :                  # 	Return the sum along diagonals of the array.
        if isinstance(self, operator) :
            return self.array.trace(*args, **kwargs)
        else :
            return NotImplementedError
    def transpose(self, *axes) :                        # 	Returns a view of the array with axes transposed.
        return self.T
    def var(self, *args, **kwargs) :                    # 	Returns the variance of the array elements, along given axis.
        return self.array.var(*args, **kwargs)          #   TODO: Implement the non-biased standard deviation as default
    # def view(self, *args) :                           # 	New view of array with the same data.

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


    # Define the addition operators
    def __add__(self, other) :

        # Compare the types
        if type(self) == type(other) :
            # If they have same type, addition can be made
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

    # Define equal operator
    def __eq__(self, other) :
        # Compare the types
        if isinstance(other, self.__class__) and np.all(self.array==other.array) :
            return True
        elif isinstance(other, numbers.Number) :
            return self.array == other
        elif isinstance(other, list) :
            return [a==b for a, b in zip(self, other)]
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

    # Define representation
    def __repr__(self):
        if isinstance(self, bra) or isinstance(self, ket) :
            return self.__name__() + '(' + np.array_str(self.array) + ')'
        elif isinstance(self, operator) :
            return self.__name__() + '(' +str.replace(np.array_str(self.array),'\n','\n         ') + ')'
        else :
            raise NotImplementedError

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

    # Define name
    def __name__(self) :
        return 'bra'

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

    # Define name
    def __name__(self) :
        return 'ket'

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
        v = operator(v.T)

        return w, v

    # Define print string
    def __str__(self) :

        # Format the output to show ket notation
        return u'O = ' + str.replace(np.array_str(self.array),'\n','\n    ')

    # Define name
    def __name__(self) :
        return 'operator'

    # Overwrite the index operators
    def __getitem__(self, index) :
        if isinstance(index, tuple) :
            return self.array[index]
        else :
            return bra(self.array[index])

# Define function to verify the data inputs
def verify_data_format(data, type='1D') :

    # Convert lists to numpy arrays
    if isinstance(data, list) :
        data = np.array(data)

    if isinstance(data, np.ndarray) : # In numpy array format, reshape to be shape (N,) or (N,N)
        # Get current shape
        shape = np.shape(data)

        # data is numpy 1-D type and already the right format
        if np.size(shape) == 1 and type == '1D':
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

            # Type not recognized
            else :
                raise NotImplementedError

        # Data type unknown
        else :
            raise NotImplementedError

# Define video writer function
def make_video(fmtstr, framerate=30) :
    # fmtstr includes information of where the images are stored and how they are named
    # e.g. fmtstr = 'video/%3d.png'

    # Generate an output path
    outputpath = 'video/video.mp4'

    # Check that ffmpeg is installed
    try :
        os.system('ffmpeg -r %d -f image2 -i %s -vcodec libx264 -crf 25 -pix_fmt yuv420p %s' % (framerate, fmtstr, outputpath))
    except:
        raise Exception('Could not find ffmpeg. Are you sure it is installed?')

# Define a few useful quantities
c           = 299792458         # [m / s]   Speed of light
h           = 6.62606896e-34    # [J]       Planck Constant
hbar        = 1.054571628e-34   # [J]       Planck Constant / 2 pi (Diracs constant)
eV          = 1.602176565e-19   # [J]       Electron Volt
m_electron  = 9.10938356e-31    # [kg]      Mass of electron
pi          = np.pi             #           Pi