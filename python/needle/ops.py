"""Operator implementations."""

from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a**self.scalar

    def gradient(self, out_grad, node):
        return (self.scalar*out_grad**(self.scalar - 1), )


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a/b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / rhs**2


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return (out_grad / self.scalar,)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is not None:
            return numpy.swapaxes(a, *self.axes)
        else:
            return numpy.swapaxes(a, -1, -2)

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return numpy.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        t, = node.inputs
        return reshape(out_grad, t.shape)


def reshape(a, shape):
    return Reshape(shape)(a)


def br_grad(out_grad, sh0, sh1):
    sh0 = list(sh0)
    sh1 = list(sh1)
    while len(sh1) > len(sh0):
        out_grad = summation(out_grad, 0)
        sh1.pop(0)

    for i in range(len(sh0)):
        if sh0[i] != sh1[i]:
            sh1[i] = sh0[i]
            out_grad = reshape(summation(out_grad, i), sh1)
    return out_grad

class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        t = node.inputs[0]
        return br_grad(out_grad, t.shape, self.shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return numpy.sum(a, self.axes)

    def gradient(self, out_grad, node):
        t, = node.inputs
        shape = list(t.shape)

        if self.axes is None:
            self.axes = [i for i in range(len(shape))]

        transpositions = []
        need_to_transpose = [i in self.axes for i in range(len(shape))]
        #not_axes = [i for i in range(len(shape)) if i not in self.axes]
        for i in range(len(self.axes)):
            if not need_to_transpose[i]:
                for j in range(len(shape)-1, i, -1):
                    if need_to_transpose[j]:
                        break
                transpositions.append((i,j))
                shape[i], shape[j] = shape[j], shape[i]

        res = broadcast_to(out_grad, shape)

        for ax in transpositions:
            res = transpose(res, ax)

        return res


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return numpy.matmul(a, b)

    def gradient(self, out_grad, node):
        # (6,6,5,4) x (4, 3) -> (6,6,5,4) x (6, 6, 4, 3)
        lhs, rhs = node.inputs
        ga = matmul(out_grad, transpose(rhs))
        gb = matmul(transpose(lhs), out_grad)
        
        ga = br_grad(ga, lhs.shape, ga.shape)
        gb = br_grad(gb, rhs.shape, gb.shape)

        return ga, gb


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

