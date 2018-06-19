"""Provide module mixins for data managing classes."""
from typing import TypeVar, Any, Callable, Union
from numbers import Number
import numpy as np

T = TypeVar('T', bound='OpDelegatorMixin')
OwnedData = TypeVar('OwnedData')


class OpDelegatorMixin(object):  # pylint:disable=R0903
    """Provides a data management class with basic operators that are delegated
    to it's values member.
    """
    values: Any = None
    dtype = None

    def create_like(self: T, values) -> T:
        raise NotImplementedError

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return False
        if np.issubdtype(self.dtype, np.float_):
            return np.allclose(self.values, other.values)
        return np.array_equal(self.values, other.values)

    def __basic_operation(self: T, member: Callable, other: Union[T, Number]) -> T:
        if isinstance(other, type(self)):
            return self.create_like(member(other.values))
        return self.create_like(member(other))

    def __add__(self: T, other: Union[T, Number]) -> T:
        return self.__basic_operation(self.values.__add__, other)

    def __sub__(self: T, other: Union[T, Number]) -> T:
        return self.__basic_operation(self.values.__sub__, other)

    def __mul__(self: T, other: Union[T, Number]) -> T:
        return self.__basic_operation(self.values.__mul__, other)

    def __pow__(self: T, other: int) -> T:
        return self.create_like(self.values ** other)

    def __truediv__(self: T, other: Union[T, Number]) -> T:
        return self.__basic_operation(self.values.__truediv__, other)

    def __floordiv__(self, other: Union[T, Number]):
        if np.issubdtype(self.values.dtype, np.integer):
            return self.__basic_operation(self.values.__floordiv__, other)
        else:
            raise ValueError("no floordiv on dtype not integer")

    def __and__(self: T, other: T) -> T:
        if np.issubdtype(self.values.dtype, np.bool_):
            return self.__basic_operation(self.values.__and__, other)
        else:
            raise ValueError("no logical operations on dtype not bool")

    def __or__(self: T, other: T) -> T:
        if np.issubdtype(self.values.dtype, np.bool_):
            return self.__basic_operation(self.values.__or__, other)
        else:
            raise ValueError("no logical operations on dtype not bool")
