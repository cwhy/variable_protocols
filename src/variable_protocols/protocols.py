from __future__ import annotations
import abc
from typing import Literal, Protocol, NamedTuple, FrozenSet, Optional
from variable_protocols.base_variables import BaseVariableType, BaseVariable, struct_hash_base_variable

VariableType = Literal['VariableGroup',
                       'VariableTensor']

TensorBaseType = Literal[BaseVariableType,
                         'VariableGroup',
                         'VariableTensor']


class TensorBase(Protocol):
    @property
    @abc.abstractmethod
    def type(self) -> TensorBaseType: ...


class Variable(Protocol):
    @property
    @abc.abstractmethod
    def id(self) -> Optional[str]: ...

    @property
    @abc.abstractmethod
    def label(self) -> Optional[str]: ...

    @property
    @abc.abstractmethod
    def type(self) -> VariableType: ...


# Dimension Group
class DimensionFamily(NamedTuple):
    n_members: int
    positioned: bool
    label: str
    len: Optional[int] = 1

    @property
    def struct_hash(self) -> str:
        return f"D[{self.len}|{self.n_members}|{int(self.positioned)}]"

    @property
    def id(self) -> str:
        return str(hash(self))[:5]

    def fmt(self) -> str:
        positioned = "shuffle-able" if self.positioned else ""
        family = f"*{self.n_members}" if self.n_members > 1 else ""
        return f"{self.label}[{self.len}|{positioned}]{family}"


# Grouping Variables of the same type
class VariableTensor(NamedTuple):
    var: TensorBase
    dims: FrozenSet[DimensionFamily]
    label: Optional[str] = None
    type: Literal['VariableTensor'] = 'VariableTensor'

    @property
    def id(self) -> str:
        return str(hash(self))[:5]

    @property
    def struct_hash(self) -> str:
        dims = "|".join([d.struct_hash for d in self.dims])
        return f"T[{self.id}|{struct_hash(self.var)}|[{dims}]]"

    def fmt(self, indent: int = 2, curr_indent: int = 0) -> str:
        var_type = fmt(self.var)
        if len(self.dims) == 0:
            return var_type
        else:
            header = "Tensor#"
            dims = ", ".join(d.fmt() for d in self.dims)
            if len(header) + len(var_type) + len(": ") + len(dims) > 70:
                indent_spaces = (curr_indent + len(header) + indent) * ' '
                return f"{header}{var_type}:\n{indent_spaces}{dims}"
            else:
                return f"{header}{var_type}: {dims}"


# Grouping Variables of different types
class VariableGroup(NamedTuple):
    vars: FrozenSet[Variable]
    label: Optional[str] = None
    type: Literal['VariableGroup'] = 'VariableGroup'

    @property
    def id(self) -> str:
        return str(hash(self))[:5]

    @property
    def struct_hash(self) -> str:
        _vars = "|".join([struct_hash(v) for v in self.vars])
        return f"G[{self.id}|[{_vars}]]"

    @classmethod
    def make(cls,
             _vars: FrozenSet[Variable],
             label: Optional[str] = None) -> VariableGroup:
        if len(_vars) > 1:
            return VariableGroup(_vars, label)
        else:
            raise ValueError("Groups must be made of two Variables")

    def fmt(self, indent: int = 2, curr_indent: int = 0) -> str:
        s = "Set{\n"
        curr_indent += indent
        for i, var in enumerate(self.vars):
            s += f"{fmt(var, indent=indent, curr_indent=curr_indent)},\n"
        s = s.strip(",\n")
        s += "}\n"
        return s


def struct_hash(var: TensorBase, ignore_names: bool = False) -> str:
    if var.type == 'BaseVariable':
        assert isinstance(var, BaseVariable)
        return struct_hash_base_variable(var, ignore_names)
    elif var.type == 'VariableTensor':
        assert isinstance(var, VariableTensor)
        return var.struct_hash
    else:
        assert isinstance(var, VariableGroup)
        return var.struct_hash


def fmt(var: TensorBase, **kwargs) -> str:
    if var.type == 'BaseVariable':
        assert isinstance(var, BaseVariable)
        return var.fmt()
    elif var.type == 'VariableTensor':
        assert isinstance(var, VariableTensor)
        return var.fmt(**kwargs)
    else:
        assert isinstance(var, VariableGroup)
        return var.fmt(**kwargs)
