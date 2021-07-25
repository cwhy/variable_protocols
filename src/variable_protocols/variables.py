from __future__ import annotations

from typing import FrozenSet, List, Dict, Set, Literal, Iterable

from variable_protocols.hashed_tree import str_hash
from variable_protocols.protocols import VariableTensor, Variable, VariableGroup, NamedVariable, VariableList, Bounded, \
    OneHot, CategoricalVector, Gaussian, Dimension, BaseVariable, UnNamedVariable, Ordinal, NamedCategorical, \
    OneSideSupported, Gamma, CategoryIds


def bounded_float(min_val: float, max_val: float) -> Bounded:
    if min_val > max_val:
        raise ValueError(f"min_val > max_val: {min_val} > {max_val}")
    return Bounded(max_val, min_val)


def one_side_supported(bound: int, max_or_min: Literal["max", "min"]) -> OneSideSupported:
    return OneSideSupported(bound, max_or_min)


def positive_float() -> OneSideSupported:
    return OneSideSupported(0, "min")


def negative_float() -> OneSideSupported:
    return OneSideSupported(0, "max")


def gamma(alpha: float, beta: float) -> Gamma:
    if alpha > 0 and beta > 0:
        return Gamma(alpha, beta)
    else:
        raise ValueError(f"Invalid alpha or beta value {alpha}, {beta}")


def erlang(k: int, beta: float) -> Gamma:
    if k > 0:
        return gamma(k, beta)
    else:
        raise ValueError(f"Invalid k:{k}")


def exponential(lambda_: float) -> Gamma:
    if lambda_ > 0:
        return erlang(1, lambda_)
    else:
        raise ValueError(f"Invalid lambda(l) value {lambda_}")


def ordinal(n_category: int) -> Ordinal:
    if n_category > 0:
        return Ordinal(n_category)
    else:
        raise ValueError(f"Invalid n_category value: {n_category}")


def one_hot(n_category: int) -> OneHot:
    if n_category > 0:
        return OneHot(n_category)
    else:
        raise ValueError(f"Invalid n_category value: {n_category}")


def cat_vec(n_category: int, n_embedding: int) -> CategoricalVector:
    if n_category > 0:
        if n_embedding > 0:
            return CategoricalVector(n_category, n_embedding)
        else:
            raise ValueError(f"Invalid n_embedding value: {n_embedding}")
    else:
        raise ValueError(f"Invalid n_category value: {n_category}")


def cat_from_names(names: Iterable[str]) -> NamedCategorical:
    return NamedCategorical(frozenset(names))


def cat_ids(max_id_len: int) -> CategoryIds:
    return CategoryIds(max_id_len)


def gaussian(mean: float, var: float) -> Gaussian:
    if var > 0:
        return Gaussian(mean, var)
    else:
        raise ValueError(f"Invalid variance(var) value: {var}")


def dim(name: str, length: int) -> Dimension:
    return Dimension(name, length)


def var_tensor(var: BaseVariable, dims: Set[Dimension]) -> VariableTensor:
    if len(dims) <= 1:
        raise ValueError((
            "len(dims) <= 1:\n"
            "Creation if 1D tensor is not allowed "
            "to avoid duplicate representations, "
            "use var_array instead. "
            "To create scalars, use var_scalar."
        ))
    return VariableTensor(var, frozenset(dims))


def var_scalar(var: BaseVariable) -> VariableTensor:
    return VariableTensor(var, frozenset())


def var_set(vars_set: Set[Variable]) -> VariableGroup:
    assert isinstance(vars_set, set)
    if len(vars_set) <= 1:
        raise ValueError("A variable set/group must contain more than one variable")
    return VariableGroup(frozenset(vars_set))


def var_dict(vars_dict: Dict[str, UnNamedVariable]) -> VariableGroup:
    return var_set({
        add_name(var, name)
        for name, var in vars_dict.items()
    })


def var_ordered(vars_list: List[UnNamedVariable]) -> VariableGroup:
    return var_set({
        add_name(var, str(i))
        for i, var in enumerate(vars_list)
    })


def var_array(var: BaseVariable, length: int) -> VariableGroup:
    # noinspection PyTypeChecker
    # because pyCharm sucks
    return var_ordered([var_scalar(var)] * length)


def add_name(var: UnNamedVariable, name: str) -> NamedVariable:
    if isinstance(var, NamedVariable):
        # noinspection PyTypeChecker
        # because pyCharm sucks
        raise ValueError(f"var {fmt_var(var)} is already named")
    return NamedVariable(var, name)


def var_named(var: BaseVariable, name: str) -> NamedVariable:
    # noinspection PyTypeChecker
    # because pyCharm sucks
    return add_name(var_scalar(var), name)


def var_flex_len(var: NamedVariable, positioned: bool) -> VariableList:
    return VariableList(var, positioned)


def fmt_dims(dims: FrozenSet[Dimension], indent: int = 0, curr_indent: int = 0) -> str:
    s = ""
    curr_indent += len(s) + indent
    for d in dims:
        s += f"{d.name}[{d.len}], "
    s = s.strip(", ")
    return s


def fmt_var(g: Variable, indent: int = 2, curr_indent: int = 0) -> str:
    if g.type == 'VariableNamed':
        assert isinstance(g, NamedVariable)
        header = curr_indent * " "
        out = f"{header}{g.name}: {fmt_var(g.var, curr_indent)}"
        if len(out) > 70:
            s = header + f"{g.name}:\n" + (curr_indent + indent) * " "
            s += fmt_var(g.var, indent, curr_indent)
            return s
        else:
            return out
    elif g.type == 'VariableGroup':
        assert isinstance(g, VariableGroup)
        s = "Set{\n"
        curr_indent += indent
        for i, var in enumerate(g.vars):
            s += f"{fmt_var(var, indent, curr_indent)},\n"
        s = s.strip(",\n")
        s += "}\n"
        return s
    elif g.type == 'VariableList':
        assert isinstance(g, VariableList)
        header = "List("
        # noinspection PyTypeChecker
        # because pyCharm sucks
        s_var = fmt_var(g.var)
        if len(header) + len(s_var) > 70:
            # noinspection PyTypeChecker
            # because pyCharm sucks
            return f"{header}{fmt_var(g.var, indent, curr_indent + len(header))}]"
        else:
            # noinspection PyTypeChecker
            # because pyCharm sucks
            return f"{header}{fmt_var(g.var)})"
    elif g.type == 'VariableTensor':
        assert isinstance(g, VariableTensor)
        var_type = g.var_type.fmt()
        if len(g.dims) == 0:
            return var_type
        else:
            header = "Tensor#"
            dims = fmt_dims(g.dims, indent, curr_indent)
            if len(header) + len(var_type) + len(": ") + len(dims) > 70:
                indent_spaces = (curr_indent + len(header) + indent) * ' '
                return f"{header}{var_type}:\n{indent_spaces}{fmt_dims(g.dims, indent, curr_indent)}"
            else:
                return f"{header}{var_type}: {fmt_dims(g.dims, indent, curr_indent)}"
    else:
        raise Exception(f"Unexpected Variable type {g.type}")
