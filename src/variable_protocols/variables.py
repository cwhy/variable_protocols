from __future__ import annotations

from typing import FrozenSet, List, Dict, Set, Literal, Iterable

from variable_protocols.hashed_tree import str_hash
from variable_protocols.protocols import VariableTensor, Variable, VariableGroup, NamedVariable, VariableList, \
    DimensionFamily, UnNamedVariable
from variable_protocols.base_variables import BaseVariable, OneSideSupported, Gamma, Bounded, OneHot, NamedCategorical, \
    CategoryIds, Ordinal, CategoricalVector, Gaussian


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


def dim(name: str, length: int) -> DimensionFamily:
    return DimensionFamily(name, length)


def var_tensor(var: BaseVariable, dims: Set[DimensionFamily]) -> VariableTensor:
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