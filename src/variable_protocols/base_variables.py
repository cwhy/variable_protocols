import abc
from typing import Literal, Protocol, NamedTuple, FrozenSet, runtime_checkable

BaseVariableType = Literal['bounded', '1hot', '2vec', 'gaussian', 'gamma',
                           'ordinal', 'named_categorical', 'one_side_supported',
                           'category_ids']


@runtime_checkable
class BaseVariable(Protocol):
    @property
    @abc.abstractmethod
    def type(self) -> BaseVariableType: ...

    @abc.abstractmethod
    def fmt(self) -> str: ...

    @abc.abstractmethod
    def _asdict(self) -> dict: ...


@runtime_checkable
class CustomHashBaseVariable(Protocol):
    @property
    @abc.abstractmethod
    def type(self) -> BaseVariableType: ...

    @abc.abstractmethod
    def fmt(self) -> str: ...

    @abc.abstractmethod
    def struct_hash(self, ignore_names: bool) -> str: ...


class OneSideSupported(NamedTuple):
    bound: float
    min_or_max: Literal["min", "max"]
    type: BaseVariableType = 'one_side_supported'

    def fmt(self) -> str:
        if self.min_or_max == 'min':
            return f"OneSideSupported(>={self.bound})"
        else:
            assert self.min_or_max == 'max'
            return f"OneSideSupported(<={self.bound})"


class Gamma(NamedTuple):
    alpha: float
    beta: float
    type: BaseVariableType = 'gamma'

    def fmt(self) -> str:
        return f"Gamma({self.alpha}, {self.beta})"


class Bounded(NamedTuple):
    max: float
    min: float
    type: BaseVariableType = 'bounded'

    def fmt(self) -> str:
        return f"Bounded({self.min}, {self.max})"


class OneHot(NamedTuple):
    n_category: int
    type: BaseVariableType = '1hot'

    def fmt(self) -> str:
        return f"OneHot({self.n_category})"


class NamedCategorical(NamedTuple):
    names: FrozenSet[str]
    type: BaseVariableType = 'named_categorical'

    def fmt(self) -> str:
        return f"Categorical({', '.join(self.names)})"

    def struct_hash(self, ignore_names: bool) -> str:
        content = (
            str(len(self.names)) if ignore_names
            else f"[{'|'.join(self.names)}]")
        return f"B[{self.type}|{content}]"


class CategoryIds(NamedTuple):
    max_id_len: int
    type: BaseVariableType = 'category_ids'

    def fmt(self) -> str:
        return f"CategoryIds(max_id_len={self.max_id_len})"


class Ordinal(NamedTuple):
    n_category: int
    type: BaseVariableType = 'ordinal'

    def fmt(self) -> str:
        return f"Ordinal({self.n_category})"


class CategoricalVector(NamedTuple):
    n_category: int
    n_embedding: int
    type: BaseVariableType = '2vec'

    def fmt(self) -> str:
        return f"CategoricalVector({self.n_category}, n_embedding={self.n_embedding})"


class Gaussian(NamedTuple):
    mean: float = 0
    var: float = 1
    type: BaseVariableType = 'gaussian'

    def fmt(self) -> str:
        return f"Gaussian({self.mean}, {self.var})"


def struct_hash_base_variable(var: BaseVariable, ignore_names: bool) -> str:
    if isinstance(var, CustomHashBaseVariable):
        return var.struct_hash(ignore_names)
    else:
        # noinspection PyProtectedMember
        # Because Pycharm sucks
        var_dict = var._asdict()
        var_type = var_dict.pop("type")
        content = "|".join(str(int(v)) if isinstance(v, bool) else str(v)
                           for _, v in var_dict.items())
        return f"B[{var_type}|{content}]"
