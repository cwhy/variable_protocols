import abc
from typing import Literal, Protocol, NamedTuple, Set, FrozenSet, List, Tuple, Optional

BaseVariableType = Literal['bounded', '1hot', '2vec', 'gaussian', 'gamma',
                           'ordinal', 'named_categorical', 'one_side_supported',
                           'category_ids']

UniqueVariableType = Literal[BaseVariableType,
                             'VariableList',
                             'VariableGroup']

VariableType = Literal[UniqueVariableType,
                       'VariableTensor']


class Variable(Protocol):
    @property
    @abc.abstractmethod
    def type(self) -> VariableType: ...


class UniqueVariable(Protocol):
    @property
    @abc.abstractmethod
    def type(self) -> UniqueVariableType: ...


class BaseVariable(Protocol):
    @property
    @abc.abstractmethod
    def type(self) -> BaseVariableType: ...

    @abc.abstractmethod
    def fmt(self) -> str: ...

    @abc.abstractmethod
    def _asdict(self) -> dict: ...


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


class Dimension(NamedTuple):
    name: str
    len: int
    positioned: bool

    def str_hash(self, ignore_names: bool) -> str:
        if ignore_names:
            return f"D[{self.len}|{int(self.positioned)}]"
        else:
            return f"D[{self.name}|{self.len}|{int(self.positioned)}]"


class VariableTensor(NamedTuple):
    var: UniqueVariable
    dims: FrozenSet[Dimension]
    name: Optional[str] = None
    type: Literal['VariableTensor'] = 'VariableTensor'

    @classmethod
    def build(cls, var: UniqueVariable, dims: FrozenSet[Dimension], name: Optional[str] = None):
        if len(dims) != 0:
            return cls(var, dims, name)
        else:
            raise ValueError("Empty dimension")


class VariableGroup(NamedTuple):
    vars: FrozenSet[Variable]
    name: Optional[str] = None
    type: Literal['VariableGroup'] = 'VariableGroup'


class VariableList(NamedTuple):
    var: Variable
    positioned: bool = True
    type: Literal['VariableList'] = 'VariableList'
