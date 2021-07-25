from abc import abstractmethod
from collections import defaultdict
from typing import Protocol, TypeVar, Dict, Set

from variable_protocols.variables import VariablePort, VarTensor
from variable_protocols.protocols import VariableGroup


class Transformation(Protocol):
    @property
    @abstractmethod
    def source(self) -> VariablePort: ...

    @property
    @abstractmethod
    def target(self) -> VariablePort: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


LookUpTable = Dict[VariablePort, Set[Transformation]]
lookup_table_source: LookUpTable = defaultdict(set)
lookup_table_target: LookUpTable = defaultdict(set)


def register_(transformation: Transformation):
    lookup_table_source[transformation.source].add(transformation)
    lookup_table_target[transformation.source].add(transformation)


def strict_check_(transformation: Transformation,
                  source: VariablePort,
                  target: VariablePort) -> bool:
    return transformation.source == source and \
           transformation.target == target


def structure_compare(a: VariablePort, b: VariablePort) -> bool:
    if len(a.variables) == 1:
        av = next(iter(a.variables))
        if isinstance(av, VarTensor):
            if len(b.variables) == 1:
                bv = next(iter(b.variables))
                if isinstance(bv, VarTensor):
                    return av.type == bv.type
                else:
                    assert isinstance(bv, VariableGroup)
                    # noinspection PyTypeChecker
                    # because pycharm sucks
                    return structure_compare(a, bv)
            else:
                return False
        else:
            assert isinstance(av, VariableGroup)
            # noinspection PyTypeChecker
            # because pycharm sucks
            return structure_compare(av, b)
    else:
        for av in a.variables:




def structure_check_(transformation: Transformation,
                     source: VariablePort,
                     target: VariablePort) -> bool:
    return structure_compare(transformation.source, source) and \
           structure_compare(transformation.target, target)

