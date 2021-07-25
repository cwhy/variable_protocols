from __future__ import annotations

from typing import List, Tuple, Optional

from variable_protocols.protocols import Variable, NamedVariable, VariableGroup, VariableList, VariableTensor, \
    BaseVariable


def str_hash_base_variable(var: BaseVariable, ignore_names: bool) -> str:
    # noinspection PyProtectedMember
    # because python sucks
    var_dict = var._asdict()
    var_type = var_dict.pop("type")
    if ignore_names:
        var_dict.pop('name', None)
        var_dict.pop('names', None)
    content = "|".join(str(int(v)) if isinstance(v, bool) else str(v)
                       for _, v in var_dict.items())
    return f"B[{var_type}|{content}]"


def str_hash(var: Variable, ignore_names: bool) -> str:
    if var.type == 'VariableNamed':
        assert isinstance(var, NamedVariable)
        content = str_hash(var.var, ignore_names)
        if ignore_names:
            # only the names are ignores, identity remains
            return f"N[{content}]"
        else:
            return f"N[{var.name}|{content}]"
    elif var.type == 'VariableGroup':
        assert isinstance(var, VariableGroup)
        hashes = sorted(str_hash(var, ignore_names) for var in var.vars)
        return f"G[{'|'.join(hashes)}]"
    elif var.type == 'VariableList':
        assert isinstance(var, VariableList)
        # noinspection PyTypeChecker
        # because pycharm sucks
        return f"L[{str_hash(var.var, ignore_names)}|{int(var.positioned)}]"
    elif var.type == 'VariableTensor':
        assert isinstance(var, VariableTensor)
        base_type = str_hash_base_variable(var.var_type, ignore_names)
        if len(var.dims) > 0:
            dims = "|".join([d.str_hash(ignore_names) for d in var.dims])
            return f"T[{base_type}|[{dims}]]"
        else:
            return f"S[{base_type}]"
    else:
        raise Exception(f"Unexpected Variable type {var.type}")


# noinspection PyTypeChecker
# because pycharm sucks
class HashedTree:
    def __init__(self, var: Variable,
                 ignore_names: bool = True,
                 is_root: bool = True) -> None:
        self.hash = str_hash(var, ignore_names)
        self.is_root = is_root
        self.id: Optional[int] = None

        if var.type in ('VariableNamed', 'VariableList', 'VariableTensor'):
            assert isinstance(var, NamedVariable)
            self.children: List[HashedTree] = []
            self.nodes: List[HashedTree] = []
        elif var.type == 'VariableGroup':
            assert isinstance(var, VariableGroup)
            self.children = frozenset(
                HashedTree(var, ignore_names, is_root=False) for var in var.vars
            )
            self.nodes = sum([c.nodes for c in self.children], [])
        else:
            raise Exception(f"Unexpected Variable type {var.type}")
        if is_root:
            self.node_idx = {node: i for i, node in enumerate(self.nodes)}
            queue = [self]
            while queue:
                node = queue.pop(0)
                node.id = self.node_idx[node]
                if var.type == 'VariableGroup':
                    assert isinstance(var, VariableGroup)
                    queue += node.children
                else:
                    if var.type not in ('VariableNamed', 'VariableList', 'VariableTensor'):
                        raise Exception(f"Unexpected Variable type {var.type}")


def compare(t1: HashedTree, t2: HashedTree) -> bool:
    return t1.hash == t2.hash


DiffNode = HashedTree
DiffResult = List[Tuple[Optional[DiffNode], Optional[DiffNode]]]


def diff(t1: HashedTree, t2: HashedTree) -> DiffResult:
    assert t1.is_root
    assert t2.is_root
    return diff_helper(t1, t2, [])


def diff_helper(t1: HashedTree, t2: HashedTree, results: DiffResult) -> DiffResult:
    if t1.children is None:
        if t2.children is not None:
            results.append((t1, t2))
        return results
    elif t2.children is not None:
        results.append((t1, t2))
        return results
    else:
        tc1 = set(t1.children)
        tc2 = set(t1.children)
        g1, g2 = set(), set()
        for n1 in t1.children:
            if n1 in t2.children:
                tc2.remove(n1)
            else:
                g1.add(n1)
        for n2 in t2.children:
            if n2 in t1.children:
                tc1.remove(n2)
            else:
                g2.add(n2)

