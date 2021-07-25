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
            return content
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


class HashedTree:
    def __init__(self, var: Variable, ignore_names: bool = True):
        if var.type == 'VariableNamed':
            assert isinstance(var, NamedVariable)
        elif var.type == 'VariableGroup':
            assert isinstance(var, VariableGroup)
        elif var.type == 'VariableList':
            assert isinstance(var, VariableList)
        elif var.type == 'VariableTensor':
            assert isinstance(var, VariableTensor)
            self.hash = str_hash(var, ignore_names)
        else:
            raise Exception(f"Unexpected Variable type {var.type}")
