from variable_protocols.protocols import fmt
from variable_protocols.morph import look_up, register_
from variable_protocols.transformations import Transformation, new_transformation
from variable_protocols.variables import dim, bounded_float, var_tensor, var_scalar, one_hot


# noinspection PyTypeChecker
# because pyCharm sucks
mnist_in = var_tensor(bounded_float(0, 1), {dim("h", 28), dim("w", 28)})
# noinspection PyTypeChecker
# because pyCharm sucks
mnist_in_flattened = var_tensor(bounded_float(0, 1), {dim("hw", 28*28)})
# noinspection PyTypeChecker
# because pyCharm sucks
flatten = new_transformation(name="flatten", source=mnist_in, target=mnist_in_flattened)
register_(flatten)
# noinspection PyTypeChecker
# because pyCharm sucks
print([x.fmt() for x in look_up(mnist_in, mnist_in_flattened)])

# noinspection PyTypeChecker
# because pyCharm sucks
mnist_out = var_scalar(one_hot(10))
