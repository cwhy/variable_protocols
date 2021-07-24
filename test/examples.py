# float MNIST input
from src.variable_protocols.variables import VariablePort, VariableTensor, Bounded, OneHot, fmt_var_group, \
    named_variable, named_variables

# noinspection PyTypeChecker
# because pyCharm sucks
mnist_in: VariablePort = named_variables(name="mnist_in",
                                         variables={
                                              VariableTensor(Bounded(max=1, min=0), (28, 28))
                                          })
# noinspection PyTypeChecker
# because pyCharm sucks
mnist_out: VariablePort = named_variables(name="mnist_out",
                                          variables={
                                               VariableTensor(OneHot(n_category=10), (1,))
                                           })

print(fmt_var_group(mnist_in, 2))
print(fmt_var_group(mnist_out, 2))
