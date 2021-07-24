# float MNIST input
from src.variable_protocols.variables import VariableGroup, VariableTensor, Bounded, OneHot, fmt_var_group

mnist_in = VariableGroup(name="mnist_in",
                         variables={
                             VariableTensor(Bounded(max=1, min=0), (28, 28))
                         })
mnist_out = VariableGroup(name="mnist_out",
                          variables={
                              VariableTensor(OneHot(n_category=10), (1,))
                          })

print(fmt_var_group(mnist_in, 2))
print(fmt_var_group(mnist_out, 2))
