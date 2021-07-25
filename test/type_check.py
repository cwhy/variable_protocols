from src.variable_protocols.variables import var_set, positive_float, var_named, var_array
from src.variable_protocols.hashed_tree import HashedTree

ft_names = [
    "sepal length in cm",
    "sepal width in cm",
    "petal length in cm",
    "petal width in cm"
]

cat_names = ["Setosa", "Versicolour", "Virginica"]

# noinspection PyTypeChecker
# because pyCharm sucks
iris_in = var_set({var_named(positive_float(), name) for name in ft_names})

# noinspection PyTypeChecker
# because pyCharm sucks
iris_in_clean = var_array(positive_float(), 4)

t1 = HashedTree(iris_in_clean)
t2 = HashedTree(iris_in).compare(t1)
print(t1)
print(t2)
