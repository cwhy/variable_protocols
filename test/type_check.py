from src.variable_protocols.variables import var_group, positive_float, var_unique
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
iris_in = var_group({var_unique(positive_float(), name) for name in ft_names})

# noinspection PyTypeChecker
# because pyCharm sucks
iris_in_clean = var_array(positive_float(), 4)

t1 = HashedTree(iris_in_clean)
t2 = HashedTree(iris_in).compare(t1)
print(t1)
print(t2)
