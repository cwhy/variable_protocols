from src.variable_protocols.variables import var_group, positive_float, var_unique
from src.variable_protocols.hashed_tree import HashedTree, check_tr, diff

ft_names = [
    "sepal length in cm",
    "sepal width in cm",
    "petal length in cm",
    "petal width in cm"
]

cat_names = ["Setosa", "Versicolour", "Virginica"]

iris_in = var_group({var_unique(positive_float(), name) for name in ft_names})
iris_in2 = var_group({var_unique(positive_float(), name + "!") for name in ft_names})
not_iris_in = var_group({var_unique(positive_float(), name) for name in ft_names if "width" in name})

t1 = HashedTree(iris_in)
t2 = HashedTree(iris_in2)
t3 = HashedTree(not_iris_in)

c = check_tr(t1, t2)
print(c)
c = check_tr(t1, t3)
print(c)
d = diff(t1, t2)
print(d)
d = diff(t1, t3)
print(d)
