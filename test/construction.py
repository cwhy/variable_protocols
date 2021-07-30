from src.variable_protocols.variables import dim, var_tensor, var_scalar, var_group, positive_float, var_unique, \
    var_ordered

iris_ft_names = [
    "sepal length in cm",
    "sepal width in cm",
    "petal length in cm",
    "petal width in cm"
]

iris_cat_names = ["Setosa", "Versicolour", "Virginica"]
# noinspection PyTypeChecker
# because pyCharm sucks
iris_in = var_group({var_unique(positive_float(), name) for name in iris_ft_names})
# noinspection PyTypeChecker
# because pyCharm sucks
iris_in_clean = var_tensor(positive_float(), {dim("iris_ft", 4)})
# noinspection PyTypeChecker
# because pyCharm sucks
iris_in_clean2 = var_ordered([var_scalar(positive_float())] * 4)
assert iris_in_clean == iris_in_clean2
# noinspection PyTypeChecker
# because pyCharm sucks
assert str_hash(iris_in_clean, ignore_names=True) == str_hash(iris_in, ignore_names=True)
# noinspection PyTypeChecker
# because pyCharm sucks
assert str_hash(iris_in_clean, ignore_names=False) != str_hash(iris_in, ignore_names=False)
