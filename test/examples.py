from variable_protocols.variables import dim, bounded_float, var_tensor, var_scalar, one_hot, \
    cat_from_names, var_group, positive_float, var_unique, cat_vec, cat_ids, var_ordered
from variable_protocols.common_variables import var_image, var_sentence_1hot, var_sentence
from variable_protocols.protocols import fmt, struct_hash

iris_ft_names = [
    "sepal length in cm",
    "sepal width in cm",
    "petal length in cm",
    "petal width in cm"
]

iris_cat_names = ["Setosa", "Versicolour", "Virginica"]

# noinspection PyTypeChecker
# because pyCharm sucks
all_test_vars = dict(
    mnist_in=var_tensor(bounded_float(0, 1), {dim("h", 28), dim("w", 28)}),
    mnist_out=var_scalar(one_hot(10)),
    imagenet_in=var_image(256, 256),
    sentence_1hot=var_sentence_1hot(n_vocab=10000),
    movie_lens_1M_embedding_in=var_group({
        var_unique(cat_vec(n_category=6000, n_embedding=64), "users"),
        var_unique(cat_vec(n_category=4000, n_embedding=64), "items")}),
    iris_in=var_group({var_unique(positive_float(), name) for name in iris_ft_names}),
    iris_in_clean=var_tensor(positive_float(), {dim("iris_ft", 4)}),
    iris_out=var_scalar(cat_from_names(iris_cat_names)),
    collaborative_filtering_id_in=var_group({
        var_unique(cat_ids(max_id_len=4), "users"),
        var_unique(cat_ids(max_id_len=4), "items")}),
    sentence_str=var_sentence(10)
)

for var_name, var in all_test_vars.items():
    print(var_name, fmt(var))
    print(struct_hash(var, ignore_names=True))
    print(struct_hash(var, ignore_names=False))
