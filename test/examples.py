# float MNIST input
from src.variable_protocols.variables import fmt_var, dim, bounded_float, var_tensor, var_scalar, one_hot, \
    cat_from_names, var_set, positive_float, var_named, cat_vec, cat_ids
from src.variable_protocols.common_variables import var_image, var_sentence_1hot, var_sentence

# noinspection PyTypeChecker
# because pyCharm sucks
mnist_in = var_tensor(bounded_float(0, 1), {dim("h", 28), dim("w", 28)})

# noinspection PyTypeChecker
# because pyCharm sucks
mnist_out = var_scalar(one_hot(10))

# noinspection PyTypeChecker
# because pyCharm sucks
imagenet_in = var_image(256, 256)

# noinspection PyTypeChecker
# because pyCharm sucks
sentence_1hot = var_sentence_1hot(n_vocab=10000)

# noinspection PyTypeChecker
# because pyCharm sucks
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
iris_out = var_scalar(cat_from_names(cat_names))

# noinspection PyTypeChecker
# because pyCharm sucks
movie_lens_1M_embedding_in = var_set({
    var_named(cat_vec(n_category=6000, n_embedding=64), "users"),
    var_named(cat_vec(n_category=4000, n_embedding=64), "items")})

# noinspection PyTypeChecker
# because pyCharm sucks
collaborative_filtering_id_in = var_set({
    var_named(cat_ids(max_id_len=4), "users"),
    var_named(cat_ids(max_id_len=4), "items")})
sentence_str = var_sentence(10)

# noinspection PyTypeChecker
# because pyCharm sucks
print(fmt_var(mnist_in))
# noinspection PyTypeChecker
# because pyCharm sucks
print(fmt_var(mnist_out))
print(fmt_var(imagenet_in))
print(fmt_var(sentence_1hot))
# noinspection PyTypeChecker
# because pyCharm sucks
print(fmt_var(iris_out))
# noinspection PyTypeChecker
# because pyCharm sucks
print(fmt_var(iris_in))
# noinspection PyTypeChecker
# because pyCharm sucks
print(fmt_var(movie_lens_1M_embedding_in))
# noinspection PyTypeChecker
# because pyCharm sucks
print(fmt_var(collaborative_filtering_id_in))
print(fmt_var(sentence_str))
