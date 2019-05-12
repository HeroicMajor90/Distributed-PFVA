#!/usr/bin/env python
from global_array import GlobalArray, sort_by_first_column, qr
import numpy as np
import sys
import matplotlib.pyplot as plt

PROBABILITY_WINDOW_SIZE = 21  # Must be odd
F_INDEX = 0
FIT_DEGREE = 5

CATEGORIES = [
    "Outdoors-n-Adventures",
    "Tech",
    "Family",
    "Health-n-Wellness",
    "Sports-n-Fitness",
    "Learning",
    "Photography",
    "Food-n-Drink",
    "Writing",
    "Language-n-Culture",
    "Music",
    "Movements",
    "LGBTQ",
    "Film",
    "Sci-Fi-n-Games",
    "Beliefs",
    "Arts",
    "Book Clubs",
    "Dance",
    "Hobbies-n-Crafts",
    "Fashion-n-Beauty",
    "Social",
    "Career-n-Business",
    "Gardening-n-Outdoor housework",
    "Cooking",
    "Theatre, Show, Performance, Concerts",
    "Drinking alcohol, Partying",
    "Sex and Making Love"
]


def normalize(X):
    return (X - X.mean(axis=0).transpose()) / (np.sqrt(X.total_rows) *
                                               X.std(axis=0, zero_default=1).transpose())


def PCA(X):
    sigma = X.transpose().dot(X)
    S = GlobalArray.eye(sigma.total_rows)
    for _ in range(40):
        Q, R = qr(sigma)
        sigma = R.dot(Q)
        S = S.dot(Q)
    return sigma.diagonal(), S


def sort_by_eigen_value(e_val, e_vect):
    descending = np.argsort(-e_val)
    return e_val[descending], e_vect[:, descending]


def get_probability_distribution(data, window_size):
    offset = int(window_size / 2)
    prob_dist = np.empty(np.size(data) - 2 * offset)
    for i in range(offset, np.size(data) - offset):
        prob_dist[i - offset] = np.sum(data[i:i + window_size]) / window_size
    return prob_dist


def linearize(x, degree):
    X = GlobalArray(x.total_rows, degree + 1)
    for i in range(degree + 1):
        X[:, i] = x ** i
    return X


def poly_fit(x, y, degree):
    X = linearize(x, degree)
    I = GlobalArray.eye(np.size(x))
    C = GlobalArray(X.total_rows, X.total_cols + I.total_cols)
    C[:, :X.total_cols] = X
    C[:, C.total_cols - I.total_cols:] = I
    C.rref()
    C = C[:, X.total_cols - I.total_cols:]
    return C


def main():
    X = GlobalArray.from_file("dataM.txt")
    X = normalize(X)

    e_val, e_vect = PCA(X)
    e_val.disp()
    e_vect.disp()
    exit()

    e_val, e_vect = sort_by_eigen_value(e_val, e_vect)
    F = X.dot(GlobalArray.array(e_vect))

    cat_to_use = sys.argv[1] if len(sys.argv) > 1 else "Tech"
    cat_index = CATEGORIES.index(cat_to_use) if cat_to_use in CATEGORIES else 0
    cat_data = GlobalArray.from_file("likesM.txt")[:, cat_index]

    f_cat_data = np.stack([F[:, F_INDEX], cat_data], axis=-1)
    f_cat_data = sort_by_first_column(f_cat_data)

    prob_dist = get_probability_distribution(f_cat_data[:, 1],
                                             PROBABILITY_WINDOW_SIZE)
    offset = int(PROBABILITY_WINDOW_SIZE / 2)
    trunc_f = f_cat_data[offset:f_cat_data.shape[0] - offset, 0]
    alpha = poly_fit(trunc_f, prob_dist, degree=FIT_DEGREE)
    estimated_prob_dist = linearize(trunc_f, degree=FIT_DEGREE).dot(alpha)
    model_precision = np.mean(
        (linearize(f_cat_data[:, 0], degree=FIT_DEGREE).dot(alpha) > 0.5) == f_cat_data[:, 1])

    plt.plot(trunc_f, prob_dist)
    plt.plot(trunc_f, estimated_prob_dist)
    plt.title("Feature%i vs Likes(%s)" % (F_INDEX, cat_to_use))
    plt.xlabel("F%d" % F_INDEX)
    plt.ylabel("p(f)")
    plt.show()


if __name__ == "__main__":
    main()


