#!/usr/bin/env python
import numpy as np
import sys
import matplotlib.pyplot as plt

PROBABILITY_WINDOW_SIZE = 21  # Must be odd
F_INDEX = 0
FIT_DEGREE = 3

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
    "Sex and Making Love",
]


def normalize(X):
    return (X - np.mean(X, axis=0)) / (np.sqrt(X.shape[0]) * np.std(X, axis=0))


def PCA(X):
    sigma = X.T.dot(X)
    S = np.identity(sigma.shape[0])
    for _ in range(40):
        Q, R = np.linalg.qr(sigma, mode="complete")
        sigma = R.dot(Q)
        S = S.dot(Q)
    return np.diagonal(sigma), S


def sort_by_eigen_value(e_val, e_vect):
    descending = np.argsort(-e_val)
    return e_val[descending], e_vect[:, descending]


def sort_by_first_column(A):
    asending = np.argsort(A[:, 0])
    return A[asending, :]


def get_probability_distribution(data, window_size):
    offset = int(window_size / 2)
    prob_dist = np.empty(np.size(data) - 2 * offset)
    for i in range(offset, np.size(data) - offset):
        prob_dist[i - offset] = np.sum(data[i:i + window_size]) / window_size
    return prob_dist


def linearize(x, degree):
    X = np.empty((np.size(x), degree + 1))
    for i in range(degree + 1):
        X[:, i] = x ** i
    return X


def poly_fit(x, y, degree):
    X = linearize(x, degree)
    return np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)


def get_precision(f, cat_data, alpha):
    return np.mean(
        (linearize(f, degree=FIT_DEGREE).dot(alpha) > 0.5) == cat_data)


def get_r_squared(prob_dist, estimated_prob_dist):
    mean_prob = prob_dist.mean()
    centered_prob_dist = prob_dist - mean_prob
    ss_tot = centered_prob_dist.transpose().dot(centered_prob_dist)
    error = prob_dist - estimated_prob_dist
    ss_error = error.transpose().dot(error)
    return 1 - ss_error / ss_tot


def main():
    X = np.loadtxt("dataM.txt")
    X = normalize(X)
    e_val, e_vect = PCA(X)
    e_val, e_vect = sort_by_eigen_value(e_val, e_vect)
    F = X.dot(e_vect)

    cat_to_use = sys.argv[1] if len(sys.argv) > 1 else "Tech"
    cat_index = CATEGORIES.index(cat_to_use) if cat_to_use in CATEGORIES else 0
    cat_data = np.loadtxt("likesM.txt")[:, cat_index]

    f_cat_data = np.stack([F[:, F_INDEX], cat_data], axis=-1)
    f_cat_data = sort_by_first_column(f_cat_data)

    prob_dist = get_probability_distribution(f_cat_data[:, 1],
                                             PROBABILITY_WINDOW_SIZE)
    offset = int(PROBABILITY_WINDOW_SIZE / 2)
    trunc_f = f_cat_data[offset:f_cat_data.shape[0] - offset, 0]
    alpha = poly_fit(trunc_f, prob_dist, degree=FIT_DEGREE)
    estimated_prob_dist = linearize(trunc_f, degree=FIT_DEGREE).dot(alpha)

    print("Model Precision: %f" % get_precision(F[:, F_INDEX], cat_data, alpha))
    print("R^2: %f" % get_r_squared(prob_dist, estimated_prob_dist))

    plt.plot(trunc_f, prob_dist)
    plt.plot(trunc_f, estimated_prob_dist)
    plt.title("Feature(%d) vs Likes(%s)" % (F_INDEX, cat_to_use))
    plt.xlabel("F%d" % F_INDEX)
    plt.ylabel("p(f)")
    plt.show()


if __name__ == "__main__":
    main()
