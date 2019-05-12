#!/usr/bin/env python
import global_array as ga
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpi4py import MPI

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
	"Film" ,
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


def im_root():
    return MPI.COMM_WORLD.Get_rank() == 0


def normalize(X):
	return ((X - X.mean(axis=0).transpose()) / 
          (np.sqrt(X.total_rows) * X.std(axis=0, zero_default=1).transpose()))


def PCA(X):
    sigma = X.transpose().dot(X)
    S = ga.GlobalArray.eye(sigma.total_rows)

    for _ in xrange(40):
        Q, R = ga.qr(sigma)
        sigma = R.dot(Q)
        S = S.dot(Q)
    return sigma.diagonal(), S
  
  
def sort_by_eigen_value(e_val, e_vect):
    C = ga.GlobalArray(e_val.total_rows, e_vect.total_rows + 1)
    C[:, 0] = -1 * e_val
    C[:, 1:] = e_vect.transpose()
    descending = ga.sort_by_first_column(C)
    return -1 * descending[:,0], descending[:, 1:].transpose()
  
  
def get_probability_distribution(data, window_size):
    offset = int(window_size / 2)
    prob_dist = ga.GlobalArray(data.total_rows - 2 * offset, 1)
    for i in range(offset, data.total_rows - offset):
        prob_dist[i - offset] = data[i:i + window_size].sum() / window_size
    return prob_dist
  
  
def linearize(x, degree):
    X = ga.GlobalArray(x.total_rows, degree + 1)
    for i in range(degree + 1):
        X[:, i] = x ** i
    return X

  
def poly_fit(x, y, degree):
    X = linearize(x, degree)
    X_t = X.transpose()
    return ga.inv(X_t.dot(X)).dot(X_t).dot(y)

  
def get_precision(f, cat_data, alpha):
    return ((linearize(f, FIT_DEGREE).dot(alpha) > 0.5) == cat_data).mean()
  

def get_r_squared(prob_dist, estimated_prob_dist):
    mean_prob = prob_dist.mean()
    centered_prob_dist = prob_dist - mean_prob
    ss_tot = centered_prob_dist.transpose().dot(centered_prob_dist)
    error = prob_dist - estimated_prob_dist
    ss_error = error.transpose().dot(error)
    return 1 - ss_error / ss_tot


def main():
    X = ga.GlobalArray.from_file("dataM.txt")
    X = normalize(X)
    e_val, e_vect = PCA(X)
    e_val, e_vect = sort_by_eigen_value(e_val, e_vect)
    F = X.dot(e_vect)

    cat_to_use = sys.argv[1] if len(sys.argv) > 1 else "Tech"
    cat_index = CATEGORIES.index(cat_to_use) if cat_to_use in CATEGORIES else 0
    cat_data = ga.GlobalArray.from_file("likesM.txt")[:, cat_index]

    f_cat_data = ga.hstack([F[:, F_INDEX], cat_data])
    f_cat_data = ga.sort_by_first_column(f_cat_data)

    prob_dist = get_probability_distribution(f_cat_data[:, 1],
                                             PROBABILITY_WINDOW_SIZE)
    offset = int(PROBABILITY_WINDOW_SIZE / 2)
    trunc_f = f_cat_data[offset:f_cat_data.total_rows - offset, 0]
    alpha = poly_fit(trunc_f, prob_dist, degree=FIT_DEGREE)
    estimated_prob_dist = linearize(trunc_f, degree=FIT_DEGREE).dot(alpha)
    model_precision = get_precision(F[:, F_INDEX], cat_data, alpha).to_np()
    r_squared = get_r_squared(prob_dist, estimated_prob_dist).to_np()
    
    trunc_f = trunc_f.to_np()
    prob_dist = prob_dist.to_np()
    estimated_prob_dist = estimated_prob_dist.to_np()
    
    if im_root():
        print("Model Precision: %f" % model_precision)
        print("R^2: %f" % r_squared)
        plt.plot(trunc_f, prob_dist)
        plt.plot(trunc_f, estimated_prob_dist)
        plt.title("Feature%i vs Likes(%s)" % (F_INDEX, cat_to_use))
        plt.xlabel("F%d" % F_INDEX)
        plt.ylabel("p(f)")
        plt.show()
  
  
if __name__ == "__main__":
    main()