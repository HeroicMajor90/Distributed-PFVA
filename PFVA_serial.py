#!/usr/bin/env python
import numpy as np


def PCA(X):
	print "Starting PCA"
	rows, cols = X.shape
	X = (X - np.mean(X, axis=0)) / (np.sqrt(rows)*np.std(X, axis=0))
	sigma = X.T.dot(X)
	S = np.identity(sigma.shape[0])

	for _ in xrange(40):
		print _
		Q, R = np.linalg.qr(sigma, mode="complete")
		sigma = R.dot(Q)
		S = S.dot(Q)

	eigen_values = np.diagonal(sigma)
	eigen_vectors = S
	descending = cols - 1 - np.argsort(eigen_values)

	return eigen_values[descending], eigen_vectors[:, descending]

	
def main():
	D, Q = PCA(np.random.randint(0, 256, (300, 300), dtype=np.uint8))
	print

if __name__ == "__main__":
	main()
