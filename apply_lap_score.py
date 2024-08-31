import numpy as np
from FS_package.function.similarity_based import lap_score
from FS_package.utility import construct_W
from Method2 import df

def main():
    X = np.array(df, dtype=object)
    X = X.astype(float)
    n_samples, n_features = X.shape
    np.savetxt("X.csv", X, delimiter=",")
    kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
    W = construct_W.construct_W(X, **kwargs_W)
    score = lap_score.lap_score(X, W = W)
    print ('score:', score)
    np.savetxt("score.csv", score, delimiter=",")


if __name__ == '__main__':
    main()