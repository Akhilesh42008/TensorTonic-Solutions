import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """
    # Write code here
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1,1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1,1)
    n_train = len(X_train)
    result = np.full((len(X_test),k),-1,dtype=int)
    for i in range(len(X_test)):
        distances = np.sqrt(np.sum((X_train-X_test[i])**2,axis=1))
        indices = np.argsort(distances,kind='stable')
        count = min(k,n_train)
        result[i,:count] = indices[:count]
    return result
    pass