import numpy as np

def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as X.
    """
    # Write code here
    arr=np.array(X,dtype=float,copy=True)
    if arr.ndim==1:
        arr=arr.reshape(-1,1)
        for j in range(arr.shape[1]):
            col=arr[:,j]
            valid=~np.isnan(col)
            fill=np.mean(col[valid]) if strategy=="mean" else np.median(col[valid]) if np.any(valid) else 0.0
            arr[~valid,j]=fill
        return arr.ravel()
    for j in range(arr.shape[1]):
        col=arr[:,j]
        valid=~np.isnan(col)
        fill=(np.mean(col[valid]) if strategy=="mean" else np.median(col[valid])) if np.any(valid) else 0.0
        arr[~valid,j]=fill
    return arr
    pass