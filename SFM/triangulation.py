import numpy as np


# corrs: x1, y1, x2, y2, ...
def triangulate(Ps, corrs):
    
    # A = np.zeros((corrs.shape[0] * corrs.shape[1], 4 * corrs.shape[0]))
    pts = []
    for i in range(corrs.shape[0]):
        A = np.zeros((corrs.shape[1], 4))
        for j in range(corrs.shape[1] // 2):
            A[j * 2] = corrs[i, j * 2] * Ps[j][2] - Ps[j][0]
            A[j * 2 + 1] = corrs[i, j * 2 + 1] * Ps[j][2] - Ps[j][1]
            
        _, _, sol = np.linalg.svd(A)
        X = sol[-1]
        X /= X[-1]
        pts.append(X)
    # A = np.array([[-1, 0, corr[0], 0], [0, -1, corr[1], 0], corr[2] * P[2] - P[0], corr[3] * P[2] - P[1]])
    # print(A)
    return np.stack(pts)