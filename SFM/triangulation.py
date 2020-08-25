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


def compute_F_from_P(P1, P2):
    C = -P1[:3, :3].T @ P1[:3, -1]
    P2C = P2[:3, :3] @ C + P2[:3, -1]
    F = np.array([[0, -P2C[2], P2C[1]], 
                  [P2C[2], 0, -P2C[0]],
                  [-P2C[1], P2C[0], 0]]) @ P2 @ np.linalg.pinv(P1)
    return F

def opt_triangulate(Ps, corrs):
    assert len(Ps) == 2
    P1, P2 = Ps[0], Ps[1]
    F = compute_F_from_P(P1, P2)
    
    x1, y1, x2, y2 = corrs[0]
    T1 = np.eye(3)
    T1[:2, -1] = np.array([-x1, -y1])
    T2 = np.eye(3)
    T2[:2, -1] = np.array([-x2, -y2])
    F = np.linalg.inv(T2).T @ F @ np.linalg.inv(T1)
    
    u, _, vt = np.linalg.svd(F)
    e1 = vt[-1]
    e2 = u[:, -1]
    
    e1 /= np.linalg.norm(e1[:2])
    e2 /= np.linalg.norm(e2[:2])
    
    R1 = np.array([[e1[0], e1[1], 0],
                   [-e1[1], e1[0], 0],
                   [0, 0, 1]])
    R2 = np.array([[e2[0], e2[1], 0],
                   [-e2[1], e2[0], 0],
                   [0, 0, 1]])
    
    F = R2 @ F @ R1.T
    f1 = e1[2]
    f2 = e2[2]
    a, b, c, d = F[1, 1], F[1, 2], F[2, 1], F[2, 2]
    
    admbc = a * d - b * c
    adpbc = a * d + b * c
    coeff = np.array([-admbc * f1 ** 4 * a * c, # 6
                      -admbc * f1 ** 4 * adpbc + a ** 2 + f2 ** 2 * c ** 2, # 5
                      -admbc * (f1 ** 4 * b * d + 2 * f1 ** 2 * a * c) + 2 * (a ** 2 + f2 ** 2 * c ** 2) * (2 * a * b + 2 * f2 ** 2 * c * d),  # 4
                      -admbc * 2 * f1 ** 2 * adpbc + 2 * (a ** 2 + f2 ** 2 * c ** 2) * (b ** 2 + f2 ** 2 * d ** 2) + (2 * a * b + 2 * f2 ** 2 * c * d) ** 2, # 3
                      -admbc * (2 * f1 ** 2 * b * d + a * c) + 2 * (2 * a * b + 2 * f2 ** 2 * c * d) * (b ** 2 + f2 ** 2 * d ** 2), # 2
                      -admbc * adpbc + (b ** 2 + f2 ** 2 * d ** 2) ** 2,  # 1
                      -admbc * b * d])
    roots = np.real(np.roots(coeff))
    st = roots ** 2 / (1 + f1 ** 2 * roots ** 2) + (c * roots + d) ** 2 / ((a * roots + b) ** 2 + f2 ** 2 * (c * roots + d) ** 2)
    t = roots[np.argmin(st)]
    best_cost = roots.min()
    t_inf = False
    if 1 / f1 ** 2 + c ** 2 / (a ** 2 + f2 ** 2 * c ** 2) < best_cost:
        t_inf = True
    
    if t_inf:
        l1 = np.array([-f1, 0, 1])
        l2 = np.array([-f2, a / c, 1])
    else:
        l1 = np.array([t * f1, 1, -t])
        l2 = np.array([-f2 * (c * t + d), a * t +   b, c * t + d])
        
    def closest2origin(line):
        return np.array([-line[0] * line[2], - line[1] * line[2], line[0] ** 2 + line[1] ** 2])
    
    p1 = closest2origin(l1)
    p2 = closest2origin(l2)
    
    p1 /= p1[-1]
    p2 /= p2[-1]
    p1 = np.linalg.inv(T1) @ R1.T @ p1
    p2 = np.linalg.inv(T2) @ R2.T @ p2
    return triangulate(Ps, np.array([[p1[0], p1[1], p2[0], p2[1]]]))
    
    
if __name__ == "__main__":
        
    P2 = [
        np.array([[0.450927, -0.0945642, -0.887537 , -7.28137], [-0.892535, -0.0401974, -0.449183 , -7.57667], [0.00679989, 0.994707, -0.102528, 0.204446]]),
        np.array([[0.582226, -0.0983866, -0.807052, -8.31326], [-0.813027, -0.0706383, -0.577925, -6.3181], [-0.000148752, 0.992638, -0.121118, 0.16107]]),
        np.array([[0.666779, -0.0831384, -0.740603, -9.46627], [-0.74495, -0.0459057, -0.665539, -5.58174], [0.021334, 0.99548, -0.0925429, 0.147736 ]]),
    ]
    P2[1][:3, -1] -= P2[0][:3, -1]
    P2[2][:3, -1] -= P2[0][:3, -1]
    P2[0][:3, -1] = 0
    for i in range(len(P2)):
        P2[i] = np.concatenate([P2[i][:3,:3].T, -np.dot(P2[i][:3,:3].T, P2[i][:3,-1])[:, None]], -1)
        
    # synthetic dataset
    X = np.concatenate([-np.random.uniform(size=(4, 3)), np.ones((4, 1))], -1)
    # X[:, -2] += 0.5
    
    projection1 = np.einsum('rc,nc->nr', P2[0], X)
    projection2 = np.einsum('rc,nc->nr', P2[1], X)
    # print(projection1[:, -1])
    projection1 = projection1[:, :2] / projection1[:, -1:]
    projection2 = projection2[:, :2] / projection2[:, -1:]
    
    print(opt_triangulate(P2[:2], np.array([[projection1[0, 0], projection1[0, 1], projection2[0, 0], projection2[0, 1]]])))
    print(X[0])