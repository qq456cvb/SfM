import numpy as np
from scipy.optimize import linprog
from SFM.triangulation import triangulate
from tqdm import tqdm
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False


class Kernel:
    def __init__(self, T, N):
        super().__init__()
        self.T = T
        self.N = N
        
    def fit(self, records):
        raise NotImplementedError()
    
    def compute_errors(self, model, records):
        raise NotImplementedError
    

class HomographyKernel(Kernel):
    def __init__(self, T, N):
        super().__init__(T, N)
        
    def fit(self, records):
        x, y, xp, yp = records[:, 0], records[:, 1], records[:, 2], records[:, 3]
        A = np.stack([xp * x, xp * y, xp, yp * x, yp * y, yp, x, y, np.ones_like(x)], -1)
        u, s, vt = np.linalg.svd(A)
        F = vt[-1].reshape(3, 3)
        u, s, vt = np.linalg.svd(F)
        s[-1] = 0
        return u @ np.diag(s) @ vt
    
    def compute_errors(self, F, records):
        x = np.concatenate([records[:, :2], np.ones((records.shape[0], 1))], -1)
        xp = np.concatenate([records[:, 2:], np.ones((records.shape[0], 1))], -1)
        
        Fx = (F @ x.T).T  # N x 3
        Ftxp = xp @ F
        sampson_errors = np.sqrt(np.sum(Fx * xp, -1) ** 2 / np.sum(Fx[:, :-1] ** 2 + Ftxp[:, :-1] ** 2, -1))
        # print(sampson_errors)
        # import pdb; pdb.set_trace()
        return sampson_errors
    

class TrifocalKernel(Kernel):
    def __init__(self, T, N, Rs):
        super().__init__(T, N)
        self.Rs = Rs
        
    # records: x1, y1, x2, y2, x3, y3  three views
    # variables: t2, t3, X1, X2, X3, X4
    def fit(self, records):
        # print(records.shape)
        x11, y11, x12, y12, x13, y13 = [records[0, i] for i in range(6)]
        x21, y21, x22, y22, x23, y23 = [records[1, i] for i in range(6)]
        x31, y31, x32, y32, x33, y33 = [records[2, i] for i in range(6)]
        x41, y41, x42, y42, x43, y43 = [records[3, i] for i in range(6)]
        
        lb = 0
        ub = 1
        thresh = ub
        prec = 1e-6
        res = None
        opt = np.zeros((6,))
        while ub - lb > prec:
            A_ub = np.array([
                [0, 0, 0, 0, 0, 0, *(self.Rs[0][2] * (x11 - thresh) - self.Rs[0][0]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 1
                [0, 0, 0, 0, 0, 0, *(self.Rs[0][2] * (y11 - thresh) - self.Rs[0][1]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 1
                [-1, 0, x12 - thresh, 0, 0, 0, *(self.Rs[1][2] * (x12 - thresh) - self.Rs[1][0]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 2
                [0, -1, y12 - thresh, 0, 0, 0, *(self.Rs[1][2] * (y12 - thresh) - self.Rs[1][1]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 2
                [0, 0, 0, -1, 0, x13 - thresh, *(self.Rs[2][2] * (x13 - thresh) - self.Rs[2][0]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 3
                [0, 0, 0, 0, -1, y13 - thresh, *(self.Rs[2][2] * (y13 - thresh) - self.Rs[2][1]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 3
                
                [0, 0, 0, 0, 0, 0, 0, 0, 0, *(self.Rs[0][2] * (x21 - thresh) - self.Rs[0][0]), 0, 0, 0, 0, 0, 0],  # view 1
                [0, 0, 0, 0, 0, 0, 0, 0, 0, *(self.Rs[0][2] * (y21 - thresh) - self.Rs[0][1]), 0, 0, 0, 0, 0, 0],  # view 1
                [-1, 0, x22 - thresh, 0, 0, 0, 0, 0, 0, *(self.Rs[1][2] * (x22 - thresh) - self.Rs[1][0]), 0, 0, 0, 0, 0, 0],  # view 2
                [0, -1, y22 - thresh, 0, 0, 0, 0, 0, 0, *(self.Rs[1][2] * (y22 - thresh) - self.Rs[1][1]), 0, 0, 0, 0, 0, 0],  # view 2
                [0, 0, 0, -1, 0, x23 - thresh, 0, 0, 0, *(self.Rs[2][2] * (x23 - thresh) - self.Rs[2][0]), 0, 0, 0, 0, 0, 0],  # view 3
                [0, 0, 0, 0, -1, y23 - thresh, 0, 0, 0, *(self.Rs[2][2] * (y23 - thresh) - self.Rs[2][1]), 0, 0, 0, 0, 0, 0],  # view 3
                
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(self.Rs[0][2] * (x31 - thresh) - self.Rs[0][0]), 0, 0, 0],  # view 1
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(self.Rs[0][2] * (y31 - thresh) - self.Rs[0][1]), 0, 0, 0],  # view 1
                [-1, 0, x32 - thresh, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(self.Rs[1][2] * (x32 - thresh) - self.Rs[1][0]), 0, 0, 0],  # view 2
                [0, -1, y32 - thresh, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(self.Rs[1][2] * (y32 - thresh) - self.Rs[1][1]), 0, 0, 0],  # view 2
                [0, 0, 0, -1, 0, x33 - thresh, 0, 0, 0, 0, 0, 0, *(self.Rs[2][2] * (x33 - thresh) - self.Rs[2][0]), 0, 0, 0],  # view 3
                [0, 0, 0, 0, -1, y33 - thresh, 0, 0, 0, 0, 0, 0, *(self.Rs[2][2] * (y33 - thresh) - self.Rs[2][1]), 0, 0, 0],  # view 3
                
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(self.Rs[0][2] * (x41 - thresh) - self.Rs[0][0])],  # view 1
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(self.Rs[0][2] * (y41 - thresh) - self.Rs[0][1])],  # view 1
                [-1, 0, x42 - thresh, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(self.Rs[1][2] * (x42 - thresh) - self.Rs[1][0])],  # view 2
                [0, -1, y42 - thresh, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(self.Rs[1][2] * (y42 - thresh) - self.Rs[1][1])],  # view 2
                [0, 0, 0, -1, 0, x43 - thresh, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(self.Rs[2][2] * (x43 - thresh) - self.Rs[2][0])],  # view 3
                [0, 0, 0, 0, -1, y43 - thresh, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(self.Rs[2][2] * (y43 - thresh) - self.Rs[2][1])],  # view 3
                
                # negative parts
                [0, 0, 0, 0, 0, 0, *(self.Rs[0][2] * (-x11 - thresh) + self.Rs[0][0]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 1
                [0, 0, 0, 0, 0, 0, *(self.Rs[0][2] * (-y11 - thresh) + self.Rs[0][1]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 1
                [1, 0, -x12 - thresh, 0, 0, 0, *(self.Rs[1][2] * (-x12 - thresh) + self.Rs[1][0]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 2
                [0, 1, -y12 - thresh, 0, 0, 0, *(self.Rs[1][2] * (-y12 - thresh) + self.Rs[1][1]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 2
                [0, 0, 0, 1, 0, -x13 - thresh, *(self.Rs[2][2] * (-x13 - thresh) + self.Rs[2][0]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 3
                [0, 0, 0, 0, 1, -y13 - thresh, *(self.Rs[2][2] * (-y13 - thresh) + self.Rs[2][1]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 3
                
                [0, 0, 0, 0, 0, 0, 0, 0, 0, *(self.Rs[0][2] * (-x21 - thresh) + self.Rs[0][0]), 0, 0, 0, 0, 0, 0],  # view 1
                [0, 0, 0, 0, 0, 0, 0, 0, 0, *(self.Rs[0][2] * (-y21 - thresh) + self.Rs[0][1]), 0, 0, 0, 0, 0, 0],  # view 1
                [1, 0, -x22 - thresh, 0, 0, 0, 0, 0, 0, *(self.Rs[1][2] * (-x22 - thresh) + self.Rs[1][0]), 0, 0, 0, 0, 0, 0],  # view 2
                [0, 1, -y22 - thresh, 0, 0, 0, 0, 0, 0, *(self.Rs[1][2] * (-y22 - thresh) + self.Rs[1][1]), 0, 0, 0, 0, 0, 0],  # view 2
                [0, 0, 0, 1, 0, -x23 - thresh, 0, 0, 0, *(self.Rs[2][2] * (-x23 - thresh) + self.Rs[2][0]), 0, 0, 0, 0, 0, 0],  # view 3
                [0, 0, 0, 0, 1, -y23 - thresh, 0, 0, 0, *(self.Rs[2][2] * (-y23 - thresh) + self.Rs[2][1]), 0, 0, 0, 0, 0, 0],  # view 3
                
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(self.Rs[0][2] * (-x31 - thresh) + self.Rs[0][0]), 0, 0, 0],  # view 1
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(self.Rs[0][2] * (-y31 - thresh) + self.Rs[0][1]), 0, 0, 0],  # view 1
                [1, 0, -x32 - thresh, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(self.Rs[1][2] * (-x32 - thresh) + self.Rs[1][0]), 0, 0, 0],  # view 2
                [0, 1, -y32 - thresh, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(self.Rs[1][2] * (-y32 - thresh) + self.Rs[1][1]), 0, 0, 0],  # view 2
                [0, 0, 0, 1, 0, -x33 - thresh, 0, 0, 0, 0, 0, 0, *(self.Rs[2][2] * (-x33 - thresh) + self.Rs[2][0]), 0, 0, 0],  # view 3
                [0, 0, 0, 0, 1, -y33 - thresh, 0, 0, 0, 0, 0, 0, *(self.Rs[2][2] * (-y33 - thresh) + self.Rs[2][1]), 0, 0, 0],  # view 3
                
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(self.Rs[0][2] * (-x41 - thresh) + self.Rs[0][0])],  # view 1
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(self.Rs[0][2] * (-y41 - thresh) + self.Rs[0][1])],  # view 1
                [1, 0, -x42 - thresh, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(self.Rs[1][2] * (-x42 - thresh) + self.Rs[1][0])],  # view 2
                [0, 1, -y42 - thresh, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(self.Rs[1][2] * (-y42 - thresh) + self.Rs[1][1])],  # view 2
                [0, 0, 0, 1, 0, -x43 - thresh, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(self.Rs[2][2] * (-x43 - thresh) + self.Rs[2][0])],  # view 3
                [0, 0, 0, 0, 1, -y43 - thresh, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(self.Rs[2][2] * (-y43 - thresh) + self.Rs[2][1])],  # view 3
                
                # front constraints
                [0, 0, 0, 0, 0, 0, *(-self.Rs[0][2]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 1
                [0, 0, -1, 0, 0, 0, *(-self.Rs[1][2]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 2
                [0, 0, 0, 0, 0, -1, *(-self.Rs[2][2]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # view 3
                
                [0, 0, 0, 0, 0, 0, 0, 0, 0, *(-self.Rs[0][2]), 0, 0, 0, 0, 0, 0],  # view 1
                [0, 0, -1, 0, 0, 0, 0, 0, 0, *(-self.Rs[1][2]), 0, 0, 0, 0, 0, 0], # view 2
                [0, 0, 0, 0, 0, -1, 0, 0, 0, *(-self.Rs[2][2]), 0, 0, 0, 0, 0, 0], # view 3
                
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(-self.Rs[0][2]), 0, 0, 0],  # view 1
                [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(-self.Rs[1][2]), 0, 0, 0],  # view 2
                [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, *(-self.Rs[2][2]), 0, 0, 0],  # view 3
                
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(-self.Rs[0][2])], # view 1
                [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(-self.Rs[1][2])],  # view 2
                [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, *(-self.Rs[2][2])]  # view 3
            ])
            c = np.zeros((18,))
            b_ub = np.zeros((60,))
            b_ub[-12:] = -1
            
            # res = linprog(c, A_ub=A_ub, b_ub=b_ub)
            
            res = solvers.lp(matrix(c), matrix(A_ub), matrix(b_ub))
            
            # import pdb; pdb.set_trace()
            if res['status'] == 'optimal':
                ub = thresh
                opt = np.array(res['x']).reshape(-1)[:6]
            else:
                lb = thresh
                
            thresh = (lb + ub) / 2.
        
        # print(thresh)
        # print(np.array(res['x']).reshape(-1)[6:])
        return opt
    
    # records: x1, y1, x2, y2, x3, y3  three views
    # model: t2, t3
    def compute_errors(self, model, records):
        Ps = np.stack([np.concatenate([self.Rs[0], np.zeros((3, 1))], -1), 
              np.concatenate([self.Rs[1], model[:3][:, None]], -1), 
              np.concatenate([self.Rs[2], model[3:][:, None]], -1)])
        
        X = triangulate(Ps, records)
        projection = np.einsum('vrc,nc->nvr', Ps, X)
        projection = projection[:, :, :2] / projection[:, :, -1:]  # N x V x 2
        errors = np.abs(projection - records.reshape(-1, 3, 2)).max(-1).max(-1)
        errors[np.isnan(errors)] = 1e10
        return errors
    

class RANSAC:
    def __init__(self, kernel: Kernel, prob=0.99, inlier_ratio=0.5):
        super().__init__()
        self.kernel = kernel
        self.num_trials = min(int(np.log(1 - prob) / np.log(1 - np.power(inlier_ratio, self.kernel.N))), 500)
        # print(np.log(1 - prob) / np.log(1 - np.power(inlier_ratio, self.kernel.N)))
        # self.num_trials = 1
        
    def process(self, records):
        best_model = None
        best_inlier_idxs = []
        for _ in tqdm(range(self.num_trials)):
            idxs = np.random.choice(records.shape[0], size=self.kernel.N, replace=False)
            model = self.kernel.fit(records[idxs])
            # print(self.kernel.compute_errors(model, records[idxs]))
            # import pdb; pdb.set_trace()
            errors = self.kernel.compute_errors(model, records)
            inlier_idxs = np.where(errors < self.kernel.T)[0]
            
            # import pdb; pdb.set_trace()
            if len(inlier_idxs) > len(best_inlier_idxs):
                # if self.kernel.N == 4:
                #     print(errors[inlier_idxs].max())
                best_inlier_idxs = inlier_idxs
                best_model = model
        
        best_model = self.kernel.fit(records[best_inlier_idxs])
        return best_model, best_inlier_idxs
        
    
    
