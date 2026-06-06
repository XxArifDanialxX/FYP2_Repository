import numpy as np
from scipy.optimize import minimize

# ==================== METHOD 1: q-ROF-AHP ====================
class QROFAHP:
    def __init__(self, q=3):
        self.q = q

    def calculate_weights(self, matrix):
        matrix = np.array(matrix, dtype=float)
        n = len(matrix)
        q_scores = []
        for i in range(n):
            row = []
            for j in range(n):
                val = matrix[i, j] if matrix[i, j] != 0 else 1
                mem = min(1, val / 9)
                non_mem = min(1, (1 / val) / 9)
                row.append((mem, non_mem))
            q_scores.append(row)

        geo_means = []
        for i in range(n):
            prod_m, prod_nm = 1, 1
            for j in range(n):
                prod_m *= q_scores[i][j][0]
                prod_nm *= q_scores[i][j][1]
            geo_means.append((prod_m**(1/n), prod_nm**(1/n)))

        weights = []
        tot_m = sum(x[0] for x in geo_means)
        tot_nm = sum(x[1] for x in geo_means)
        denom = tot_m + n - tot_nm
        for m, nm in geo_means:
            weights.append((m + (1 - nm)) / (denom if denom != 0 else 1))

        w_sum = sum(weights)
        return np.array(weights) / w_sum if w_sum != 0 else np.ones(n) / n

    def calculate_scores(self, matrix, weights):
        norm = np.zeros_like(matrix, dtype=float)
        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            mn, mx = np.min(col), np.max(col)
            norm[:, j] = (col - mn) / (mx - mn) if mx - mn != 0 else 1
        return np.sum(norm * weights, axis=1) * 100

# ==================== METHOD 2: BWM + VIKOR ====================
class BWM_VIKOR:
    def solve_bwm_weights(self, best, worst, b_to_o, o_to_w):
        def obj(w):
            max_dev = 0
            for i in range(len(w)):
                dev1 = abs(w[best] - b_to_o[i] * w[i])
                dev2 = abs(w[i] - o_to_w[i] * w[worst])
                max_dev = max(max_dev, dev1, dev2)
            return max_dev
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        res = minimize(obj, np.ones(len(b_to_o))/len(b_to_o), bounds=[(0.01, 1)]*len(b_to_o), constraints=cons)
        return res.x

    def calculate_vikor(self, matrix, weights):
        # 1. Best and Worst per criterion
        f_star = np.max(matrix, axis=0)
        f_minus = np.min(matrix, axis=0)
        
        # 2. S and R values
        S = np.zeros(matrix.shape[0])
        R = np.zeros(matrix.shape[0])
        for i in range(matrix.shape[0]):
            sum_val = 0
            max_val = 0
            for j in range(matrix.shape[1]):
                diff = (f_star[j] - matrix[i, j]) / (f_star[j] - f_minus[j] if f_star[j] != f_minus[j] else 1)
                weighted_diff = weights[j] * diff
                sum_val += weighted_diff
                if weighted_diff > max_val: max_val = weighted_diff
            S[i] = sum_val
            R[i] = max_val
            
        # 3. Q value (Consensus)
        S_star, S_minus = np.min(S), np.max(S)
        R_star, R_minus = np.min(R), np.max(R)
        v = 0.5
        Q = v * ((S - S_star) / (S_minus - S_star if S_minus != S_star else 1)) + \
            (1 - v) * ((R - R_star) / (R_minus - R_star if R_minus != R_star else 1))
        
        # Invert Q so higher is better (Percentage)
        return (1 - Q) * 100

# ==================== METHOD 3: SWARA + MOORA ====================
class SWARA_MOORA:
    def calculate_swara_weights(self, rank_order, s_j):
        n = len(rank_order)
        k_j = [1.0]
        for val in s_j: k_j.append(float(val) + 1.0)
        q_j = [1.0]
        for i in range(1, n): q_j.append(q_j[i-1] / k_j[i])
        total = sum(q_j)
        return dict(zip(rank_order, [val/total for val in q_j]))

    def calculate_moora(self, matrix, weights):
        # Ratio System Normalization
        norm = np.zeros_like(matrix, dtype=float)
        for j in range(matrix.shape[1]):
            denom = np.sqrt(np.sum(matrix[:, j]**2))
            norm[:, j] = matrix[:, j] / (denom if denom != 0 else 1)
        
        # Assessment Value
        y = np.sum(norm * weights, axis=1)
        mn, mx = np.min(y), np.max(y)
        return (y - mn) / (mx - mn if mx != mn else 1) * 100

# ==================== METHOD 4: LTSF-CRITIC-EDAS ====================
class CRITIC_EDAS:
    def execute(self, matrix):
        std = np.std(matrix, axis=0)
        corr = np.corrcoef(matrix, rowvar=False)
        if np.isnan(corr).any(): corr = np.eye(matrix.shape[1])
        info = std * np.sum(1 - corr, axis=1)
        w = info / np.sum(info) if np.sum(info) != 0 else np.ones(matrix.shape[1])/matrix.shape[1]
        
        avg = np.mean(matrix, axis=0)
        pda = np.maximum(0, (matrix - avg)) / (avg + 1e-9)
        nda = np.maximum(0, (avg - matrix)) / (avg + 1e-9)
        sp, sn = np.sum(pda * w, axis=1), np.sum(nda * w, axis=1)
        nsp, nsn = sp / (np.max(sp) + 1e-9), 1 - (sn / (np.max(sn) + 1e-9))
        return (nsp + nsn) / 2 * 100, w