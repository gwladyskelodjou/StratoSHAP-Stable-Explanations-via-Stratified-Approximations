import numpy as np
from scipy.optimize import root_scalar
from scipy.stats import qmc
from scipy.special import beta
# from numba import njit
from ApproxMethods.ShapleyEstimator import ShapleyEstimator

class GameWrapper:

    def __init__(self, parent_estimator):
        self.parent = parent_estimator
        self.game = parent_estimator.game
        self.players = self.game.n

    def __call__(self, S,d):
        result = np.empty(S.shape[0])

        for i in range(S.shape[0]):
            sample = S[i]
            s_set = set()
            s_set = set(np.where(sample == 1)[0])

            tmp = self.game.compute_value(s_set)
            self.parent.budget -= 1
            result[i] = tmp[:, self.parent.idx_dims][0,d]
        return result, self.parent.budget > 0



class Sobol(ShapleyEstimator):
    def __init__(self):
        self.final_budget = 0
        
    def approximate_shapley_values(self) -> dict:
        full = list(range(self.n))
        self.grand = self.game.compute_value(full)
        
        if isinstance(self.grand, np.ndarray) and len(self.grand.shape) > 1:
            
            num_classes = self.grand.shape[1]

            if num_classes > 10:  # if too much classes --> ImageNet
                top_k = min(1, num_classes)
                top_classes = np.argsort(-self.grand[0])[:top_k]

                self.dim = top_k
                self.idx_dims = top_classes
            else:
                self.dim = num_classes
                self.idx_dims = np.arange(self.dim)
        else:
            self.dim = 1
            self.idx_dims = np.arange(self.dim)


        phi = self.shapley_values = np.zeros((self.dim, self.n))
        for d in range(self.dim):
            phi_formula = self.solve(d)
            phi[d] = phi_formula

        self.shapley_values = phi

        self.final_budget = self.initial_budget if self.budget == 0 else self.initial_budget - self.budget
        return self.shapley_values, self.final_budget

    def solve(self, d):
        
        self.reset(game=self.game, budget=self.initial_budget)

        budget = self.budget - 1
        permutations_fully_explored = budget  // self.n
        remaining_budget = budget % self.n

        total_permutations = permutations_fully_explored + (1 if remaining_budget else 0)
        permutations = sobol_permutations(total_permutations, self.n)

        game_wrapper = GameWrapper(self)
        mask_empty = np.zeros(self.n, dtype=bool)
        value_empty, _ = game_wrapper(mask_empty.reshape(1,-1), d)

        phi = np.zeros(self.n)
        counts = np.zeros(self.n)

        for perm_index in range(total_permutations):
            mask = np.zeros(self.n, dtype=bool)
            pred_off = value_empty[0]

            for idx in permutations[perm_index]:
                mask[idx] = True
                pred_on, more_budget = game_wrapper(mask.reshape(1,-1), d)
                margin_contrib = pred_on[0] - pred_off
                phi[idx] += margin_contrib
                counts[idx] += 1
                pred_off = pred_on[0]

                if not more_budget:
                    break

            
        phi /= np.where(counts>0, counts, 1)
        self.sv = phi

        return self.get_estimates()
    

    def get_name(self):

        return "Sobol"
    
    def get_estimates(self):
        return self.sv
    

# Sobol-related utility functions below
# @njit
def int_sin_m(x, m):
    """
    Computes the integral of sin(x) for an integer m.
    
    Args:
        x (float): The value at which the integral is computed.
        m (int): The integer parameter of the integral.
    
    Returns:
        float: The result of the integral.
    """
    if m == 0:
        return x
    elif m == 1:
        return 1 - np.cos(x)
    else:
        return (m - 1) / m * int_sin_m(x, m - 2) - np.cos(x) * np.sin(x) ** (m - 1) / m


def equal_area_projection(X, d):
    """
    Projects points generated in a (d-1)-dimensional space onto a d-dimensional sphere.

    Args:
        X (np.array): Points generated in a (d-1)-dimensional space.
        d (int): Target dimension of the sphere.

    Returns:
        np.array: Points projected onto a d-dimensional sphere.
    """
    n = X.shape[0]
    Y = np.ones((n, d))

    for i in range(n):
        Y[i][0] *= np.sin(X[i, 0] * 2 * np.pi)
        Y[i][1] *= np.cos(X[i, 0] * 2 * np.pi)

    for j in range(2, d):
        inv_beta = 1 / beta(j / 2, 1 / 2)
        for i in range(n):
            root_function = lambda varphi: inv_beta * int_sin_m(varphi, j - 1) - X[i, j - 1]
            deg = root_scalar(root_function, bracket=[0, np.pi], xtol=1e-15).root
            for k in range(j):
                Y[i][k] *= np.sin(deg)
            Y[i][j] *= np.cos(deg)
    
    return Y


def sobol_sphere(n, d):
    """
    Generates Sobol points in (d-1)-dimensional space and projects them onto a d-dimensional sphere.

    Args:
        n (int): Number of points to generate.
        d (int): Dimension of the target sphere.

    Returns:
        np.array: Sobol points projected onto a d-dimensional sphere.
    """
    sampler = qmc.Sobol(d - 1, scramble=True)
    X = sampler.random(n)
    return equal_area_projection(X, d)


def zero_sum_projection(d):
    """
    Generates an orthonormal basis for a (d-1)-dimensional subspace of R^d where vectors sum to zero.

    Args:
        d (int): Dimension of the space.

    Returns:
        np.array: Orthonormal basis of the subspace.
    """
    basis = np.array([[1.0] * i + [-i] + [0.0] * (d - i - 1) for i in range(1, d)])
    return np.array([v / np.linalg.norm(v) for v in basis])


def sobol_permutations(n, d):
    """
    Generates permutations from Sobol points projected on a sphere.

    Args:
        n (int): Number of permutations to generate.
        d (int): Dimension of the sphere.

    Returns:
        np.array: Array of permutations for each point.
    """

    
    sphere_points = sobol_sphere(n, d - 1)
    basis = zero_sum_projection(d)
    projected_sphere_points = sphere_points.dot(basis)
    p = np.zeros((n, d), dtype=np.int64)
    
    for i in range(n):
        p[i] = np.argsort(projected_sphere_points[i])
    
    return p