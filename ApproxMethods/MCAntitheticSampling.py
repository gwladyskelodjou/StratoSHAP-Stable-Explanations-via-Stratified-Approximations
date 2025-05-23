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
    
class MonteCarloAntithetic(ShapleyEstimator):
    def __init__(self):
        self.final_budget = 0

    def approximate_shapley_values(self) -> dict:
        full = list(range(self.n))
        self.grand = self.game.compute_value(full)
        
        if isinstance(self.grand, np.ndarray) and len(self.grand.shape) > 1:
            
            num_classes = self.grand.shape[1]

            if num_classes > 10: 
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

        return self.shapley_values, self.final_budget
    
    def solve(self, d):
        
        self.reset(game=self.game, budget=self.initial_budget)

        budget = self.budget - 1
        
        n_permutations = budget // self.n
        n_antithetic_pairs = (n_permutations // 2) * 2
        remaining_budget = budget - (n_antithetic_pairs * self.n)
        
        permutations = get_antithetic_permutations(n_antithetic_pairs, self.n)
                
        game_wrapper = GameWrapper(self)
        mask_empty = np.zeros(self.n, dtype=bool)
        value_empty, _ = game_wrapper(mask_empty.reshape(1,-1), d)
        
        phi = np.zeros(self.n)
        counts = np.zeros(self.n)

        for perm_index in range(n_antithetic_pairs):
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

        if remaining_budget > 0 and more_budget:

            if remaining_budget < self.n:
                perm = np.random.permutation(self.n)
                mask = np.zeros(self.n, dtype=bool)
                pred_off = value_empty[0]

                for idx in perm:
                    mask[idx] = True
                    pred_on, more_budget = game_wrapper(mask.reshape(1,-1), d)
                    margin_contrib = pred_on[0] - pred_off
                    phi[idx] += margin_contrib
                    counts[idx] += 1
                    pred_off = pred_on[0]

                    if not more_budget:
                        break   
            else:
                perms = get_antithetic_permutations(2, self.n)

                mask = np.zeros(self.n, dtype=bool)
                pred_off = value_empty[0]

                for perm in range(2):

                    for idx in perms[perm]:
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

        self.final_budget = self.n * n_antithetic_pairs + remaining_budget + 1
        return self.get_estimates()
    
    def get_name(self):

        return "MCAntithetic"
    
    def get_estimates(self):
        return self.sv
        
#MC Antithetic utility functions
def get_antithetic_permutations(nb_perm, nb_players):
    p = np.zeros((nb_perm, nb_players), dtype=np.int64)
    # Forward samples
    
    for i in range(nb_perm // 2):
        p[i] = np.random.permutation(nb_players)
    # Reverse samples
    for i in range(nb_perm // 2):
        p[i + nb_perm // 2] = np.flip(p[i])
    return p    