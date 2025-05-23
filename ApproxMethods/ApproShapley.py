import numpy as np
from ApproxMethods.ShapleyEstimator import ShapleyEstimator


class PermutationSampling(ShapleyEstimator):
    def approximate_shapley_values(self) -> dict:
        full = list(range(self.n))
        self.grand = self.game.compute_value(full)
        final_budget = self.budget

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

        return self.shapley_values, final_budget

    def solve(self,d):
        self.reset(game=self.game, budget=self.initial_budget)
        player_set = np.arange(self.n)
        more_budget = True
        
        while more_budget:
            permutation = np.random.choice(player_set, self.n, replace=False)
            value_list = np.zeros(permutation.shape)

            for i in range(self.n):
                j = permutation[:i + 1][-1]
                is_budget, val_coalition = self.get_game_value(permutation[:i + 1])
                more_budget, value_list[i] = is_budget, val_coalition[:, self.idx_dims][0,d]

                if i == 0:
                    delta = value_list[i]
                else:
                    delta = value_list[i] - value_list[i - 1]

                self.update_shapley_value(j, delta)

                if not more_budget:
                    break

        
        return self.get_estimates()

    def get_estimates(self):

        return self.sv
    

    def get_name(self):

        return "ApproShapley"
