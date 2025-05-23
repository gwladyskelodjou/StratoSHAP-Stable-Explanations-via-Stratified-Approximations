import numpy as np

from ApproxMethods.ShapleyEstimator import ShapleyEstimator

# Stratified Sampling (Maleki et al., 2013)
class StratifiedSampling(ShapleyEstimator):

    def approximate_shapley_values(self):
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


    def solve(self,d) -> dict:
        self.reset(game=self.game, budget=self.initial_budget)

        self.phi_i_l = np.zeros(shape=(self.n, self.n))
        self.c_i_l = np.zeros(shape=(self.n, self.n))

        # distribute budget among players -> m is list of each player's budget
        m = [int(self.budget / (2 * self.n))] * self.n
        # print(m)
        rest_player_budget = self.budget % self.n
        for i in range(rest_player_budget):
            m[i] += 1

        # budget per stratum -> budget is the matrix m_i_l
        budget = np.zeros(shape=(self.n, self.n))
        denominator = np.sum([np.power(k + 1, 2 / 3) for k in range(self.n)])

        for i in range(self.n):
            for l in range(self.n):
                budget[i][l] = int((m[i] * np.power(l + 1, 2 / 3)) / denominator)

        # fill leftovers
        for i in range(self.n):
            left = int(m[i] - sum(budget[i]))
            for j in range(left):
                budget[i][j] += 1

        # calculate the strata available for each player
        available_stratum = [[i for i in range(self.n)] for _ in range(self.n)]
        for i in range(len(available_stratum)):
            for j in range(len(available_stratum[i])):
                if budget[i][j] == 0:
                    available_stratum[i].remove(j)

        # sample coalitions
        active_player = -1
        more_budget = True
        while more_budget:
            active_player = (active_player + 1) % self.n
            if len(available_stratum[active_player]) == 0:
                if sum(len(i) for i in available_stratum) == 0:
                    break
                else:
                    continue

            # sample stratum
            sampled_stratum = np.random.choice(available_stratum[active_player])
            budget[active_player][sampled_stratum] -= 1
            if budget[active_player][sampled_stratum] == 0:
                available_stratum[active_player].remove(sampled_stratum)

            # sample S
            S_i_l = list(range(self.n))
            S_i_l.remove(active_player)
            if sampled_stratum == 0:
                S1 = []
                S2 = [active_player]
            else:
                S1 = np.random.choice(S_i_l, sampled_stratum, replace=False)
                S2 = np.append(S1, active_player)

            _, first_value = self.get_game_value(S2)
            _, second_value = self.get_game_value(S1)
            
            delta_i_l = first_value[:,self.idx_dims][0,d] - second_value[:,self.idx_dims][0,d]

            # update shapley value
            c = self.c_i_l[active_player][sampled_stratum]
            self.phi_i_l[active_player][sampled_stratum] = (self.phi_i_l[active_player][sampled_stratum] * (
                        c) + delta_i_l) / (c+1)
            self.c_i_l[active_player][sampled_stratum] += 1

        return self.get_estimates()

    def get_estimates(self):
        self.shapley_values = 1 / np.sum(np.where(self.c_i_l > 0, 1, 0), axis=1) * np.sum(self.phi_i_l, axis=1)
        na_indices = np.argwhere(np.isnan(self.shapley_values))
        self.shapley_values[na_indices] = 0
       
        return self.shapley_values

    def get_name(self):
        return 'StratifiedSampling'
