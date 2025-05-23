import copy
import itertools
import random
import numpy as np
from scipy.special import binom
from ApproxMethods.ShapleyEstimator import ShapleyEstimator

class KernelSHAP(ShapleyEstimator):
    def __init__(self, pairing=False):
        super().__init__()
        self.pairing = pairing
        self.big_M = 10_000_000_000

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
    

    def solve(self, d) -> dict:

        weights = self.__get_weights()
        sampling_weight = (np.asarray([0] + [*weights] + [0])) / sum(weights)
        regression_weights = (np.asarray([self.big_M] + [*weights] + [self.big_M]))
        game_function = self.game.compute_value
        sampling_budget = self.budget - 2

        S_list, game_values, kernel_weights = self.__get_S_and_game_values(
            budget=sampling_budget,
            num_players=self.n,
            weight_vector=sampling_weight,
            N=self.get_all_players(),
            pairing=self.pairing,
            game_fun=game_function,
            d=d
        )

        empty_value = game_function({})[:, self.idx_dims][0,d]
        full_value = game_function(self.get_all_players())[:, self.idx_dims][0,d]

        S_list.append(set())
        S_list.append(self.get_all_players())
        game_values.append(empty_value)
        game_values.append(full_value)
        kernel_weights[()] = self.big_M
        kernel_weights[tuple(self.get_all_players())] = self.big_M

        self.all_S = np.zeros(shape=(len(S_list), self.n), dtype=bool)
        for i, subset in enumerate(S_list):
            if len(subset) != 0:
                subset = np.asarray(list(subset))
                self.all_S[i, subset] = 1
        self.game_values = np.asarray(game_values) - empty_value

        self.W = np.zeros(shape=np.array(game_values).shape, dtype=float)
        for i, S in enumerate(self.all_S):
            weight = kernel_weights[tuple(sorted(np.where(S)[0]))]
            self.W[i] = weight

        return self.get_estimates()

    def __get_weights(self):
        weights = np.arange(1, self.n)
        weights = (self.n - 1) / (weights * (self.n - weights))
        return weights / np.sum(weights)

    def __get_S_and_game_values(self, budget, num_players, weight_vector, N, pairing, game_fun, d):
        complete_subsets, incomplete_subsets, budget = self.__determine_complete_subsets(
            budget=budget, n=num_players, s=1, q=weight_vector)

        all_subsets_to_sample = []
        kernel_weights = {}

        for complete_subset in complete_subsets:
            combinations = itertools.combinations(N, complete_subset)
            for subset in combinations:
                subset = set(subset)
                all_subsets_to_sample.append(subset)
                kernel_weights[tuple(sorted(subset))] = weight_vector[len(subset)] / binom(num_players, len(subset))

        remaining_weight = weight_vector[incomplete_subsets] / sum(weight_vector[incomplete_subsets])
        kernel_weights_sampling = {}

        if len(incomplete_subsets) > 0:
            sampled_subsets = set()
            n_sampled_subsets = 0
            while len(sampled_subsets) < budget:
                subset_size = random.choices(incomplete_subsets, remaining_weight, k=1)
                ids = np.random.choice(num_players, size=subset_size, replace=False)
                sampled_subset = tuple(sorted(ids))
                if sampled_subset not in sampled_subsets:
                    sampled_subsets.add(sampled_subset)
                    kernel_weights_sampling[sampled_subset] = 1.
                else:
                    kernel_weights_sampling[sampled_subset] += 1.
                n_sampled_subsets += 1
                if pairing and len(sampled_subsets) < budget:
                    sampled_subset_paired = tuple(sorted(set(N) - set(ids)))
                    if sampled_subset_paired not in sampled_subsets:
                        sampled_subsets.add(sampled_subset_paired)
                        kernel_weights_sampling[sampled_subset_paired] = 1.
                    else:
                        kernel_weights_sampling[sampled_subset_paired] += 1.
            for subset in sampled_subsets:
                all_subsets_to_sample.append(set(subset))

            weight_left = np.sum(weight_vector[incomplete_subsets])
            kernel_weights_sampling = {subset: weight * (weight_left / n_sampled_subsets) for subset, weight in kernel_weights_sampling.items()}
            kernel_weights.update(kernel_weights_sampling)

        game_values = [game_fun(subset)[:, self.idx_dims][0,d] for subset in all_subsets_to_sample]
        return all_subsets_to_sample, game_values, kernel_weights

    def __determine_complete_subsets(self, s, n, budget, q):
        complete_subsets = []
        paired_subsets, unpaired_subset = self.__get_paired_subsets(s, n)

        incomplete_subsets = list(range(s, n - s + 1))

        weight_vector = copy.copy(q)
        weight_vector = np.divide(weight_vector, np.sum(weight_vector), out=weight_vector, where=np.sum(weight_vector) != 0)
        allowed_budget = weight_vector * budget

        for subset_size_1, subset_size_2 in paired_subsets:
            subset_budget = int(binom(n, subset_size_1))
            if allowed_budget[subset_size_1] >= subset_budget and allowed_budget[subset_size_1] > 0:
                complete_subsets.extend((subset_size_1, subset_size_2))
                incomplete_subsets.remove(subset_size_1)
                incomplete_subsets.remove(subset_size_2)
                weight_vector[subset_size_1] = 0
                weight_vector[subset_size_2] = 0
                if not np.sum(weight_vector) == 0:
                    weight_vector /= np.sum(weight_vector)
                budget -= subset_budget * 2
            else:
                return complete_subsets, incomplete_subsets, budget

            allowed_budget = weight_vector * budget
        if unpaired_subset is not None:
            subset_budget = int(binom(n, unpaired_subset))
            if budget - subset_budget >= 0:
                complete_subsets.append(unpaired_subset)
                incomplete_subsets.remove(unpaired_subset)
                budget -= subset_budget

        return complete_subsets, incomplete_subsets, budget

    def __get_paired_subsets(self, s, n):
        subset_sizes = list(range(s, n - s + 1))
        n_paired_subsets = int(len(subset_sizes) / 2)
        paired_subsets = [(subset_sizes[subset_size - 1], subset_sizes[-subset_size]) for subset_size in range(1, n_paired_subsets + 1)]
        unpaired_subset = int(np.median(subset_sizes)) if n_paired_subsets < len(subset_sizes) / 2 else None
        return paired_subsets, unpaired_subset

    def get_estimates(self):
        if self.W is None:
            return 0
        A = self.all_S
        B = self.game_values
        
        W = np.sqrt(self.W) 
        Aw = A * W[:, np.newaxis]
        Bw = B * W
        phi, residuals, rank, singular_values = np.linalg.lstsq(Aw, Bw, rcond=None)
 
        return phi

    def get_name(self):
        return 'KernelSHAP'

