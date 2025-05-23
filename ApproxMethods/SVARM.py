import numpy as np
from ApproxMethods.ShapleyEstimator import ShapleyEstimator


# SVARM
class SVARM(ShapleyEstimator):
  def __init__(self, warm_up=True):
    super().__init__()
    self.warm_up = warm_up

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


  def solve(self,d):
    self.reset(game=self.game, budget=self.initial_budget)

    self.phi_i_plus = np.zeros(self.n)
    self.phi_i_minus = np.zeros(self.n)
    self.c_i_plus = np.zeros(self.n)
    self.c_i_minus = np.zeros(self.n)
    self.H_n = sum([1/s for s in range(1, self.n+1)])

    if self.warm_up:
      self.__conduct_warmup(d)

    more_budget = True
    while more_budget:
      A_plus = self.__sample_A_plus()
      more_budget = self.__positive_update(A_plus,d)
      if not more_budget:
        break

      A_minus = self.__sample_A_minus()
      more_budget = self.__negative_update(A_minus,d)

    return self.get_estimates()


  def __sample_A_plus(self):
    s_plus = np.random.choice(range(1, self.n+1), 1, p=[1/(s*self.H_n) for s in range(1, self.n+1)])
    return np.random.choice(self.get_all_players(), s_plus, replace=False)


  def __sample_A_minus(self):
    s_minus = np.random.choice(range(0, self.n), 1, p=[1/((self.n-s)*self.H_n) for s in range(0, self.n)])
    return np.random.choice(self.get_all_players(), s_minus, replace=False)


  def __positive_update(self, A, d):
    more_budget, value = self.get_game_value(A)
    for i in A:
      self.phi_i_plus[i] = (self.phi_i_plus[i]*self.c_i_plus[i] + value[:, self.idx_dims][0,d]) / (self.c_i_plus[i] + 1)
      self.c_i_plus[i] += 1
    return more_budget


  def __negative_update(self, A, d):
    more_budget, value = self.get_game_value(A)
    players = [i for i in self.get_all_players() if i not in A]
    for i in players:
      self.phi_i_minus[i] = (self.phi_i_minus[i]*self.c_i_minus[i] + value[:, self.idx_dims][0,d]) /(self.c_i_minus[i] + 1)
      self.c_i_minus[i] += 1
    return more_budget


  def __conduct_warmup(self,d):
    for i in self.get_all_players():
      players_without_i = [j for j in self.get_all_players() if j != i]

      # sample A_plus
      size_of_A_plus = np.random.choice(self.n, 1)
      A_plus = np.random.choice(players_without_i, size_of_A_plus, replace=False)

      # sample A_minus
      size_of_A_minus = np.random.choice(self.n, 1)
      A_minus = np.random.choice(players_without_i, size_of_A_minus, replace=False)

      # set values
      _, value = self.get_game_value(np.append(A_plus, i))
      self.phi_i_plus[i] = value[:, self.idx_dims][0,d]
      self.c_i_plus[i] = 1

      _, value = self.get_game_value(A_minus)
      self.phi_i_minus[i] = value[:, self.idx_dims][0,d]
      self.c_i_minus[i] = 1

  def get_estimates(self):
    return  self.phi_i_plus - self.phi_i_minus
    
  def get_name(self) -> str:
    if self.warm_up:
      return 'SVARM_warmup'
    return 'SVARM'
