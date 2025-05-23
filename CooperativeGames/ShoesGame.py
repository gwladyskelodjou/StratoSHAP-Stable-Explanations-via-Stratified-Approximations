from CooperativeGames.CooperativeGameBase import CooperativeGameBase
import numpy as np

class ShoesGame(CooperativeGameBase):

    def __init__(self, n):
        assert n%2 ==0, "n must be even"
        self.n = n

    def describe_game(self):

        return {
            "name": "Sum of Unanimity Game",
            "number_of_players": self.n,
            "ground_truth_shapley_values": self.calculate_shapley_values()
        }

    def fetch_game_name(self):
        return "Shoes"

    def compute_value(self, S):
        S = list(S)
        S_left = len([i for i in S if i >= self.n/2])
        S_right = len([i for i in S if i < self.n/2])
        tmp = min(S_left, S_right)
        tmp = np.array(tmp)

        return tmp.reshape(1,-1)

    def number_of_players(self):
        return self.n

    def calculate_shapley_values(self):
        return [0.5 for i in range(self.n)]
