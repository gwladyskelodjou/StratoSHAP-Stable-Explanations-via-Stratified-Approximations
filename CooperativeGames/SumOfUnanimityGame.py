import random
import numpy as np
from CooperativeGames.CooperativeGameBase import CooperativeGameBase



class SumOfUnanimityGame(CooperativeGameBase):
    """
    This class represents the Sum of Unanimity Game. 
    The game is a weighted sum of unanimity games, where each unanimity game gives a value only to coalitions that contain a specific set of players.

    Campenn 2018 present a sum of unanimity game as ...
    """

    def __init__(self, n, num_unanimity_sets):
        """ 
        Initializes the Sum of Unanimity Game with n players and num_unanimity_sets sets.

        Parameters:
            n (int): The number of players in the game.
            num_unanimity_sets (int): The number of unanimity sets (should be between 1 and 2^n)
        """

        assert 1 <= num_unanimity_sets <= 2 ** n, "num_unanimity_sets must be between 1 and 2^n"
        
        self.n = n
        self.total_game_value = 100
        self.min_set_size = 1
        self.max_set_size = int(self.n)
        self.unanimity_sets, self.set_weights = self.__generate_random_unanimity_sets(num_unanimity_sets=num_unanimity_sets, seed=42)
        self.exact_shapley_values = self.calculate_shapley_values()
        

    def describe_game(self):
        """
        Describe the game setup.
        """
        return {
            "name": "Sum of Unanimity Game",
            "number_of_players": self.n,
            "ground_truth_shapley_values": self.exact_shapley_values.tolist(),
            "unanimity_sets": self.unanimity_sets,
            "coefficients": self.set_weights
        }

    def fetch_game_name(self):
        return "SOUG"

    def compute_value(self, S):
        """
        Computes the value of a coalition S based on the unanimity sets and coefficients.

        Parameters:
            S (set): The coalition of players.
        Returns: 
            (float): The value of the coalition S. 
        """

        value = 0

        for i, set_ in enumerate(self.unanimity_sets):
            if self.__issubset(set_, S):
                value += self.set_weights[i]

        tmp = np.array(value)
        return tmp.reshape(1,-1)

    @staticmethod
    def __issubset(subset, main_set):
        """
        Helper method to check if a set is a subset of another set.
        """
        return all((player in main_set) for player in subset)
    
    def __generate_random_unanimity_sets(self, num_unanimity_sets, seed=None):
        """ 
        Randombly generates a specified number of unanimity sets and their associated weights.

        Parameters:
            num_unanimity_sets (int): The number of unanimity sets to generate.
        Returns:
            (list, list): A tuple of the generated unanimity sets and their associated weights.
        """

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        unanimity_sets = []
        set_weigths = []

        for _ in range(num_unanimity_sets):
          set_size = random.randint(self.min_set_size, self.max_set_size)
          unanimity_sets.append(np.sort(np.random.choice(self.n, set_size, replace=False)))
          set_weigths.append(np.random.random_sample())

        # Normalize the weights to sum up to the total game value
        set_weigths = [self.total_game_value * weight / sum(set_weigths) for weight in set_weigths]

        return unanimity_sets, set_weigths

    
    def number_of_players(self):
        return self.n

    def calculate_shapley_values(self):
        """
        Computes the Shapley values for all players based on the unanimity sets and their associated weights.

        Returns:
            (np.array): The Shapley values for all players.
        """
        shapley_values = np.zeros(self.n)

        for player in range(self.n):
            shapley_values[player] = sum(
                self.set_weights[j]/len(subset) if player in subset else 0
                for j, subset in enumerate(self.unanimity_sets)
            )
            
        return shapley_values
