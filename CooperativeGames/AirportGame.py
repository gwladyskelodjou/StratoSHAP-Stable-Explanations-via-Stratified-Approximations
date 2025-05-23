from CooperativeGames.CooperativeGameBase import CooperativeGameBase
import numpy as np


class AirportGame(CooperativeGameBase):

    def __init__(self):
        
        self.c = [0] + [1]*2 + [2]*2 + [3] + [4]*4 + [5]*3 + [6]*2 + [7]*3 + [8]*2 
        self.true_shapley_values = [
            0.0,
            0.052631578947,
            0.052631578947,
            0.111455108359,
            0.111455108359,
            0.178121775026,
            0.249550346454,
            0.249550346454,
            0.249550346454,
            0.249550346454,
            0.349550346454,
            0.349550346454,
            0.349550346454,
            0.492407489312,
            0.492407489312,
            0.692407489312,
            0.692407489312,
            0.692407489312,
            1.192407489312,
            1.192407489312
           
        ]
        self.n = len(self.c)


    def describe_game(self):
        information = dict()
        information["name"] = "Airport Game"
        information["number_of_players"] = self.n
        information["ground_truth_shapley_values"] = self.true_shapley_values
        information["c"]= self.c

        return information

    def fetch_game_name(self):
        return "Airport"

    def compute_value(self, S)-> float:
        if len(S) ==0:
            return np.array([0]).reshape(1, -1)
        
        tmp = np.array(max([self.c[i] for i in range(self.n) if i in S]))
        return tmp.reshape(1,-1)

    def number_of_players(self):
        return self.n

    def calculate_shapley_values(self):
        return self.true_shapley_values


    def __str__(self) -> str:
        return (
            f"Airpoort Object: \n"
            f"Information about the game : {self.describe_game()}"
        )
