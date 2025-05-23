from CooperativeGames.CooperativeGameBase import CooperativeGameBase
import utils.utils as utils
import numpy as np

class MachineLearningGame(CooperativeGameBase):
    """
    """
    
    def __init__(self, model=None, model_name = None, dataset_name = None,explained_instance=None, instance_id = None, background_dataset=None):
        self.nb_features = self.n = background_dataset.shape[1]
        self.explained_instance = explained_instance
        self.background_dataset = background_dataset
        self.game_name = "MachineLearning"
        self.model = model
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.instance_id = instance_id
        
        d = self.model(self.explained_instance.reshape(1, -1))
        self.dim = d.shape[1] if len(d.shape) > 1 else 1


    def describe_game(self) -> dict:
        """
        Provides a description of the game, including key information such as the 
        type of game, rules, and any relevant details.
        
        Returns:
            dict: A dictionary containing the game description.
        """

        description = {
            "type": "Cooperative Game",
            "game_name": self.game_name,
            "rules": " Explain the prediction of a model for a given instance. ",
            "number_of_players": self.nb_features
        }
        return description
    


    def fetch_game_name(self) -> str:
        """
        Returns the name of the game.
        """
        return f"MachineLearning"
    
    def number_of_players(self):
        return self.nb_features
    
    def compute_value(self, coalition) -> float:
        """
        Computes the value of the game for a given coalition of players.
        
        Args:
            coalition (list or set): A coalition of players.
        
        Returns:
            float: The computed value for the given coalition.
        """
        mask, synthdata, weight = utils.reconstruct_coalitions(list(coalition), self.explained_instance, self.nb_features, self.background_dataset)

        y = self.model(synthdata[0])

        return np.mean(y, axis=0).reshape(1,-1)
    
    def calculate_shapley_values(self) -> list:
        """
        Computes and returns the Shapley values for each player in the game.
        
        Returns: 
            list: A list of Shapley values for each player in the game.
        """

        pass
