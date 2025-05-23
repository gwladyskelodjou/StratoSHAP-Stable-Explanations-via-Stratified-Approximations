from abc import ABC, abstractmethod

class CooperativeGameBase(ABC):
    """
    Abstract base class for defining cooperative games. Subclasses must implement 
    the abstract methods to provide game-specific information and logic.
    """

    @abstractmethod
    def describe_game(self) -> dict:
        """
        Provides a description of the game, including key information such as the 
        type of game, rules, and any relevant details.
        
        Returns:
            dict: A dictionary containing the game description.
        """
        pass
    
    @abstractmethod
    def fetch_game_name(self) -> str:
        """
        Returns the name of the game.
        """
        pass

    @abstractmethod
    def compute_value(self, coalition) -> float:
        """
        Computes the value of the game for a given coalition of players.
        
        Args:
            coalition (list or set): A coalition of players.
        
        Returns:
            float: The computed value for the given coalition.
        
        Raises:
            NotImplementedError: If this method is not implemented in the subclass.
        """

        raise NotImplementedError("The compute_value method must be overridden in a subclass.")

    @abstractmethod
    def number_of_players(self) -> int:
        """
        Retrieves the number of players involved in the game.
        """

        pass

    @abstractmethod
    def calculate_shapley_values(self) -> list:
        """
        Computes and returns the Shapley values for each player in the game.
        
        Returns: 
            list: A list of Shapley values for each player in the game.
        """

        pass
