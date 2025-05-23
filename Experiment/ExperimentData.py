from CooperativeGames.CooperativeGameBase import CooperativeGameBase


class ExperimentStorage:

  def __init__(self, game: CooperativeGameBase):
    self.game = game
    self.shapley_value = list()
    self.player_selected = list()
    self.duration_list = list()
    self.mse_history = list()
    self.shapley_mean_list = list()
    self.shapley_var_list = list()



  def add_shapley_values(self, shapley_value):
    """
    Add shapley values to the storage.
    :param shapley_value: list of shapley values

    NB: shapley_value is a list of shapley values for each player
    By replacing self.shapley_value with shapley_value, this suppose that shapley_value is a list of shapley values for each player.
    """
    self.shapley_value = shapley_value


  def add_time_evaluation(self, duration):
    self.duration_list.append(duration)


  def add_mse_value(self, mse):
    self.mse_history.append(mse)


  def add_mean_shapley_value(self, mean_value):
    self.shapley_mean_list.append(mean_value)


  def add_var_shapley_value(self, var_value):
      self.shapley_var_list.append(var_value)


  def to_json(self):
    result_dict = dict()
    result_dict['GameInformation'] = self.game.get_game_information()
    result_dict['v(N)'] = self.game.get_value(list(range(self.game.get_player_number())))
    result_dict['shapley_value'] = self.shapley_value
    result_dict['durations'] = self.duration_list
    result_dict['mse_history'] = self.mse_history
    result_dict['mean_shapley_values'] = self.shapley_mean_list
    result_dict['var_shapley_values'] = self.shapley_var_list
    return result_dict
  
  def __str__(self) -> str:
    result_dict = self.to_json()
    return (f"ExperimentStorage Object:\n"
            f"\tGame: {result_dict['GameInformation']['name']}\n"
            f"\tNumber of Players: {result_dict['GameInformation']['number_of_players']}\n"
            f"\tGround Truth Shapley Value: {result_dict['GameInformation']['ground_truth_shapley_value']}\n"
            f"\tValue of v(N): {result_dict['v(N)']}\n"
            f"\tShapley Value: {result_dict['shapley_value']}\n"
            f"\tDurations: {result_dict['durations']}\n"
            f"\tMSE History: {result_dict['mse_history']}\n"
            f"\tMean Shapley Values: {result_dict['mean_shapley_values']}\n"
            f"\tVar Shapley Values: {result_dict['var_shapley_values']}\n")