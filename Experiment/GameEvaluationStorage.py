import numpy as np

from utils.utils import calculate_mse



class GameEvaluationStorage:

  def __init__(self,
               number_of_runs,
               approx_methods,
               budget):
    self.approx_methods = approx_methods
    self.number_of_runs = number_of_runs
    self.budget = budget

    shape = (len(approx_methods), number_of_runs)
    self.mse = np.zeros(shape=shape)
    self.runtimes = np.zeros(shape=shape)
    self.shapley_values = np.zeros(shape=shape, dtype=list)

    self.experiment = dict()

    # for method in approx_methods:
    #   self.experiment[f'method: {method.get_name()}'] = {}



  def add_experiment_information(self, experiment_storage, method_number, run, game):

    print(f"=== fonction 'adding experiment information' ===")
    
    shapley_value = experiment_storage['shapley_value']
    self.mse[method_number, run] = calculate_mse(shapley_value, game.get_shapley_values())
    self.runtimes[method_number, run] = experiment_storage['overall_time']
    self.shapley_values[method_number, run] = shapley_value.tolist()

    # add experiment to json
    self.experiment[f'method: {self.approx_methods[method_number].get_name()}'][f'run: {run}'] = experiment_storage


  def to_json(self):
    info_dict = dict()
    info_dict['ShapleyValues'] = self.shapley_values
    info_dict['MeanSquaredError'] = self.mse
    info_dict['runtimes'] = self.runtimes
    info_dict['budget'] = self.budget
    info_dict['number_of_runs'] = self.number_of_runs
    info_dict['approx_methods'] = [i.get_name() for i in self.approx_methods]
    info_dict['experiments'] = self.experiment

    print(f"Resultat obtenu (fonction to_json appelée à la fin de evaluate_game) : \n\t{info_dict}\n")

    return info_dict
  

  def __str__(self):
    approx_methods_names = [method.get_name() for method in self.approx_methods]

    return (f"GameEvaluationStorage Object:\n"
            f"  \tApprox Methods: {approx_methods_names}\n"
            f"  \tBudget: {self.budget}\n"
            f"  \tNumber of Runs: {self.number_of_runs}\n"
            f"  \tMSE Shape: {self.mse.shape}\n"
            f"  \tRuntimes Shape: {self.runtimes.shape}\n"
            f"  \tShapley Values Shape: {self.shapley_values.shape}\n"
            f"  \tExperiment: {self.experiment}"
    )

   
