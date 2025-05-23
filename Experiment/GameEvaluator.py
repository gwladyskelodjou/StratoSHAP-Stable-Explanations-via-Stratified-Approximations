import time
import os
import csv
import numpy as np

class GameEvaluator:
    def __init__(self, game, approx_methods, budgets, number_of_runs):
        self.game = game
        self.approx_methods = approx_methods
        self.budgets = budgets
        self.number_of_runs = number_of_runs

    def evaluate_fixed_game(self):
        if not os.path.exists('results'):
            os.makedirs('results')

        game_name = self.game.fetch_game_name()
        game_dir = f'results/{game_name}'
        os.makedirs(game_dir, exist_ok=True)

        for method in self.approx_methods:
            method_name = method.get_name()
            method_dir = f'{game_dir}/{method_name}'
            os.makedirs(method_dir, exist_ok=True)
            for budget in self.budgets:
                try:
                    for _ in range(self.number_of_runs):
                        start_time = time.time()
                        method.reset(game=self.game, budget=budget)
                        shapley_values, final_budget = method.approximate_shapley_values()
                        end_time = time.time()
                        overall_time = end_time - start_time

                        explanation = list(shapley_values[0])
                        explanation.append(overall_time)
                        csv_file_path = f"{method_dir}/{game_name}_{method_name}_{final_budget}.csv"

                        with open(csv_file_path, 'a', newline='') as csv_file:
                            csv_writer = csv.writer(csv_file, delimiter=',')
                            csv_writer.writerow(explanation)
                except Exception as e:
                    print(f"Error: {e} - {game_name} - {method_name} - {budget}")


    def evaluate_ml_game(self):
        if not os.path.exists('results'):
            os.makedirs('results')

        game_name = self.game.fetch_game_name()
        model_name = self.game.model_name
        dataset_name = self.game.dataset_name
        instance_id = self.game.instance_id

        game_dir = f'results/{game_name}/{dataset_name}'
        os.makedirs(game_dir, exist_ok=True)

        for method in self.approx_methods:
            method_name = method.get_name()
            method_dir = f'{game_dir}/{method_name}/{model_name}'
            os.makedirs(method_dir, exist_ok=True)

            for budget in self.budgets:
                budget_dir = f'{method_dir}/{str(budget)}'
                os.makedirs(budget_dir, exist_ok=True)

                pred = self.game.model(self.game.explained_instance.reshape(1, -1))

                try:
                    for run in range(self.number_of_runs):
                        start_time = time.time()
                        method.reset(game=self.game, budget=budget)
                        shapley_values, final_budget = method.approximate_shapley_values()
                        end_time = time.time()
                        overall_time = end_time - start_time
                        
                        if isinstance(pred, np.ndarray) and pred.ndim > 1:
                            predicted_class = np.argmax(pred)
                            explanation = list(shapley_values[predicted_class])
                            explanation.insert(0, predicted_class)
                        else:
                            explanation = list(shapley_values[0])
                            explanation.insert(0, -1)

                        explanation.append(overall_time)
                        # explanation.insert(0, run)

                        csv_file_path = f"{budget_dir}/{dataset_name}_{method_name}_{model_name}_{final_budget}_ins{instance_id}.csv"

                        with open(csv_file_path, 'a', newline='') as csv_file:
                            csv_writer = csv.writer(csv_file, delimiter=',')
                            csv_writer.writerow(explanation)

                except Exception as e:
                    print(f"Error: {e} - {dataset_name}- {method_name} -{model_name} - {budget} - {instance_id}")


    def evaluate_image_game(self):
        if not os.path.exists('results'):
            os.makedirs('results')

        game_name = self.game.fetch_game_name()
        model_name = self.game.model_name
        image_id = self.game.image_id

        game_dir = f'results/{game_name}'
        os.makedirs(game_dir, exist_ok=True)

        for method in self.approx_methods:
            method_name = method.get_name()
            method_dir = f'{game_dir}/{method_name}/{model_name}'
            os.makedirs(method_dir, exist_ok=True)

            for budget in self.budgets:
                budget_dir = f'{method_dir}/{str(budget)}'
                os.makedirs(budget_dir, exist_ok=True)

                pred = self.game.model(self.game.explained_image)

                try:
                    for run in range(self.number_of_runs):
                        start_time = time.time()
                        method.reset(game=self.game, budget=budget)
                        shapley_values, final_budget = method.approximate_shapley_values()
                        end_time = time.time()
                        overall_time = end_time - start_time
                        
                        predicted_class = np.argmax(pred)
                        explanation = list(shapley_values[0])
                        explanation.insert(0, predicted_class)

                        explanation.append(overall_time)
                        # explanation.insert(0, run)

                        csv_file_path = f"{budget_dir}/{method_name}_{model_name}_{final_budget}_ins{image_id}.csv"

                        with open(csv_file_path, 'a', newline='') as csv_file:
                            csv_writer = csv.writer(csv_file, delimiter=',')
                            csv_writer.writerow(explanation)

                except Exception as e:
                    print(f"Error: {e} - {method_name} -{model_name} - {budget} - {image_id}")