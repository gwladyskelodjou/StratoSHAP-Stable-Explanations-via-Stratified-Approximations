from enum import Enum
from Experiment.GameEvaluator import GameEvaluator
from ApproxMethods.ApproShapley import PermutationSampling
from ApproxMethods.SVARM import SVARM
from ApproxMethods.SobolSampling import Sobol
from ApproxMethods.MCAntitheticSampling import MonteCarloAntithetic
from ApproxMethods.KernelShap import KernelSHAP
from ApproxMethods.StratoShap import StratoSHAP
from ApproxMethods.StratifiedSampling import StratifiedSampling
from ApproxMethods.UnbiasedKernelShap import UnbiasedKernelSHAP
from CooperativeGames.ShoesGame import ShoesGame
from CooperativeGames.SumOfUnanimityGame import SumOfUnanimityGame
from CooperativeGames.AirportGame import AirportGame
from CooperativeGames.MachineLearningGame import MachineLearningGame
from CooperativeGames.ImageGame import ImageGame
from utils.data_loading import Data
import numpy as np

NB_FEATURES = {
    "adult": 12,
    "bank": 15,
    "german": 20,

    "thoraric": 16,
    "dtcr": 16,

    "thyroid": 16,
    "parkinson": 19,
    "compas": 13,
    "bike": 12,

    "fixed": 20,
    "soug": 20,
    "shoes": 20,
    "airport": 20,

    "img": 16,
    "image": 16
}

def get_nb_features(dataset_name: str) -> int:

    return NB_FEATURES.get(dataset_name.lower(), 0)

def total_stratum_number(dataset_name: str) -> int:

    nb_features = get_nb_features(dataset_name)
    if nb_features == 0:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return int(np.ceil((nb_features - 1) / 2.0))


class Method(Enum):
    STRATOSHAP = StratoSHAP()
    SOBOL = Sobol()
    MCAntithetic = MonteCarloAntithetic()
    KERNELSHAP = KernelSHAP()
    SVARM = SVARM()
    UNBIASED_KSHAP = UnbiasedKernelSHAP()
    APPROSHAPLEY = PermutationSampling()
    STRATIFIED = StratifiedSampling()


class DatasetName(Enum):
    PARKINSON = "parkinson"
    BANK = "bank"
    COMPAS = "compas"
    BIKE = "bike"
    ADULT = "adult"
    GERMAN = "german"
    DTCR = "thyroid"
    THORACIC = "thoraric"

class ModelName(Enum):
    SVM = "SVM"
    MLP = "MLP"
    XGB = "XGB"


class Game(Enum):
    MACHINE_LEARNING_GAME = "ml"
    IMAGE_GAME = "img"
    SHOES = ShoesGame(20)
    SOUG = SumOfUnanimityGame(20, 50)
    AIRPORT = AirportGame()


def start_experiment(method, game, budgets, model=None, dataset=None, instance_id=0, number_of_runs=30, **kwargs):
    approx_method = method.value

    if game.value == "ml":
        dataset_name = dataset.value
        model_name = model.value
        baseline = kwargs.get("baseline", False)

        machine_learning_game(
            dataset_name=dataset_name,
            model_name=model_name,
            instance_id=instance_id,
            approx_methods=[approx_method],
            budgets=budgets,
            number_of_runs=number_of_runs,
            baseline=baseline
        )

    elif game.value == "img":
        image_id = kwargs.get("image_id", 0)
        tile_size = kwargs.get("tile_size", 14)
        img_dims = kwargs.get("img_dims", 224)

        image_game(
            image_id=image_id,
            img_dims=img_dims,
            tile_size=tile_size,
            approx_methods=[approx_method],
            budgets=budgets,
            number_of_runs=number_of_runs,
        )

    else:
        run_experiment(
            game=game.value,
            approx_methods=[approx_method],
            number_of_runs=number_of_runs,
            budgets=budgets,
        )


def run_experiment(game, approx_methods, budgets, number_of_runs, game_type = "fixed_game"):
    
    game_evaluator = GameEvaluator(
        game=game,
        approx_methods=approx_methods,
        budgets=budgets,
        number_of_runs=number_of_runs,
    )

    if game_type == "fixed_game":
        evaluation = game_evaluator.evaluate_fixed_game()
    elif game_type == "ml_game":
        evaluation = game_evaluator.evaluate_ml_game()
    elif game_type == "image_game":
        evaluation = game_evaluator.evaluate_image_game()
    else:
        raise ValueError(f"Game type '{game_type}' not found.")

    return evaluation


def machine_learning_game(dataset_name, model_name, instance_id, approx_methods, budgets, number_of_runs, game_type="ml_game", baseline=False):

    data = Data(dataset_name=dataset_name, model_name=model_name)
    X_train, X_test, Y_train, Y_test = data.load_data()
    model_to_explain = data.load_model()

    explained_instance = X_test.iloc[instance_id, :].values
    background_dataset = (
        X_train.iloc[:100, :].values if not baseline else X_train.values.mean(0).reshape(1, -1)
    )

    game = MachineLearningGame(
        model=model_to_explain,
        model_name=model_name,
        dataset_name=dataset_name,
        explained_instance=explained_instance,
        instance_id=instance_id,
        background_dataset=background_dataset,
    )

    run_experiment(
        game=game,
        approx_methods=approx_methods,
        number_of_runs=number_of_runs,
        budgets=budgets,
        game_type=game_type,
    )


def image_game(image_id, img_dims, tile_size, approx_methods, budgets, number_of_runs, game_type="image_game"):

    data = Data(dataset_name="image", model_name="ResNet50")
    X, Y = data.load_data()
    model, class_names = data.load_model()
    explained_image = X[image_id]

    game = ImageGame(
        model=model,
        explained_image=explained_image,
        image_id=image_id,
        img_dims=img_dims,
        tile_size=tile_size,
    )

    run_experiment(
        game=game,
        approx_methods=approx_methods,
        number_of_runs=number_of_runs,
        budgets=budgets,
        game_type=game_type
    )