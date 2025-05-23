import os
import json
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from tensorflow.keras.preprocessing import image  # type: ignore
from sklearn import metrics

from CooperativeGames.CooperativeGameBase import CooperativeGameBase
import utils.utils as utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ImageGame(CooperativeGameBase):
    """
    
    """

    def __init__(
        self, 
        model, 
        explained_image, 
        image_id, 
        model_name="ResNet50", 
        img_dims=224, 
        tile_size=14
    ):
        """
        Initializes the ImageGame instance.

        Args:
            model (callable): The model to be explained.
            model_name (str): The name of the model.
            explained_image (np.ndarray): The image to be explained.
            image_id (int): The index of the image in the dataset.
            img_dims (int): The dimensions of the image.
            tile_size (int): The size of the tiles.

        """

        self.nb_features = self.n = (img_dims // tile_size) ** 2
        self.tile_size = tile_size
        self.img_dims = img_dims
        self.model_name = model_name
        self.game_name = "ImageClassification"
        self.image_id = image_id
        self.model = model
        
        #Preprocess image
        self.explained_image = utils.preprocess_image(explained_image, img_dims)[0]
        
        #Segment the image into tiles
        self.all_tiles = utils.segment_image(self.explained_image, self.tile_size)

        # Run the model once on the image to determine output dimension
        d = self.model(self.explained_image, verbose=0)

        if isinstance(d, np.ndarray):
            self.dim = d.shape[1] if len(d.shape) > 1 else 1
        elif isinstance(d, tf.Tensor):
            self.dim = d.shape[-1]
            d = d.numpy()
        else:
            raise ValueError("Unexpected model output type. Expected numpy array or Tensorflow tensor.")

        
        logging.info(f"Predicted max confidence: {np.max(d)}")
        logging.info(f"Model output dimension: {self.dim}")


    def describe_game(self):

        """
        Provides a description of the cooperative game.

        Returns:
            dict: A dictionary describing the game.
        """

        description = {
            "type": "Cooperative Game",
            "game_name": self.game_name,
            "rules": "Explain the prediction of a model for a given image.",
            "number_of_players": self.nb_features
        }
        
        return description
    
    def fetch_game_name(self):

        """Returns the name of the game."""

        return self.game_name
    

    def number_of_players(self):
        return self.nb_features
    
    def calculate_shapley_values(self):
        """
        Computes and returns the Shapley values for each superpixel in the game.

        Returns:
            list: A list of Shapley values for each superpixel in the image.
        """
        pass


    def compute_value(self, coalition):
        """
        Computes the value of the game for a given coalition of superpixels.
        
        Args:
            coalition (list or set): A coalition of players.
        
        Returns:
            float: The computed value for the given coalition.
        """
        

        mask, perturb_img, weight = utils.construct_disturbed_images(
            list(coalition), self.img_dims, self.all_tiles
            )
        
        output = self.model(perturb_img, verbose=0) 

        if isinstance(output, tf.Tensor):
            output = output.numpy()

        return output


