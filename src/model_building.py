import logging
from abc import ABC , abstractmethod
from typing import Any

import pandas as pd  
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# set up logging configuration
logging.basicConfig(level = logging.INFO,format = "%(asctime)s-%(levelname)s-%(message)s")


# Abstarct Base Class for Model Building Strategy
class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self,X_train:pd.DataFrame,y_train:pd.Series)->RegressorMixin:
        """
        Abstract method to build and train a model
        Args:
            X_train (_type_): The training data features.  
            y_train (pd.Series): The training data labels/target.  
        Returns:
        RegressorMixin: A trained scikit-learn model instance.  
        """
        
        pass

# Concrete Strategy for Linear Regression using scikit-learn
class LinearRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Build and trains alinear regression model using scikit learn

        Args:
            X_train (pd.DataFrame): The trainibg data features 
            y_train (pd.Series): The training data labels / target.
        Returns:
        Pipeline: A scikit-learn pipeline with a trained Linear  Regression model
        """
        # Ensure the inputs are of the correct type
        if not isinstance(X_train,pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame")
        if not isinstance(y_train,pd.Series):
            raise TypeError("y_train must be a pandas Series")
        logging.info("Initializing Linear Regression model with scale")
        
        # Creating a pipeline with standrad scaling and linear regression model
        pipeline = Pipeline(
            [
                ("scaler",StandardScaler()), # feature scaling
                ("model",LinearRegression()), #  Linear regression
            ]
        )
        
        logging.info("Training Linear Regression model.")
        pipeline.fit(X_train,y_train) # Fit the pipeline to the model
        
        logging.info("Model Training Completed")
        return pipeline
    
# Context Class for model building

class ModelBuilder:
    def __init__(self,strategy:ModelBuildingStrategy):
        """
        Initializes the ModelBuilder with a specific model building strategy
        Args:
            strategy (ModelBuildingStrategy): The strategy to be 
        """
        self._strategy = strategy
        
    def set_strategy(self,strategy:ModelBuildingStrategy):
        """
        Sets a new strategy for the ModelBuilder

        Args:
            strategy (ModelBuildingStrategy): The new strategy to be set
        """
        logging.info("Switching model building strategy to {}",strategy)
        self._strategy = strategy
        
    def build_model(self,X_train : pd.DataFrame,y_train:pd.Series)->RegressorMixin:
        """
        executes the model building and trainibg using the current 

        Args:
            X_train (pd.DataFrame): The training data features
            y_train (pd.Series): The training data labels/target

        Returns:
            RegressorMixin: A trained scikit-learn model instance
        """
        logging.info("Building and training model using the scikit learn model")
        return self._strategy.build_and_train_model(X_train,y_train)

# Example usage
if __name__ =="__main__":
    # Example DataFrame (replace with actual data loading)
    # df = pd.read_csv("data/your_data.csv")
    # X_train = df.drop(columns = ['target_column'])
    # y_train = df['target_column']
    
    
    # Example usage of linear Regression strategy
    # model_builder = ModelBuilder(LinearRegressionStrategy)
    # trained model = model_builder.build_model(X_train,y_train)
    # print(trained_model.named_steps['model'].coef_)
    
    pass