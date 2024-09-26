import logging
from typing import Annotated

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from zenml import ArtifactConfig, step  
from zenml.client import Client

# Get the active experiment tracker from zenml
experiment_tracker = Client().active_stack.experiment_tracker
from zenml import Model


model = Model(
    name= "price_predictor",
    version= None,
    license = "Apache 2.0",
    description = 'price prediction model for houses.',
)


@step(enable_cache = False,experiment_tracker=experiment_tracker.name,model=model)
def model_building_step(
    X_train:pd.DataFrame,y_train:pd.Series
)->Annotated[Pipeline,ArtifactConfig(name = "sklearn_pipeline",is_model_artifact = True)]:
    """
    Builds and trains a linear Regression model using scikit-learn
    
    Parameters :
    X_train (pd.DataFrame): The trainig data features.
    y_train (pd.Series): The training data labels/target.  
    Returns:
    Pipeline: The trained scikit-learn pipleine including preparation
    """
    # Ensure the inputs are of the correct type
    if not isinstance(X_train,pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame")
    if not isinstance(y_train,pd.Series):
        raise TypeError("y_train must be pandas Series")
    
    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include= ['object',"category"]).columns
    numerical_cols = X_train.select_dtypes(exclude= ['object','category']).columns
    
    logging.info(f"Categorical columns :{categorical_cols.tolist()}")
    logging.info(f"Numerical columns :{numerical_cols.tolist()}")
    
    # define preprocessing for categorical and numerical features
    numerical_transformer = SimpleImputer(strategy = 'mean')
    categorical_transformer = Pipeline(
        steps = [
            ("imputer",SimpleImputer(strategy = "most_frequent")),
            ("onehot",OneHotEncoder(handle_unknown = "ignore")),
        ]
    )

    # Bundle preprocessing for numerical and categorical column
    preprocessor = ColumnTransformer(
        transformers = [
            ("num",numerical_transformer,numerical_cols),
            ("cat",categorical_transformer,categorical_cols),
            ]
    ) 
    
    # Define the model training pipeline
    pipeline = Pipeline(steps = [("preprocessor",preprocessor),("model",LinearRegression())])
    
    # Start an Mlflow run to log model training process
    if not mlflow.active_run():
        mlflow.start_run() # starts a new Mlflow ru if there is no running mlflow
        
    try:
        # enable autologging for scikit -learn to automatically captures model metrics parameters and artifacts
        mlflow.sklearn.autolog()
        
        logging.info("Building and training the Linear Regression model")
        pipeline.fit(X_train,y_train)
        logging.info("Model training completed")
        
        #Log th coilumns that the models expects
        onehot_encoder = (
            pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot']
        )
        onehot_encoder.fit(X_train[categorical_cols])
        expected_columns = numerical_cols.tolist()+list(
            onehot_encoder.get_feature_names_out(categorical_cols)
        )
        logging.info(f"Model expects the following columns : {expected_columns}")
    except Exception as e:
        logging.error(f"An error occurred during model training: {str(e)}")
        # mlflow.end_run(status=mlflow.entities.RunStatus.FAILED)
        raise e
    finally:
        # end the mlflow run
        mlflow.end_run()
    return pipeline 
        
        
        # zenml model-deployer register mlflow_prices_new --flavor=mlflow
        # zenml stack register mlflow_stack_prices_new -a default -o default -d mlflow -e mlflow_tracker_prices_new --set