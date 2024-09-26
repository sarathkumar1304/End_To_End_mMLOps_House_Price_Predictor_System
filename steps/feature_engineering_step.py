from src.feature_engineering import OneHotEncoding,StandardScaling,MinMaxScaling,FeatureEngineer,LogTransformation
from zenml import step
import pandas as pd 

@step
def feature_engineering_step(
    df:pd.DataFrame,strategy:str='log',features:list=None
)->pd.DataFrame:
    """
    Performs feature engineering using FeatureEngineer and select
    """
    # Ensure features is a list , evn if not provided
    if features is None:
        features = [] # or raise an error if features are required
        
    if strategy =='log':
        engineer = FeatureEngineer(LogTransformation(features))
    elif strategy == 'standard_scaling':
        engineer = FeatureEngineer(StandardScaling(features))
    elif strategy == "minmax_scaling":
        engineer = FeatureEngineer(MinMaxScaling(features))
    elif strategy == "onehot_encoding":
        engineer = FeatureEngineer(OneHotEncoding(features))
    else:
        raise ValueError(f"Unsupported feature engineering strategy :{strategy}")
    transformed_df = engineer.apply_feature_engeering(df)  
    return transformed_df 