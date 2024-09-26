import logging
from abc import ABC, abstractmethod
import numpy as np 
import pandas as pd  
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,StandardScaler

# setup loggig configuration

logging.basicConfig(level = logging.INFO,format = "%(asctime)s- %(levelname)s - %(message)s")

# Abstract Base Class for feature engineering
# ----------------------------------------------
# This class defines a common interface for different feature engineering
# Subclasses must implement the apply_transformation method.
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self,df:pd.DataFrame)->pd.DataFrame:
        """
        Abstract method to apply feature engineering transformation

        Args:
            df (pd.DataFrame): The dataframe containing features to the transform

        Returns:
            pd.DataFrame: A dataframe with the applied transformation
        """
        pass
    
# Concrete Strategy for Log Transformation
# ------------------------------------------
# This strategy applies a logarithmic transformation to skwed feature
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self,features):
        """
        Initailizes the LogTransfomation with the specific feature

        Args:
            features (list): the list of features to apply the log transformation
        """
        self.features = features
        
    def apply_transformation(self,df:pd.DataFrame)->pd.DataFrame:
        """
        Applies a log transformation to the specified feature in the data

        Args:
            df (pd.DataFrame): The dataframe containing features to transform 

        Returns:
            pd.DataFrame: The dataframe with log-transformed features
        """
        logging.info(f"Applying log transformation to features :{self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(
                df[feature]
            ) # log1p handles log(0)by calculating log(1+x)
        logging.info("Log Transformation completed")
        return df_transformed
    
# Concrete Strategy for StandardScaling
# ----------------------------------------
# This strategy applies standrad scaling (z-score normalization)
class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self,features):
        """
        Initailizes the standard scaling with specific feature in the dataframe

        Args:
            features (list):The list of features to apply the standard scaling normaliation 
        """
        self.features = features
        self.scaler  = StandardScaler()
    
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies standard scaling to the specified features in the dataframe

        Args:
            df (pd.DataFrame): The dataframe containing features to transform
        Returns:
            pd.DataFrame: The dataframe with scaled features
        """
        logging.info(f"Applying standard scaling to features : {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Standard Scaling completed")
        return df_transformed
    
    
# conceret strategy fpor Min-Max Scaling
# ---------------------------------------
# This strategy applies Min-max scaling to features , scaling the  features 
class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self,features,feature_range= (0,1)):
        """
        Initializes the MinMaxScaling with the specific fetaures
        Args:
            features (_type_): The list of features to apply the Min-Max scaling
            feature_range (tuple, optional): The target range for scaling, Defaults to (0,1).
        """
        self.features = features
        self.scaler = MinMaxScaler(features_range = feature_range)
        
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Min-Max scaling to the specified features in the data

        Args:
            df (pd.DataFrame): The dataframe containing fetaures to the data

        Returns:
            pd.DataFrame: The dataframe with Min-Max scaled features.
        """
        logging.info(f"Applying Min-Max scaling to features :{self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df_transformed[self.features])
        logging.info("Min-Max scaling copleted")
        return df_transformed
    
# Concrete Strategy for One-Hot Encoding
# ----------------------------------------
# This Strategy applies one-hot encoding to categorical features
class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self,features):
        """
        Initaializes the OneHotEncoding with the specific features

        Args:
            features (list): The list of categorical features to apply transformation 
        """
        self.features = features
        self.encoder = OneHotEncoder(sparse= False,drop='first')
        
    def apply_transformation(self,df:pd.DataFrame)->pd.DataFrame:
        """
        Applies one-hot encoding to the specified categorical features in the dataframe

        Args:
            df (pd.DataFrame): The dataframe containing categorical features to transform

        Returns:
            pd.DataFrame: The dataframe with one-hot encoded features
        """
        logging.info(f"Applying one-hot encoding to features :{self.features}")
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df_transformed[self.features]),
            columns=self.encoder.get_feature_names_out(self.features)
        )
        df_transformed = df_transformed.drop(columns=self.features, axis=1)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)  # Reattach encoded features to the dataframe  # Reattach encoded features to the dataframe  # Reattach encoded features to the dataframe  # Reattach encoded features to the dataframe  # Reattach encoded features to the dataframe  # Reattach encoded features to the dataframe  # Reattach encoded features to the dataframe  # Reattach encoded features to the dataframe  # Reattach encoded features to the
        logging.info("One-Hot encoding completed")
        return df_transformed
    
# Context Class for Feature Engineering
# -------------------------------------
# This class uses a FeatureEngineeringStrategy to apply transfomation
class FeatureEngineer:
    def __init__(self,strategy:FeatureEngineeringStrategy):
        """ 
        Initializes the FeatureEngineer with a specified strategy.

        Args:
            strategy (FeatureEngineeringStrategy): The strategy to be used for feature engineering.
        """
        self._strategy = strategy
        
    def set_strategy(self, strategy:FeatureEngineeringStrategy):
        """
        Sets a new strategy for the FeatureEngineer.

        Args:
            strategy (FeatureEngineeringStrategy): The new strategy to be used for feature engineering.
        """
        logging.info("Switching feature engineering strategy.")
        self._strategy = strategy
    def apply_feature_engeering(self,df:pd.DataFrame)->pd.DataFrame:
        """
        Executes the feature engineering transformation using the strategy

        Args:
            df (pd.DataFrame): The dataframe containing features to analyze

        Returns:
            pd.DataFrame: The dataframe with applied feature engineering 
        """
        logging.info("Applying feature engineering strategy")
        return self._strategy.apply_transformation(df)
    
    
    
# Example usage
if __name__ == "__main__":
    # Example dataframe
    # df = pd.read_csv("extracted_data/AmesHousing.csv")
    
    # Log Transformation Example
    # log_transformer = FeatureEngineer(LogTransformation(features= ['SalePrice','Gr liv Area']))
    # df_log_transformed = log_transformer.apply_feature_engineering(df)
    
    # Standard Scaling Example
    # standard_scaler = FeatureEngineer(StandardScaler(features)= ['SalePrice','Gr liv Area'])
    # df_standard_scaled = standard_scaler.apply_feature_engineering(df)
    
    # Min Max scaling
    # minmax_scaler = FeatureEngineer(MinMaxScaling(features = ['SalePrice','Gr liv Area']))
    # df_minmax_scale = minmax_scaler.apply_feature_engineering(df) 
    
    # One-hot Encoding Example
    # onehot_encoder = FeaturesEngineer(OneHotEncoding(features = ['SalePrice','Gr liv Area']))
    # df_onehot_encoded = onehot_encoder.apply_feature_engineer(df)
    pass
    
    
    
    
    
    
    
    
    
    
