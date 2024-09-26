import logging
from abc import ABC ,abstractmethod

import pandas as pd


# Setup logging configuration

logging.basicConfig(level = logging.INFO,format = "%(asctime)s- %(levelname)s - %(message)s")

#Abstarct Base Class for Missing value Handling Strategy
class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self,df:pd.DataFrame)->pd.DataFrame:
        """
        Abstarct method to handle missing values in the dataframe
        
        parameters:
        df (pd.DataFrame): The input Dataframe containing missing values
        
        Returns:
        pd.DataFrame: The Dataframe with missing values handled. 
        """
        pass
    
    
#Concerete Strategy for Dropping missing Values
class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self,axis =0, thresh= None):
        """
        Initializes the DropMissingValuesStrategy with specific axis
        
        Parameters:
        axis (int): 0 to drop rows with missing values, 1 to drop column
        thres (int): The threshold for non-NA values. Rows/columns 
        """
        self.axis = axis
        self.thresh = thresh
        
    def handle(self,df:pd.DataFrame)->pd.DataFrame:
        """
        Drops rows/columns with missing values based on the specified axis and threshold
        
        Parameters:
        df (pd.DataFrame): The input Dataframe containing missing values
        
        Returns:
        pd.DataFrame: The Dataframe with missing values dropped. 
        """
        logging.info(f"Dropping {self.axis} with missing values, threshold: {self.thresh}")
        df_cleaned = df.dropna(axis = self.axis, thresh =self.thresh )
        logging.info("Missing values dropped")
        return df_cleaned
    
# Concrete Strategy for Filling Missing values
class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self,method = 'mean',fill_values = None):
        """
        initailizes the FillmissingValuesStrategy with specific method
        

        Args:
            method (str, optional): The method to fill missing values(mean,median,mode) Defaults to 'mean'.
            fill_values (any, optional): The constant value to fill missing values. Defaults to None.
        """
        self.method = method
        self.fill_values = fill_values
        
    def handle(self,df:pd.DataFrame)->pd.DataFrame:
        """
        Fills missing values using the specigied method or constant values

        Args:
            df (pd.DataFrame): The input Dataframe containibg missing values 

        Returns:
            pd.DataFrame: The Dataframe with missing values filled.  
        """
        logging.info(f"Filling missing values using method :{self.method}")
        df_cleaned = df.copy()
        if self.method =="mean":
            numeric_columns = df_cleaned.select_dtypes(include= "number").columns
            df_cleaned[numeric_columns]  = df[numeric_columns].fillna(
                df[numeric_columns].mean())
        elif self.method == "median":
            numeric_columns = df_cleaned.select_dtypes(include= "number").columns
            df_cleaned[numeric_columns]  = df[numeric_columns].fillna(
                df[numeric_columns].median())
        elif self.method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(df[column].mode().iloc[0],inplace=True)
        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)
        else:
            logging.warning(f"Unknown method '{self.method}'.No missing values handled.")
        logging.info("Missing values filled.")
        return df_cleaned
    
# Context Class for Handling missing Values
class MissingvalueHandler:
    def __init__(self,strategy:MissingValueHandlingStrategy):
        """
        Intializes the MissingHandler with a specific missing values

        Args:
            strategy (MissingValueHandlingStrategy): The strategy to fill missing values. 
        """
        self._strategy = strategy
    def set_strategy(self,strategy:MissingValueHandlingStrategy):
        """
        Sets a new strategy for the MissingValueHandler.  

        Args:
            strategy (MissingValueHandlingStrategy): the new strategy
        """
        logging.info("Switching missing value handling strategy")
        self._strategy = strategy
        
    def handle_missing_values(self,df:pd.DataFrame)->pd.DataFrame:
        """
        Executes the missing values handling using the current stragey
        
        Args:
            df (pd.DataFrame): The input DataFrame containing missing values

        Returns:
            pd.DataFrame: The DataFrame with missing values handled.
        """
        logging.info("Executes missing value handling strategy")
        return self._strategy.handle(df) 
    
    
    
# Example usage
if __name__ == "__main__":
    # read csv
    # df = pd.read_csv('extracted_data/AmesHousing.csv')
    
    # Initialize missing value handler with a specific strategy
    # missing_value_handler = MissingValueHandler(DropMissingValuesStrategy)
    # df_cleaned = missing_value_handler.handle_missing_values(df)
    
    # switch to filling missing values with mean
    # missing_value_handler.set_strategy(FillMissingValuesStrategy(method = "mean"))
    # df_filled = missing_value_handler.handle_missing_values(df)
    pass