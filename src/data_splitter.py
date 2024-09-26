from abc import ABC , abstractmethod
import pandas as pd  
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level = logging.INFO,format = "%(asctime)s- %(levelname)s - %(message)s")

# Abstract Base Class for Data splitting Strategy
# --------------------------------------------------
# This class defines a common interface for different data split
# Subclasses must iplement the split_data method
class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self,df:pd.DataFrame,target_column:str):
        """
        Abstarct method to split the data into training and testing 

        Args:
            df (pd.DataFrame): The input DataFrame to be split.  
            target_column (str): The name of the target column
            
        Returns:
        X_train,X_test,y_train,y_test: The training and test 
        """

# concrete Strategy for Simple train-Test Split
# --------------------------------------------------
# This strategy implements a simple train- test split
class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self,test_size = 0.2,random_state = 42):
        """
        Initializes the SimpleTrainTestsplitStrategy with specific features

        Args:
            test_size (float, optional): The proportion of the dataset to include . Defaults to 0.2.
            random_state (int, optional): The seed used by the random number. Defaults to 42.
            
        """
        self.test_size= test_size
        self.random_state = random_state
        
    def split_data(self,df:pd.DataFrame,target_column:str):
        """
        Splits the data into training and testing sets using a split

        Args:
            df (pd.DataFrame): The input DataFrame to be split
            target_column (str): The name of the target column.  
        Returns:
        X_train,X_test,y_train,y_test : The training and testing data
        """
        logging.info("Performing simple train test split")
        X = df.drop(columns = [target_column])
        y = df[target_column]
        
        X_train,X_test, y_train,y_test = train_test_split(X,y,test_size =self.test_size,random_state= self.random_state)
        
        logging.info("Train -test split completed")
        return X_train,X_test,y_train,y_test 
    
# Context Class for Data Splitting
# -----------------------------------
# The class uses a DataSplittingStrategy to split the data
class DataSplitter:
    def __init__(self,strategy:DataSplittingStrategy):
        """
        Intiailizes the DataSplitter with a specific data splitting
        Args:
            strategy (DataSplittingStrategy): The strategy to be use
        """
        self._strategy = strategy
        
    def set_strategy(self,strategy:DataSplittingStrategy):
        """
        sets a new startegy for the DataSplitter.  

        Args:
            strategy (DataSplittingStrategy): The new strategy to be set
        """
        logging.info("switching data splitting strategy")
        self._strategy = strategy
        
    def split(self,df:pd.DataFrame,target_column:str):
        """
        Executes the data splitting using the current strategy

        Args:
            df (pd.DataFrame): The input DataFrame to be split
            target_column (str): The name of the target column
        Returns :
        X_train,X_test,y_train,y_test : the training and testing part of the data
        """
        logging.info("Splitting data using the selected strategy")
        return self._strategy.split_data(df,target_column)
    
        
# Example usage
if __name__ == "__main__":
    # Example dataframe (replace with actual data loading)
    # df = pd.read_csv('your_data.csv')

    # Initialize data splitter with a specific strategy
    # data_splitter = DataSplitter(SimpleTrainTestSplitStrategy(test_size=0.2, random_state=42))
    # X_train, X_test, y_train, y_test = data_splitter.split(df, target_column='SalePrice')

    pass
