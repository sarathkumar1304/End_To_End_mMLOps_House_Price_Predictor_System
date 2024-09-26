from abc import ABC , abstractmethod

import pandas as pd

# Abstract Base Class for Data Inspection Strategies
# -------------------------------------------------
# This class defines a common interface for data inspection strategy
# Subclasses must implemnt the inspect method
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self,df:pd.DataFrame):
        """
        perfrom a specifictype of data inspection

        Parameters:
        df (pd.DataFrame) : the dataframe on which the inspection is to be performed

        Returns:
        None: This method prints the inspection result directly
        """
        pass

# Concrete Strategy for Data types inspection
# --------------------------------------------
# This Strategy inspects the data types and of each column  counts  non null values
class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self,df:pd.DataFrame):
        """
        Inspects and prints the data types and non null counts

        Args:
            df (pd.DataFrame): The DataFrame to be inspected.
        Returns :
        None: prints the data types and non null counts to the 

        """
        print("\nData Types and Non-null Counts :")
        print(df.info())

# Concrete Strategy for Summary Statistics Inspection 
# --------------------------------------------------
# This Strategy provides summary statistics for both numerical and categorical
class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self,df:pd.DataFrame):
        """
        Prints summary statistics for numerical and categorical 


        Args:
            df (pd.DataFrame): The Dataframe to be inspected.

        Returns:
        None: prints the summary statistics to the console.
        """

        print("\nSummary Statistics (Numerical Features)")
        print(df.describe)
        print("\nSummary Statistics (Categorical Features) :")
        print(df.describe(include= ["O"]))



# Context class that uses a Datainspection Strategy
# -------------------------------------------------
# This class aloows you to switch between different Data inspection
class DataInspector:
    def __init__(self, strategy:DataInspectionStrategy):
        """ 
        Initializes the DataInspector with a specified strategy.

        Args:
            strategy (DataInspectionStrategy): The strategy to be used for data inspection.
        Return :
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy:DataInspectionStrategy):
        """
        Sets a new strategy for the Data inspection.

        Parameters:
        strategy (DataInspectionStrategy): The new strategy to be used for data inspection.

        Returns:
        None 
        """
        self._strategy = strategy

    def execute_inspection(self,df:pd.DataFrame):
        """
        Executes the strategy's inspect method on the given DataFrame.

        Args:
        df (pd.DataFrame): The DataFrame to be inspected.

        Returns:
        None: The strategy's inspect method is called.
        """
        self._strategy.inspect(df)

# Example Usage

if __name__ == '__main__':
    # Example usage of the DataInspector with different strategies
    # Load the data
    df = pd.read_csv("extracted_data/AmesHousing.csv")

    # Initialize the data inspector with a specific strategy
    inspector = DataInspector(DataTypesInspectionStrategy())
    inspector.execute_inspection(df)
    
    # Change Stragety to get summary statistics and execute
    inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    inspector.execute_inspection(df)
    # pass