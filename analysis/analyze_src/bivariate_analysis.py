from abc import ABC , abstractmethod


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Abstract Base class for Bivariate analysis Strategy 
# -----------------------------------------------------
# This class defines a common interface for bivariate analysis strategies
# Subclasses must implemnt the analyze method
class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Perform bivariate analysis on two features of the dataframe. 

        Parameters:
            df (pd.DataFrame): The dataframe containing the data.
            x_feature (str): The name of the first feature.
            y_feature (str): The name of the second feature.
        Returns:
        None: This method visualizes the relationship between the features.
        """
        pass

# Concrete Strategy for Numerical vs Numerical Analysis
# -------------------------------------------------------
# This strategy analyzes the relationship between two numerical features

class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between two numerical features 

        Args:
            df (pd.DataFrame): The dataframe containing the data
            feature1 (str): The name of the first numerical feature
            feature2 (str): The name of the second numerical feature
        Returns:
            None: Displays a scatter plot showing relationship between them.
        """
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=feature1, y=feature2,data = df)
        plt.title(f" {feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()


# Concrete Strategy for Categorical vs Numerical Analysis
# ------------------------------------------------------
# This strategy analyzes the relationship between a categorical and numerical data
class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self,df:pd.DataFrame, feature1:str,feature2:str):
        """
        Plots the relationship between a categorical and numerical data 

        Args:
            df (pd.DataFrame): The dataframe containing the data
            feature1 (str): The name of the categorical feature
            feature2 (str): The name of the numerical feature
        Returns:
            None: Displays a boxplot showing relationship between them.
        """
        plt.figure(figsize=(10,6))
        sns.boxplot(x=feature1, y=feature2, data = df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation = 45)
        plt.show()

# Context class that uses a BivariateAnalysisStrategy
# ----------------------------------------------------
# This class allows you to switch between different bivariate analysis
class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        """ 
        Initializes the BivariateAnalyzer with a specified strategy.

        Args:
            strategy (BivariateAnalysisStrategy): The strategy to be used for bivariate analysis.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        """
        Sets a new strategy for the BivariateAnalyzer.

        Args:
            strategy (BivariateAnalysisStrategy): The new strategy to be used for bivariate analysis.

        Returns:
        None 
        """
        self._strategy = strategy

    def execute_analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Executes the strategy's analyze method on the given DataFrame.

        Args:
            df (pd.DataFrame): The dataframe containing the data.
            feature1 (str): The name of the first feature.
            feature2 (str): The name of the second feature.
            """
        self._strategy.analyze(df,feature1, feature2)
        

# Example
if __name__ == '__main__':
    # Create a Numerical vs Numerical Analysis strategy
    df = pd.read_csv('extracted_data/AmesHousing.csv')
    numerical_strategy = NumericalVsNumericalAnalysis()

    # Create a BivariateAnalyzer with the Numerical vs Numerical Analysis strategy
    analyzer = BivariateAnalyzer(numerical_strategy)

    # Execute the analysis
    analyzer.execute_analyze(df, 'Gr Liv Area', 'SalePrice')