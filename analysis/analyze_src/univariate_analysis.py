from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns


# Abstarct base class for Univariate Analysis Strategy
# -----------------------------------------------------
# This class defines a common interface for univariate analysis strategies.
# Subclasses must implement the analyze method.
class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Perform univariate analysis on a specific features of the dataframe. 

        Parameters:
            df (pd.DataFrame): The dataframe containing the data.
            feature (str): the name of the feature/column to be analyzed.
        Returns:
        None: this method visualizes the distribution of the features.
        """
        pass


# Concrete Strategy for Numerical Features
# ------------------------------------------
# This strategy analyzes numerical features by plotting their distribution.
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a numerical features using histogram and KDE.
        
        Parameters:
            df (pd.DataFrame): The dataframe containing the data.
            feature (str): The name of the numerical features / columns to be analyzed.
        Returns:
        None: Displays a histogram with a KDE 
        """
        plt.figure(figsize=(10,6))
        sns.histplot(df[feature],kde = True,bins = 30)
        plt.title(f"distibution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("frequency")
        plt.show()


# Concrete Strategy for Categorical Features
# ------------------------------------------
# Strategy analyzes categorical features by ploting their frequency distribution.
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the frequency distribution of a categorical features.
        
        Parameters:
            df (pd.DataFrame): The dataframe containing the data.
            feature (str): The name of the categorical features / columns to be analyzed.
        Returns:
        None: Displays a bar chart with frequency distribution 
        """
        plt.figure(figsize=(10,6))
        sns.countplot(x = feature, data = df,palette = 'muted')
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation = 45)
        plt.show()

class UnivariateAnalyzer:
    def __init__(self, strategy:UnivariateAnalysisStrategy):
        """ 
        Initializes the DataInspector with a specified strategy.

        Args:
            strategy (UnivariateAnalysisStrategy): The strategy to be used for Univariate analysis.
        Return :
        None
        """
        self._strategy = strategy
    def set_strategy(self, strategy:UnivariateAnalysisStrategy):
        """
        Sets a new strategy for the Data inspection.

        Parameters:
        strategy (DataInspectionStrategy): The new strategy to be used for data inspection.

        Returns:
        None 
        """
        self._strategy = strategy

    def execute_analyze(self,df:pd.DataFrame,feature:str):
        """
        Executes the strategy's inspect method on the given DataFrame.

        Args:
        df (pd.DataFrame): The DataFrame to be inspected.

        Returns:
        None: The strategy's inspect method is called.
        """
        self._strategy.analyze(df,feature)



# Example usage
if __name__ == '__main__':
    # Example usage of the UnivariateAnalyzer with different strategies
    df = pd.read_csv("extracted_data/AmesHousing.csv")
    num_strategy = NumericalUnivariateAnalysis()
    cat_strategy = CategoricalUnivariateAnalysis()

    univariate_analyzer = UnivariateAnalyzer(num_strategy)
    univariate_analyzer.execute_analyze(df, 'SalePrice')

    
    