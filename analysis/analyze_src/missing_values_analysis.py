import pandas as pd
import seaborn as sns

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


# Abstarct Base Class for Misiing Values Analysis
# -----------------------------------------------
# This class defines a template for missing values analysis
# Subclasses must implement the methods to identify and visualize
class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """ 
        Performs a complete missing values analysis by identifying the data

        Parameters :
        df (pd.DataFrame) : The dataframe to be analyzed.

        Returns :
        None: This method performs the analysis and visualizes  
        """
        self.identify_missing_values(df)
        self.visualize_missing_values(df)


        @abstractmethod
        def identify_missing_values(self, df:pd.DataFrame):
            """
            Prints the count of missing values for each column in the data 

            Parameters :
            df (pd.DataFrame): The dataframe to be analyzed.

            Returns :
            None : Prints the missing values count to the console.
            """

            pass
        @abstractmethod
        def visualize_missing_values(self, df:pd.DataFrame):
            """
            Creates a heatmap to visualize the missing values in the data

            Parameters:
                df (pd.DataFrame): The dataframe to be visualized.

            Returns:
            None: displaying a heatmap of missing values.
            """

            pass


class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df:pd.DataFrame):
            """
            Prints the count of missing values for each column in the data 

            Parameters :
            df (pd.DataFrame): The dataframe to be analyzed.

            Returns :
            None : Prints the missing values count to the console.
            """

            print("\nMisiing Values Count by column :")
            missing_values = df.isnull().sum()
            print(missing_values[missing_values >0])

    def visualize_missing_values(self,df:pd.DataFrame):
        """
            Creates a heatmap to visualize the missing values in the data

            Parameters:
                df (pd.DataFrame): The dataframe to be visualized.

            Returns:
            None: displaying a heatmap of missing values.
        """

        print("\nVisualizing Missing values...")
        plt.figure(figsize=(12,8))
        sns.heatmap(df.isnull(),cbar = False,cmap = 'viridis')
        plt.title("Missing values Heatmap")
        plt.show()


# Example Usage 

if __name__ == '__main__':
    # # Example usage of the SimpleMissingValuesAnalysis class.
    # # Load the data 
    # df = pd.read_csv("extracted_data/AmesHousing.csv")
    
    # # Perform Misiing Values Analysis
    # missing_values_analyzer = SimpleMissingValuesAnalysis()
    # missing_values_analyzer.analyze(df)
    pass  