from abc import ABC , abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Abstract Base class for Multivariate Analysis
# -----------------------------------------------------
# This class defines a template for performing multivriate analysis
# subclasses can override specific steps like correlation heatmap 
class MultivariateAnalysisTemplate(ABC):
    def analyze(self, df:pd.DataFrame):
        """
        Performs a comprehensive multivaraite analysis by generating heat map
        
        Parameters:
        df (pd.DataFrame): The dataframe containing the data to perform analysis
        
        Returns:
        None: This Method orchestrates the multivariate analysis
 
        """
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)
        
    @abstractmethod
    def generate_correlation_heatmap(self, df:pd.DataFrame):
        """
        Generate and display a heatmap of the correlations between the fetaures

        Args:
            df (pd.DataFrame): The dataframe containing the data to perform analysis
        Returns:
        None: This method should generate and display a correlation between the features
        """
        pass
    
    @abstractmethod
    def generate_pairplot(self,df:pd.DataFrame):
        """
        Generate and display a pairplot of the selected features
        
        Args:
            df (pd.DataFrame): The Dataframe containing  the data to analyse
        
        Returns:
        None: This method should generate and displays pair plot of selected features
        """
        pass


# Concrete Class for Multivariate Analysis with Correlation Heatmap
# ------------------------------------------------------------------
# This class implements the methods to generate a correlation heatmap.
class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generates and displays a correlation heatmap for the numerical features
        

        Args:
            df (pd.DataFrame): The dataframe containing the data to analyse
            
        Returns:
        None: Displays a heatmap showing correlation between numerical features 
        """
        plt.figure(figsize= (12,10))
        sns.heatmap(df.corr(),annot = True,fmt=".2f",cmap='coolwarm',linewidths = 0.5)
        plt.title("Correlation Heatmap")
        plt.show()
        
    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generates and display a pairplot for the selected features in the dataframe
        

        Args:
            df (pd.DataFrame): The dataframe containing the data to be analyzed
        
        Returns:
        None: Displays a pair plot for the selected features. 
        """
        sns.pairplot(df)
        plt.suptitle("Pair Plot of Selected Features",y = 1.02)
        plt.show()
        
# Example usuage
if __name__ == "__main__":
     # Load the data
     # df = pd.read_csv("extracted_data/AmesHousing.csv")
     
    
    
    # Create an instance of SimpleMultivariateAnalysis
    # multivariate_analyzer = SimpleMultivariateAnalysis()
    
    # Perform multivariate analysis
    # multivariate_analyzer.analyze(df)
    pass