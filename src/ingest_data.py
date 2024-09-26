import os 
import zipfile
from abc import ABC, abstractmethod

import pandas as pd

# Define a abstarct class for Data Ingestor
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self,file_path:str)->pd.DataFrame:
        """Abstarct method to ingest data from a given file. """
        pass

# Implement a concrete class for ZIP ingestion
class ZipDataIngestor(DataIngestor):
    def ingest(self,file_path:str)->pd.DataFrame:
        """Extracts a .zip file and return the content as as pandas DataFrame. """
        # Ensure the file is a .zip
        if not file_path.endswith(".zip"):
            raise ValueError("Input file must be a .zip file.")
        
        # Extract the zip file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall("extracted_data")
        
        #find the extracted CSV file (assuming there is one csv file inside the zip)
        extracted_files = os.listdir("extracted_data")
        csv_files = [f for f in extracted_files if f.endswith(".csv")]
        

        if len(csv_files) == 0:
            raise FileNotFoundError("No CVS file found in the extracted data.")
        if len(csv_files)>1:
            raise ValueError("Multiple CSV files found. please specify which one to use")
        
        # Read the CVS into a DataFrame 
        csv_file_path = os.path.join("extracted_data",csv_files[0])
        df = pd.read_csv(csv_file_path)

        return df
    

class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension:str)->DataIngestor:
        """Factory method to return the appropriate DataIngestor based on the file extension. """
        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension : {file_extension}")

# Example usage
if __name__ == "__main__":
    # # Specify the file path
    # file_path = "data/archive.zip"
    
    
    # # Determine the file extension
    # file_extension = os.path.splitext(file_path)[1]
    # # Get the appropraite data ingestor
    # data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    
    # # Ingest the data and load it into a DataFrame
    # df = data_ingestor.ingest(file_path)
    
    # print(df.head())
    pass