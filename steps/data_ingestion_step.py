import pandas as pd 
from src.ingest_data import DataIngestorFactory
from zenml import step

@step
def data_ingestion_step(file_path:str)->pd.DataFrame:
    """
    ingest data from a ZIP file using the appropriate DataIng
    
    """
    # Detemine the file extension
    file_extension = '.zip'  # since we're dealing with ZIP file.
    
    #Get the appropriate DataIngestor
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    
    # ingest the data and load it into a dataframe
    df = data_ingestor.ingest(file_path)
    return df