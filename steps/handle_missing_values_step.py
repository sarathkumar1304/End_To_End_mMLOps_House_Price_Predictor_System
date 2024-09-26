import pandas as pd
from src.handle_missing_values import (
    DropMissingValuesStrategy,
    FillMissingValuesStrategy,
    MissingvalueHandler,
)
from zenml import step

@step
def handle_missing_values_step(df:pd.DataFrame,strategy:str = "mean")->pd.DataFrame:
    if strategy =="drop":
        handler = MissingvalueHandler(DropMissingValuesStrategy(axis= 0))
    elif strategy in ["mean","median","mode","constant"]:
        handler = MissingvalueHandler(FillMissingValuesStrategy(method = strategy))
    else:
        raise ValueError(f"Unsupported missing value handling strategy : {strategy}")
    cleaned_df = handler.handle_missing_values(df)
    return cleaned_df