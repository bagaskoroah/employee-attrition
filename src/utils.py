import yaml
import pandas as pd
import joblib
from typing import Any

def load_config(config_path: str) -> dict:
    """
    Load the configuration file.

    Parameters:
    -----------
    config_path (str): The path to the configuration file.

    Returns:
    --------
    config (dict): The loaded configuration as a dictionary.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        if config is None:
            raise ValueError("Configuration file is empty.")
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    
    return config

def load_data(data_path: str) -> pd.DataFrame:
    """ 
    Load the raw data from the path specified in the config.

    Parameters:
    -----------
    data_path (str): The path to the raw data file.

    Returns:
    --------
    df (pd.DataFrame): The loaded raw data as a pandas DataFrame.
    """

    # make the validation layers 
    try:
        df = pd.read_csv(data_path)
        if df.empty:
            raise ValueError("The raw data file is empty. Please check the file content.")
        else:
            print(f"Raw data loaded successfully from: {data_path}")

    except Exception as e:
        raise ValueError(f"Error loading raw data: {e}")

    return df

def serialize_object(path: str, obj: Any) -> None:
    """ 
    Serialize the object into specified path.

    Parameters:
    -----------
    path (str): The path to save the serialized object.
    obj (Any): The object to be serialized.

    Returns:
    --------
    None
    """
    # save the object to the specified path
    joblib.dump(value=obj, filename=path)
    print(f'Saving object. . . .')
    print(f'Object serialized successfully to: {path}')

def deserialize_object(path: str) -> Any:
    """
    Deserialize the object into specified path.

    Parameters:
    -----------
    path (str) : The path to deserialize the object.

    Returns:
    --------
    object (Any) : Any object will be deserialized.
    """

    obj = joblib.load(path)
    return obj
