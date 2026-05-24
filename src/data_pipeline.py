import pandas as pd
from sklearn.model_selection import train_test_split
from utils import *
from path import *
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

def data_validation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Do data validation for removing bad data.

    Parameters:
    ----------
    data : pd.DataFrame
        The loaded raw dataset.

    Returns:
    -------
    validated_data : pd.DataFrame
        The validated data.
    """
    # copy dataframe so keep remain immutable
    data = data.copy()

    # encode attrition to biner numbers
    data['Attrition'] = data['Attrition'].replace({
    'No': 0,
    'Yes': 1
    })

    # change education column to categorical
    data['Education'] = data['Education'].astype('object')
    data['Education'] = data['Education'].replace({
        1: 'Below College',
        2: 'College',
        3: 'Bachelor',
        4: 'Master',
        5: 'Doctor'
    })

    # drop irrelevant columns
    data = data.drop(columns=['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'])

    return data

def data_defense(data: pd.DataFrame, config: Dict) -> None:
    """
    Do data defense for checking the data types and range of data.

    Parameters:
    ----------
    data : pd.DataFrame
        The data to be checked.

    config : dict
        Loaded configuration parameters.

    Returns:
    -------
    None, it's a void function.
    """
    if data.empty:
        raise ValueError("Dataframe is empty")
    
    expected_cols = set(config['ALL_COLS'])
    actual_cols = set(data.columns)

    missing_cols = expected_cols - actual_cols
    extra_cols = actual_cols - expected_cols

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    if extra_cols:
        raise ValueError(f"Unexpected cols: {extra_cols}")
    
    # missing values
    if data.isna().sum().sum() > 0:
        raise ValueError('Missing value detected.')
    
    # duplicate rows
    if data.duplicated().any():
        raise ValueError("Duplicate rows detected.")
    
    # check numeric ranges
    for col in config['NUM_COLS']:
        min_val, max_val = config[f'RANGE_{col.upper()}']
        invalid_val_mask = ~data[col].between(min_val, max_val)

        if invalid_val_mask.any():
            bad_values = data.loc[invalid_val_mask, col].head()

            raise ValueError(
                f"{col} contains invalid values:\n{bad_values}"
            )

def split_data(data: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    """

    Split the data into training and testing sets.

    Parameters:
    -----------
    data (pd.DataFrame): The data to be split.
    config (dict): The configuration file.

    Returns:
    --------
    X_train (pd.DataFrame): The training input features.
    X_test (pd.DataFrame): The testing input features.
    y_train (pd.Series): The training target variable.
    y_test (pd.Series): The testing target variable.

    """

    # split input and output
    X = data.drop(columns=[config['TARGET_COL']]).copy()
    y = data[config['TARGET_COL']].copy()

    # split train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['TEST_SIZE'], random_state=config['RANDOM_STATE'], stratify=y
    )

    return X_train, X_test, y_train, y_test

# main function
def main():
    # load config
    config = load_config(config_path=CONFIG_PATH)

    # load the raw dataset
    PATH_DATA_RAW = RAW_DATA_DIR/config['PATH_RAW_DATA']
    raw_dataset = load_data(data_path=PATH_DATA_RAW)

    # data validation
    validated_data = data_validation(raw_dataset)

    # data defense
    data_defense(data=validated_data, config=config)

    # split data
    X_train, X_test, y_train, y_test = split_data(data=validated_data, config=config)

    serialize_object(PROCESSED_DATA_DIR/config['PATH_TRAIN_DATA'][0], X_train)
    serialize_object(PROCESSED_DATA_DIR/config['PATH_TEST_DATA'][0], X_test)
    serialize_object(PROCESSED_DATA_DIR/config['PATH_TRAIN_DATA'][1], y_train)
    serialize_object(PROCESSED_DATA_DIR/config['PATH_TEST_DATA'][1], y_test)

if __name__ == '__main__':
    main()

