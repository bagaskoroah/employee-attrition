# Import the required libraries
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from catboost import CatBoostClassifier
from sklearn.metrics import recall_score, precision_score
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from utils import *
from path import *

# build preprocessing pipeline defined function

def build_pipeline() -> Tuple[Pipeline, Pipeline]:
    """
    Build numerical and categorical pipeline.

    The numerical pipeline applies feature scaling using StandardScaler,
    while the categorical pipeline applies One Hot Encoding for handling
    categorical variables.

    Returns
    -------
    Tuple[Pipeline, Pipeline]
        A tuple containing:
        - num_pipeline : sklearn.pipeline.Pipeline
            Pipeline for preprocessing numerical features.
        - cat_pipeline : sklearn.pipeline.Pipeline
            Pipeline for preprocessing categorical features.
    """

    # build numerical features pipeline
    num_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return num_pipeline, cat_pipeline

def build_preprocessing(config, num_pipe, cat_pipe):
    # combine numerical and categorical pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('cat', cat_pipe, config['CAT_COLS']),
        ('num', num_pipe, config['NUM_COLS'])
    ])

    """
    Combine numerical and categorical preprocessing pipelines
    into a single ColumnTransformer object.

    Parameters
    ----------
    config : Dict
        Configuration dictionary containing numerical and
        categorical column names.

    num_pipe : Pipeline
        Preprocessing pipeline for numerical features.

    cat_pipe : Pipeline
        Preprocessing pipeline for categorical features.

    Returns
    -------
    ColumnTransformer
        Combined preprocessing transformer for numerical
        and categorical features.
    """
    
    return preprocessor

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Any,
    config: Dict
) -> ImbPipeline:
    """
    Train the final machine learning model using the
    best configuration obtained from the experimentation phase.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature dataset.

    y_train : pd.Series
        Training target dataset.

    preprocessor : Any
        Preprocessor pipeline obtained from preprocessing numerical and categorical column.

    config : Dict
        Configuration dictionary containing model parameters.

    Returns
    -------
    model : ImbPipeline
        The chosen best model based on notebook exploration.
    """

    estimator = CatBoostClassifier(
        depth=config['BEST_DEPTH'],
        learning_rate=config['BEST_LEARNING_RATE'],
        iterations=config['BEST_ITERATIONS'],
        l2_leaf_reg=config['BEST_L2_LEAF_REG'],
        random_strength=config['BEST_RANDOM_STRENGTH'],
        verbose=False
    )

    model = ImbPipeline(steps=[
        ('preprocessing', preprocessor),
        ('smote', SMOTE(random_state=config['RANDOM_STATE'])),
        ('model', estimator)
    ])

    model.fit(X_train, y_train)

    return model

def build_test(
        estimator: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.5) -> Tuple[np.ndarray, float, float]:
    """
    Evaluate trained model on test data and log the recall score to MLflow.

    Parameters
    ----------
    estimator : Any
        Trained model or pipeline (already fitted) used for prediction.
    threshold : float
        Model threshold default point in 0.5.
    X_test : pd.DataFrame
        Test feature set.
    y_test : pd.Series
        Ground truth labels for test data.

    Returns
    -------
    y_pred: np.ndarray
        Series of label predicted data by estimator.
    recall_test: float
        Recall score computed on the test dataset.
    precision_test: float
        Precision score computed on the test dataset.
    """

    y_pred = (estimator.predict_proba(X_test)[:, 1] >= threshold).astype(int)

    # evaluate test score
    recall_test = recall_score(y_test, y_pred)
    precision_test = precision_score(y_test, y_pred)

    return y_pred, recall_test, precision_test

def main():
    # load config
    config = load_config(config_path=CONFIG_PATH)

    # build the preprocessor
    num_pipeline, cat_pipeline = build_pipeline()
    preprocessor = build_preprocessing(config=config, num_pipe=num_pipeline, cat_pipe=cat_pipeline)

    # deserialize all processed data
    X_train = deserialize_object(path=PROCESSED_DATA_DIR/config['PATH_TRAIN_DATA'][0])
    y_train = deserialize_object(path=PROCESSED_DATA_DIR/config['PATH_TRAIN_DATA'][1])
    X_test = deserialize_object(path=PROCESSED_DATA_DIR/config['PATH_TEST_DATA'][0])
    y_test = deserialize_object(path=PROCESSED_DATA_DIR/config['PATH_TEST_DATA'][1])
    
    best_model = train_model(
        X_train=X_train,
        y_train=y_train,
        config=config,
        preprocessor=preprocessor
    )

    # save best model to models path
    serialize_object(path=BEST_MODEL_DIR/'best_model.pkl', obj=best_model)

    # predict & evaluate on 0.35 threshold point
    y_pred_tuned, recall_test_tuned, precision_test_tuned = build_test(
        estimator=best_model,
        X_test=X_test,
        y_test=y_test,
        threshold=config['BEST_THRESHOLD_POINT']
    )

    print('Recall Score Using Test Data:', recall_test_tuned)
    print('Precision Score Using Test Data:', precision_test_tuned)

if __name__ == "__main__":
    main()