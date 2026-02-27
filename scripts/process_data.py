import json
import os

import mlflow
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from constants import (
    DATASET_NAME, DATASET_PATH_PATTERN, FEATURE_NAMES_PATH,
    RANDOM_STATE, TEST_SIZE,
)
from utils import get_logger, load_params

STAGE_NAME = 'process_data'


def process_data(params_override: dict | None = None):
    logger = get_logger(logger_name=STAGE_NAME)
    params = params_override if params_override is not None else load_params(stage_name=STAGE_NAME)

    logger.info('Начали скачивать данные')
    dataset = load_dataset(DATASET_NAME)
    logger.info('Успешно скачали данные!')

    logger.info('Делаем предобработку данных')
    df = dataset['train'].to_pandas()
    columns = params['features']
    target_column = 'income'
    X, y = df[columns], df[target_column]
    logger.info(f'    Используемые фичи: {columns}')

    all_cat_features = [
        'workclass', 'education', 'marital.status', 'occupation', 'relationship',
        'race', 'sex', 'native.country',
    ]
    cat_features = list(set(columns) & set(all_cat_features))
    num_features = list(set(columns) - set(all_cat_features))

    preprocessor = OrdinalEncoder()
    X_transformed = np.hstack([X[num_features], preprocessor.fit_transform(X[cat_features])])
    feature_names = num_features + cat_features
    y_transformed: pd.Series = (y == '>50K').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y_transformed, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    train_size = params.get('train_size')
    if train_size is not None:
        train_size = int(train_size)
        X_train = X_train[:train_size]
        y_train = y_train.iloc[:train_size]

    logger.info(f'    Размер тренировочного датасета: {len(y_train)}')
    logger.info(f'    Размер тестового датасета: {len(y_test)}')

    logger.info('Начали сохранять датасеты')
    os.makedirs(os.path.dirname(DATASET_PATH_PATTERN.format(split_name='X_train')), exist_ok=True)
    for split, split_name in zip(
        (X_train, X_test, y_train, y_test),
        ('X_train', 'X_test', 'y_train', 'y_test'),
    ):
        pd.DataFrame(split).to_csv(
            DATASET_PATH_PATTERN.format(split_name=split_name), index=False
        )

    with open(FEATURE_NAMES_PATH, 'w') as f:
        json.dump(feature_names, f)
    logger.info('Успешно сохранили датасеты!')

    mlflow.log_params({
        'features': columns,
        'num_features': len(columns),
        'train_size': len(y_train),
        'test_size': len(y_test),
        'train_size_param': train_size if train_size is not None else 'full',
    })

    logger.info('Логируем датасеты в MLflow')
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['target'] = y_train.values
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['target'] = y_test.values

    train_dataset = mlflow.data.from_pandas(
        train_df,
        source=DATASET_NAME,
        name='train',
    )
    test_dataset = mlflow.data.from_pandas(
        test_df,
        source=DATASET_NAME,
        name='test',
    )
    mlflow.log_input(train_dataset, context='training')
    mlflow.log_input(test_dataset, context='evaluation')
    logger.info('Датасеты залогированы!')


if __name__ == '__main__':
    from constants import EXPERIMENT_NAME, MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        process_data()
