import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, RANDOM_STATE
from utils import get_logger, load_params

STAGE_NAME = 'train'

MODEL_MAP = {
    'LogisticRegression': LogisticRegression,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'GradientBoostingClassifier': GradientBoostingClassifier,
    'XGBClassifier': XGBClassifier,
}


def train(params_override: dict | None = None):
    logger = get_logger(logger_name=STAGE_NAME)
    params = params_override if params_override is not None else load_params(stage_name=STAGE_NAME)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    logger.info('Успешно считали датасеты!')

    model_type = params.get('model_type', 'LogisticRegression')
    if model_type not in MODEL_MAP:
        raise ValueError(f'Неизвестный тип модели: {model_type}. Доступные: {list(MODEL_MAP.keys())}')

    model_params = {k: v for k, v in params.items() if k != 'model_type'}
    model_params['random_state'] = RANDOM_STATE

    logger.info(f'Создаём модель: {model_type}')
    logger.info(f'    Параметры модели: {model_params}')

    model_class = MODEL_MAP[model_type]
    model = model_class(**model_params)

    logger.info('Обучаем модель')
    model.fit(X_train, y_train.values.ravel())

    logger.info('Сохраняем модель')
    dump(model, MODEL_FILEPATH)
    logger.info('Успешно!')

    mlflow.log_param('model_type', model_type)
    mlflow.log_params({f'model_{k}': v for k, v in model_params.items()})

    if model_type == 'XGBClassifier':
        mlflow.xgboost.log_model(model, 'model')
    else:
        mlflow.sklearn.log_model(model, 'model')


if __name__ == '__main__':
    from constants import EXPERIMENT_NAME, MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        train()
