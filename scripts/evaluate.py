import json
import os
import tempfile

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from constants import DATASET_PATH_PATTERN, FEATURE_NAMES_PATH, MODEL_FILEPATH
from utils import get_logger, load_params

STAGE_NAME = 'evaluate'


def evaluate(params_override: dict | None = None):
    logger = get_logger(logger_name=STAGE_NAME)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    y_test_arr = y_test.values.ravel()
    logger.info('Успешно считали датасеты!')

    logger.info('Загружаем обученную модель')
    if not os.path.exists(MODEL_FILEPATH):
        raise FileNotFoundError(
            'Не нашли файл с моделью. Убедитесь, что был запущен шаг с обучением'
        )
    model = load(MODEL_FILEPATH)

    logger.info('Скорим модель на тесте')
    y_proba = model.predict_proba(X_test.values)[:, 1]
    y_pred = np.where(y_proba >= 0.5, 1, 0)

    logger.info('Считаем метрики на тесте')
    metrics = {
        'accuracy': accuracy_score(y_test_arr, y_pred),
        'precision': precision_score(y_test_arr, y_pred, zero_division=0),
        'recall': recall_score(y_test_arr, y_pred, zero_division=0),
        'f1': f1_score(y_test_arr, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test_arr, y_proba),
        'pr_auc': average_precision_score(y_test_arr, y_proba),
    }
    logger.info(f'Значения метрик - {metrics}')
    mlflow.log_metrics(metrics)

    with tempfile.TemporaryDirectory() as tmp_dir:
        _log_classification_report(y_test_arr, y_pred, tmp_dir, logger)
        _log_confusion_matrix(y_test_arr, y_pred, tmp_dir, logger)
        _log_pr_curve(y_test_arr, y_proba, tmp_dir, logger)
        _log_feature_importances(model, X_test, tmp_dir, logger)
        _log_error_csv(X_test, y_test_arr, y_pred, y_proba, tmp_dir, logger)


def _log_classification_report(y_true, y_pred, tmp_dir, logger):
    report = classification_report(y_true, y_pred, target_names=['<=50K', '>50K'])
    logger.info(f'Classification report:\n{report}')
    report_path = os.path.join(tmp_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    mlflow.log_artifact(report_path, artifact_path='artifacts')


def _log_confusion_matrix(y_true, y_pred, tmp_dir, logger):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['<=50K', '>50K'])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    cm_path = os.path.join(tmp_dir, 'confusion_matrix.png')
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(cm_path, artifact_path='artifacts')


def _log_pr_curve(y_true, y_proba, tmp_dir, logger):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title('Precision-Recall Curve')
    plt.tight_layout()
    pr_path = os.path.join(tmp_dir, 'pr_curve.png')
    fig.savefig(pr_path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(pr_path, artifact_path='artifacts')


def _log_feature_importances(model, X_test, tmp_dir, logger):
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])

    if importances is None:
        return

    feature_names = list(range(X_test.shape[1]))
    if os.path.exists(FEATURE_NAMES_PATH):
        with open(FEATURE_NAMES_PATH) as f:
            feature_names = json.load(f)

    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=False)

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
    ax.barh(fi_df['feature'], fi_df['importance'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importances')
    plt.tight_layout()
    fi_path = os.path.join(tmp_dir, 'feature_importances.png')
    fig.savefig(fi_path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(fi_path, artifact_path='artifacts')


def _log_error_csv(X_test, y_true, y_pred, y_proba, tmp_dir, logger):
    errors_mask = y_true != y_pred
    errors_df = pd.DataFrame(X_test.values, columns=range(X_test.shape[1]))
    errors_df['y_true'] = y_true
    errors_df['y_pred'] = y_pred
    errors_df['y_proba'] = y_proba
    errors_df = errors_df[errors_mask]
    errors_path = os.path.join(tmp_dir, 'errors.csv')
    errors_df.to_csv(errors_path, index=False)
    mlflow.log_artifact(errors_path, artifact_path='artifacts')
    logger.info(f'    Количество ошибок: {len(errors_df)}')


if __name__ == '__main__':
    from constants import EXPERIMENT_NAME, MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        evaluate()
