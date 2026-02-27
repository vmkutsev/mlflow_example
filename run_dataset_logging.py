import mlflow
from constants import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, ALL_FEATURES
from scripts.process_data import process_data
from scripts.train import train
from scripts.evaluate import evaluate

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']

GB_PARAMS = {'model_type': 'GradientBoostingClassifier', 'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}

experiments = [
    {
        'name': 'dataset-logging-gb-train_size-1000',
        'process': {'features': ALL_FEATURES, 'train_size': 1000},
        'train': GB_PARAMS,
        'evaluate': {'metrics': METRICS},
    },
    {
        'name': 'dataset-logging-gb-train_size-5000',
        'process': {'features': ALL_FEATURES, 'train_size': 5000},
        'train': GB_PARAMS,
        'evaluate': {'metrics': METRICS},
    },
    {
        'name': 'dataset-logging-gb-train_size-15000',
        'process': {'features': ALL_FEATURES, 'train_size': 15000},
        'train': GB_PARAMS,
        'evaluate': {'metrics': METRICS},
    },
]

print(f'Запускаем {len(experiments)} экспериментов | experiment: {EXPERIMENT_NAME}')
for i, exp in enumerate(experiments, 1):
    run_name = exp['name']
    print(f'[{i}/{len(experiments)}] {run_name}', flush=True)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag('series', 'dataset_logging')
        process_data(exp['process'])
        train(exp['train'])
        evaluate(exp['evaluate'])
    print(f'    Готово', flush=True)

print('Все эксперименты завершены!')
