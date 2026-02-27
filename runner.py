import mlflow

from constants import ALL_FEATURES, EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from scripts import evaluate, process_data, train

METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']

# fmt: off
EXPERIMENTS = [
    # ── Series 1: Vary model type ─────────────────────────────────────────────
    # Fixed: all features, train_size=10000
    {
        'name': 'model_type-LogisticRegression',
        'process': {'features': ALL_FEATURES, 'train_size': 10000},
        'train':   {'model_type': 'LogisticRegression', 'penalty': 'l2', 'C': 1.0,
                    'solver': 'lbfgs', 'max_iter': 1000},
        'evaluate': {'metrics': METRICS},
    },
    {
        'name': 'model_type-DecisionTree',
        'process': {'features': ALL_FEATURES, 'train_size': 10000},
        'train':   {'model_type': 'DecisionTreeClassifier', 'max_depth': 10},
        'evaluate': {'metrics': METRICS},
    },
    {
        'name': 'model_type-RandomForest',
        'process': {'features': ALL_FEATURES, 'train_size': 10000},
        'train':   {'model_type': 'RandomForestClassifier', 'n_estimators': 100},
        'evaluate': {'metrics': METRICS},
    },
    {
        'name': 'model_type-GradientBoosting',
        'process': {'features': ALL_FEATURES, 'train_size': 10000},
        'train':   {'model_type': 'GradientBoostingClassifier',
                    'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
        'evaluate': {'metrics': METRICS},
    },

    # ── Series 2: Vary train_size ─────────────────────────────────────────────
    # Fixed: RandomForest, n_estimators=100, all features
    {
        'name': 'train_size-500',
        'process': {'features': ALL_FEATURES, 'train_size': 500},
        'train':   {'model_type': 'RandomForestClassifier', 'n_estimators': 100},
        'evaluate': {'metrics': METRICS},
    },
    {
        'name': 'train_size-2000',
        'process': {'features': ALL_FEATURES, 'train_size': 2000},
        'train':   {'model_type': 'RandomForestClassifier', 'n_estimators': 100},
        'evaluate': {'metrics': METRICS},
    },
    {
        'name': 'train_size-8000',
        'process': {'features': ALL_FEATURES, 'train_size': 8000},
        'train':   {'model_type': 'RandomForestClassifier', 'n_estimators': 100},
        'evaluate': {'metrics': METRICS},
    },
    {
        'name': 'train_size-full',
        'process': {'features': ALL_FEATURES, 'train_size': None},
        'train':   {'model_type': 'RandomForestClassifier', 'n_estimators': 100},
        'evaluate': {'metrics': METRICS},
    },

    # ── Series 3: Vary n_estimators ───────────────────────────────────────────
    # Fixed: RandomForest, train_size=5000, all features
    {
        'name': 'n_estimators-10',
        'process': {'features': ALL_FEATURES, 'train_size': 5000},
        'train':   {'model_type': 'RandomForestClassifier', 'n_estimators': 10},
        'evaluate': {'metrics': METRICS},
    },
    {
        'name': 'n_estimators-50',
        'process': {'features': ALL_FEATURES, 'train_size': 5000},
        'train':   {'model_type': 'RandomForestClassifier', 'n_estimators': 50},
        'evaluate': {'metrics': METRICS},
    },
    {
        'name': 'n_estimators-100',
        'process': {'features': ALL_FEATURES, 'train_size': 5000},
        'train':   {'model_type': 'RandomForestClassifier', 'n_estimators': 100},
        'evaluate': {'metrics': METRICS},
    },
    {
        'name': 'n_estimators-200',
        'process': {'features': ALL_FEATURES, 'train_size': 5000},
        'train':   {'model_type': 'RandomForestClassifier', 'n_estimators': 200},
        'evaluate': {'metrics': METRICS},
    },
]
# fmt: on


def run_all():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f'Запускаем {len(EXPERIMENTS)} экспериментов в эксперименте "{EXPERIMENT_NAME}"')

    for i, exp in enumerate(EXPERIMENTS, start=1):
        run_name = exp['name']
        print(f'\n[{i}/{len(EXPERIMENTS)}] Запускаем: {run_name}')

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag('series', run_name.split('-')[0])
            process_data(exp['process'])
            train(exp['train'])
            evaluate(exp['evaluate'])

        print(f'    Готово: {run_name}')

    print('\nВсе эксперименты завершены!')


if __name__ == '__main__':
    run_all()
