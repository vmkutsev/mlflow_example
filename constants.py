DATASET_PATH_PATTERN = '/app/data/{split_name}.csv'
DATASET_NAME = 'scikit-learn/adult-census-income'
MODEL_FILEPATH = '/app/model.joblib'
FEATURE_NAMES_PATH = '/app/data/feature_names.json'
RANDOM_STATE = 42
TEST_SIZE = 0.3

MLFLOW_TRACKING_URI = 'http://158.160.2.37:5000/'
EXPERIMENT_NAME = 'homework_kutsev'

ALL_FEATURES = [
    'age', 'workclass', 'fnlwgt', 'education', 'education.num',
    'marital.status', 'occupation', 'relationship', 'race', 'sex',
    'capital.gain', 'capital.loss', 'hours.per.week', 'native.country',
]
