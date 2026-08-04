"""Default operational settings for ``run_pipeline.sh`` and its DAG."""

DEFAULT_SEED = 42
DEFAULT_YEAR = 2024
DEFAULT_PIPELINE = "legacy"
DEFAULT_REGRESSOR = "sgd"
DEFAULT_GPR_MAX_TRAIN_ROWS = 500

DEFAULT_ANNUAL_INPUT = (
    "data_csv_parquet/andria_distribuzione_annuale_zone_inferite_e_zone_costruzione.csv"
)
DEFAULT_COMPLETE_ANNUAL_INPUT = "data_csv_parquet/andria_distribuzione_annuale_completa.parquet"
DEFAULT_PRODUCTION_INPUT = "data_csv_parquet/produzione_giornaliera_completa.parquet"
DEFAULT_COLLECTIONS_INPUT = "data_csv_parquet/conferimenti_giornalieri_completi.parquet"
DEFAULT_CALENDAR_INPUT = "data_csv_parquet/calendario_raccolta.csv"

KMEANS_CLUSTERS = 6
KMEANS_COORD_WEIGHT = 8
KMEANS_ANNUAL_KG_WEIGHT = 0.5
KMEANS_OTHER_NUMERIC_WEIGHT = 1
KMEANS_CATEGORICAL_WEIGHT = 1

LEGACY_SPLIT_RATIOS = "80,10,10"
LEGACY_SGD_EPOCHS = 3
LEGACY_SGD_BATCH_SIZE = 4_096
LEGACY_ID_COLUMN = "id_civico"
LEGACY_ZONE_COLUMN = "zona_raccolta"
MISSING_CALENDAR_MODE = "daily"

DEFAULT_ANNUAL_SEED = 42
DEFAULT_SCENARIO_SEED = 42
DEFAULT_SPILLOVER_DAYS = 14
DEFAULT_MISSING_CALENDAR_MODE = "error"
DEFAULT_GENERATOR_VERSION = "unversioned-current"
QUANTUMLEAP_GENERATOR_VERSION = "quantumleap"
DEFAULT_HGB_LOSS = "poisson"
DEFAULT_VALIDATION_STRATEGY = "fixed"
DEFAULT_VALIDATION_FOLDS = 5

DEFAULT_QL_URL = "http://localhost:8668"
DEFAULT_QL_ENTITY_TYPE = "RifiutiConferimento"
DEFAULT_FIWARE_SERVICE = "openiot"
DEFAULT_FIWARE_SERVICE_PATH = "/"
DEFAULT_QL_TIMEOUT_SECONDS = 30.0
DEFAULT_ORION_URL = "http://localhost:1026"
DEFAULT_ORION_ENTITY_TYPE = "RifiutiDigitalTwin"
DEFAULT_ORION_TIMEOUT_SECONDS = 30.0
DEFAULT_ORION_BATCH_SIZE = 50
