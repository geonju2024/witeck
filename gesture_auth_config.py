SEQUENCE_LENGTH = 60
FEATURE_SIZE = 63

MODEL_PATH = "gesture_auth_gru_model.keras"
METADATA_PATH = "gesture_dataset/metadata.csv"
THRESHOLD_PATH = "gesture_auth_threshold.json"

# Calibrated from the current dataset/model on 2026-07-23.
# This gives 16/17 = 94.12% resubstitution accuracy with the existing model.
DEFAULT_AUTH_THRESHOLD = 0.35

PERSON_MODEL_PATH = "gesture_person_gru_model.keras"
PERSON_LABELS_PATH = "gesture_person_labels.json"


PERSON_REPORT_PATH = "gesture_person_evaluation_report.json"
PERSON_DATASET_AUDIT_PATH = "gesture_person_dataset_audit.json"
PERSON_TARGET_ACCURACY = 0.90
MIN_PERSON_USERS = 2
MIN_PERSON_SAMPLES_PER_USER = 3

RECOMMENDED_PERSON_SAMPLES_PER_USER = 10
PERSON_REPEATED_EVAL_SPLITS = 5
PERSON_TEST_SIZE = 0.25
