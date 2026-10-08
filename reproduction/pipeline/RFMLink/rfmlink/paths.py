from pathlib import Path

RFMLINK_ROOT = Path(__file__).resolve().parent.parent
SBM_TFM_ROOT = RFMLINK_ROOT.parent
ULTRA_ROOT = RFMLINK_ROOT / "external" / "ULTRA"
MODELS_CKPT = Path("/home/jupyter-alessandro/remote/data/models_ckpt")
TFMS_PYTHON = Path("/home/jupyter-alessandro/.conda/envs/tfms/bin/python")

TABPFN3_CLASSIFIER_CKPT = MODELS_CKPT / "tabpfn-v3-classifier-v3_default.ckpt"

DATA_PROCESSED = RFMLINK_ROOT / "data" / "processed"
DATA_RAW = RFMLINK_ROOT / "data" / "raw"
CONTEXTS_DIR = RFMLINK_ROOT / "contexts"
FEATURES_DIR = RFMLINK_ROOT / "features"
ULTRA_EXPORTS_DIR = RFMLINK_ROOT / "ultra_exports"
RESULTS_DIR = RFMLINK_ROOT / "results"
