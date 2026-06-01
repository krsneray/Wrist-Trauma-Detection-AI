from pathlib import Path

# Yollar (Paths)
ROOT       = Path('/content/drive/MyDrive/Wrist Dataset')
LABELS_DIR = ROOT / 'folder_structure/yolov5/labels'
IMAGES_DIR = ROOT / 'images'
CSV_PATH   = ROOT / 'dataset.csv'
LOCAL_DIR  = Path('/content')
CACHE_PATH = ROOT / 'annotations_cache.csv'

# Konfigürasyon
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.10
TEST_RATIO    = 0.20
SEED          = 42
FORCE_REBUILD = False

CLASSES = ['boneanomaly', 'bonelesion', 'foreignbody_metal',
           'fracture', 'periostealreaction', 'pronatorsign', 'softtissue']

AGE_BINS   = [0, 5, 9, 13, 99]
AGE_LABELS = ['0-4', '5-8', '9-12', '13-19']