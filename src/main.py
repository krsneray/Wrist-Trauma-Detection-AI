from src.services.preprocessing_service import extract_labels_and_classes
from src.services.dataset_service import build_patient_splits
from src.data.yolo_dataset_builder import prepare_yolo_directories
from src.services.training_service import run_training_pipeline

def main():
    # 1. Aşama: Ham Etiketlerin Okunması
    triples = extract_labels_and_classes()
    
    # 2. Aşama: Train/Val/Test Split İşlemi (Stratified)
    build_patient_splits(triples)
    
    # 3. Aşama: YOLO formatına kopyalama ve symlink
    yolo_yaml_path = prepare_yolo_directories(triples)
    
    # 4. Aşama: Eğitim ve Test
    run_training_pipeline(str(yolo_yaml_path))

if __name__ == "__main__":
    main()