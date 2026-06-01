from ultralytics import YOLO
from pathlib import Path
from src.config.settings import SEED

def run_training_pipeline(yolo_yaml_path: str):
    print("YOLOv11m (Attention Katmanlı) başlatılıyor...")

    model = YOLO('/content/yolov11m-cbam.yaml')

    results = model.train(
        data      = yolo_yaml_path,
        epochs    = 100,
        imgsz     = 640,
        batch     = 16,
        patience  = 20,
        optimizer = 'AdamW',
        lr0       = 1e-3,
        cos_lr    = True,
        cache     = 'ram',
        fliplr    = 0.0,
        flipud    = 0.0,
        degrees   = 5,
        translate = 0.05,
        scale     = 0.3,
        device    = 0,
        project   = '/content/runs',
        name      = f'yolov11m_cbam_seed{SEED}',
        exist_ok  = True,
        save      = True,
        plots     = True,
    )

    run_dir = Path(f'/content/runs/yolov11m_cbam_seed{SEED}')
    best_pt = run_dir / 'weights/best.pt'

    print(f'\nEn iyi ağırlıklar yükleniyor: {best_pt}')
    test_model = YOLO(best_pt)

    test_res = test_model.val(
        data      = yolo_yaml_path,
        split     = 'test',
        device    = 0,
        plots     = True,
        save_json = True,
    )
    
    return test_res