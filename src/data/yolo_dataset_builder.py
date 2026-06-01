import shutil
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from src.config.settings import LOCAL_DIR, SEED, IMAGES_DIR, CLASSES

def prepare_yolo_directories(triples):
    LOCAL_IMGS = Path('/content/images_local')
    LOCAL_IMGS.mkdir(exist_ok=True)

    all_stems = set()
    for sn in ['train', 'val', 'test']:
        all_stems.update(pd.read_csv(LOCAL_DIR / f'{sn}_split{SEED}.csv')['image_stem'])

    to_copy = [s for s in all_stems if not (LOCAL_IMGS / f'{s}.png').exists()]
    if to_copy:
        print(f'{len(to_copy)} resim kopyalanıyor (Bu işlem biraz sürebilir)...')
        def _copy_img(stem): shutil.copy2(IMAGES_DIR / f'{stem}.png', LOCAL_IMGS / f'{stem}.png')
        with ThreadPoolExecutor(max_workers=32) as ex: list(ex.map(_copy_img, to_copy))

    anno_map = {}
    for stem, _, annotations in triples:
        if annotations:
            anno_map[stem] = [f"{c} {x} {y} {w} {h}" for c, x, y, w, h in annotations]

    YOLO = Path(f'/content/yolo_data_{SEED}')
    for split in ['train', 'val', 'test']:
        (YOLO / 'images' / split).mkdir(parents=True, exist_ok=True)
        (YOLO / 'labels' / split).mkdir(parents=True, exist_ok=True)

    def _prep(args):
        stem, sn = args
        img_dst = YOLO / 'images' / sn / f'{stem}.png'
        if not img_dst.exists(): img_dst.symlink_to(LOCAL_IMGS / f'{stem}.png')
        lbl_dst = YOLO / 'labels' / sn / f'{stem}.txt'
        if not lbl_dst.exists(): lbl_dst.write_text('\n'.join(anno_map.get(stem, [])))

    tasks = [(stem, sn) for sn in ['train', 'val', 'test'] for stem in pd.read_csv(LOCAL_DIR / f'{sn}_split{SEED}.csv')['image_stem']]
    with ThreadPoolExecutor(max_workers=8) as ex: list(ex.map(_prep, tasks))

    YOLO_YAML = YOLO / 'dataset.yaml'
    YOLO_YAML.write_text(
        f'path: {YOLO}\n'
        f'train: images/train\n'
        f'val:   images/val\n'
        f'test:  images/test\n\n'
        f'nc: {len(CLASSES)}\n'
        f'names: {CLASSES}\n'
    )
    return YOLO_YAML