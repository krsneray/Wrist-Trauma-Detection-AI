from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from src.utils.annotation_utils import remap_class
from src.config.settings import LABELS_DIR

def extract_labels_and_classes():
    def _read_label(p):
        class_ids, annotations = set(), []
        for line in p.read_text().split('\n'):
            tok = line.split()
            if not tok: continue
            orig = int(tok[0])
            nid = remap_class(orig) 
            if nid is not None:
                class_ids.add(nid)
                if len(tok) >= 5:
                    annotations.append((nid, tok[1], tok[2], tok[3], tok[4]))
        return p.stem, class_ids, annotations

    label_files = sorted(LABELS_DIR.glob('*.txt'))
    print(f'Etiketler taranıyor ({len(label_files)} dosya)...', flush=True)

    with ThreadPoolExecutor(max_workers=8) as ex:
        triples = list(ex.map(_read_label, label_files))
        
    return triples