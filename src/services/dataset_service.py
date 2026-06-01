import pandas as pd
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from skmultilearn.model_selection import IterativeStratification
from src.config.settings import *
from src.utils.patient_utils import check_patient_leakage

def build_patient_splits(triples):
    img_records = {k: {'new_classes': v, 'is_empty': len(v) == 0} for k, v, _ in triples}

    pid_of  = {s: s.split('_')[0] for s in img_records}
    pid_cls = defaultdict(set)
    pid_ann = defaultdict(bool)
    
    for stem, rec in img_records.items():
        pid = pid_of[stem]
        pid_cls[pid].update(rec['new_classes'])
        pid_ann[pid] = pid_ann[pid] or (not rec['is_empty'])

    valid_pids = {p for p, v in pid_ann.items() if v}
    df_m = pd.read_csv(CSV_PATH, dtype={'patient_id': str})
    df_m['patient_id'] = df_m['patient_id'].str.zfill(4)
    df_m = df_m[df_m['patient_id'].isin(valid_pids)].copy()

    agg = (df_m.groupby('patient_id', sort=False)
               .agg(gender  = ('gender', 'first'),
                    min_age = ('age', 'min'),
                    lat_set = ('laterality', lambda x: frozenset(x.dropna())))
               .reset_index())

    agg['age_group']  = pd.cut(agg['min_age'], bins=AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
    agg['laterality'] = agg['lat_set'].apply(lambda s: 'both' if len(s) > 1 else f'{list(s)[0]}_only')

    for cid, cname in enumerate(CLASSES):
        agg[cname] = agg['patient_id'].apply(lambda pid, c=cid: int(c in pid_cls.get(pid, set())))

    patient_df = agg.set_index('patient_id')[['gender', 'age_group', 'laterality'] + CLASSES]

    enc  = pd.get_dummies(patient_df, columns=['gender', 'age_group', 'laterality'], dtype=int)
    pids = enc.index.to_numpy()
    lbl  = enc.values.astype(float)
    idx  = np.random.default_rng(SEED).permutation(len(pids))
    pids, lbl = pids[idx], lbl[idx]

    s1 = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[1 - TRAIN_RATIO, TRAIN_RATIO])
    f0, f1 = next(s1.split(sp.lil_matrix(lbl), sp.lil_matrix(lbl)))
    tr_i, tp_i = (f1, f0) if len(f1) > len(f0) else (f0, f1)
    train_pids  = pids[tr_i]
    tp, tp_lbl  = pids[tp_i], lbl[tp_i]

    vf = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    s2 = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[1 - vf, vf])
    g0, g1 = next(s2.split(sp.lil_matrix(tp_lbl), sp.lil_matrix(tp_lbl)))
    v_i, te_i = (g1, g0) if len(g1) < len(g0) else (g0, g1)
    val_pids   = tp[v_i]
    test_pids  = tp[te_i]

    pid_split = {'train': set(train_pids), 'val': set(val_pids), 'test': set(test_pids)}

    check_patient_leakage(pid_split['train'], pid_split['val'], pid_split['test'])
    print('✓ Hasta sızıntısı (Leakage) testi başarılı. Sızıntı yok!')

    for sn, pid_set in pid_split.items():
        rows = []
        for stem, rec in img_records.items():
            if rec['is_empty']: continue
            pid = pid_of[stem]
            if pid not in pid_set: continue
            p = patient_df.loc[pid]
            rows.append({'image_path': str(IMAGES_DIR / f'{stem}.png'), 'image_stem': stem, 'patient_id': pid,
                         'gender': p['gender'], 'age_group': p['age_group'], 'laterality': p['laterality'],
                         **{c: int(p[c]) for c in CLASSES}})
        pd.DataFrame(rows).sort_values('image_path').reset_index(drop=True).to_csv(LOCAL_DIR / f'{sn}_split{SEED}.csv', index=False)