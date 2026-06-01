import pandas as pd
from src.config.settings import AGE_BINS, AGE_LABELS

def get_age_group(age):
    res = pd.cut([age], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    return str(res[0])

def check_patient_leakage(train_set, val_set, test_set):
    train_set, val_set, test_set = set(train_set), set(val_set), set(test_set)
    if train_set & val_set: raise ValueError("SIZINTI: Train ve Val ortak!")
    if train_set & test_set: raise ValueError("SIZINTI: Train ve Test ortak!")
    if val_set & test_set: raise ValueError("SIZINTI: Val ve Test ortak!")
    return True