def remap_class(original_id):
    CLASS_REMAP = {0:0, 1:1, 2:2, 3:3, 4:2, 5:4, 6:5, 7:6, 8:None}
    if original_id not in CLASS_REMAP:
        raise KeyError(f'Bilinmeyen sınıf ID: {original_id}')
    return CLASS_REMAP[original_id]