_NEU_MAP = {0: 2, 1: 4, 2: 6, 3: 3, 4: 4, 5: 1}
_SEVERSTAL_MAP = {4: 0, 0: 3, 1: 4, 2: 1, 3: 2}
_DAGM_MAP = {i: 7 for i in range(10)}
_DEFECT_SPECTRUM_MAP = {0: 7}
_MVTEC_DEFECT_TO_UNIFIED = {"broken": 2, "broken_large": 2, "broken_small": 2, "crack": 2, "cut": 2, "scratch": 1, "rough": 1, "thread": 1, "hole": 3, "poke": 3, "contamination": 4, "glue": 4, "missing": 5, "color": 6, "oil": 6, "bent": 7, "fold": 7, "squeeze": 7}
_DIRECT_MAP_DATASETS = {"neu_det": _NEU_MAP, "severstal": _SEVERSTAL_MAP, "dagm2007": _DAGM_MAP, "defect_spectrum": _DEFECT_SPECTRUM_MAP}
_META_MAP_DATASETS = {"mvtec_ad", "mvtec_ad_2"}

def map_to_unified(dataset_name, class_label, meta=None):
    if dataset_name in _DIRECT_MAP_DATASETS:
        return _DIRECT_MAP_DATASETS[dataset_name][class_label]
    if dataset_name in _META_MAP_DATASETS:
        defect = meta.get("defect", "")
        return _MVTEC_DEFECT_TO_UNIFIED.get(defect, 7)
    raise ValueError(f"Unknown dataset '{dataset_name}'")

def map_dataset_item(item):
    return {**item, "class_label": map_to_unified(item["source_dataset"], item["class_label"], item.get("meta"))}