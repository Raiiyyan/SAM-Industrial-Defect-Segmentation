"""Map per-dataset class labels to the unified 8-class defect vocabulary.

The ``class_label`` field in ``UniversalIndustrialDataset`` items uses
each dataset's *own* label scheme (category IDs, steel-defect class numbers,
etc.).  The ``IndustrialSAM`` classifier head predicts the unified 8-class
vocabulary defined in ``classifier_head.CLASS_VOCAB``.

This module provides the mapping layer between the two.

.. code-block:: python

    from label_mapping import map_dataset_item

    item = ds[0]
    item = map_dataset_item(item)  # now item["class_label"] is in [0..7]
"""

from typing import Dict, Optional
from classifier_head import NUM_CLASSES

# -------------------------------------------------------------------
# Unified vocabulary (copied from classifier_head.py for reference):
#   0: Flawless
#   1: Surface Scratch
#   2: Structural Crack
#   3: Hole / Puncture
#   4: Inclusion
#   5: Missing Component
#   6: Discoloration / Stain
#   7: Geometric Deformation
# -------------------------------------------------------------------


# ── NEU Surface Defect ───────────────────────────────────────────
# Official 6-class scheme: crazing, inclusion, patches,
# pitted_surface, rolled-in_scale, scratches.
_NEU_MAP: Dict[int, int] = {
    0: 2,  # crazing        → Structural Crack
    1: 4,  # inclusion      → Inclusion
    2: 6,  # patches        → Discoloration / Stain
    3: 3,  # pitted_surface → Hole / Puncture
    4: 4,  # rolled-in_scale → Inclusion
    5: 1,  # scratches      → Surface Scratch
}

# ── Severstal Steel Defect ──────────────────────────────────────
# ClassId 1-4  (dataset stores as 0-indexed after subtracting 1).
# REVIEW: these mappings are approximate — validate against your
# domain labels before deployment.
_SEVERSTAL_MAP: Dict[int, int] = {
    4: 0,  # no defect   → Flawless
    0: 3,  # Class 1     → Hole / Puncture   (pitted / crater)
    1: 4,  # Class 2     → Inclusion          (oxide patch)
    2: 1,  # Class 3     → Surface Scratch
    3: 2,  # Class 4     → Structural Crack
}

# ── DAGM2007 ─────────────────────────────────────────────────────
# Classes 0-9 are texture-background variants with elliptical defects.
# All are essentially a "geometric deformation" of the surface pattern.
# REVIEW: if your DAGM classes have distinguishable defect semantics,
# override individual entries here.
_DAGM_MAP: Dict[int, int] = {
    i: 7 for i in range(10)  # all → Geometric Deformation
}

# ── DefectSpectrum ──────────────────────────────────────────────
# Currently binary (0 = defect).  Without finer-grained labels we
# cannot map to a specific unified class.  Default:
#   0 → 7 (Geometric Deformation — most generic defect class)
# REVIEW: update when per-sample labels become available.
_DEFECT_SPECTRUM_MAP: Dict[int, int] = {
    0: 7,
}

# ── MVTec AD defect-type → unified mapping ──────────────────────
# MVTec AD / AD 2 labels are by *category* (object) not defect type,
# but each sample's ``meta`` contains a ``"defect"`` string.
# We use that to determine the unified label.
# REVIEW: unknown / unlisted defect names fall back to Geometric Deformation (7).
_MVTEC_DEFECT_TO_UNIFIED: Dict[str, int] = {
    # Structural Crack (2)
    "broken": 2,
    "broken_large": 2,
    "broken_small": 2,
    "crack": 2,
    "cracked": 2,
    "cut": 2,
    "rupture": 2,
    # Surface Scratch (1)
    "scratch": 1,
    "scratched": 1,
    "rough": 1,
    "thread": 1,
    # Hole / Puncture (3)
    "hole": 3,
    "puncture": 3,
    "poke": 3,
    # Inclusion (4)
    "contamination": 4,
    "glue": 4,
    "foreign": 4,
    "dirt": 4,
    # Missing Component (5)
    "missing": 5,
    "chip": 5,
    # Discoloration / Stain (6)
    "color": 6,
    "discoloration": 6,
    "stain": 6,
    "oil": 6,
    # Geometric Deformation (7)
    "bent": 7,
    "fold": 7,
    "squeeze": 7,
    "wrinkle": 7,
    "misalignment": 7,
    # MVTec AD 2 specific (position shifts, lighting)
    "shift_1": 7,
    "shift_2": 7,
    "shift_3": 7,
    "shift_4": 7,
    "overexposed": 6,
    "underexposed": 6,
    "regular": 7,
}


# ── Combined registry ───────────────────────────────────────────

# Datasets whose class_label is directly mappable via an int→int dict.
_DIRECT_MAP_DATASETS = {
    "neu_det": _NEU_MAP,
    "severstal": _SEVERSTAL_MAP,
    "dagm2007": _DAGM_MAP,
    "defect_spectrum": _DEFECT_SPECTRUM_MAP,
}

# Datasets that need meta-based mapping (category/defect-type lookup).
_META_MAP_DATASETS = {"mvtec_ad", "mvtec_ad_2"}


def map_to_unified(
    dataset_name: str,
    class_label: int,
    meta: Optional[Dict] = None,
) -> int:
    """Convert a per-dataset class label to a unified [0..7] label.

    Parameters
    ----------
    dataset_name : str
        One of the ``SUPPORTED_DATASETS`` names.
    class_label : int
        The original label from the dataset sample.
    meta : dict or None
        Sample metadata (required for MVTec AD / AD 2).

    Returns
    -------
    unified_label : int  in [0, NUM_CLASSES)

    Raises
    ------
    ValueError
        If the label cannot be mapped (unrecognised dataset or value).
    """
    if dataset_name in _DIRECT_MAP_DATASETS:
        mapping = _DIRECT_MAP_DATASETS[dataset_name]
        if class_label not in mapping:
            raise ValueError(
                f"{dataset_name}: unknown class_label={class_label}. "
                f"Known labels: {sorted(mapping)}"
            )
        return mapping[class_label]

    if dataset_name in _META_MAP_DATASETS:
        if meta is None or "defect" not in meta:
            raise ValueError(
                f"{dataset_name}: mapping requires meta with 'defect' key, "
                f"but got meta={meta}"
            )
        defect = meta["defect"]
        if defect in _MVTEC_DEFECT_TO_UNIFIED:
            return _MVTEC_DEFECT_TO_UNIFIED[defect]
        # REVIEW: unknown defect type — fallback to Geometric Deformation
        return 7

    raise ValueError(
        f"Unknown dataset '{dataset_name}'. "
        f"Supported: {sorted(_DIRECT_MAP_DATASETS)} | {sorted(_META_MAP_DATASETS)}"
    )


def map_dataset_item(item: dict) -> dict:
    """Return a copy of *item* with ``class_label`` mapped to unified [0..7].

    The input dict must contain at least ``"source_dataset"`` and
    ``"class_label"``.  For MVTec AD / AD 2, ``"meta"`` must include
    a ``"defect"`` key.
    """
    dataset = item["source_dataset"]
    label = item["class_label"]
    meta = item.get("meta")
    unified = map_to_unified(dataset, label, meta)
    return {**item, "class_label": unified}


# ── Smoke test ──────────────────────────────────────────────────

def _smoke_test():
    """Verify every dataset has a plausible mapping for at least one label."""
    print("label_mapping smoke test\n" + "=" * 30)

    for ds, mapping in sorted(_DIRECT_MAP_DATASETS.items()):
        sample_orig = next(iter(mapping.keys()))
        unified = mapping[sample_orig]
        assert 0 <= unified < NUM_CLASSES, f"{ds}: {unified} out of range"
        print(f"  {ds:20s}  {sample_orig} -> {unified}  OK")

    for ds in sorted(_META_MAP_DATASETS):
        sample_defect = next(iter(_MVTEC_DEFECT_TO_UNIFIED.keys()))
        unified = _MVTEC_DEFECT_TO_UNIFIED[sample_defect]
        assert 0 <= unified < NUM_CLASSES, f"{ds}: {unified} out of range"
        print(f"  {ds:20s}  '{sample_defect}' -> {unified}  OK")

    print(f"\nAll maps valid ({NUM_CLASSES} classes).")


if __name__ == "__main__":
    _smoke_test()
