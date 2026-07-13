"""
Dataset constants for DESR.

This file keeps only the datasets used in the DESR paper:
MVTec-AD, VisA, BTAD, MPDD, DTD-Synthetic, and DAGM.

Dataset paths are intentionally configurable for GitHub release.
Set DESR_DATA_ROOT or dataset-specific environment variables if your
datasets are stored elsewhere.

Example:
    set DESR_DATA_ROOT=C:\\code\\MoEAD_lion_adaptor\\data
    set VISA_PATH=C:\\dataset\\VisA_20220922
"""

import os


def _data_path(env_name: str, default_relative_path: str) -> str:
    """Return dataset path from an environment variable or DESR_DATA_ROOT."""
    data_root = os.environ.get("DESR_DATA_ROOT", "./data")
    return os.environ.get(env_name, os.path.join(data_root, default_relative_path))


# =============================================================================
# Dataset paths
# =============================================================================
DATA_PATH = {
    # Main source / target datasets
    "MVTec": _data_path("MVTEC_PATH", os.path.join("MVTec-AD", "anomaly_detection")),
    "VisA": _data_path("VISA_PATH", "VisA_20220922"),

    # Cross-dataset evaluation datasets used in the paper
    "BTAD": _data_path("BTAD_PATH", "BTAD"),
    "MPDD": _data_path("MPDD_PATH", "MPDD"),
    "DTD": _data_path("DTD_PATH", "DTD-Synthetic"),
    "DAGM": _data_path("DAGM_PATH", "DAGM"),
}


# =============================================================================
# Dataset class names
# =============================================================================
CLASS_NAMES = {
    "MVTec": [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "transistor",
        "toothbrush",
        "wood",
        "zipper",
    ],
    "VisA": [
        "candle",
        "pcb3",
        "capsules",
        "pipe_fryum",
        "pcb4",
        "macaroni2",
        "pcb2",
        "chewinggum",
        "macaroni1",
        "cashew",
        "fryum",
        "pcb1",
    ],
    "BTAD": ["01", "02", "03"],
    "MPDD": [
        "connector",
        "tubes",
        "metal_plate",
        "bracket_white",
        "bracket_brown",
        "bracket_black",
    ],
    "DTD": [
        "Woven_001",
        "Woven_127",
        "Woven_104",
        "Stratified_154",
        "Blotchy_099",
        "Woven_068",
        "Woven_125",
        "Marbled_078",
        "Perforated_037",
        "Mesh_114",
        "Fibrous_183",
        "Matted_069",
    ],
    "DAGM": [
        "Class1",
        "Class2",
        "Class3",
        "Class4",
        "Class5",
        "Class6",
        "Class7",
        "Class8",
        "Class9",
        "Class10",
    ],
}


# =============================================================================
# Domain type
# =============================================================================
DOMAINS = {
    "MVTec": "Industrial",
    "VisA": "Industrial",
    "BTAD": "Industrial",
    "MPDD": "Industrial",
    "DTD": "Industrial",
    "DAGM": "Industrial",
}


# =============================================================================
# Human-readable object names for prompt construction
# =============================================================================
REAL_NAMES = {
    "MVTec": {
        "bottle": "dark bottle",
        "cable": "top view of three cables",
        "capsule": "black and orange capsule",
        "carpet": "gray carpet",
        "grid": "metal or plastic mesh",
        "hazelnut": "single brown hazelnut",
        "leather": "brown leather",
        "metal_nut": "metal nut which has four notched edges",
        "pill": "oval white pill with small red speckles and the letters 'FF' engraved",
        "screw": "screw",
        "tile": "speckled tile surface",
        "transistor": "a three-legged transistor placed vertically",
        "toothbrush": "toothbrush head",
        "wood": "wood surface",
        "zipper": "a black zipper",
    },
    "VisA": {
        "candle": "candle",
        "pcb3": "infrared sensor pcb module",
        "capsules": "capsules",
        "pipe_fryum": "pipe-shaped fryum",
        "pcb4": "battery charging pcb module",
        "macaroni2": "scattered yellow macaroni",
        "pcb2": "integrated circuits board",
        "chewinggum": "chewing gum",
        "macaroni1": "orange macaroni",
        "cashew": "cashew nut",
        "fryum": "wheel-shaped fryum snack",
        "pcb1": "dual ultrasonic distance sensor pcb module",
    },
    "BTAD": {
        "01": "bright concentric rings in neon yellow and blue tones against a dark blue background",
        "02": "vertical fabric lines in warm, dusty pink and beige tones",
        "03": "oval concentric circular rings in gradient shades of blue and white",
    },
    "MPDD": {
        "connector": "metal clamps with black adjustment knobs",
        "tubes": "scattered metal objects",
        "metal_plate": "blue rectangular metal plate with a notch on one side",
        "bracket_white": "white elongated triangular metal bracket with a smooth matte finish",
        "bracket_brown": "brown L-shaped metal bracket with a glossy finish and mounting holes",
        "bracket_black": "black ornamental metal bracket with spiral design",
    },
    "DTD": {
        "Woven_001": "woven fabric with regular interlacing pattern, light beige tone",
        "Woven_127": "woven pattern with fine, tight threads, gray-beige colors",
        "Woven_104": "woven fabric with irregular gaps, light brown shade",
        "Stratified_154": "stratified layers of fibers with alternating light and dark bands",
        "Blotchy_099": "blotchy irregular color patches on fabric",
        "Woven_068": "coarse woven texture with slightly uneven threads",
        "Woven_125": "dense woven material with uniform pattern",
        "Marbled_078": "marbled pattern with swirls of contrasting colors",
        "Perforated_037": "fabric with small regular perforations",
        "Mesh_114": "mesh-like texture with open grid pattern",
        "Fibrous_183": "random fibrous structure with tangled fibers",
        "Matted_069": "matted surface with compressed fibers and irregular patterns",
    },
    "DAGM": {
        "Class1": "fabric surface with bright concentric rings",
        "Class2": "vertical fabric lines in warm pink and beige tones",
        "Class3": "oval concentric circular rings in blue and white shades",
        "Class4": "fine textile surface with irregular woven patterns",
        "Class5": "layered textile with alternating light and dark bands",
        "Class6": "fabric with subtle random defects",
        "Class7": "dense woven material with uniform thread pattern",
        "Class8": "marbled textile pattern with swirling contrast colors",
        "Class9": "perforated fabric with small evenly spaced holes",
        "Class10": "open mesh-like fabric with grid structure",
    },
}


# =============================================================================
# Text prompts
# =============================================================================
PROMPTS = {
    "prompt_normal": [
        "{}",
        "a {}",
        "the {}",
    ],
    "prompt_abnormal": [
        "a damaged {}",
        "a broken {}",
        "a {} with flaw",
        "a {} with defect",
        "a {} with damage",
    ],
    "prompt_templates": [
        "{}.",
        "a photo of {}.",
    ],
}
