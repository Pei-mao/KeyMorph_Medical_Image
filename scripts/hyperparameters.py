import torchio as tio
from keymorph.utils import rescale_intensity

TRANSFORM = tio.Compose(
    [
        tio.ToCanonical(),
        tio.Mask(masking_method="mask"),
        tio.Resize(128),
        tio.Lambda(rescale_intensity),
    ]
)

EVAL_METRICS = [
    "mse",
    "softdice",
    "harddice",
    "hausd",
    "jdstd",
    "jdlessthan0",
]

EVAL_UNI_NAMES = [
    ("T1", "T1"),
    ("T2", "T2"),
    ("PD", "PD"),
]
EVAL_MULTI_NAMES = [
    ("T1", "T2"),
    ("T1", "PD"),
    ("T2", "PD"),
]
EVAL_LESION_NAMES = None
EVAL_GROUP_NAMES = None
EVAL_LONG_NAMES = None

EVAL_AUGS = [
    "rot0",
    "rot45",
    "rot90",
    "rot135",
    "rot180",
]

EVAL_KP_ALIGNS = [
    "rigid",
    "affine",
    "tps_10",
    "tps_1",
    "tps_0.1",
    "tps_0.01",
    "tps_0",
]
