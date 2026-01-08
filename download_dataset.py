import os

# Set HuggingFace home to local disk which has enough space
os.environ["HF_HOME"] = os.path.join("/datasets/DL3DV-DSO", ".cache", "huggingface")

from datasets import load_dataset
from datasets.config import HF_DATASETS_CACHE

cache_dir = os.path.join("/datasets/DL3DV-DSO/fineweb-edu")
os.makedirs(cache_dir, exist_ok=True)

# Load a subset with roughly 100B tokens, suitable for small- or medium-sized experiments
dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    # name="default",
    name="sample-100BT",
    # num_proc=None,  # Disable multiprocessing to avoid subprocess crashes
    num_proc=4,
    cache_dir=os.path.join(cache_dir, "sample-100BT"),
    trust_remote_code=True,
)