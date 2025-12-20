import os

# Set HuggingFace home to scratch directory
os.environ["HF_HOME"] = os.path.join(os.environ.get("SCRATCH", "/scratch/junchen"), ".cache", "huggingface")

from datasets import load_dataset

cache_dir = os.path.join(os.environ.get("SCRATCH", "/scratch/junchen"), "datasets/fineweb-edu")
os.makedirs(cache_dir, exist_ok=True)

# load fineweb-edu with parallel processing
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="default", num_proc=1, cache_dir=os.path.join(cache_dir, "default"))

# or load a subset with roughly 100B tokens, suitable for small- or medium-sized experiments
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", num_proc=1, cache_dir=os.path.join(cache_dir, "sample-100BT"))