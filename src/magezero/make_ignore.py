from pyroaring import BitMap

from dataset import load_dataset_from_directory, create_redundancy_ignore_list
from train import GLOBAL_MAX

combined_ds = load_dataset_from_directory("data/UWTempo/ver11/training")
print("Generating ignore list for combined dataset to use for model")
ignore_list = create_redundancy_ignore_list(combined_ds, GLOBAL_MAX)
print("Saving ignore list to ignore.roar")
ignore = BitMap(ignore_list)  # iterable of ints
with open("ignore.roar", "wb") as f:
    f.write(ignore.serialize())