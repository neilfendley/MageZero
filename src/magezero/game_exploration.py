from magezero.dataset import H5Indexed
DECK_NAME='BWBats'
VER_NUMBER=0
DATA_DIR = f"data/{DECK_NAME}/ver{VER_NUMBER}/training"

def explore_game():
    ds = H5Indexed(DATA_DIR)
    breakpoint()