
import tensorflow as tf

from interfaces import Model





if __name__ == "__main__":
    from model import SynethesiaModel
    with SessionHandler(model=SynethesiaModel(64), model_name="synethesia") as sess_handler:
        sess_handler.load_weights_or_init()
