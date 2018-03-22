import numpy as np

from synethesia_model import SynethesiaModel
from model_interactor import Trainable, Inferable


class Synethesia(Trainable, Inferable):

    def __init__(self, feature_dim, img_size=(256, 128)):
        self.img_size = img_size
        self.model = SynethesiaModel(feature_dim=feature_dim, img_size=img_size)
        Trainable.__init__(self, model=self.model)
        Inferable.__init__(self, model=self.model)

    def generate_train_dict(self, learning_rate, input_features):
        batch_size = input_features.shape[0]
        feed_dict = {self.model.learning_rate: learning_rate,
                     self.model.data_input: input_features,
                     self.model.base_img: self.random_start_img(batch_size=batch_size)}
        return feed_dict

    def generate_inference_dict(self, input_features):
        batch_size = input_features.shape[0]
        feed_dict = {self.model.data_input: input_features,
                     self.model.base_img: self.random_start_img(batch_size=batch_size)}
        return feed_dict

    def random_start_img(self, batch_size):
        generation_shape = (batch_size, *self.img_size, 3)
        img = np.random.uniform(size=generation_shape)
        return img



if __name__ == "__main__":
    from audio_chunk_loader import StaticSongLoader
    from PIL import Image
    from feature_creators import logfbank_features

    train = False
    batch_size = 32
    train_loader = StaticSongLoader(song_files=[#"/home/veith/Projects/PartyGAN/data/Bearded Skull - 420 [Hip Hop Instrumental]/audio/soundtrack.mp3",
                                                "/home/veith/Projects/PartyGAN/data/Gorillaz - Feel Good Inc. (Official Video)/audio/soundtrack.mp3",
                                                ],
                                    batch_size=batch_size, load_n_songs_at_once=2,
                                    to_infinity=train, feature_extractor=logfbank_features, allow_shuffle=train)

    synethesia = Synethesia(feature_dim=train_loader.feature_dim)
    model_name = "overfit_multiple"
    if train:
        synethesia.training_session.train(model_name=model_name, data_provider=train_loader, learning_rate=0.0001)
    else:
        img_id = 0
        # TODO make inference go song by song
        for i, (imgs, sounds) in enumerate(synethesia.inferece_session.infer(model_name=model_name,
                                                                             data_provider=train_loader)):
                for j, img in enumerate(imgs):
                    img_id += 1
                    Image.fromarray((img * 255).astype(np.uint8)).save(f"/tmp/test_feel_good_inc/{img_id}.png")
