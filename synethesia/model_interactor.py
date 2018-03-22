import time

from interfaces import CustomSession, SessionHook, Trainable, Inferable


def time_diff(start_time):
    m, s = divmod(time.time() - start_time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return "%d:%02d:%02d:%02d" % (d, h, m, s)


class TrainingSession(CustomSession):

    def __init__(self, model, learning_rate, trainable):
        if not isinstance(trainable, Trainable):
            raise ValueError(f"{trainable} does not implement the Trainable interface!")

        self.learning_rate = learning_rate
        self.generate_train_dict = trainable.generate_train_dict
        super().__init__(model=model)

    def train(self, model_name, data_provider, save_every_n_steps=1000):
        for _ in self.utilize_session(model_name=model_name, data_provider=data_provider,
                                      save_every_n_steps=save_every_n_steps):
            pass

    @SessionHook
    def _train_once(self, session_handler, input_feature):
        feed_dict = self.generate_train_dict(input_features=input_feature)
        feed_dict.update({self.model.learning_rate: self.learning_rate})
        step, _ = session_handler.training_step(feed_dict=feed_dict)
        return step

    @_train_once.provides
    def _train_once(self):
        return ("step",)

    @SessionHook
    def _maybe_save(self, session_handler, step, start_time, save_every_n_steps):
        if step % save_every_n_steps == 0 and step > 0:
            session_handler.save(step=step)
            print(f"Step {step}, time: {time_diff(start_time)}: Saving in {session_handler.checkpoint_dir}")


class InferenceSession(CustomSession):

    def __init__(self, model, inferable):
        if not isinstance(inferable, Inferable):
            raise ValueError(f"{inferable} does not implement Inferable interface!")

        self.generate_inference_dict = inferable.generate_inference_dict
        super().__init__(model=model)

    def infer(self, model_name, data_provider):
        yield from self.utilize_session(model_name=model_name, data_provider=data_provider)

    @SessionHook
    def _infer_once(self, session_handler, input_feature):
        feed_dict = self.generate_inference_dict(input_features=input_feature)
        results = session_handler.inference_step(feed_dict=feed_dict)
        return results

    @_infer_once.provides
    def _infer_once(self):
        return ("results",)

    @_infer_once.yields
    def _infer_once(self, results):
        return results


if __name__ == "__main__":
    from audio_chunk_loader import StaticSongLoader
    from synethesia_model import SynethesiaModel
    from PIL import Image
    from feature_creators import logfbank_features

    train = False
    train_loader = StaticSongLoader(song_files=["/home/veith/Projects/PartyGAN/data/Bearded Skull - 420 [Hip Hop Instrumental]/audio/soundtrack.mp3"],
                                    batch_size=16, load_n_songs_at_once=1,
                                    to_infinity=train, feature_extractor=logfbank_features)

    synethesia = SynethesiaModel(feature_dim=41)

    sess = TestSession(model=synethesia)
    sess.print_input(input_feature="Hi")
    sess.utilize_session(model_name="test", data_provider=train_loader)
