import time
import abc
import inspect
import functools

from session_handler import SessionHandler


def time_diff(start_time):
    m, s = divmod(time.time() - start_time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return "%d:%02d:%02d:%02d" % (d, h, m, s)


class Trainable(object, metaclass=abc.ABCMeta):

    def __init__(self, model, learning_rate):
        self.training_session = TrainingSession(model=model, generate_train_dict=self.generate_train_dict,
                                                 learning_rate=learning_rate)

    @abc.abstractmethod
    def generate_train_dict(self, learning_rate, input_features, batch_size):
        pass


class Inferable(object, metaclass=abc.ABCMeta):

    def __init__(self, model):
        self.inferece_session = InferenceSession(model=model, generate_inference_dict=self.generate_inference_dict)

    @abc.abstractmethod
    def generate_inference_dict(self, input_features, batch_size):
        pass


class SessionHook():

    def __init__(self, f, p=None, y=None):
        self.f = f
        self.p = (lambda _: ()) if p is None else p
        self.has_final_result = y is not None
        self.y = y
        self.y_requirements = list(inspect.signature(y).parameters)[1:] if y is not None else ()
        self.f_requirements = list(inspect.signature(f).parameters)[1:]

    def __get__(self, obj, objtype):
        # Needed to be able to decorate instance methods
        return functools.partial(self.__call__, obj)

    def __call__(obj, self, **kwargs):
        kwargs.pop("self", None)
        arguments = {key: kwargs[key] for key in obj.f_requirements}
        provided = obj.p(self)
        provisions = obj.f(self, **arguments)
        if not isinstance(provisions, tuple):
            provisions = (provisions, )
        return {var: provisions[i] for i, var in enumerate(provided)}

    def get_yieldables(obj, self, **kwargs):
        if obj.y is None:
            return None
        kwargs.pop("self", None)
        arguments = {key: kwargs[key] for key in obj.y_requirements}
        provisions = obj.y(self, **arguments)
        return provisions

    def provides(self, p):
        return SessionHook(f=self.f, p=p, y=self.y)

    def yields(self, y):
        return SessionHook(f=self.f, p=self.p, y=y)


class Hookable(type):

    def __init__(cls, name, bases, nmspc):
        cls.hooks = []
        super().__init__(name, bases, nmspc)
        for name, func in nmspc.items():
            if isinstance(func, SessionHook):
                cls.hooks.append(func)


class CustomSession(object, metaclass=Hookable):

    def __init__(self, model):
        self.model = model

    def utilize_session(self, model_name, data_provider, **kwargs):
        with SessionHandler(model=self.model, model_name=model_name) as session_handler:
            session_handler.load_weights_or_init()
            start_time = time.time()
            step = session_handler.step
            available = locals()
            available.pop("self", None)
            available.update(kwargs)
            print(f"{'Resuming' if step > 0 else 'Starting'} {model_name}: at step {step}")

            for input_feature in data_provider:
                available["input_feature"] = input_feature
                for hook in self.hooks:
                    provided = hook(self=self, **available)
                    available.update(provided)

                    if hook.has_final_result:
                        yield hook.get_yieldables(self=self, **available)


class TrainingSession(CustomSession):

    def __init__(self, model, learning_rate, generate_train_dict):
        self.learning_rate = learning_rate
        self.generate_train_dict = generate_train_dict
        super().__init__(model=model)

    def train(self, model_name, data_provider, save_every_n_steps=10):
        self.utilize_session(model_name=model_name, data_provider=data_provider,
                             save_every_n_steps=save_every_n_steps)

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

    def __init__(self, model, generate_inference_dict):
        self.generate_inference_dict = generate_inference_dict
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
