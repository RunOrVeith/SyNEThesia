import time
import abc

import tensorflow as tf
import numpy as np

from synethesia.framework.session_management import CustomSession, SessionHook


def time_diff(start_time):
    m, s = divmod(time.time() - start_time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return "%d:%02d:%02d:%02d" % (d, h, m, s)


class Trainable(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def generate_train_dict(self, input_features):
        pass


class Inferable(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def generate_inference_dict(self, input_features):
        pass


class TrainingSession(CustomSession):

    def __init__(self, model, learning_rate, trainable):
        if not isinstance(trainable, Trainable):
            raise ValueError(f"{trainable} does not implement the Trainable interface!")

        self.learning_rate = learning_rate
        self.generate_train_dict = trainable.generate_train_dict
        self.prev_img_handle = None
        self.previous_shape = None
        super().__init__(model=model)

    def train(self, model_name, data_provider, save_every_n_steps=1000):
        for _ in self.utilize_session(model_name=model_name, data_provider=data_provider,
                                      save_every_n_steps=save_every_n_steps):
            pass

    @SessionHook
    def _train_once(self, session_handler, input_feature):
        feed_dict = self.generate_train_dict(input_features=input_feature)
        feed_dict.update({self.model.learning_rate: self.learning_rate})

        if self.prev_img_handle is None or input_feature.shape[0] != self.previous_shape:
            mean = 0.5
            std = 0.3
            previous_img = np.random.normal(loc=mean, scale=2*std,
                                            size=(input_feature.shape[0], *self.model.img_size, 3))
        else:
            previous_img = self.prev_img_handle

        feed_dict.update({self.model.previous_img: previous_img})
        step, self.prev_img_handle = session_handler.training_step(feed_dict=feed_dict,
                                                                   additional_ops=[self.model.gen_handle])

        self.previous_shape = input_feature.shape[0]
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
