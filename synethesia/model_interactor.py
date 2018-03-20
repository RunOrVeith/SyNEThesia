import time
import abc

from session_handler import SessionHandler


def time_diff(start_time):
    m, s = divmod(time.time() - start_time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return "%d:%02d:%02d:%02d" % (d, h, m, s)


class Trainable(object, metaclass=abc.ABCMeta):

    def __init__(self, model):
        self.training_session = TrainingSession(model=model, generate_train_dict=self.generate_train_dict)

    @abc.abstractmethod
    def generate_train_dict(self, learning_rate, input_features, batch_size):
        pass


class Inferable(object, metaclass=abc.ABCMeta):

    def __init__(self, model):
        self.inferece_session = InferenceSession(model=model, generate_inference_dict=self.generate_inference_dict)

    @abc.abstractmethod
    def generate_inference_dict(self, input_features, batch_size):
        pass


class TrainingSession(object):

    # TODO implement training validation and inference as same loop with customizable hooks

    def __init__(self, model, generate_train_dict):
        self.model = model
        self.generate_train_dict = generate_train_dict

    def train(self, model_name, data_provider, learning_rate=0.0001, save_every_n_steps=1000):
        with SessionHandler(model=self.model, model_name=model_name) as session_handler:
            session_handler.load_weights_or_init()
            start_time = time.time()
            step = session_handler.step
            print(f"{'Resuming' if step > 0 else 'Starting'} to train {model_name}: at step {step}")

            for input_feature in data_provider:
                feed_dict = self.generate_train_dict(input_features=input_feature,
                                                     learning_rate=learning_rate)
                step, _ = session_handler.training_step(feed_dict=feed_dict)

                if step % save_every_n_steps == 0 and step > 0:
                    session_handler.save(step=step)
                    print(f"Step {step}, time: {time_diff(start_time)}: Saving in {session_handler.checkpoint_dir}")


class InferenceSession(object):

    def __init__(self, model, generate_inference_dict):
        self.model = model
        self.generate_inference_dict = generate_inference_dict

    def infer(self, model_name, data_provider):
        start_time = time.time()

        with SessionHandler(model=self.model, model_name=model_name) as session_handler:
            session_handler.load_weights_or_init()

            for input_feature in data_provider:

                feed_dict = self.generate_inference_dict(input_features=input_feature)
                results = session_handler.inference_step(feed_dict=feed_dict)

                yield results
