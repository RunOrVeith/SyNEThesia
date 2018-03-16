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
        self.model = model

    @abc.abstractmethod
    def generate_train_feed_dict(self, learning_rate, input_feature, batch_size):
        pass

    def train(self, model_name, data_provider, learning_rate=0.0001, save_every_n_steps=1000):
        start_time = time.time()

        with SessionHandler(model=self.model, model_name=model_name) as session_handler:
            session_handler.load_weights_or_init()

            for input_feature in data_provider:
                feed_dict = self.generate_train_feed_dict(learning_rate=learning_rate,
                                                          input_feature=input_feature,
                                                          batch_size=data_provider.batch_size)
                step, _ = session_handler.training_step(feed_dict=feed_dict)

                if step % save_every_n_steps == 0 and step > 0:
                    session_handler.save(step=step)
                    print(f"Step {step}, time: {time_diff(start_time)}: Saving in {pth}")

    # TODO integrate validation and inference
