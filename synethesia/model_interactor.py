import time
import abc

from session_handler import SessionHandler


def time_diff(start_time):
    m, s = divmod(time.time() - start_time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return "%d:%02d:%02d:%02d" % (d, h, m, s)


class TrainingSession(object):

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
                                                     learning_rate=learning_rate,
                                                     batch_size=data_provider.batch_size)
                step, _ = session_handler.training_step(feed_dict=feed_dict)

                if step % save_every_n_steps == 0 and step > 0:
                    session_handler.save(step=step)
                    print(f"Step {step}, time: {time_diff(start_time)}: Saving in {pth}")
    # TODO integrate validation


class InferenceSession(object):

    def __init__(self, model, generate_inference_dict):
        self.model = model
        self.generate_inference_dict = generate_inference_dict

    def infer(self, model_name, data_provider):
        start_time = time.time()

        with SessionHandler(model=self.model, model_name=self.model_name) as session_handler:
            session_handler.load_weights_or_init()

            for input_feature in data_provider:
                feed_dict = self.generate_inference_dict(input_features=input_feature,
                                                         batch_size=data_provider.batch_size)
                results = session_handler.inference_step(feed_dict=feed_dict)

                yield results
