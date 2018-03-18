from pathlib import Path

import tensorflow as tf

from interfaces import Model


class SessionHandler(object):

    def __init__(self, model, model_name, checkpoint_dir="./checkpoints", logdir="./logs"):

        if not isinstance(model, Model):
            raise ValueError(f"Model must be of type 'Model', not {type(model)}")

        self.model_name = model_name
        self._checkpoint_dir = checkpoint_dir
        self._logdir = logdir

        self.model = model

        self._graph = None
        self._session = None
        self._saver = None
        self._running_model = None
        self._summary_writer = None

    def _raise_on_uninitialized(func):

        def _assert_initialization(self, *args, **kwargs):
            if (self._session is None or self._saver is None
            or self._summary_writer is None or self._graph is None):
               raise AttributeError("Can not use SessionHandler without active context manager.")
            return func(self, *args, **kwargs)

        return _assert_initialization

    @property
    @_raise_on_uninitialized
    def graph(self):
        return self._graph

    @property
    @_raise_on_uninitialized
    def session(self):
        return self._session

    @property
    @_raise_on_uninitialized
    def saver(self):
        return self._saver

    @property
    @_raise_on_uninitialized
    def running_model(self):
        return self._running_model

    @property
    @_raise_on_uninitialized
    def summary_writer(self):
        return self._summary_writer

    @property
    def step(self):
        return tf.train.global_step(sess=self.session, global_step_tensor=self.model.global_step)

    @property
    def checkpoint_dir(self):
        return str(Path(self._checkpoint_dir) / self.model_name)

    @property
    def log_dir(self):
        return str(Path(self._logdir) / self.model_name)

    def __enter__(self):
        self._graph = tf.Graph()
        self._graph.as_default().__enter__()
        self.model.initialize()
        session = tf.Session().__enter__()
        summary_writer = tf.summary.FileWriter(self.log_dir)
        saver = tf.train.Saver(max_to_keep=5)
        self._session = session
        self._saver = saver
        self._summary_writer = summary_writer
        return self

    def __exit__(self, *args, **kwargs):
        self._session.__exit__(*args, **kwargs)
        # self._graph.as_default().__exit__(*args, **kwargs)  # TODO find why this raises a Runtime exception

    def training_step(self, feed_dict, additional_ops=()):
        ops_to_run = [self.model.training_summary, self.model.optimizer].extend(additional_ops)
        results = self.session.run(ops_to_run, feed_dict=feed_dict)
        summary = results[0]
        step = self.step
        self.summary_writer.add_summary(summary, step)
        if len(additional_ops) > 0:
            return step, results[2:]
        return step, None

    def inference_step(self, feed_dict, additional_ops=()):
        ops_to_run = [self.model.data_output].extend(additional_ops)
        results = self.session.run(ops_to_run, feed_dict=feed_dict)
        return results

    def save(self, step=None):
        step = self.step if step is None else step
        pth = self.saver.save(self.session, self.checkpoint_file, step)
        return pth

    def load_weights_or_init(self):
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(f"Loading existing model {self.model_name} from {self.checkpoint_dir}")
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
        else:
            print(f"Initializing new model {self.model_name}")
            self.session.run(tf.global_variables_initializer())


if __name__ == "__main__":
    from model import SynethesiaModel
    with SessionHandler(model=SynethesiaModel(64), model_name="synethesia") as sess_handler:
        sess_handler.load_weights_or_init()
