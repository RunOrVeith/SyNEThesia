import tensorflow as tf


class SessionHandler(object):

    def __init__(self, model, model_name, checkpoint_dir="./checkpoints", logdir="./logs"):

        self.model_name = model_name
        self._checkpoint_dir = checkpoint_dir
        self._logdir = logdir

        self.model = model

        self._session = None
        self._saver = None
        self._running_model = None
        self._summary_writer = None


    def _raise_on_uninitialized(func):
        def assert_initialization(self, *args, **kwargs):
            if (self._session is None or self._saver is None
            or self._running_model is None or self._summary_writer is None):
               raise AttributeError("Can not use SessionHandler without active context manager.")
            return func(*args, **kwargs)
        return assert_initialization

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
    def checkpoint_dir(self):
        return str(Path(self._checkpoint_dir) / self.model_name)

    @property
    def logdir(self):
        return str(Path(self._logdir) / self.model_name)

    def __enter__(self):
        graph = tf.Graph()
        session = tf.Session(graph=graph)
        running_model = self.model(graph=graph, **kwargs)
        summary_writer = tf.summary.FileWriter(self.log_dir, graph)
        saver = tf.train.Saver(max_to_keep=5)
        self._session = session
        self._saver = saver
        self._running_model = running_model
        self._summary_writer = summary_writer
        return self

    def __exit__(self, *args, **kwargs):
        return self._session.__exit(*args, **kwargs)

    @_raise_on_uninitialized
    def load_weights_or_init(self):
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(f"Loading existing model {self.model_name} from {self.checkpoint_dir}")
            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print(f"Initializing new model {self.model_name}")
            sess.run(tf.global_variables_initializer())


if __name__ == "__main__":
    from model import SynethesiaModel
    with SessionHandler(model=SynethesiaModel, model_name="synethesia") as sess_handler:
        sess_handler.load_weights_or_init()
