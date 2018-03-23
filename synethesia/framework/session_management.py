from pathlib import Path
import abc
import inspect
import time

import tensorflow as tf


class SessionHandler(object):

    def __init__(self, model, model_name, checkpoint_dir="./checkpoints", logdir="./logs", max_saves_to_keep=5):

        if not isinstance(model, Model):
            raise ValueError(f"Model must be of type 'Model', not {type(model)}")

        self.model_name = model_name
        self._checkpoint_dir = checkpoint_dir
        self._logdir = logdir
        self.max_saves_to_keep = max_saves_to_keep

        self.model = model

        self._graph = None
        self._session = None
        self._saver = None
        self._running_model = None
        self._summary_writer = None

        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

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
    def checkpoint_file(self):
        return str((Path(self._checkpoint_dir) / self.model_name) / "checkpoint.ckpt")

    @property
    def log_dir(self):
        return str(Path(self._logdir) / self.model_name)

    def __enter__(self):
        self._graph = tf.Graph()
        self._graph.as_default().__enter__()
        self.model.initialize()
        # TODO allow a debug session instead
        session = tf.Session().__enter__()
        summary_writer = tf.summary.FileWriter(self.log_dir, graph=self._graph)
        saver = tf.train.Saver(max_to_keep=self.max_saves_to_keep)
        self._session = session
        self._saver = saver
        self._summary_writer = summary_writer
        return self

    def __exit__(self, *args, **kwargs):
        self._session.__exit__(*args, **kwargs)
        # self._graph.as_default().__exit__(*args, **kwargs)
        # TODO find why this raises a Runtime exception, maybe need to get context manager of graph from
        # session before closing it?

    def training_step(self, feed_dict, additional_ops=()):
        ops_to_run = [self.model.training_summary, self.model.optimizer]
        ops_to_run.extend(additional_ops)
        results = self.session.run(ops_to_run, feed_dict=feed_dict)
        summary = results[0]
        step = self.step
        self.summary_writer.add_summary(summary, step)

        return (step, results[2:]) if additional_ops else (step, None)

    def inference_step(self, feed_dict, additional_ops=()):
        ops_to_run = [self.model.data_output]
        ops_to_run.extend(additional_ops)
        results = self.session.run(ops_to_run, feed_dict=feed_dict)
        return results if additional_ops else results[0]

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


class SessionHook():

    def __init__(self, f, p=None, y=None):
        self.f = f
        self.p = (lambda _: ()) if p is None else p
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
            return lambda **_: None
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
        print("Utilizing session")
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

                    yield hook.get_yieldables(self=self, **available)
