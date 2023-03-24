# coding=utf-8
# Copyright (c) 2021 Alibaba PAI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
import tensorflow.compat.v1 as tf
from abc import ABCMeta, abstractmethod
import six

__all__ = ['apply_grad_processors', 'ProxyOptimizer',
           'PostProcessOptimizer', 'VariableAssignmentOptimizer',
           'AccumGradOptimizer']

def HIDE_DOC(func):
    func.__HIDE_SPHINX_DOC__ = True
    return func


def get_tf_version_tuple():
    """
    Return TensorFlow version as a 2-element tuple (for comparison).
    """
    return tuple(map(int, tf.__version__.split('.')[:2]))


@six.add_metaclass(ABCMeta)
class GradientProcessor(object):
    """
    Base class for all gradient processors.
    Gradient processors can be applied to optimizers by
    :func:`optimizer.apply_grad_processors`.

    Subclass should override the ``_process()`` method.
    """
    _name_scope = None

    def process(self, grads):
        """
        Process the symbolic gradients.

        Args:
            grads (list): list of (grad, var).
        Returns:
            list: processed gradients, with the same type as input.
        """

        # reuse the old name_scope, if process() is called multiple times
        if self._name_scope is None:
            with tf.name_scope(type(self).__name__) as scope:
                self._name_scope = scope
                return self._process(grads)
        else:
            with tf.name_scope(self._name_scope):
                return self._process(grads)

    @abstractmethod
    def _process(self, grads):
        pass


class FilterNoneGrad(GradientProcessor):
    """
    Skip the update and print a warning (instead of crashing),
    when the gradient of certain variable is None.
    """
    def __init__(self, verbose=True):
        """
        Args:
            verbose (bool): whether to print warning about None gradients.
        """
        super(FilterNoneGrad, self).__init__()
        self._verbose = verbose

    def _process(self, grads):
        g = []
        to_print = []
        for grad, var in grads:
            if grad is None:
                to_print.append(var.op.name)
            else:
                g.append((grad, var))
        if self._verbose and len(to_print):
            message = ', '.join(to_print)
            logger.warn("No gradient w.r.t {} trainable variables: {}".format(len(to_print), message))
        return g

class ProxyOptimizer(tf.train.Optimizer):
    """
    A transparent proxy which delegates all methods of :class:`tf.train.Optimizer`
    """
    def __init__(self, opt, name='ProxyOptimizer'):
        assert isinstance(opt, tf.train.Optimizer), opt
        super(ProxyOptimizer, self).__init__(False, name)
        self._opt = opt

    @HIDE_DOC
    def compute_gradients(self, *args, **kwargs):
        return self._opt.compute_gradients(*args, **kwargs)

    @HIDE_DOC
    def get_slot(self, *args, **kwargs):
        return self._opt.get_slot(*args, **kwargs)

    @HIDE_DOC
    def get_slot_names(self, *args, **kwargs):
        return self._opt.get_slot_names(*args, **kwargs)

    @HIDE_DOC
    def apply_gradients(self, *args, **kwargs):
        return self._opt.apply_gradients(*args, **kwargs)


def apply_grad_processors(opt, gradprocs):
    """
    Wrapper around optimizers to apply gradient processors.
    Args:
        opt (tf.train.Optimizer):
        gradprocs (list[GradientProcessor]): gradient processors to add to the
            optimizer.
    Returns:
        a :class:`tf.train.Optimizer` instance which runs the gradient
        processors before updating the variables.
    """
    assert isinstance(gradprocs, (list, tuple)), gradprocs
    for gp in gradprocs:
        assert isinstance(gp, GradientProcessor), gp

    class _ApplyGradientProcessor(ProxyOptimizer):
        def __init__(self, opt, gradprocs):
            self._gradprocs = gradprocs[:]
            super(_ApplyGradientProcessor, self).__init__(opt)

        def apply_gradients(self, grads_and_vars,
                            global_step=None, name=None):
            g = self._apply(grads_and_vars)
            return self._opt.apply_gradients(g, global_step, name)

        def _apply(self, g):
            for proc in self._gradprocs:
                g = proc.process(g)
            return g

    return _ApplyGradientProcessor(opt, gradprocs)


class PostProcessOptimizer(ProxyOptimizer):
    """
    An optimizer which applies some "post-processing operation" per variable
    (e.g. clipping, quantization) after the gradient update.
    """
    def __init__(self, opt, func, colocate=True):
        """
        Args:
            opt (tf.train.Optimizer):
            func (tf.Variable -> tf.Operation or None): the operation needed
                to perform for this variable after the gradient update.
            colocate (boolean): colocate the function with the variable. No effect since TF 1.13.
        """
        super(PostProcessOptimizer, self).__init__(opt)
        self._func = func
        self._colocate = colocate

    @HIDE_DOC
    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        update_op = super(PostProcessOptimizer, self).apply_gradients(
            grads_and_vars, global_step)
        ops = []
        with tf.control_dependencies([update_op]):
            for _, var in grads_and_vars:
                with self._maybe_colocate(var):
                    op = self._func(var)
                    if op is not None:
                        assert isinstance(op, tf.Operation), op
                        ops.append(op)
        update_op = tf.group(update_op, *ops, name=name)
        return update_op

    @contextmanager
    def _maybe_colocate(self, var):
        G = tf.get_default_graph()
        if self._colocate and get_tf_version_tuple() <= (1, 12):
            with G.colocate_with(var):
                yield
        else:
            yield


class VariableAssignmentOptimizer(PostProcessOptimizer):
    """
    An optimizer which assigns each variable a new value (e.g. clipping,
    quantization) after the gradient update.
    """
    def __init__(self, opt, func):
        """
        Args:
            opt (tf.train.Optimizer):
            func (tf.Variable -> tf.Tensor or None): the new value to be
                assigned to this variable after the gradient update.
        """
        def f(v):
            t = func(v)
            if t is None:
                return t
            return tf.assign(v, t, use_locking=False).op
        super(VariableAssignmentOptimizer, self).__init__(opt, f)


class AccumGradOptimizer(ProxyOptimizer):
    """
    An optimizer which accumulates gradients across :math:`k` :meth:`minimize` executions,
    and apply them together in every :math:`k` th :meth:`minimize` execution.
    This is roughly the same as using a :math:`k` times larger batch size plus a
    :math:`k` times larger learning rate, but uses much less memory.
    This optimizer can be used in any TensorFlow code (with or without tensorpack).
    Example:
    .. code-block:: python
        from tensorpack.tfutils.optimizer import AccumGradOptimizer
        myopt = tf.train.GradientDescentOptimizer(0.01)
        myopt = AccumGradOptimizer(myopt, niter=5)
        train_op = myopt.minimize(loss)
    """

    def __init__(self, opt, niter):
        """
        Args:
            opt (tf.train.Optimizer): the underlying sub-optimizer.
            niter (int): number of iterations to accumulate gradients.
        """
        super(AccumGradOptimizer, self).__init__(opt, 'AccumGrad')
        self._niter = int(niter)

    def _create_accum_slots(self, var_list):
        slots = []
        for v in var_list:
            # TODO an option to not colocate the accumulators with variables (to save more memory)
            s = self._zeros_slot(v, "accum", self._name)
            slots.append(s)
        return slots

    @HIDE_DOC
    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        tf.logging.info("Call apply gradients....")
        grads_and_vars = FilterNoneGrad().process(grads_and_vars)
        vs = []
        for g, v in grads_and_vars:
            assert isinstance(g, (tf.Tensor, tf.IndexedSlices)) and isinstance(v, tf.Variable), \
                "AccumGradOptimizer does not work for the gradient of {}! " \
                "Types of v and g are {} and {}".format(v.op.name, type(v), type(g))
            vs.append(v)

        with tf.control_dependencies(None):
            slots = self._create_accum_slots(vs)
            slots_and_vars = [(s, gv[1]) for s, gv in zip(slots, grads_and_vars)]

            with tf.variable_scope(self._name), tf.device('/cpu:0'):
                counter = tf.Variable(
                    0, name="counter", trainable=False, dtype=tf.int32)

        with tf.name_scope('AccumGradOptimizer'):
            ops = []
            for s, gv in zip(slots, grads_and_vars):
                g, v = gv
                ops.append(s.assign_add(g))
            update_counter = tf.assign_add(counter, 1, name='update_counter')
            update_slot_op = tf.group(update_counter, *ops, name='update_slot')

            def update_grad():
                #update_op = self._opt.apply_gradients(slots_and_vars)
                # 30 grad tensors for each group.
                n = 30
                grouped_grads_and_vars = [slots_and_vars[i * n:(i + 1) * n] for i in range((len(slots_and_vars) + n - 1) // n)]
                train_ops = []
                for idx, grads_and_vars in enumerate(grouped_grads_and_vars):
                    if train_ops:
                        with tf.control_dependencies([train_ops[-1]]):
                            train_op = self._opt.apply_gradients(grads_and_vars, global_step=global_step if (idx == len(grouped_grads_and_vars) - 1) else None)
                    else:
                        train_op = self._opt.apply_gradients(grads_and_vars, global_step=global_step if (idx == len(grouped_grads_and_vars) - 1) else None)
                    train_ops.append(train_op)

                with tf.control_dependencies(train_ops):
                    clear_ops = [tf.assign(s, tf.zeros_like(s)) for s in slots]
                return tf.group(*clear_ops, name='update_grad')

            pred = tf.equal(tf.mod(counter, self._niter), 0)
            with tf.control_dependencies([update_slot_op]):
                if name is None:
                    name = 'cond_update_grad'
                op = tf.cond(pred, update_grad, tf.no_op)

            op = tf.identity(op, name=name).op
        return op


if __name__ == '__main__':
    # run it with "python -m tensorpack.tfutils.optimizer"

    x = tf.get_variable('x', shape=[6])
    cost = tf.reduce_sum(tf.abs(x), name='cost')
    opt = tf.train.GradientDescentOptimizer(0.01)
    opt = AccumGradOptimizer(opt, 5)
    min_op = opt.minimize(cost, global_step=tf.train.get_or_create_global_step())

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    with sess.as_default():
        for _ in range(20):
            min_op.run()
            print(x.eval())
        print(tf.train.get_or_create_global_step().eval())
