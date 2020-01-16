#Referred:  http://n-s-f.github.io/2017/07/10/rnn-tensorflow.html
#           https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/RNNCell
#           http://theanets.readthedocs.io/en/stable/api/generated/theanets.layers.recurrent.SCRN.html
#           Mikolov, T., Joulin, A., Chopra, S., Mathieu, M., & Ranzato, M. A. (2014). Learning longer memory in recurrent neural networks. arXiv preprint arXiv:1412.7753.

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
import collections

#This is same structure to the LSTMStateTuple
_SCRNStateTuple = collections.namedtuple("SCRNStateTuple", ("s", "h"))
class SCRNStateTuple(_SCRNStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        (s, h) = self
        if s.dtype != h.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" % (str(s.dtype), str(h.dtype)))
        return s.dtype

class SCRNCell(RNNCell):
    def __init__(
        self,
        num_units,
        initializer= None,
        leakage = 0.95,
        use_bias = False,
        bias_initializer = tf.zeros_initializer,
        activation=None,    #Baisc is tanh. Paper used softmax, but softmax cannot be used before the projection.
        state_is_tuple=True,
        reuse=None,
        name=None
        ):
        super(SCRNCell, self).__init__(_reuse=reuse, name=name)
        self._num_units = num_units
        self._initializer= initializer
        self._leakage = leakage
        self._use_bias = use_bias
        self._bias_initializer = bias_initializer
        self._activation = activation or tf.nn.tanh
        self._reuse = reuse
        self._name = name

        if state_is_tuple:
            self._state_size = SCRNStateTuple(num_units, num_units)
        else:
            self._state_size = num_units * 2
        self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):
        input_Size = inputs.get_shape().with_rank(2)[1]

        with tf.variable_scope(self._name or type(self).__name__):
            if input_Size.value is None:
                raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

            weights_X_SH = tf.get_variable( #A,B
                name="weights_X_to_SH",
                shape=[input_Size, self._num_units * 2],
                initializer=self._initializer
                )
            weights_S_H = tf.get_variable(  #P
                name="weights_S_to_H",
                shape=[self._num_units, self._num_units],
                initializer=self._initializer
                )

            weights_H_H = tf.get_variable( #R
                name="weights_H_to_H",
                shape=[self._num_units, self._num_units],
                initializer=self._initializer
                )

            weights_SH_Y = tf.get_variable( #U,V
                name="weights_SH_to_Y",
                shape=[self._num_units * 2, self._num_units],
                initializer=self._initializer
                )

            s_from_X, h_from_X = tf.split(
                value = tf.matmul(inputs, weights_X_SH),
                num_or_size_splits= 2,
                axis = 1
                )
            s = (1 - self._leakage) * s_from_X + self._leakage * state.s
            h = tf.nn.sigmoid(
                h_from_X +
                tf.matmul(s, weights_S_H) +
                tf.matmul(state.h, weights_H_H)
                )

            output = tf.matmul(tf.concat([s,h], axis=1), weights_SH_Y)

            if self._use_bias:
                bias = tf.get_variable( #there is a bias in theanet, not the paper.
                    name="bias_Y",
                    shape=[1, self._num_units],
                    initializer=self._bias_initializer
                    )
                output += bias

            output = self._activation(output)

            new_State = SCRNStateTuple(s=s, h=h)

        return output, new_State
