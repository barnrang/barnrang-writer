import tensorflow as tf

class RNN(object):
    """docstring for RNN."""
    def __init__(self, num_layers = 2):
        super(RNN, self).__init__()
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
