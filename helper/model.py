import tensorflow as tf

class RNN(object):
    """docstring for RNN.
    input: (T,N,L)
    T - time-step
    N - batch-size
    L - lenght(n_char)
    """
    def __init__(self,hidden_size,hid_layers = 2):
        super(RNN, self).__init__()
        def lstm_cell_mid():
            return tf.contrib.rnn.BasicLSTMCell(512)
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(hidden_size)
        if hid_layers > 1:
            self.stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell_mid() for _ in range(hid_layers)] + [lstm_cell()])

    def batchnorm(self, x, training=True):
        with tf.variable_scope("batch"):
            return tf.layers.batch_normalization(x, training=training)
    def forward(self,input_val, output_val, out_l, step, batch_size, training):
        state = self.stacked_lstm.zero_state(batch_size, tf.float32)
        out = tf.reshape(input_val[0],[1,batch_size,-1])
        loss = tf.Variable(0., tf.float32)
        for i in range(step):
            tmp, state = self.stacked_lstm(input_val[i], state)
            if i > 0:
                with tf.variable_scope("norm") as scope:
                    scope.reuse_variables()
                    tmp = self.batchnorm(tmp, training=training)
            else:
                with tf.variable_scope("norm") as scope:
                    tmp = self.batchnorm(tmp, training=training)
            out = tf.concat([out, tf.reshape(tmp,[1,batch_size,-1])],0)
            if i >= step - out_l:
                loss += tf.reduce_mean(tf.losses.softmax_cross_entropy(output_val[i-(step-out_l)], tmp))
        return tf.slice(out,[step-out_l,0,0],[out_l,-1,-1]), loss/step
    def tensorOnehot(self, x):
        '''
        x:(1,nchar)
        '''
        pos = tf.argmax(x,axis=1)
        return tf.sparse_tensor_to_dense(tf.SparseTensor([[0,pos[0]]],[1.],tf.cast(tf.shape(x),dtype=tf.int64)))

    def test_forward(self, input_val, out_l, T, nchar):
        '''
        input_val: (T,1,nchar)
        '''
        state = self.stacked_lstm.zero_state(1, tf.float32)
        out = tf.reshape(input_val[0],[1,1,nchar])
        tmp = None
        for i in range(T):
            tmp, state = self.stacked_lstm(input_val[i], state)
        inp = self.tensorOnehot(tmp)
        for i in range(out_l):
            tmp, state = self.stacked_lstm(inp, state)
            try:
                with tf.variable_scope("norm") as scope:
                    scope.reuse_variables()
                    tmp = self.batchnorm(tmp, training=False)
            except:
                with tf.variable_scope("norm") as scope:
                    tmp = self.batchnorm(tmp, training=False)
            out = tf.concat([out, tf.reshape(tmp,[1,1,-1])],0)
            inp = self.tensorOnehot(tmp)
        return out
