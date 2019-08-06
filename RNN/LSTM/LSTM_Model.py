import tensorflow as tf

TRAIN_PROB = 0.5
PREDICT_PROB = 1.0

FC_NODE = [10, 10, 5, 5]

# tf.reset_default_graph()


class LSTM_Model(object):
    def __init__(self, n_steps, n_inputs, n_hid_units, n_outputs, forget_bias, leraning_rate, isRestore = False, model_name = None):
        
        config = tf.ConfigProto() # 設置 session 運行參數
#         config.gpu_options.per_process_gpu_memory_fraction = 0.4 # 佔用 40% 顯存
        self.sess = tf.Session(config = config)
        
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hid_units = n_hid_units
        
        self.X = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.Y = tf.placeholder(tf.float32, [None, self.n_outputs])
        self.prob = tf.placeholder_with_default(1.0, shape=())
        
#         with tf.variable_scope('RNN', reuse = reuse):
        self.output = self.Build_Model_1(X = self.X,
                                         forget_bias = forget_bias,
                                         activation = tf.nn.tanh,
                                         initializer = tf.contrib.layers.xavier_initializer())

        self.loss, self.train_op = self.Train_Model_1(self.Y, self.output, leraning_rate)
        self.softmax_out, self.labelIndex, self.predictIndex, self.accuracy = self.Evaluate_Model(self.Y, self.output)
        
        self.saver = tf.train.Saver()
        if isRestore:
            self.saver.restore(self.sess, model_name)
        else:
            self.sess.run(tf.global_variables_initializer())

            
    def Build_Model_1(self, X, forget_bias, activation, initializer):
        weights = tf.get_variable('rnn_w', [self.n_hid_units, self.n_outputs], initializer = initializer)
        biases = tf.get_variable('rnn_b', [self.n_outputs, ], initializer = initializer)
        
        rnn, outputs = self.RNN_LSTM(X, weights, biases, forget_bias, activation)
        
        fc_node = FC_NODE

        dense = tf.layers.dense(inputs = outputs[-1], 
                                units = fc_node[0], 
                                activation = activation,
                                kernel_initializer = initializer,
                                bias_initializer = initializer)

        for n in fc_node[1:]:
            dense = tf.layers.dense(inputs = dense, 
                                    units = n, 
                                    activation = activation,
                                    kernel_initializer = initializer,
                                    bias_initializer = initializer)
        
        out = tf.layers.dense(inputs=dense, units=self.n_outputs, activation = activation)
        
        return out

        
    def RNN_LSTM(self, X, weight, bias, forget_bias, activation, num_layers = 2, **params):
#         lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hid_units, forget_bias=forget_bias, state_is_tuple=True)
        
        lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.n_hid_units, forget_bias=forget_bias,
                                                          dropout_keep_prob = self.prob,
                                                          activation = activation)
        
#         lstm_cell = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(self.n_hid_units,
#                                                                   forget_bias=forget_bias,
#                                                                   use_peepholes = True,
#                                                                   proj_clip = 15.0,
#                                                                   initializer = tf.contrib.layers.xavier_initializer(),
#                                                                   activation = activation)

        init_state = lstm_cell.zero_state(tf.shape(X)[0], dtype=tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X, initial_state=init_state)
        
        outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
        
        result = tf.matmul(outputs[-1], weight) + bias

        return tf.matmul(outputs[-1], weight) + bias, outputs

    
    def Train_Model_1(self, label, prediction, leraning_rate):
        loss = tf.losses.softmax_cross_entropy(onehot_labels = label, logits = prediction)
        op = tf.train.AdamOptimizer(leraning_rate).minimize(loss)
        
        return loss, op

    
    def Evaluate_Model(self, label, prediction): 
        softmax = tf.nn.softmax(prediction)
        labIndex = tf.argmax(label, axis = 1)
        preIndex = tf.argmax(prediction, axis = 1)
        
        correct = tf.equal(labIndex, preIndex)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
        return softmax, labIndex, preIndex, accuracy
    
    
    def Training(self, train_x, train_y):       
        batches = len(train_x)
        
        for i in range(batches):
            tra_x = train_x[i]
            tra_y = train_y[i]
            
            tra_x = self.sess.run(tra_x)
            tra_y = self.sess.run(tra_y)
            
            self.sess.run(self.train_op, feed_dict = {self.X:tra_x, self.Y:tra_y, self.prob : TRAIN_PROB})
            
            loss, accuracy = self.sess.run((self.loss, self.accuracy), feed_dict = {self.X:tra_x, 
                                                                                    self.Y:tra_y, 
                                                                                    self.prob : TRAIN_PROB})
            
        return loss, accuracy
        
        
    def Testing(self, test_x, test_y):
        test_x = self.sess.run(test_x)
        test_y = self.sess.run(test_y)
            
        loss, accuracy = self.sess.run((self.loss, self.accuracy), feed_dict = {self.X:test_x, 
                                                                                self.Y:test_y, 
                                                                                self.prob : PREDICT_PROB})
        
        return loss, accuracy
    
    
    def Predicting(self, test_x):
        test_x = self.sess.run(test_x)

        predict, prob = self.sess.run((self.predictIndex, self.softmax_out), feed_dict = {self.X : test_x, 
                                                                                          self.prob : PREDICT_PROB})
        
        return predict, prob
    
    
    def Save_Model(self, file_name):
        self.saver.save(self.sess, file_name)
#         self.saver.save(self.sess, file_name, write_meta_graph=False)
        
        
    def Close(self):
        self.sess.close()
        tf.reset_default_graph()
