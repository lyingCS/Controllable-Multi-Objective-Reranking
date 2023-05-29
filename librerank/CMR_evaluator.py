from librerank.reranker import *
from tensorflow.contrib import layers
from tensorflow.python.ops import rnn
from librerank.prada_util_attention import *


class CMR_evaluator(EGR_evaluator):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                 profile_num, max_norm=None):
        super(CMR_evaluator, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                            itm_spar_num, itm_dens_num, profile_num, max_norm)
        with self.graph.as_default():
            # add model components
            self.all_feature_concatenation = None
            self.sum_pooling_layer = None
            self.concatenation_layer = None
            self.multi_head_self_attention_layer = None
            self.rnn_layer = None
            self.pair_wise_comparison_layer = None
            self.name = 'CMR_evaluator'
            self.label_type = 'zero_one'
            self.feature_batch_norm = True
            self.N = self.item_size = self.pv_size = self.max_time_len
            self.use_BN = True
            self.dnn_hidden_units = [512, 256, 128]

            self.enc_input = tf.concat([self.itm_enc_input, tf.tile(self.usr_enc_input, [1, self.item_size, 1])],
                                       axis=-1)
            # self.all_feature_concatenation = self.enc_input
            self.is_training = tf.placeholder(tf.bool)
            self.batch_size = tf.shape(self.itm_enc_input)[0]
            self.score_format = 'iv'

            self.build_model()

    def build_loss(self):
        with tf.name_scope("CMR_evaluator_Loss_Op"):
            if self.score_format == 'pv':
                loss_weight = tf.ones([self.batch_size, 1])  # [B,1]
                if self.label_type == "total_num":  # label_ph: [B, N(0 or 1)]
                                                      loss_weight = tf.reduce_sum(self.label_ph, axis=1)
                                                      loss_weight = tf.where(loss_weight > 1, loss_weight, tf.ones_like(loss_weight))  # [B,1]
                # label = tf.reshape(tf.minimum(tf.reduce_sum(self.label_ph, axis=1), 1.0), [-1, 1])  # [B,1]
                label = tf.reshape(tf.reduce_sum(self.label_ph, axis=1), [-1, 1])  # [B,1]
                self.print_loss = tf.print("label: ", tf.reshape(label, [1, -1]),
                                        "\nlogits", tf.reshape(self.logits, [1, -1]),
                                        "\nb_logits", self.before_sigmoid,
                                        summarize=-1, output_stream=sys.stderr)
                self.loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits,
                    labels=label)
                self.loss = self.loss * loss_weight  # [B,1]
                self.loss = tf.reduce_mean(self.loss)
                self.gap = self.loss
            elif self.score_format == 'iv':
                self.loss = tf.losses.log_loss(self.label_ph, self.logits)
                self.gap = self.loss

        self.opt()

    def dnn_layer(self):
        dnn_layer = [self.sum_pooling_layer, self.concatenation_layer,
                     self.multi_head_self_attention_layer, self.rnn_layer, self.pair_wise_comparison_layer]
        dnn_layer = tf.concat(values=dnn_layer, axis=-1)
        if self.feature_batch_norm:
            with tf.variable_scope(name_or_scope="{}_Input_BatchNorm".format(self.name)):
                dnn_layer = tf.contrib.layers.batch_norm(dnn_layer, is_training=self.is_train, scale=True)
        self.dnn_input = dnn_layer
        self.final_neurons = self.get_dnn(self.dnn_input, self.dnn_hidden_units, [tf.nn.relu, tf.nn.relu, tf.nn.relu],
                                          "evaluator_dnn"),

    def build_model(self):
        # if self.restore_embedding and not self.is_local:
        #     self.feature_columns = self.setup_feature_columns()
        # self.embedding_layer()
        # self.reshape_input()
        self.feature_augmentation()
        # self.all_feature_concatenation = self.enc_input

        self.sum_pooling_channel()
        self.concatenation_channel()
        self.multi_head_self_attention_channel()
        self.rnn_channel()
        self.pair_wise_comparison_channel()

        self.dnn_layer()
        self.logits_layer()
        self.build_loss()



    def sum_pooling_channel(self):
        with tf.variable_scope(name_or_scope="{}_Sum_Pooling_Channel".format(self.name)):
            self.sum_pooling_layer = tf.reduce_sum(self.all_feature_concatenation, axis=1)

    def concatenation_channel(self):
        with tf.variable_scope(name_or_scope="{}_Concatenation_Channel".format(self.name)) as scope:
            running_layer = layers.fully_connected(
                self.all_feature_concatenation,
                16,
                tf.nn.relu,
                scope=scope,
                normalizer_fn=layers.batch_norm if self.use_BN else None,
                normalizer_params={"scale": True, "is_training": self.is_train})
            self.concatenation_layer = tf.reshape(running_layer,
                                                  [-1, self.pv_size * running_layer.get_shape().as_list()[2]])

    def multi_head_self_attention_channel(self):
        with tf.variable_scope(name_or_scope="{}_Multi_Head_Self_Attention_Channel".format(self.name)):
            shape_list = self.all_feature_concatenation.get_shape().as_list()
            all_feature_concatenation = tf.reshape(self.all_feature_concatenation, [-1, self.pv_size, shape_list[2]])
            queries = all_feature_concatenation
            keys = all_feature_concatenation
            mask = tf.cast(tf.ones_like(keys[:, :, 0]), dtype=tf.bool)
            outputs, _ = multihead_attention(queries=queries,
                                             keys=keys,
                                             num_heads=8,
                                             num_units=128,
                                             num_output_units=2 * 128,
                                             activation_fn="relu",
                                             scope="multi_head_att",
                                             atten_mode="ln",
                                             reuse=tf.AUTO_REUSE,
                                             key_masks=mask,
                                             query_masks=mask,
                                             is_target_attention=False)
            self.multi_head_self_attention_layer = tf.reduce_sum(outputs, axis=1)

    def rnn_channel(self):
        with tf.variable_scope(name_or_scope="{}_RNN_Channel".format(self.name)):
            # one can reverse self.all_feature_concatenation and make it a Bi-GRU
            encoder_cell = tf.nn.rnn_cell.GRUCell(64)
            rnn_inputs = tf.transpose(self.all_feature_concatenation, perm=[1, 0, 2])  # [N,B,E]
            rnn_inputs = tf.unstack(rnn_inputs, num=self.pv_size, axis=0)  # [B,E]*N
            outputs, final_state = rnn.static_rnn(encoder_cell, rnn_inputs, dtype=tf.float32)

            output = [tf.reshape(output, [-1, 1, encoder_cell.output_size]) for output in outputs]
            output = tf.concat(axis=1, values=output)
            self.rnn_layer = tf.reduce_sum(output, axis=1)

    def pair_wise_comparison_channel(self):
        with tf.variable_scope(name_or_scope="{}_Pair_Wise_Comparison_Channel".format(self.name)):
            input_transposed = tf.transpose(self.all_feature_concatenation, perm=[0, 2, 1])
            output = tf.matmul(self.all_feature_concatenation, input_transposed)
            self.pair_wise_comparison_layer = tf.reshape(output, [-1, self.pv_size * self.pv_size])

    def get_dnn(self, x, layer_nums, layer_acts, name="dnn"):
        input_ft = x
        assert len(layer_nums) == len(layer_acts)
        with tf.variable_scope(name):
            for i, layer_num in enumerate(layer_nums):
                input_ft = tf.contrib.layers.fully_connected(
                    inputs=input_ft,
                    num_outputs=layer_num,
                    scope='layer_%d' % i,
                    activation_fn=layer_acts[i],
                    reuse=tf.AUTO_REUSE)
        return input_ft

    def logits_layer(self):
        # with tf.variable_scope(name_or_scope="{}_Logits".format(self.name)) as dnn_logits_scope:
        #     logits = layers.linear(self.final_neurons, 1, scope=dnn_logits_scope)
        if self.score_format == 'pv':
            logits = layers.linear(self.final_neurons, 1)
            self.before_sigmoid = logits
            logits = tf.sigmoid(logits)
            predictions = tf.reshape(logits, [-1, 1])  # [B,1]
            self.logits = predictions
        elif self.score_format == 'iv':
            logits = layers.linear(self.final_neurons, self.max_time_len)
            logits = tf.reshape(tf.nn.softmax(logits), [-1, self.max_time_len])
            seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
            predictions = seq_mask*logits
            self.logits = predictions
        return predictions

    def predict(self, usr_ft, item_spar_fts, item_dens_fts, seq_len):
        with self.graph.as_default():
            ctr_probs = self.sess.run(self.logits, feed_dict={
                self.usr_profile: np.reshape(usr_ft, [-1, self.profile_num]),
                self.itm_spar_ph: item_spar_fts.reshape([-1, self.max_time_len, self.itm_spar_num]),
                self.itm_dens_ph: item_dens_fts.reshape([-1, self.max_time_len, self.itm_dens_num]),
                self.seq_length_ph: seq_len,
                self.is_train: False,
                self.keep_prob: 1.0})
            return ctr_probs

    def train(self, batch_data, lr, reg_lambda, keep_prob=0.8, train_prefer=1):
        with self.graph.as_default():
            loss, _ = self.sess.run([self.loss, self.train_step], feed_dict={
                self.usr_profile: np.reshape(np.array(batch_data[1]), [-1, self.profile_num]),
                self.itm_spar_ph: batch_data[2],
                self.itm_dens_ph: batch_data[3],
                self.label_ph: batch_data[4],
                self.seq_length_ph: batch_data[6],
                self.lr: lr,
                self.reg_lambda: reg_lambda,
                self.keep_prob: keep_prob,
                self.is_train: True,
                self.controllable_auc_prefer: train_prefer,
                self.controllable_prefer_vector: [[train_prefer, 1 - train_prefer]],
            })
            return loss

    def eval(self, batch_data, reg_lambda, eval_prefer=0, keep_prob=1, no_print=True):
        with self.graph.as_default():
            pred, loss = self.sess.run([self.logits, self.loss], feed_dict={
                self.usr_profile: np.reshape(np.array(batch_data[1]), [-1, self.profile_num]),
                self.itm_spar_ph: batch_data[2],
                self.itm_dens_ph: batch_data[3],
                self.label_ph: batch_data[4],
                self.seq_length_ph: batch_data[6],
                self.reg_lambda: reg_lambda,
                self.keep_prob: keep_prob,
                self.is_train: False,
                self.controllable_auc_prefer: eval_prefer,
                self.controllable_prefer_vector: [[eval_prefer, 1 - eval_prefer]],
            })
            return pred.tolist(), loss
