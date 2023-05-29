import itertools
import sys

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, static_bidirectional_rnn, LSTMCell, MultiRNNCell
import numpy as np
import heapq


def tau_function(x):
    return tf.where(x > 0, tf.exp(x), tf.zeros_like(x))


def attention_score(x):
    return tau_function(x) / tf.add(tf.reduce_sum(tau_function(x), axis=1, keepdims=True), 1e-20)


class BaseModel(object):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                 profile_num, max_norm=None, acc_prefer=1.0, is_controllable=False):
        # reset graph
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            # input placeholders
            with tf.name_scope('inputs'):
                self.acc_prefer = acc_prefer
                self.is_controllable = is_controllable
                self.controllable_prefer_vector = tf.placeholder(tf.float32, [1, 2])
                self.itm_spar_ph = tf.placeholder(tf.int32, [None, max_time_len, itm_spar_num], name='item_spar')
                self.itm_dens_ph = tf.placeholder(tf.float32, [None, max_time_len, itm_dens_num], name='item_dens')
                self.usr_profile = tf.placeholder(tf.int32, [None, profile_num], name='usr_profile')
                # self.usr_spar_ph = tf.placeholder(tf.int32, [None, max_seq_len, hist_spar_num], name='user_spar')
                # self.usr_dens_ph = tf.placeholder(tf.float32, [None, max_seq_len, hist_dens_num], name='user_dens')
                self.seq_length_ph = tf.placeholder(tf.int32, [None, ], name='seq_length_ph')
                # self.hist_length_ph = tf.placeholder(tf.int32, [None, ], name='hist_length_ph')
                self.label_ph = tf.placeholder(tf.float32, [None, max_time_len], name='label_ph')
                # self.time_ph = tf.placeholder(tf.float32, [None, max_seq_len], name='time_ph')
                self.is_train = tf.placeholder(tf.bool, [], name='is_train')
                self.cate_id = tf.placeholder(tf.int32, [None, max_time_len], name='cate_id')

                # controllable_auc_prefer
                self.controllable_auc_prefer = tf.placeholder(tf.float32, [])
                # lr
                self.lr = tf.placeholder(tf.float32, [])
                # reg lambda
                self.reg_lambda = tf.placeholder(tf.float32, [])
                # keep prob
                self.keep_prob = tf.placeholder(tf.float32, [])
                self.max_time_len = max_time_len
                # self.max_seq_len = max_seq_len
                self.hidden_size = hidden_size
                self.emb_dim = eb_dim
                self.itm_spar_num = itm_spar_num
                self.itm_dens_num = itm_dens_num
                # self.hist_spar_num = hist_spar_num
                # self.hist_dens_num = hist_dens_num
                self.profile_num = profile_num
                self.max_grad_norm = max_norm
                self.ft_num = itm_spar_num * eb_dim + itm_dens_num
                self.feature_size = feature_size
                self.epsilon = 0.00001
                self.augment_feature_normalization = "divide_mean"

            # embedding
            with tf.name_scope('embedding'):
                self.emb_mtx = tf.get_variable('emb_mtx', [feature_size + 1, eb_dim],
                                               initializer=tf.truncated_normal_initializer)
                self.itm_spar_emb = tf.gather(self.emb_mtx,
                                              self.itm_spar_ph)  # [? ,10(max_time_len), 5(itm_spar_num), 16(eb_dim)]
                # self.usr_spar_emb = tf.gather(self.emb_mtx, self.usr_spar_ph)
                self.usr_prof_emb = tf.gather(self.emb_mtx, self.usr_profile)  # [?, 8(profile_num), 16(eb_dim)]

                self.item_seq = tf.concat(
                    [tf.reshape(self.itm_spar_emb, [-1, max_time_len, itm_spar_num * eb_dim]), self.itm_dens_ph],
                    axis=-1)  # [?, 10, ft_num]

                self.usr_seq = tf.reshape(self.usr_prof_emb, [-1, profile_num * eb_dim])

                self.itm_enc_input = tf.reshape(self.item_seq, [-1, self.max_time_len, self.ft_num])  # [B, N, ft_num]
                self.usr_enc_input = tf.reshape(self.usr_seq, [-1, 1, self.profile_num * self.emb_dim])

    def divide_mean_normalization(self, raw_feature):
        raw_feature_mean = tf.reduce_mean(raw_feature, axis=1, keep_dims=True)  # (B, 1, emb_dims)
        matrix_f_global = tf.divide(raw_feature, raw_feature_mean + self.epsilon)  # (B, N, emb_dims)
        return matrix_f_global

    def min_max_normalization(self, raw_feature, tensor_global_min_tile, tensor_global_max_tile):
        matrix_f_global = tf.where(tensor_global_max_tile - tensor_global_min_tile < self.epsilon,
                                   tf.fill(tf.shape(raw_feature), 0.5),
                                   tf.div(tf.subtract(raw_feature, tensor_global_min_tile),
                                          tf.subtract(tensor_global_max_tile,
                                                      tensor_global_min_tile) + self.epsilon))  # (B, N, emb_dims)
        return matrix_f_global

    def feature_augmentation(self):
        with tf.variable_scope(name_or_scope="Feature_Augmentation"):
            if False:
                block_layer_dict_augmented = self.augment_context_features_with_mask(self.itm_dens_ph)
                # self.ccc, self.ddd = self.embed_list[0], self.augment_context_features_with_mask(self.embed_list[0])

                self.all_feature_concatenation = tf.concat(values=block_layer_dict_augmented, axis=-1)
                self.all_feature_concatenation = tf.concat([self.itm_spar_emb, self.all_feature_concatenation], axis=-1)
                self.all_feature_concatenation = tf.nn.tanh(self.all_feature_concatenation, name='FEATURE_TANH')
            else:
                self.all_feature_concatenation = self.itm_enc_input

            # position feature.
            self.batch_size = tf.shape(self.all_feature_concatenation)[0]
            position_feature = self.get_position_feature(self.max_time_len)  # [B,N,1]
            self.all_feature_concatenation = tf.concat([tf.tile(self.usr_enc_input, [1, self.max_time_len, 1])
                                                           , self.all_feature_concatenation, position_feature], axis=-1)
            mask = tf.reshape(tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
                              , [self.batch_size, self.max_time_len, 1])
            self.all_feature_concatenation *= tf.tile(mask,
                                                      [1, 1, self.all_feature_concatenation.get_shape()[-1].value])

    def get_position_feature(self, length):
        position_feature = tf.range(1, length + 1, 1.0) / tf.cast(self.max_time_len, tf.float32)
        position_feature = tf.reshape(position_feature, [-1, length, 1])
        position_feature = tf.tile(position_feature, [self.batch_size, 1, 1])  # (B, N, 1)
        mask = tf.reshape(tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
                          , [self.batch_size, self.max_time_len, 1])
        position_feature *= mask
        return position_feature

    def augment_context_features(self, raw_feature):
        with tf.name_scope("{}_Context_Augmentation".format(self.name)):
            N = tf.shape(raw_feature)[1]
            tensor_global_max = tf.reduce_max(raw_feature, axis=1, keep_dims=True)  # (B, 1, d2)
            tensor_global_min = tf.reduce_min(raw_feature, axis=1, keep_dims=True)  # (B, 1, d2)
            tensor_global_max_tile = tf.tile(tensor_global_max, [1, N, 1])  # (B, N, d2)
            tensor_global_min_tile = tf.tile(tensor_global_min, [1, N, 1])  # (B, N, d2)

            if self.augment_feature_normalization == "divide_mean":
                matrix_f_global = self.divide_mean_normalization(raw_feature)
            elif self.augment_feature_normalization == "min_max":
                matrix_f_global = self.min_max_normalization(raw_feature, tensor_global_min_tile,
                                                             tensor_global_max_tile)
                matrix_f_global = matrix_f_global - 0.5

            tensor_global_mean = tf.divide(tf.reduce_sum(raw_feature, axis=1, keep_dims=True),
                                           tf.cast(N, dtype=tf.float32))  # (B, 1, emb_dims)
            tensor_global_mean_tile = tf.tile(tensor_global_mean, [1, N, 1])  # (B, 17, d2)

            tensor_global_sigma = tf.reduce_mean(tf.square(raw_feature - tensor_global_mean_tile), axis=1,
                                                 keep_dims=True)
            tensor_global_sigma_tile = tf.tile(tensor_global_sigma, [1, N, 1])  # (B, 17, d2)
            tensor_global_sigma_tile = tf.where(tf.equal(tensor_global_sigma_tile, 0),
                                                tensor_global_sigma_tile + self.epsilon,
                                                tensor_global_sigma_tile)

            raw_feature_pv_norm = tf.where(tf.sqrt(tensor_global_sigma_tile) < self.epsilon,
                                           tf.fill(tf.shape(raw_feature), 0.0),
                                           (raw_feature - tensor_global_mean_tile) / (
                                                   tf.sqrt(tensor_global_sigma_tile) + self.epsilon))  # [B,N,D]

            augmented_feature_list = [raw_feature, tensor_global_mean_tile, tensor_global_sigma_tile,
                                      tensor_global_max_tile, tensor_global_min_tile, matrix_f_global,
                                      raw_feature_pv_norm]

        return tf.concat(augmented_feature_list, axis=-1)

    def augment_context_features_with_mask(self, raw_feature):
        mask = tf.reshape(tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32),
                          [-1, self.max_time_len, 1])
        mask = tf.tile(mask, [1, 1, raw_feature.get_shape()[-1].value])
        inf_mask = tf.where(tf.equal(1 - mask, 0), tf.fill(tf.shape(raw_feature), 0.0),
                            tf.fill(tf.shape(raw_feature), float('inf')))
        N = tf.shape(raw_feature)[1]
        seq_len_num = tf.cast(
            tf.tile(tf.reshape(self.seq_length_ph, [-1, 1, 1]), [1, 1, raw_feature.get_shape()[-1].value])
            , dtype=tf.float32)
        tensor_global_max = tf.reduce_max(raw_feature - inf_mask, axis=1, keep_dims=True)  # (B, 1, d2)
        # return inf_mask
        tensor_global_min = tf.reduce_min(raw_feature + inf_mask, axis=1, keep_dims=True)  # (B, 1, d2)
        tensor_global_max_tile = tf.tile(tensor_global_max, [1, N, 1])  # (B, N, d2)
        tensor_global_min_tile = tf.tile(tensor_global_min, [1, N, 1])  # (B, N, d2)

        tensor_global_mean = tf.divide(tf.reduce_sum(raw_feature * mask, axis=1, keep_dims=True),
                                       seq_len_num)  # (B, 1, emb_dims)
        tensor_global_mean_tile = tf.tile(tensor_global_mean, [1, N, 1])  # (B, 17, d2)

        matrix_f_global = tf.divide(raw_feature * mask, tensor_global_mean + self.epsilon)  # (B, N, emb_dims)

        tensor_global_sigma = tf.divide(tf.reduce_sum(tf.square(raw_feature - tensor_global_mean_tile) * mask, axis=1,
                                                      keep_dims=True), seq_len_num)
        tensor_global_sigma_tile = tf.tile(tensor_global_sigma, [1, N, 1])  # (B, 17, d2)
        tensor_global_sigma_tile = tf.where(tf.equal(tensor_global_sigma_tile, 0),
                                            tensor_global_sigma_tile + self.epsilon,
                                            tensor_global_sigma_tile)

        raw_feature_pv_norm = tf.where(tf.sqrt(tensor_global_sigma_tile) < self.epsilon,
                                       tf.fill(tf.shape(raw_feature), 0.0),
                                       (raw_feature - tensor_global_mean_tile) / (
                                               tf.sqrt(tensor_global_sigma_tile) + self.epsilon))  # [B,N,D]

        augmented_feature_list = [raw_feature, tensor_global_mean_tile, tensor_global_sigma_tile,
                                  tensor_global_max_tile, tensor_global_min_tile, matrix_f_global,
                                  raw_feature_pv_norm]
        return tf.concat(augmented_feature_list, axis=-1)

    def build_fc_net(self, inp, scope='fc'):
        with tf.variable_scope(scope):
            bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1', training=self.is_train)
            fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.relu, name='fc1')
            dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
            fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu, name='fc2')
            dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
            fc3 = tf.layers.dense(dp2, 2, activation=None, name='fc3')
            score = tf.nn.softmax(fc3)
            score = tf.reshape(score[:, :, 0], [-1, self.max_time_len])
            # output
            seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
            y_pred = seq_mask * score
        return y_pred

    def build_mlp_net(self, inp, layer=(500, 200, 80), scope='mlp'):
        with tf.variable_scope(scope):
            inp = tf.layers.batch_normalization(inputs=inp, name='mlp_bn', training=self.is_train)
            for i, hidden_num in enumerate(layer):
                fc = tf.layers.dense(inp, hidden_num, activation=tf.nn.relu, name='fc' + str(i))
                inp = tf.nn.dropout(fc, self.keep_prob, name='dp' + str(i))
            final = tf.layers.dense(inp, 1, activation=None, name='fc_final')
            score = tf.reshape(final, [-1, self.max_time_len])
            seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
            y_pred = seq_mask * score
        return y_pred

    def build_hyper_mlp_net(self, inp, inp_last_dim, units, activation=tf.nn.relu):  # [B, N, ft_num]
        # prefer_vector = tf.constant([self.controllable_auc_prefer, 1 - self.controllable_auc_prefer], dtype=tf.float32)
        w_output_dim = inp_last_dim * units
        hyper_w = tf.reshape(tf.layers.dense(self.controllable_prefer_vector, w_output_dim, reuse=tf.AUTO_REUSE),
                             [-1, units])
        hyper_b = tf.layers.dense(self.controllable_prefer_vector, units, reuse=tf.AUTO_REUSE)
        # self.print_loss = tf.print("prefer_vector:", self.controllable_prefer_vector, output_stream=sys.stderr)
        ret = tf.add(tf.matmul(inp, hyper_w), hyper_b)
        if activation:
            ret = activation(ret)
        return ret

    def get_hyper_dnn(self, x, layer_nums, layer_acts, name="hyper_dnn"):
        input_ft = x
        assert len(layer_nums) == len(layer_acts) + 1
        with tf.variable_scope(name):
            for i, layer_act in enumerate(layer_acts):
                input_ft = self.build_hyper_mlp_net_scope(input_ft, layer_nums[i], layer_nums[i + 1], 'layer_%d' % i,
                                                          layer_act)
        return input_ft

    def build_hyper_mlp_net_scope(self, inp, inp_last_dim, units, scope_name, activation=tf.nn.relu):  # [B, N, ft_num]
        # prefer_vector = tf.constant([self.controllable_auc_prefer, 1 - self.controllable_auc_prefer], dtype=tf.float32)
        w_output_dim = inp_last_dim * units
        hyper_w = tf.reshape(tf.contrib.layers.fully_connected(
            inputs=self.controllable_prefer_vector,
            num_outputs=w_output_dim,
            scope=scope_name + '_w',
            activation_fn=None,
            reuse=tf.AUTO_REUSE), [-1, units])
        hyper_b = tf.contrib.layers.fully_connected(
            inputs=self.controllable_prefer_vector,
            num_outputs=units,
            scope=scope_name + '_b',
            activation_fn=None,
            reuse=tf.AUTO_REUSE)
        # self.print_loss = tf.print("prefer_vector:", self.controllable_prefer_vector, output_stream=sys.stderr)
        ret = tf.add(tf.matmul(inp, hyper_w), hyper_b)
        if activation:
            ret = activation(ret)
        return ret

    def build_diversity_loss(self, y_pred, t=1e-3, prefer_div=0.5, balance_coef=1e4):
        y_pred = y_pred / (tf.reshape(tf.reduce_sum(y_pred, axis=1), [-1, 1]) + 1e-5)
        # gather [B, N, spar_dim] -> [B, N, 1]
        # cate_id = tf.divide(tf.cast(tf.gather(self.itm_spar_ph, [1], axis=2), dtype=tf.float32), t)
        cate_id = tf.cast(tf.gather(self.itm_spar_ph, [1], axis=2), dtype=tf.float32)
        # [B, N, 1] -> [B, N, N] - [B, 1, N] -> [B, N, N]
        cate_id_2 = tf.reshape(cate_id, [-1, 1, self.max_time_len])
        cate_id, cate_id_2 = tf.tile(cate_id, [1, 1, self.max_time_len]), tf.tile(cate_id_2,
                                                                                  [1, self.max_time_len, 1])
        cate_sub_1 = tf.abs(tf.subtract(cate_id, cate_id_2))
        # 1/tf.exp([B, N, N])+1
        similar_metric = tf.divide(1, tf.add(tf.exp(cate_sub_1), 1))
        # cate_0 = tf.zeros_like(cate_sub_1)
        # similar_metric = tf.where(tf.equal(cate_sub_1, 0), 1 - cate_sub_1, cate_0)
        # y_pred: [B, N, 1] -> [B, N, N] * [B, 1, N] -> [B, N, N]
        y_pred_1 = tf.cast(tf.reshape(y_pred, [-1, 1, self.max_time_len]), dtype=tf.float32)
        y_pred_2 = tf.cast(tf.reshape(y_pred, [-1, self.max_time_len, 1]), dtype=tf.float32)
        y_pred_1, y_pred_2 = tf.tile(y_pred_1, [1, self.max_time_len, 1]), tf.tile(y_pred_2,
                                                                                   [1, 1, self.max_time_len])
        div_loss = tf.reduce_mean(
            tf.multiply(tf.multiply(similar_metric, tf.multiply(y_pred_2, y_pred_1)),
                        1.0 - (self.controllable_auc_prefer if self.is_controllable else self.acc_prefer)))
        # self.print_loss = tf.print("diversity loss:", div_loss, "log loss:", self.loss, output_stream=sys.stderr)
        self.div_loss = div_loss
        self.loss = tf.add(self.loss, tf.multiply(div_loss, balance_coef))

    def build_logloss(self, y_pred):
        # loss
        self.loss = tf.multiply(tf.losses.log_loss(self.label_ph, y_pred),
                                self.controllable_auc_prefer if self.is_controllable else self.acc_prefer)
        self.auc_loss = self.loss
        self.build_diversity_loss(y_pred)
        self.opt()

    def build_norm_logloss(self, y_pred):
        self.loss = - tf.reduce_sum(
            self.label_ph / (tf.reduce_sum(self.label_ph, axis=-1, keepdims=True) + 1e-8) * tf.log(y_pred))
        self.opt()

    def build_mseloss(self, y_pred):
        self.loss = tf.losses.mean_squared_error(self.label_ph, y_pred)
        self.opt()

    def build_attention_loss(self, y_pred):
        self.label_wt = attention_score(self.label_ph)
        self.pred_wt = attention_score(y_pred)
        # self.pred_wt = y_pred
        self.loss = tf.losses.log_loss(self.label_wt, self.pred_wt)
        # self.loss = tf.losses.mean_squared_error(self.label_wt, self.pred_wt)
        self.opt()

    def opt(self):
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
                # self.loss += self.reg_lambda * tf.norm(v, ord=1)

        # self.lr = tf.train.exponential_decay(
        #     self.lr_start, self.global_step, self.lr_decay_step,
        #     self.lr_decay_rate, staircase=True, name="learning_rate")

        self.optimizer = tf.train.AdamOptimizer(self.lr)

        if self.max_grad_norm > 0:
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            for idx, (grad, var) in enumerate(grads_and_vars):
                if grad is not None:
                    grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)
            self.train_step = self.optimizer.apply_gradients(grads_and_vars)
        else:
            self.train_step = self.optimizer.minimize(self.loss)

    def multihead_attention(self,
                            queries,
                            keys,
                            num_units=None,
                            num_heads=2,
                            scope="multihead_attention",
                            reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]

            inp_dim = self.ft_num + 1 if self.ft_num % 2 else self.ft_num
            Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
            # if not self.is_controllable:
            #     Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
            #     K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
            #     V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
            # else:
            #     Q = self.build_hyper_mlp_net(queries, inp_dim, num_units, activation=None)
            #     K = self.build_hyper_mlp_net(keys, inp_dim, num_units, activation=None)
            #     V = self.build_hyper_mlp_net(keys, inp_dim, num_units, activation=None)

            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)
            outputs = tf.nn.dropout(outputs, self.keep_prob)
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        return outputs

    def positionwise_feed_forward(self, inp, d_hid, d_inner_hid, dropout=0.9):
        with tf.variable_scope('pos_ff'):
            inp = tf.layers.batch_normalization(inputs=inp, name='bn1', training=self.is_train)
            l1 = tf.layers.conv1d(inp, d_inner_hid, 1, activation='relu')
            l2 = tf.layers.conv1d(l1, d_hid, 1)
            dp = tf.nn.dropout(l2, dropout, name='dp')
            dp = dp + inp
            output = tf.layers.batch_normalization(inputs=dp, name='bn2', training=self.is_train)
        return output

    def bilstm(self, inp, hidden_size, scope='bilstm', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, name='cell_fw')
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, name='cell_bw')

            outputs, state_fw, state_bw = static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inp, dtype='float32')
        return outputs, state_fw, state_bw

    def train(self, batch_data, lr, reg_lambda, keep_prob=0.8, train_prefer=1):
        with self.graph.as_default():
            loss, _ = self.sess.run([self.loss, self.train_step], feed_dict={
                self.usr_profile: np.reshape(np.array(batch_data[1]), [-1, self.profile_num]),
                self.itm_spar_ph: batch_data[2],
                self.itm_dens_ph: batch_data[3],
                # self.usr_spar_ph: batch_data[3],
                # self.usr_dens_ph: batch_data[4],
                self.label_ph: batch_data[4],
                self.seq_length_ph: batch_data[6],
                # self.hist_length_ph: batch_data[8],
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
            pred, loss = self.sess.run([self.y_pred, self.loss], feed_dict={
                self.usr_profile: np.reshape(np.array(batch_data[1]), [-1, self.profile_num]),
                self.itm_spar_ph: batch_data[2],
                self.itm_dens_ph: batch_data[3],
                # self.usr_spar_ph: batch_data[3],
                # self.usr_dens_ph: batch_data[4],
                self.label_ph: batch_data[4],
                self.seq_length_ph: batch_data[6],
                # self.hist_length_ph: batch_data[8],
                self.reg_lambda: reg_lambda,
                self.keep_prob: keep_prob,
                self.is_train: False,
                self.controllable_auc_prefer: eval_prefer,
                self.controllable_prefer_vector: [[eval_prefer, 1 - eval_prefer]],
            })
            return pred.reshape([-1, self.max_time_len]).tolist(), loss

    def save(self, path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, save_path=path)
            print('Save model:', path)

    def load(self, path):
        with self.graph.as_default():
            ckpt = tf.train.get_checkpoint_state(path)
            if ckpt and ckpt.model_checkpoint_path:
                saver = tf.train.Saver()
                saver.restore(sess=self.sess, save_path=ckpt.model_checkpoint_path)
                print('Restore model:', ckpt.model_checkpoint_path)

    def set_sess(self, sess):
        self.sess = sess


class GSF(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                 profile_num, max_norm=None, group_size=1, activation='relu', hidden_layer_size=[512, 256, 128]):
        super(GSF, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                  itm_spar_num, itm_dens_num, profile_num, max_norm)

        with self.graph.as_default():
            self.group_size = group_size
            input_list = tf.unstack(self.item_seq, axis=1)
            input_data = tf.concat(input_list, axis=0)
            output_data = input_data
            if activation == 'elu':
                activation = tf.nn.elu
            else:
                activation = tf.nn.relu

            input_data_list = tf.split(output_data, self.max_time_len, axis=0)
            output_sizes = hidden_layer_size + [group_size]
            #
            output_data_list = [0 for _ in range(max_time_len)]
            group_list = []
            self.get_possible_group([], group_list)
            for group in group_list:
                group_input = tf.concat([input_data_list[idx]
                                         for idx in group], axis=1)
                group_score_list = self.build_gsf_fc_function(group_input, output_sizes, activation)
                for i in range(group_size):
                    output_data_list[group[i]] += group_score_list[i]
            self.y_pred = tf.concat(output_data_list, axis=1)
            self.y_pred = tf.nn.softmax(self.y_pred, axis=-1)
            self.build_norm_logloss(self.y_pred)

    def build_gsf_fc_function(self, inp, hidden_size, activation, scope="gsf_nn"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            for j in range(len(hidden_size)):
                bn = tf.layers.batch_normalization(inputs=inp, name='bn' + str(j), training=self.is_train)
                if j != len(hidden_size) - 1:
                    inp = tf.layers.dense(bn, hidden_size[j], activation=activation, name='fc' + str(j))
                else:
                    inp = tf.layers.dense(bn, hidden_size[j], activation=tf.nn.sigmoid, name='fc' + str(j))
        return tf.split(inp, self.group_size, axis=1)

    def get_possible_group(self, group, group_list):
        if len(group) == self.group_size:
            group_list.append(group)
            return
        else:
            for i in range(self.max_time_len):
                self.get_possible_group(group + [i], group_list)


class miDNN(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                 profile_num, is_controllable, max_norm=None, acc_prefer=1, hidden_layer_size=[512, 256, 128]):
        super(miDNN, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                    itm_spar_num, itm_dens_num, profile_num, max_norm, acc_prefer, is_controllable)

        with self.graph.as_default():
            fmax = tf.reduce_max(tf.reshape(self.item_seq, [-1, self.max_time_len, self.ft_num]), axis=1,
                                 keep_dims=True)
            fmin = tf.reduce_min(tf.reshape(self.item_seq, [-1, self.max_time_len, self.ft_num]), axis=1,
                                 keep_dims=True)
            global_seq = (self.item_seq - fmin) / (fmax - fmin + 1e-8)
            inp = tf.concat([self.item_seq, global_seq], axis=-1)

            if self.is_controllable:
                inp = self.build_hyper_mlp_net_scope(inp, inp.get_shape()[-1].value, inp.get_shape()[-1].value,
                                                     "hyper_dnn_midnn_1")
            self.y_pred = self.build_miDNN_net(inp, hidden_layer_size)
            if self.is_controllable:
                inp = self.build_hyper_mlp_net_scope(inp, inp.get_shape()[-1].value, inp.get_shape()[-1].value,
                                                     "hyper_dnn_midnn_2")
            self.build_logloss(self.y_pred)

    def build_miDNN_net(self, inp, layer, scope='mlp'):
        with tf.variable_scope(scope):
            inp = tf.layers.batch_normalization(inputs=inp, name='mlp_bn', training=self.is_train)
            for i, hidden_num in enumerate(layer):
                fc = tf.layers.dense(inp, hidden_num, activation=tf.nn.relu, name='fc' + str(i))
                inp = tf.nn.dropout(fc, self.keep_prob, name='dp' + str(i))
            final = tf.layers.dense(inp, 1, activation=tf.nn.sigmoid, name='fc_final')
            score = tf.reshape(final, [-1, self.max_time_len])
            seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
            y_pred = seq_mask * score
        return y_pred


class PRM(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                 profile_num, max_norm=None, is_controllable=False, acc_prefer=1.0, d_model=64, d_inner_hid=128,
                 n_head=1):
        super(PRM, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                  itm_spar_num, itm_dens_num, profile_num, max_norm, acc_prefer, is_controllable)

        with self.graph.as_default():
            pos_dim = self.item_seq.get_shape().as_list()[-1]
            self.d_model = d_model
            self.pos_mtx = tf.get_variable("pos_mtx", [max_time_len, pos_dim],
                                           initializer=tf.truncated_normal_initializer)
            self.item_seq = self.item_seq + self.pos_mtx
            if pos_dim % 2:
                self.item_seq = tf.pad(self.item_seq, [[0, 0], [0, 0], [0, 1]])

            inp_dim = self.ft_num + 1 if self.ft_num % 2 else self.ft_num
            self.item_seq = self.build_hyper_mlp_net_scope(self.item_seq, inp_dim, inp_dim, "before_attention")
            self.item_seq = self.multihead_attention(self.item_seq, self.item_seq, num_units=d_model, num_heads=n_head)
            self.item_seq = self.positionwise_feed_forward(self.item_seq, self.d_model, d_inner_hid, self.keep_prob)
            # self.item_seq = tf.layers.dense(self.item_seq, self.d_model, activation=tf.nn.tanh, name='fc')

            mask = tf.expand_dims(tf.sequence_mask(self.seq_length_ph, maxlen=max_time_len, dtype=tf.float32), axis=-1)
            seq_rep = self.item_seq * mask

            self.y_pred = self.build_prm_fc_function(seq_rep)
            # self.y_pred = self.build_fc_net(seq_rep)
            self.build_logloss(self.y_pred)

    def build_prm_fc_function(self, inp):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1', training=self.is_train)
        # self.print_loss = tf.print("inp shape:", tf.shape(bn1), output_stream=sys.stderr)
        if not self.is_controllable:
            fc1 = tf.layers.dense(bn1, self.d_model, activation=tf.nn.relu, name='fc1')
        else:
            fc1 = self.build_hyper_mlp_net_scope(bn1, self.d_model, self.d_model, "hyper_dnn_prm_1")
            # fc1 = self.build_hyper_mlp_net(bn1, self.d_model, self.d_model, activation=tf.nn.relu)
        dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
        if not self.is_controllable:
            fc2 = tf.layers.dense(dp1, 1, activation=None, name='fc2')
        else:
            fc2 = self.build_hyper_mlp_net_scope(dp1, self.d_model, 1, "hyper_dnn_prm_2", activation=None)
            # fc2 = self.build_hyper_mlp_net(dp1, self.d_model, 1, activation=None)
        score = tf.nn.softmax(tf.reshape(fc2, [-1, self.max_time_len]))
        # output
        seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
        return seq_mask * score


class SetRank(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                 profile_num, max_norm=None, d_model=256, n_head=8, d_inner_hid=64):
        super(SetRank, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                      itm_spar_num, itm_dens_num, profile_num, max_norm)

        with self.graph.as_default():
            self.item_seq = self.multihead_attention(self.item_seq, self.item_seq, num_units=d_model, num_heads=n_head)
            self.item_seq = self.positionwise_feed_forward(self.item_seq, d_model, d_inner_hid, dropout=self.keep_prob)

            mask = tf.expand_dims(tf.sequence_mask(self.seq_length_ph, maxlen=max_time_len, dtype=tf.float32), axis=-1)
            seq_rep = self.item_seq * mask

            self.y_pred = self.build_fc_net(seq_rep)
            self.build_attention_loss(self.y_pred)


class DLCM(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                 profile_num, max_norm=None, acc_prefer=1.0):
        super(DLCM, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                   itm_spar_num, itm_dens_num, profile_num, max_norm, acc_prefer)

        with self.graph.as_default():
            with tf.name_scope('gru'):
                seq_ht, seq_final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=self.item_seq,
                                                            sequence_length=self.seq_length_ph, dtype=tf.float32,
                                                            scope='gru1')
            self.y_pred = self.build_phi_function(seq_ht, seq_final_state, hidden_size)
            self.build_attention_loss(self.y_pred)
            # self.build_logloss()

    def build_phi_function(self, seq_ht, seq_final_state, hidden_size):
        bn1 = tf.layers.batch_normalization(inputs=seq_final_state, name='bn1', training=self.is_train)
        seq_final_fc = tf.layers.dense(bn1, hidden_size, activation=tf.nn.tanh, name='fc1')
        dp1 = tf.nn.dropout(seq_final_fc, self.keep_prob, name='dp1')
        seq_final_fc = tf.expand_dims(dp1, axis=1)
        bn2 = tf.layers.batch_normalization(inputs=seq_ht, name='bn2', training=self.is_train)
        # fc2 = tf.layers.dense(tf.multiply(bn2, seq_final_fc), 2, activation=None, name='fc2')
        # score = tf.nn.softmax(fc2)
        # score = tf.reshape(score[:, :, 0], [-1, self.max_time_len])
        fc2 = tf.layers.dense(tf.multiply(bn2, seq_final_fc), 1, activation=None, name='fc2')
        score = tf.reshape(fc2, [-1, self.max_time_len])
        # sequence mask
        seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
        score = score * seq_mask
        score = score - tf.reduce_min(score, 1, keep_dims=True)
        return score


class EGR_base(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                 profile_num, max_norm=None):
        super(EGR_base, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                       itm_spar_num, itm_dens_num, profile_num, max_norm)
        # global feature
        # new_shop_feature = self.get_global_feature(self.item_seq)
        with self.graph.as_default():
            new_shop_feature = self.item_seq

            with tf.variable_scope("network"):
                layer1 = new_shop_feature
                # fn = tf.nn.relu
                # layer1 = tf.layers.dense(dense_feature_normed, 128, name='layer1', activation=fn)
                new_dense_feature, final_state = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=layer1,
                                                                   sequence_length=self.seq_length_ph, dtype=tf.float32,
                                                                   scope='gru')
                new_feature = tf.concat([new_shop_feature, new_dense_feature], axis=-1)

                self.y_pred = self.build_fc_net(new_feature)

    def get_global_feature(self, inputph):
        tensor_global_max = tf.reduce_max(inputph, axis=1, keep_dims=True)  # (B, 1, d2)
        tensor_global_min = tf.reduce_min(inputph, axis=1, keep_dims=True)  # (B, 1, d2)
        tensor_global_max_tile = tf.tile(tensor_global_max, [1, self.max_time_len, 1])  # (B, 17, d2)
        tensor_global_min_tile = tf.tile(tensor_global_min, [1, self.max_time_len, 1])  # (B, 17, d2)
        matrix_f_global = tf.where(tf.equal(tensor_global_max_tile, tensor_global_min_tile),
                                   tf.fill(tf.shape(inputph), 0.5),
                                   tf.div(tf.subtract(inputph, tensor_global_min_tile),
                                          tf.subtract(tensor_global_max_tile, tensor_global_min_tile)))

        tensor_global_mean = tf.divide(tf.reduce_sum(matrix_f_global, axis=1, keep_dims=True),
                                       tf.cast(self.max_time_len, dtype=tf.float32))  # (B, 1, d2)
        tensor_global_mean_tile = tf.tile(tensor_global_mean, [1, self.max_time_len, 1])  # (B, 17, d2)
        tensor_global_sigma = tf.square(matrix_f_global - tensor_global_mean_tile)  # (B, 1, d2)

        new_shop_feature = tf.concat(
            [inputph, tensor_global_max_tile, tensor_global_min_tile, matrix_f_global, tensor_global_mean_tile,
             tensor_global_sigma], axis=2)
        return new_shop_feature


class EGR_evaluator(EGR_base):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                 profile_num, max_norm=None):
        super(EGR_evaluator, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                            itm_spar_num, itm_dens_num, profile_num, max_norm)
        with self.graph.as_default():
            self.build_logloss(self.y_pred)

    def predict(self, item_spar_fts, item_dens_fts, seq_len):
        with self.graph.as_default():
            ctr_probs = self.sess.run(self.y_pred, feed_dict={
                self.itm_spar_ph: item_spar_fts.reshape([-1, self.max_time_len, self.itm_spar_num]),
                self.itm_dens_ph: item_dens_fts.reshape([-1, self.max_time_len, self.itm_dens_num]),
                self.seq_length_ph: seq_len,
                self.keep_prob: 1.0,
                self.is_train: False})
            return ctr_probs


class EGR_discriminator(EGR_base):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                 profile_num, max_norm=None, c_entropy_d=0.001):
        super(EGR_discriminator, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                                itm_spar_num, itm_dens_num, profile_num, max_norm)
        with self.graph.as_default():
            self.d_reward = -tf.log(1 - self.y_pred + 1e-8)
            pred = self.pred + (self.seq_mask - 1) * 1e9
            self.build_discrim_loss(pred, c_entropy_d)

    def predict(self, item_spar_fts, item_dens_fts, seq_len):
        with self.graph.as_default():
            return self.sess.run([self.y_pred, self.d_reward], feed_dict={
                self.itm_spar_ph: item_spar_fts.reshape([-1, self.max_time_len, self.itm_spar_num]),
                self.itm_dens_ph: item_dens_fts.reshape([-1, self.max_time_len, self.itm_dens_num]),
                self.seq_length_ph: seq_len,
                self.keep_prob: 1.0,
                self.is_train: False})

    def train(self, batch_data, lr, reg_lambda, keep_prob=0.8):
        with self.graph.as_default():
            loss, _ = self.sess.run([self.loss, self.train_step], feed_dict={
                self.itm_spar_ph: batch_data[0].reshape([-1, self.max_time_len, self.itm_spar_num]),
                self.itm_dens_ph: batch_data[1].reshape([-1, self.max_time_len, self.itm_dens_num]),
                self.label_ph: batch_data[2].reshape([-1, self.max_time_len]),
                self.seq_length_ph: batch_data[3],
                self.lr: lr,
                self.reg_lambda: reg_lambda,
                self.keep_prob: keep_prob,
                self.is_train: True,
            })
            return loss

    def build_discrim_loss(self, logits, c_entropy_d):
        y_ = self.label_ph
        y = self.y_pred
        self.d_loss = -tf.reduce_mean(
            y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))

        self.entropy_loss = tf.reduce_mean(self.logit_bernoulli_entropy(logits))
        self.loss = self.d_loss - c_entropy_d * self.entropy_loss

        self.opt()

    def logit_bernoulli_entropy(self, logits):
        ent = (1. - tf.nn.sigmoid(logits)) * logits - self.logsigmoid(logits)
        return ent

    def logsigmoid(self, a):
        return -tf.nn.softplus(-a)
