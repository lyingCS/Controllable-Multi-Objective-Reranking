# -*- coding: utf-8 -*-

import json
import tensorflow as tf
from tensorflow.contrib import layers

def attention_summary(key_masks, key_len, query_len, outputs, scope,
                      first_n_att_weight_report,
                      atten_weights_collections):
    keys_masks_tmp = tf.reshape(tf.cast(key_masks, tf.float32), [-1, key_len])
    defined_length = tf.constant(first_n_att_weight_report, dtype=tf.float32, name="%s_defined_length" % (scope))
    greater_than_define = tf.cast(tf.greater(tf.reduce_sum(keys_masks_tmp, axis=1), defined_length), tf.float32)
    greater_than_define_exp = tf.tile(tf.expand_dims(greater_than_define, -1), [1, key_len])

    weight = tf.reshape(outputs, [-1, key_len]) * greater_than_define_exp
    weight_mean = tf.reduce_sum(weight, 0) / (tf.reduce_sum(greater_than_define_exp, axis=0) + 1e-5)

    if atten_weights_collections:
        for i in range(key_len):
            for collection in atten_weights_collections:
                tf.add_to_collection(collection, tf.slice(weight_mean, [i], [1], name='%s_atten_weight_%d' % (scope, i)))

    weight_map = tf.reshape(weight, [-1, query_len, key_len])  # BxL1xL2
    greater_than_define_exp_map = tf.reshape(greater_than_define_exp, [-1, query_len, key_len])  # BxL1xL2
    weight_map_mean = tf.reduce_sum(weight_map, 0) / (
        tf.reduce_sum(greater_than_define_exp_map, axis=0) + 1e-5)  # L1xL2
    report_image = tf.expand_dims(tf.expand_dims(weight_map_mean, -1), 0)  # 1xL1xL2x1
    tf.summary.image("%s_attention" % (scope), report_image[:, :80, :80, :])  # 1x10x10x1

def multihead_attention(queries, keys, num_units=None, num_output_units=None,
                        num_heads=8, scope="multihead_attention", reuse=None,
                        query_masks=None, key_masks=None, atten_mode='base',
                        linear_projection=True, is_target_attention=False,
                        variables_collections=None, outputs_collections=None,
                        activation_fn=None, first_n_att_weight_report=9,
                        atten_weights_collections=None):
    '''Applies multihead attention.

    Args:
      queries: attention的query [N, T_q, C_q].
      keys: attention的key和value，一般是一样的 [N, T_k, C_k].
      num_units: A scalar. Attention size.
      num_output_units: A scalar. Output Value size.
      num_heads: multi head的参数，>1时表示multi head
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.
      query_masks: A mask to mask queries with the shape of [N, T_k], if query_masks is None, use queries_length to mask queries
      key_masks: A mask to mask keys with the shape of [N, T_Q],  if key_masks is None, use keys_length to mask keys
      key_projection: A boolean, use projection to keys
      is_target_attention: 是否为target attention, 不是target attention 通常就是self attention和user attention

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        query_len = queries.get_shape().as_list()[1]  # T_q
        key_len = keys.get_shape().as_list()[1]  # T_k

        # Linear projections, C = # dim or column, T_x = # vectors or actions
        if linear_projection:
            queries_2d = tf.reshape(queries, [-1, queries.get_shape().as_list()[-1]])
            keys_2d = tf.reshape(keys, [-1, keys.get_shape().as_list()[-1]])
            Q = layers.fully_connected(queries_2d,
                                       num_units,
                                       activation_fn=None,
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="Q")  # (N, T_q, C)
            Q = tf.reshape(Q, [-1, queries.get_shape().as_list()[1], Q.get_shape().as_list()[-1]])
            K = layers.fully_connected(keys_2d,
                                       num_units,
                                       activation_fn=None,
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="K")  # (1, T_k, C)
            K = tf.reshape(K, [-1, keys.get_shape().as_list()[1], K.get_shape().as_list()[-1]])
            V = layers.fully_connected(keys_2d,
                                       num_output_units,
                                       activation_fn=None,
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="V")  # (1, T_k, C)
            V = tf.reshape(V, [-1, keys.get_shape().as_list()[1], V.get_shape().as_list()[-1]])
        else:
            Q = queries
            K = keys
            V = keys

        # fix rtp bug
        if is_target_attention:
            K = tf.reshape(K, [tf.shape(Q)[0], K.get_shape().as_list()[1], K.get_shape().as_list()[-1]])
            V = tf.reshape(V, [tf.shape(Q)[0], V.get_shape().as_list()[1], V.get_shape().as_list()[-1]])

        # Split and concat
        if num_heads > 1:
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C'/h)
        else:
            Q_ = Q
            K_ = K
            V_ = V

        # Multiplication & Scale
        if atten_mode == 'cos':
            # Multiplication
            Q_cos = tf.nn.l2_normalize(Q_, dim=-1)
            K_cos = tf.nn.l2_normalize(K_, dim=-1)
            outputs = tf.matmul(Q_cos, K_cos, transpose_b=True)  # (h*N, T_q, T_k)
            # Scale
            outputs = outputs * 20
        elif atten_mode == 'ln':
            # Layer Norm
            Q_ = layers.layer_norm(Q_, begin_norm_axis=-1, begin_params_axis=-1)
            K_ = layers.layer_norm(K_, begin_norm_axis=-1, begin_params_axis=-1)
            # Multiplication
            outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
            # Scale
            outputs = outputs * (K_.get_shape().as_list()[-1] ** (-0.5))
        elif atten_mode == 'din':
            din_all = tf.concat([Q_, K_, Q_ - K_, Q_ * K_], axis=-1)
            d_layer_1_all = layers.fully_connected(din_all, 80, activation_fn=tf.sigmoid, scope='f1_att')
            d_layer_2_all = layers.fully_connected(d_layer_1_all, 40, activation_fn=tf.sigmoid, scope='f2_att')
            d_layer_3_all = layers.fully_connected(d_layer_2_all, 1, activation_fn=None, scope='f3_att')
            outputs = tf.reshape(d_layer_3_all, [-1, query_len, key_len])
        else:
            # Multiplication
            outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
            # Scale
            outputs = outputs * (K_.get_shape().as_list()[-1] ** (-0.5))

        # key Masking
        key_masks = tf.tile(tf.reshape(key_masks, [-1, 1, key_len]), [num_heads, query_len, 1])  # (h*N, T_q, T_k)
        paddings = tf.fill(tf.shape(outputs), -2 ** 32 + 1.0)
        outputs = tf.where(key_masks, outputs, paddings)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        if not is_target_attention:
            # Query Masking
            query_masks = tf.tile(tf.reshape(query_masks, [-1, query_len]), [num_heads, 1])  # (h*N, T_q)
            outputs = tf.reshape(outputs, [-1, key_len])  # (h*N*T_q, T_k)
            paddings = tf.zeros_like(outputs, dtype=tf.float32)  # (h*N*T_q, T_k)
            outputs = tf.where(tf.reshape(query_masks, [-1]), outputs,
                               paddings)  # tf.where((h*N*T_q), (h*N*T_q, T_k), (h*N*T_q, T_k)) => (h*N*T_q, T_k)
            outputs = tf.reshape(outputs, [-1, query_len, key_len])  # (h*N, T_q, T_k)

        # Attention vector
        att_vec = outputs

        # Weighted sum (h*N, T_q, T_k) * (h*N, T_k, C/h)
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        if num_heads > 1:
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

    return outputs, att_vec
