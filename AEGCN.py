#!/usr/local/bin/python
# -*- coding: UTF-8
'''
Thanks authors of NGCF and LightGCN for open their codes
Based on their code
'''
import tensorflow as tf
import os
import sys

# np.random.seed(0)
# random.seed(0)
tf.set_random_seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utility.helper import *
from utility.batch_test import *


class SGCN(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.model_type = 'sgcn'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.pretrain_data = pretrain_data
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_fold = 100
        self.norm_adj = data_config['norm_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.lr = args.lr
        self.lr1 = args.lr1
        self.lr2 = args.lr2
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.log_dir = self.create_model_str()
        self.verbose = args.verbose
        '''
        *********************************************************
        var for MTL task 
        '''
        # self.dim1 = data_config['dim1']# the category number of users
        self.dim2 = data_config['dim2']# the category number of items
        self.mtl_dropout = data_config['mtl_dropout']# dropout for dense layer
        # self.profile1_pre_dim = data_config['pre_dim1']# hidden size for user profiling dense layer
        self.profile2_pre_dim = data_config['pre_dim2']#..item profiling
        # self.profile1_dim = self.dim1
        self.profile2_dim = self.dim2
        self.Train_flag = tf.placeholder(tf.bool, shape=None)#batch norm
        # self.user_profile = tf.placeholder(tf.int64, shape=(None,))  # user attributes
        self.item_profile = tf.placeholder(tf.int64, shape=(None,))  # item attributes
        # self.user_profile_mask = tf.placeholder(tf.int32, shape=(None,))  # mask for missing attributes
        self.item_profile_mask = tf.placeholder(tf.int32, shape=(None,))  #
        # self.lamda1 = data_config['lamda1']  #  lamda_u
        self.lamda2 = data_config['lamda2']  #  lamda_i
        # self.pos_weight1 = data_config['pos_weight1']  # weight for uneven user attributes distribution
        self.pos_weight2 = data_config['pos_weight2']  # weight for uneven item attributes distribution
        # self.user_profile_feature = data_config['user_profile_feature'] # user feature input
        # self.item_profile_feature = data_config['item_profile_feature'] # item feature input
        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()

        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        Different Convolutional Layers:
            1. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
            2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
            3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
        """
        if self.alg_type in ['sgcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_sgcn_embed()
            # self.ua_embeddings, self.ia_embeddings = self._create_sgcn_embed_feature()
        elif self.alg_type in ['ngcf']:
            self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()

        elif self.alg_type in ['gcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcn_embed()

        elif self.alg_type in ['gcmc']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcmc_embed()

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)#user  embedding
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)# item pos
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)# item neg
        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)

        """
        *********************************************************
        Inference for the testing phase.(recommending)
        """
        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False,
                                       transpose_b=True)

        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.(recommending)
        """
        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.rating_loss = self.mf_loss + self.emb_loss

        # self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.rating_loss)

        """
        *********************************************************
        profile estimation for user attribute(dense layer)
        """
        # y1 = tf.matmul(self.u_g_embeddings, self.weights['W_p1']) + self.weights['b_p1']
        # y_bn1 = tf.layers.batch_normalization(y1, training=self.Train_flag, momentum=0.9)
        # y1_droput = tf.nn.dropout(tf.nn.relu(y_bn1), 1 - self.mtl_dropout)
        # y2 = tf.matmul(y1_droput, self.weights['W_p2']) + self.weights['b_p2']
        # y_bn2 = tf.layers.batch_normalization(y2, training=self.Train_flag, momentum=0.9)
        # y2_droput = tf.nn.dropout(tf.nn.relu(y_bn2), 1 - self.mtl_dropout)
        # y1_ = self.user_profile
        # y1_one_hot = tf.one_hot(y1_, self.dim1)
        # weight_per_class1 = tf.constant(self.pos_weight1)  # shape (, num_classes)
        # weights1 = tf.reduce_sum(tf.multiply(y1_one_hot, weight_per_class1), axis=1)  # shape (batch_size, num_classes)
        # reduction = tf.losses.Reduction.MEAN  # this ensures that we get a weighted mean
        # cross_entropy1 = tf.losses.softmax_cross_entropy(
        #     onehot_labels=y1_one_hot, logits=y2_droput + 1e-10,weights=weights1,reduction=reduction)#
        # user_profile_mask = tf.cast(self.user_profile_mask, dtype=tf.float32)
        # user_profile_mask /= tf.reduce_mean(user_profile_mask)
        # cross_entropy1 *= user_profile_mask
        # self.profile_loss1 = tf.reduce_mean(cross_entropy1)  # / self.batch_size
        # # 计算user profile准确率
        # self.correct_counts1 = self.masked_accuracy(y2_droput, y1_, user_profile_mask)
        """
        *********************************************************
        new estimation for item profile1  profile estimation for user attribute(dense layer)
        """
        yi1 = tf.matmul(self.pos_i_g_embeddings, self.weights['W_p3']) + self.weights['b_p3']#dense layer
        yi_bn1 = tf.layers.batch_normalization(yi1, training=self.Train_flag, momentum=0.9)#bn
        yi1_droput = tf.nn.dropout(tf.nn.relu(yi_bn1), 1 - self.mtl_dropout)#dropout
        yi2 = tf.matmul(yi1_droput, self.weights['W_p4']) + self.weights['b_p4']#2nd dense layer
        yi_bn2 = tf.layers.batch_normalization(yi2, training=self.Train_flag, momentum=0.9)
        yi2_droput = tf.nn.dropout(tf.nn.relu(yi_bn2), 1 - self.mtl_dropout)
        yi1_ = self.item_profile
        yi1_one_hot = tf.one_hot(yi1_, self.dim2)
        weight_per_class2 = tf.constant(self.pos_weight2)  #
        weights2 = tf.reduce_sum(tf.multiply(yi1_one_hot, weight_per_class2), axis=1)  # shape (batch_size, num_classes)
        reduction2 = tf.losses.Reduction.MEAN  # this ensures that we get a weighted mean
        cross_entropy2 = tf.losses.softmax_cross_entropy(onehot_labels=yi1_one_hot, logits=yi2_droput + 1e-10,
                                                         weights=weights2, reduction=reduction2)  #
        item_profile_mask = tf.cast(self.item_profile_mask, dtype=tf.float32)  #masked loss
        item_profile_mask /= tf.reduce_mean(item_profile_mask)
        cross_entropy2 *= item_profile_mask
        self.profile_loss2 = tf.reduce_mean(cross_entropy2)  # / self.batch_size
        # 计算item profile准确率
        self.correct_counts2 = self.masked_accuracy(yi2_droput, yi1_, item_profile_mask)# esimated profile and acc

        '''
        regs
        '''
        # self.y1_embeddings_pre1 = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        # self.y1_embeddings_pre2 = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        # self.y1_embeddings_pre3 = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        # self.y1_embeddings_pre4 = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        #
        # self.regs2 = tf.nn.l2_loss(y1_embeddings_pre1) + tf.nn.l2_loss(y1_embeddings_pre2) + tf.nn.l2_loss(y1_embeddings_pre3) + tf.nn.l2_loss(y1_embeddings_pre4)
        # self.regs2 = self.regs2 / self.batch_size
        # self.emb_loss2 = self.decay * self.regs2
        """
        *********************************************************
        joint loss
        """
        self.joint_loss1 = self.rating_loss + self.lamda2 * self.profile_loss2  # + self.lamda1 * self.profile_loss1

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     self.opti = tf.train.AdamOptimizer(learning_rate=self.lr1).minimize(self.joint_loss1)
        with tf.control_dependencies(update_ops):
            self.opti1 = tf.train.AdamOptimizer(learning_rate=self.lr1).minimize(self.joint_loss1)
        self.opti2 = tf.train.AdamOptimizer(learning_rate=self.lr1).minimize(self.rating_loss)

    # func for MTL
    def profile_evaluation(self, logits, labels):  #
        correct = tf.nn.in_top_k(logits, labels, 1)
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def masked_accuracy(self, preds, labels, mask):
        """Accuracy with masking."""
        correct_prediction = tf.equal(tf.argmax(preds, 1), labels)
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        # mask = tf.cast(mask, dtype=tf.float32)
        # mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return [tf.reduce_mean(accuracy_all), preds, labels, mask]

    # func for gcn
    def create_model_str(self):
        log_dir = '/' + self.alg_type + '/layers_' + str(self.n_layers) + '/dim_' + str(self.emb_dim)
        log_dir += '/' + args.dataset + '/lr_' + str(self.lr) + '/reg_' + str(self.decay)
        return log_dir

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer(seed=0)

        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                        name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                        name='item_embedding')

            # all_weights['user_feature_embeddings'] = tf.Variable(initializer([self.dim1+1, self.emb_dim]),name='user_feature_embeddings', dtype=tf.float32)  # (202, 64)
            # all_weights['item_feature_embeddings'] = tf.Variable(initializer([self.dim2+1, self.emb_dim]),name='item_feature_embeddings', dtype=tf.float32)

            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            all_weights['W_gc_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_mlp_%d' % k)

        # weights for mtl
        if self.alg_type in ['sgcn']:
            #
            # all_weights['W_p1'] = tf.Variable(initializer([self.emb_dim, self.profile1_pre_dim]), name='W_p1')
            # all_weights['b_p1'] = tf.Variable(initializer([1, self.profile1_pre_dim]), name='b_p1')
            # all_weights['W_p2'] = tf.Variable(initializer([self.profile1_pre_dim, self.profile1_dim]), name='W_p2')
            # all_weights['b_p2'] = tf.Variable(initializer([1, self.profile1_dim]), name='b_p2')

            all_weights['W_p3'] = tf.Variable(initializer([self.emb_dim, self.profile2_pre_dim]), name='W_p3')
            all_weights['b_p3'] = tf.Variable(initializer([1, self.profile2_pre_dim]), name='b_p3')
            all_weights['W_p4'] = tf.Variable(initializer([self.profile2_pre_dim, self.profile2_dim]), name='W_p4')
            all_weights['b_p4'] = tf.Variable(initializer([1, self.profile2_dim]), name='b_p4')

        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_sgcn_embed(self):
        # Generate a set of adjacency sub-matrix.
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            # transformed sum messages of neighbors.
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_sgcn_embed_feature(self):
        # Generate a set of adjacency sub-matrix.
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        # user_feature_embedding = tf.nn.embedding_lookup(self.weights['user_feature_embeddings'],self.user_profile_feature)
        # self.encoded_user_embedding  = tf.add(self.weights['user_embedding'], user_feature_embedding)  # [num_users, 3, h]
        # self.encoded_user_embedding = tf.reduce_mean(all_user_embedding, axis=1)

        # item feature
        item_feature_embedding = tf.nn.embedding_lookup(self.weights['item_feature_embeddings'],
                                                        self.item_profile_feature)  # [num_items, 3, h]
        self.encoded_item_embedding = tf.add(self.weights['item_embedding'],
                                             item_feature_embedding)  # [num_items, 4, h]
        # self.encoded_item_embedding = tf.reduce_mean(all_item_embedding, axis=1)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.encoded_item_embedding], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            # transformed sum messages of neighbors.
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_ngcf_embed(self):
        # Generate a set of adjacency sub-matrix.
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            # transformed sum messages of neighbors.
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(side_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])
            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.nn.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcn_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))

            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(
                tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcmc_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)

        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = []

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)
            # convolutional layer.
            embeddings = tf.nn.leaky_relu(
                tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            # dense layer.
            mlp_embeddings = tf.matmul(embeddings, self.weights['W_mlp_%d' % k]) + self.weights['b_mlp_%d' % k]
            mlp_embeddings = tf.nn.dropout(mlp_embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [mlp_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(
            self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        regularizer = regularizer / self.batch_size

        # In the first version, we implement the bpr loss via the following codes:
        # We report the performance in our paper using this implementation.
        #         maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        #         mf_loss = tf.negative(tf.reduce_mean(maxi))

        ## In the second version, we implement the bpr loss via the following codes to avoid 'NAN' loss during training:
        ## However, it will change the training performance and training performance.
        ## Please retrain the model and do a grid search for the best experimental setting.
        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))

        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)


def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    epoch_flag = 1#for debug, test for every epoch_flag
    early_flag = 3#early stop after early_flag x epoch_flag
    args.epoch = 5#whole epochs
    print_flag = 1#print infor or not

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    # config['dim1'] = args.dim1
    config['dim2'] = data_generator.dim2
    config['mtl_dropout'] = args.mtl_dropout
    # config['pre_dim1'] = args.pre_dim1
    config['pre_dim2'] = args.pre_dim2
    # config['lamda1'] = args.lamda1  #
    config['lamda2'] = args.lamda2  #
    # config['pos_weight1'] = args.pos_weight1  #
    config['pos_weight2'] = data_generator.pos_weight2  #
    # config['user_profile_feature'] = data_generator.user_profile_list #feautre input
    # config['item_profile_feature'] = data_generator.item_profile_list #


    filepath = os.path.split(os.path.realpath(__file__))[0]
    stamp = int(time())
    result_path1 = filepath + '/output/%s/'%(args.dataset) + str(stamp) + '_loss.txt'  #
    # result_path2 = filepath + '/output/profile/' + str(stamp) + 'unlabel_final_.txt'  #
    # result_path3 = filepath + '/output/profile/' + str(stamp) + '_detail.txt'  #

    fw1 = open(result_path1, 'w')
    # fw2 = open(result_path2, 'w')
    # fw3 = open(result_path3, 'w')
    perf_str = 'dataset =%s, lamda1 = %s, lamda2 = %s, dim1=%s,dim2=%s,pre_dim1=%s,pre_dim2=%s, mtl_dropout=%s,unlabel_rate=%s, dataset=%s,layer_size=%s, embed_size=%d, lr1=%.4f, lr2=%.4f, regs=%s, adj_type=%s\n\t\n' \
               % (
                   args.dataset, args.lamda1, args.lamda2, args.dim1, data_generator.dim2, args.pre_dim1, args.pre_dim2,
                   args.mtl_dropout, args.unlabel_rate, args.dataset, args.layer_size,
                   args.embed_size, args.lr1, args.lr2,
                   args.regs,
                   args.adj_type)

    fw1.write(perf_str)
    if (print_flag):
        print(perf_str)
    fw1.flush()
    # fw2.write(perf_str)
    # fw2.flush()
    # fw3.write(perf_str)
    # fw3.flush()
    #####################debug

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj, pre_adj = data_generator.get_adj_mat()

    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')
    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        print('use the normalized adjacency matrix')
    elif args.adj_type == 'gcmc':
        config['norm_adj'] = mean_adj
        print('use the gcmc adjacency matrix')
    elif args.adj_type == 'pre':
        config['norm_adj'] = pre_adj
        print('use the pre adjcency matrix')
    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
        print('use the mean adjacency matrix')
    t0 = time()

    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None

    model = SGCN(data_config=config, pretrain_data=pretrain_data)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    """
      *********************************************************
      Reload the pretrained model parameters.
      """

    sess.run(tf.global_variables_initializer())
    cur_best_pre_0 = 0.
    print('without pretraining.')

    loss_loger, rec_loger, ndcg_loger, hr_loger, profile_acc_loger1, profile_acc_loger2, all_result_loger, profile_detail_loger,mcc_logger,f1_logger,gmean_logger = [], [], [], [], [], [], [], [], [], [], []
    # loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    # main
    should_stop = False
    for epoch in range(args.epoch):
        t1 = time()
        # loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        # 2loss
        mf_loss, emb_loss, reg_loss, profile_loss1, profile_loss2, joint_loss, rating_loss = 0., 0., 0., 0., 0., 0., 0.
        joint_loss_test, rating_loss_test, profile_loss_test1, profile_loss_test2, mf_loss_test, emb_loss_test, reg_loss_test = 0., 0., 0., 0., 0., 0., 0.

        for idx in range(n_batch):
            users, pos_items, neg_items, item_profile, item_profile_mask = data_generator.sample()
            # user profiling batch training mask
            # train_user_profile_mask = []
            # for i, temp in enumerate(user_profile_mask):
            #     if (temp == 1):
            #         train_user_profile_mask.append(1)
            #     else:
            #         train_user_profile_mask.append(0)
            # item profiling batch training mask
            train_item_profile_mask = []
            for i, temp in enumerate(item_profile_mask):
                if (temp == 1):  # 如果是train的flag
                    train_item_profile_mask.append(1)
                else:
                    train_item_profile_mask.append(0)
            if (sum(train_item_profile_mask) != 0):
                _, batch_joint_loss, batch_rating_loss, batch_profile_loss2, batch_mf_loss, batch_emb_loss, batch_reg_loss = \
                    sess.run([model.opti1, model.joint_loss1, model.rating_loss, model.profile_loss2,
                              model.mf_loss, model.emb_loss, model.reg_loss],
                             feed_dict={model.users: users, model.pos_items: pos_items,
                                        # model.user_profile: user_profile,
                                        model.item_profile: item_profile,
                                        # model.user_profile_mask: train_user_profile_mask,
                                        model.item_profile_mask: train_item_profile_mask,
                                        model.Train_flag: True,
                                        model.neg_items: neg_items})
            else:
                _, batch_rating_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = \
                    sess.run([model.opti2, model.rating_loss,
                              model.mf_loss, model.emb_loss, model.reg_loss],
                             feed_dict={model.users: users, model.pos_items: pos_items, model.neg_items: neg_items})
                batch_joint_loss = batch_rating_loss
                batch_profile_loss2 = 0
            # _, batch_joint_loss, batch_rating_loss,batch_profile_loss2, batch_mf_loss, batch_emb_loss, batch_reg_loss = \
            #     sess.run([model.opti, model.joint_loss1, model.rating_loss, model.profile_loss2,
            #               model.mf_loss, model.emb_loss, model.reg_loss],
            #              feed_dict={model.users: users, model.pos_items: pos_items,
            #                         # model.user_profile: user_profile,
            #                         model.item_profile: item_profile,
            #                         # model.user_profile_mask: train_user_profile_mask,
            #                         model.item_profile_mask: train_item_profile_mask,
            #                         model.Train_flag: True,
            #                         model.neg_items: neg_items})

            joint_loss += batch_joint_loss / n_batch  #
            rating_loss += batch_rating_loss / n_batch  #
            # profile_loss1 += batch_profile_loss1 / n_batch  #
            profile_loss2 += batch_profile_loss2 / n_batch
            mf_loss += batch_mf_loss / n_batch
            emb_loss += batch_emb_loss / n_batch

        if np.isnan(joint_loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % epoch_flag != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f = %.5f +  %.5f]' % (
                    epoch, time() - t1, joint_loss, rating_loss, profile_loss2)
                if (print_flag):
                    print(perf_str)
                fw1.write(perf_str + '\n')
                fw1.flush()
            continue

        # users_to_test = list(data_generator.train_items.keys())
        # ret = test(sess, model, users_to_test,drop_flag=True,train_set_flag=1)
        perf_str = 'Epoch %d [%.1fs]: train==[%.5f = %.5f + %.5f]' % (
            epoch, time() - t1, joint_loss, rating_loss, profile_loss2)
        if (print_flag):
            print(perf_str)

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        # focus on recommending
        ret = test(sess, model, users_to_test)
        # focus on profiling
        # test_items = data_generator.test_items#focus on profiling
        # test_items_profile=data_generator.test_items_profile
        # test_item_profile_mask2 = [1]*len(test_items) #

        # ret = test2(sess, model,  test_items_profile,
        #             test_item_profile_mask2,
        #             )

        t3 = time()

        loss_loger.append(joint_loss)
        '''
        metrics for rec
        '''
        rec_loger.append(ret['recall'])
        # hr_loger.append(ret['hit_ratio'])
        ndcg_loger.append(ret['ndcg'])
        # profile_acc_loger1.append(ret['profile_acc1'])
        # profile_acc_loger2.append(ret['profile_acc2'])
        # all_result_loger.append(ret['all_result'])
        # profile_detail_loger.append(ret['detail_profile'])
        '''
        metrics when focusing profiling
        '''
        # mcc_logger.append(ret['mcc'])
        # f1_logger.append(ret['f1'])
        # gmean_logger.append(ret['gmean'])
        '''
        no early stop for profiling
        '''
        # perf_str = 'TEST Epoch %d [%.1fs] : mcc=%.5f,f1=%.5f,gmean=%.5f' % \
        #            (epoch, t3 - t2, ret['mcc'], ret['f1'],ret['gmean'],)
        # if (print_flag):
        #     print(perf_str)
        # fw1.write(perf_str + '\n')
        # fw1.flush()
        '''
         early stop for rec
        '''

        if args.verbose > 0:
            perf_str = 'TEST Epoch %d [%.1fs] : recall=%.5f,ndcg=%.5f' % \
                       (epoch, t3 - t2, ret['recall'][0], ret['ndcg'][
                           0])
            if (print_flag):
                print(perf_str)
            fw1.write(perf_str + '\n')
            fw1.flush()
        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc',
                                                                    flag_step=early_flag)


    recs = np.array(rec_loger)
    ndcgs = np.array(ndcg_loger)
    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s],ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)
    fw1.write(final_perf)
    fw1.close()


