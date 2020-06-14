#!/usr/local/bin/python
# -*- coding: UTF-8
'''
based on ngcf
'''
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
from collections import *
np.random.seed(0)
rd.seed(0)
# tf.set_random_seed(0)

class Data(object):
    def __init__(self, path, batch_size,unlabel_rate):
        self.path = path
        self.batch_size = batch_size
        train_file = path + '/train.dat'
        # test_file = path + '/val.dat'
        test_file = path + '/test.dat'
        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        # self.user_profile1 = {}#MTL
        # self.user_profile_list = []  # MTL
        # self.user_profile2 = {}  # MTL
        # self.user_profile_mask = {}#self.create_mask()#MTL
        self.item_profile_mask = {}
        self.item_profile1={}
        self.item_profile_list = []
        rng = np.random.RandomState(seed=0)  #



        self.neg_pools = {}

        self.exist_users = [] #

        self.remap_dict_user = defaultdict(int)
        self.remap_dict_item = defaultdict(int)
        # self.remap_dict_user_feature = defaultdict(int)
        self.remap_dict_item_feature = defaultdict(int)

        self.train_dict = defaultdict(list)#
        self.test_dict = defaultdict(list)#

        with open(train_file) as f: #
            for l in f.readlines(): #
                if len(l) > 0:
                    l = l.strip('').split(',') #
                    if(l[0]=='user_id'):
                        continue
                    uid = int(l[0]) # user id
                    self.remap_dict_user.setdefault(uid,len(self.remap_dict_user))
                    new_uid = self.remap_dict_user[uid]
                    # u_profile = int(l[2])  # user attributes for yelp-oh
                    # self.remap_dict_user_feature.setdefault(u_profile, len(self.remap_dict_user_feature))
                    # new_u_profile= self.remap_dict_user_feature[u_profile]
                    item_id = int(l[1])
                    self.remap_dict_item.setdefault(item_id, len(self.remap_dict_item))
                    new_itemid = self.remap_dict_item[item_id]
                    item_profile =  int(l[2]) #i_city
                    self.remap_dict_item_feature.setdefault(item_profile, len(self.remap_dict_item_feature))
                    new_item_profile = self.remap_dict_item_feature[item_profile]
                    # self.user_profile1[new_uid]=new_u_profile #MTL
                    self.item_profile1[new_itemid]=new_item_profile

                    if((new_uid in self.exist_users)==False):
                        self.exist_users.append(new_uid) # save user  id
                    # u_len.setdefault(new_uid, 0)
                    # u_len[new_uid]+=1
                    self.n_train += 1 #
                    self.train_dict.setdefault(new_uid,[])
                    self.train_dict[new_uid].append(new_itemid)



        self.n_items =len(self.remap_dict_item) #

        with open(test_file) as f:
            for l in f.readlines():

                        l = l.strip('').split(',')  #

                        if (l[0] == 'user_id'):
                            continue
                        uid = int(l[0])  # user id
                        # remap_dict_user.setdefault(uid, len(remap_dict_user) - 1)
                        new_uid = self.remap_dict_user[uid]
                        item_profile =  int(l[2]) #i_city
                        item_id = int(l[1])
                        # self.n_users += 1  #
                        if(item_id not in self.remap_dict_item):#
                            # continue
                            self.remap_dict_item.setdefault(item_id, len(self.remap_dict_item))
                            new_itemid = self.remap_dict_item[item_id]
                            self.remap_dict_item_feature.setdefault(item_profile, len(self.remap_dict_item_feature))
                            new_item_profile = self.remap_dict_item_feature[item_profile]
                            # self.user_profile1[new_uid] = new_u_profile  #
                            self.item_profile1[new_itemid] = new_item_profile

                        new_itemid = self.remap_dict_item[item_id]
                        self.n_test += 1  # 训
                        self.test_dict.setdefault(new_uid, [new_itemid])#
                        # self.test_dict[new_uid].append()
        self.n_items = len(self.remap_dict_item)  #
        self.n_users = len(self.remap_dict_user)

        # for k,v in self.user_profile1.items():
        #     self.user_profile_list.append(v)
        for k, v in self.item_profile1.items():
            self.item_profile_list.append(v)


        # self.num_user_features = len(self.remap_dict_user_feature)#for feature embedding
        self.num_item_features = len(self.remap_dict_item_feature)



        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32) #

        self.train_items, self.test_set = {}, {} #
        for key,val in self.train_dict.items(): #
                for temp in val:
                    self.R[key, temp] = 1.
                self.train_items[key] = val #

        for key, val in self.test_dict.items():
                self.test_set[key] = val #

        # user_profile_mask={}

        # user_split_list = [-1 for s in range(self.n_users)]  #
        # rand_list = rng.rand(len(user_split_list))
        # # user_split_list = []
        # # unlabel_rate = 0.5
        # up_test_rate = 0.05 #
        # up_val_rate =  0.#
        # # up_train_rate2 = 0.7
        # up_test_rate2 = (1 - unlabel_rate) * (1 - up_test_rate)  #
        # up_val_rate2 = up_test_rate2 - (1 - unlabel_rate) * up_val_rate  #
        # self.test_users = []  #
        # self.test_users_profile = []
        # for i, temp in enumerate(rand_list):
        #     if (temp >(1 - unlabel_rate)):
        #         self.user_profile_mask[i]=0
        #         self.user_profile1[i] = self.num_user_features#unknown tag for missing profiles
        #     elif (temp > up_test_rate2):
        #         self.user_profile_mask[i] = 3 # tag for user profiling training phase, loss calculation
        #         self.test_users.append(i)  # tag for user profiling test phase, user id
        #         self.test_users_profile.append(self.user_profile1[i])  #user profile
        #     elif (temp > up_val_rate2):
        #         self.user_profile_mask[i] = 2
        #     else:
        #         self.user_profile_mask[i] = 1

        item_split_list = [-1 for s in range(self.n_items)]  #
        rand_list2 = rng.rand(len(item_split_list))
        up_test_rate = 0.05  #
        up_val_rate = 0.  #
        up_test_rate2 = (1 - unlabel_rate) * (1 - up_test_rate)  #
        up_val_rate2 = up_test_rate2 - (1 - unlabel_rate) * up_val_rate  #
        self.test_items=[]#
        self.test_items_profile = []
        for i, temp in enumerate(rand_list2):
            if (temp > (1 - unlabel_rate)):
                item_split_list[i] = 0  #
                self.item_profile_mask[i] = 0
                self.item_profile1[i]=self.num_item_features # unknown tag for missing attribute
            elif (temp > up_test_rate2):
                self.item_profile_mask[i] = 3
                self.test_items.append(i)
                self.test_items_profile.append(self.item_profile1[i])
            elif (temp > up_val_rate2):
                self.item_profile_mask[i] = 2
            else:
                self.item_profile_mask[i] = 1
        '''
        some categories may lose after random missing, need to remap category from 0
        '''
        if(unlabel_rate!=0):
            self.temp_remap_item_feature=defaultdict()
            for id,profile in self.item_profile1.items():#
                self.temp_remap_item_feature.setdefault(profile, len(self.temp_remap_item_feature))
                self.item_profile1[id] = self.temp_remap_item_feature[profile]
            for temp_i,test_item_id in enumerate(self.test_items):#
                self.test_items_profile[temp_i]=self.item_profile1[test_item_id]
        self.dim2, self.pos_weight2 = self.print_statistics(unlabel_rate)  # generate weight for weighted softmax


    def print_statistics(self,unlabel_rate):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))
        if_list = []
        for key,val in self.item_profile1.items():
            if_list.append(val)
        print('item_feature_size '+str(len(set(if_list))))
        if_count = Counter(if_list)
        if(unlabel_rate!=0):
            if(if_count.most_common()[0][0]==self.temp_remap_item_feature[self.num_item_features]):
                max_value = if_count.most_common()[1][1]
            else:
                max_value = if_count.most_common()[0][1]
        else:
            max_value = if_count.most_common()[0][1]
        count_list2 = []
        for i, temp in if_count.items():  # 8
            num = float(max_value) / float(temp)
            if (num > 50):# the ratio is too large, we have to gain an balance
                num = 50
            elif(num <1):# for missing and masked attributes
                num=0.0001#

            count_list2.append(num)
        print(count_list2)
        # # uf_list = []
        # # for key, val in self.user_profile1.items():
        # #     uf_list.append(val)
        # # print('user_feature_size '+str(len(set(uf_list))))
        # uf_count = Counter(uf_list)
        # max_value = uf_count[max(uf_count, key=uf_count.get)]  #
        # count_list2 = []
        # for i, temp in uf_count.items():  #
        #     # count_list1.append(str(i) + ' ' + str(temp[1]))
        #     num = float(max_value) / float(temp)
        #     if(num>50):
        #         num=50
        #     count_list2.append(num)
        # print(count_list2)
        return len(count_list2),count_list2

    def get_adj_mat(self): #
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        # return adj_mat, norm_adj_mat, mean_adj_mat

        try:
            pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
        except Exception:
            adj_mat = adj_mat
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            print('generate pre adjacency matrix.')
            pre_adj_mat = norm_adj.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

        return adj_mat, norm_adj_mat, mean_adj_mat, pre_adj_mat

    def create_adj_mat(self): #
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32) #
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj): #
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj): #
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def negative_pool(self): #
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample(self): # 采样 主函数中用的
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]#

        def sample_pos_items_for_u(u, num): #
            pos_items = self.train_items[u] #
            n_pos_items = len(pos_items) #
            pos_batch = []
            while True:
                if len(pos_batch) == num: break #
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0] #
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch: #
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num): #
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0] #
                if neg_id not in self.train_items[u] and neg_id not in neg_items: #
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num): #
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        def sample_profile(id,profile):#
            target_profile=[]
            target_profile.append(profile[id]) #
            return target_profile

        def sample_profile_mask(id,mask):
            profile_mask = []
            profile_mask.append(mask[id])  #
            return profile_mask


        pos_items, neg_items,user_profile, item_profile, user_profile_mask ,item_profile_mask= [], [] ,[], [],[],[]#
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1) #
            neg_items += sample_neg_items_for_u(u, 1)
            # user_profile += sample_profile(u,self.user_profile1)
            # user_profile_mask += sample_profile_mask(u,self.user_profile_mask)
        for item in pos_items:
            item_profile += sample_profile(item, self.item_profile1)
            item_profile_mask += sample_profile_mask(item,self.item_profile_mask)

        return users, pos_items, neg_items, item_profile, item_profile_mask #MTL,user_profile

    def get_num_users_items(self): #
        return self.n_users, self.n_items



    def get_sparsity_split(self): #
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)



        return split_uids, split_state


    def sample_test(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.test_set.keys(), self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.test_set[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)

            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in (self.test_set[u] + self.train_items[u]) and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        def sample_profile(id, profile):  #
            target_profile = []
            target_profile.append(profile[id])  #
            return target_profile

        def sample_profile_mask(id, mask):
            profile_mask = []
            profile_mask.append(mask[id])  #
            return profile_mask

        pos_items, neg_items, user_profile, item_profile, user_profile_mask, item_profile_mask = [], [], [], [], [], []  #
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)  #
            neg_items += sample_neg_items_for_u(u, 1)
            # user_profile += sample_profile(u, self.user_profile1)
            # user_profile_mask += sample_profile_mask(u,self.user_profile_mask)
        for item in pos_items:
            item_profile += sample_profile(item, self.item_profile1)
            item_profile_mask += sample_profile_mask(item,self.item_profile_mask)

        return users, pos_items, neg_items,  item_profile, item_profile_mask  # MTL,user_profile

    # def getallprofile(self):##MTL
    #     return self.user_profile1