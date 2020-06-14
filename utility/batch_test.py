#!/usr/local/bin/python
# -*- coding: UTF-8
'''
based ngcf
'''
import utility.metrics as metrics
from utility.parser import parse_args
from utility.load_data import *
import multiprocessing
import heapq

cores = multiprocessing.cpu_count() // 2
print(cores)
args = parse_args()
Ks = eval(args.Ks)
import os
filepath = os.path.split(os.path.realpath(__file__))[0]
args.data_path = filepath+args.data_path
data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size, unlabel_rate=args.unlabel_rate)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:#
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)#

    r = []
    for i in K_max_item_score:#
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc


def myrecall(r, K, l):
    r=r[:20]
    if(sum(r)>0):
        recall = 1
    else:
        recall = 0
    temp_ndcg = metrics.ndcg_at_k(r, 20)
    return recall,temp_ndcg

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))

    return {'recall': np.array(recall), 'ndcg': np.array(ndcg)}


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]#uid
    #user u's items in the training set
    try:
        training_items = data_generator.train_items[u]#
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = data_generator.test_set[u]#

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))#
    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)
    temp_dict=get_performance(user_pos_test, r, auc, Ks)
    temp_dict['user_id']=u
    return temp_dict

def test_one_user_train(x):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set

    training_items = []
    # user u's items in the test set
    user_pos_test = data_generator.train_items[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)
'''
focus on recommending
'''
def test(sess, model, users_to_test, drop_flag=False, batch_test_flag=False,train_set_flag=0):
    result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE * 2
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        if batch_test_flag:

            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                item_batch = range(i_start, i_end)

                if drop_flag == False:
                    i_rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch})
                else:
                    i_rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch,
                                                                model.node_dropout: [0.]*len(eval(args.layer_size)),
                                                                model.mess_dropout: [0.]*len(eval(args.layer_size))})
                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM

        else:
            item_batch = range(ITEM_NUM)

            if drop_flag == False:
                rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                              model.pos_items: item_batch})
            else:
                rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                              model.pos_items: item_batch,
                                                              model.node_dropout: [0.] * len(eval(args.layer_size)),
                                                              model.mess_dropout: [0.] * len(eval(args.layer_size))})

        user_batch_rating_uid = zip(rate_batch, user_batch)
        if train_set_flag==0:
            batch_result = pool.map(test_one_user, user_batch_rating_uid)
        else:
            batch_result = pool.map(test_one_user_train, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users


    assert count == n_test_users
    pool.close()
    return result

'''
focus on profiling
'''
def test2(sess, model, item_profile, item_profile_mask):
    result = {'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),'all_result':None,'mcc':None,'f1':None,'gmean':None}#

    profile_estimation_batch1 = sess.run(model.correct_counts2,
                                         {
                                         # model.users: test_users,
                                         model.pos_items: data_generator.test_items,
                                         model.item_profile: item_profile,
                                         model.Train_flag: False,
                                         model.item_profile_mask: item_profile_mask,
                                         })  #


    result['profile_acc1'] = float(profile_estimation_batch1[0])# / n_test_users

    preds1 = profile_estimation_batch1[1]#
    labels1 = profile_estimation_batch1[2]#
    mask1 = profile_estimation_batch1[3]#
    y_pre = []

    for i,temp in enumerate(mask1):
        # if(temp>0):
            pred1 = np.argmax(preds1[i])
            y_pre.append(pred1)

    from sklearn.metrics import f1_score,matthews_corrcoef
    from sklearn.metrics.cluster import fowlkes_mallows_score
    gmean=fowlkes_mallows_score(labels1, y_pre)
    f1 = f1_score(labels1, y_pre, average='macro')
    mcc = matthews_corrcoef(labels1, y_pre)
    print('gmean:%.5f'%gmean)
    print('f1:%.5f'%f1)
    print('mcc:%.5f'%mcc)
    result['gmean'] = gmean
    result['f1'] = f1
    result['mcc'] = mcc

    return result
