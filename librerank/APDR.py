import math

import numpy as np
import heapq
import os
# from sklearn.metrics import log_loss, roc_auc_score
import random
import time

from numpy import mean

from librerank.utils import *
from librerank.reranker import *
from librerank.rl_reranker import *
import datetime
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


class APDR(object):
    def __init__(self, max_time_len, coef):
        self.max_time_len = max_time_len
        self.coef = coef

    def auc_score(self, spar_ft, dens_ft, coef):
        return dens_ft[0]*coef[0]+coef[1]
        # return spar_ft[1]*coef[0]+spar_ft[2]*coef[1]+spar_ft[3]*coef[2]+spar_ft[4]*coef[3]+dens_ft[0]*coef[4]+coef[5]

    def div_score(self, cate_id, mp):
        return math.log2(2 + mp[cate_id]) - math.log2(1 + mp[cate_id])

    def predict(self, data_batch, lamda=1):
        cate_ids = list(map(lambda a: [i[1] for i in a], data_batch[2]))
        spar_fts = data_batch[2]
        dens_fts = data_batch[3]
        labels = data_batch[4]
        seq_lens = data_batch[6]
        ret_labels, ret_cates = [], []
        for i in range(len(seq_lens)):
            ret_label, ret_cate = [], []
            cate_mp = defaultdict(int)
            cate_id, label, seq_len = cate_ids[i], labels[i], seq_lens[i]
            spar_ft, dens_ft = spar_fts[i], dens_fts[i]
            # mean_score = sum(rank_score[:seq_len]) / seq_len
            mean_score = 1
            mask = [0 if k < seq_len else float('-inf') for k in range(self.max_time_len)]
            pred_score = [self.auc_score(spar_ft[k], dens_ft[k], self.coef) + mask[k] for k in range(self.max_time_len)]
            sorted_idx = sorted(range(self.max_time_len), key=lambda k: pred_score[k], reverse=True)
            mask[sorted_idx[0]] = float('-inf')
            ret_label.append(label[sorted_idx[0]])
            ret_cate.append(cate_id[sorted_idx[0]])
            cate_mp[cate_id[sorted_idx[0]]] += 1
            for j in range(1, seq_len):
                pred_score = [mask[k] + lamda * self.auc_score(spar_ft[k], dens_ft[k], self.coef) +
                              (1 - lamda) * abs(mean_score) * self.div_score(cate_id[k], cate_mp)
                              for k in range(self.max_time_len)]
                sorted_idx = sorted(range(self.max_time_len),
                                    key=lambda k: pred_score[k],
                                    reverse=True)
                mask[sorted_idx[0]] = float('-inf')
                ret_label.append(label[sorted_idx[0]])
                ret_cate.append(cate_id[sorted_idx[0]])
                cate_mp[cate_id[sorted_idx[0]]] += 1
            ret_labels.append(ret_label)
            ret_cates.append(ret_cate)
        return ret_labels, ret_cates


def eval_controllable_10(model, data, batch_size, isrank, metric_scope, _print=False):
    labels = [[] for i in range(11)]
    cates = [[] for i in range(11)]

    data_size = len(data[0])
    batch_num = data_size // batch_size
    print('eval', batch_size, batch_num)

    t = time.time()
    cate_ids = list(map(lambda a: [i[1] for i in a], data[2]))
    for i in range(11):
        for batch_no in range(batch_num):
            data_batch = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no)
            label, cate = model.predict(data_batch, float(i) / 10)
            labels[i].extend(label)
            # labels.extend(label)
            cates[i].extend(cate)

    res = [[] for i in range(5)]  # [5, 11, 4]
    for label, cate in zip(labels, cates):
        r = evaluate_multi(label, label, cate, metric_scope, isrank, _print)
        for j in range(5):
            res[j].append(r[j])

    print("EVAL TIME: %.4fs" % (time.time() - t))
    return res


def linear_regression(X, y):
    # Equation for linear regression coefficients
    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y).reshape([-1])
    return beta


def cal_linear_reg(data_batch):
    spar_fts = data_batch[2]
    dens_fts = data_batch[3]
    labels = data_batch[4]
    seq_lens = data_batch[6]
    mp_ft = dict()
    mp_click, mp_exposure = defaultdict(int), defaultdict(int)
    for i in range(len(seq_lens)):
        for j in range(seq_lens[i]):
            # mp_ft[spar_fts[i][j][0]] = spar_fts[i][j][1:] + dens_fts[i][j] + [1]
            mp_ft[spar_fts[i][j][0]] = dens_fts[i][j] + [1]
            mp_exposure[spar_fts[i][j][0]] += 1
            if labels[i][j] == 1:
                mp_click[spar_fts[i][j][0]] += 1
    X, y = [], []
    for key in mp_ft:
        X.append(mp_ft[key])
        y.append(mp_click[key] / mp_exposure[key])

    X = np.array(X)
    y = np.array(y).reshape([-1, 1])
    coef = linear_regression(X, y)
    print(coef)
    return coef

if __name__ == '__main__':
    processed_dir = '../Data/toy'
    processed_dir2 = '../Data/ad'
    stat_dir = os.path.join(processed_dir, 'data.stat')
    max_time_len = 10
    initial_ranker = 'lambdaMART'
    with open(stat_dir, 'r') as f:
        stat = json.load(f)

    num_item, num_cate, num_ft, profile_fnum, itm_spar_fnum, itm_dens_fnum, = stat['item_num'], stat['cate_num'], \
                                                                              stat['ft_num'], stat['profile_fnum'], \
                                                                              stat['itm_spar_fnum'], stat[
                                                                                  'itm_dens_fnum']
    print('num of item', num_item, 'num of list', stat['train_num'] + stat['val_num'] + stat['test_num'],
          'profile num', profile_fnum, 'spar num', itm_spar_fnum, 'dens num', itm_dens_fnum)

    with open(stat_dir, 'r') as f:
        stat = json.load(f)
    train_dir = os.path.join(processed_dir, initial_ranker + '.data.train')

    if os.path.isfile(train_dir):
        train_lists = pkl.load(open(train_dir, 'rb'))
    else:
        train_lists = construct_list(os.path.join(processed_dir, initial_ranker + '.rankings.train'), max_time_len)
        pkl.dump(train_lists, open(train_dir, 'wb'))
    test_dir = os.path.join(processed_dir, initial_ranker + '.data.test')
    test_dir_2 = os.path.join(processed_dir2, initial_ranker + '.data.test')
    if os.path.isfile(test_dir):
        test_lists = pkl.load(open(test_dir, 'rb'))
    else:
        test_lists = construct_list(os.path.join(processed_dir2, initial_ranker + '.rankings.test'),
                                    max_time_len)
        pkl.dump(test_lists, open(test_dir, 'wb'))
    test_lists2 = pkl.load(open(test_dir_2, 'rb'))

    coef = cal_linear_reg(train_lists)
    model = APDR(max_time_len, coef)
    res = eval_controllable_10(model, test_lists, 16, False, [1, 3, 5, 10], False)
    map_5_l = list(map(lambda a: a[2], res[0]))
    map_l = list(map(lambda a: a[3], res[0]))
    ndcg_5_l = list(map(lambda a: a[2], res[1]))
    ndcg_l = list(map(lambda a: a[3], res[1]))
    ilad_l = list(map(lambda a: a[2], res[3]))
    err_ia_5_l = list(map(lambda a: a[2], res[4]))
    err_ia_l = list(map(lambda a: a[3], res[4]))
    all_data_dict = {"all_data": [map_5_l, map_l, ndcg_5_l, ndcg_l, ilad_l, err_ia_5_l, err_ia_l]}
    print(all_data_dict["all_data"])
    for i in [0, 5, 10]:
        print(map_5_l[i], map_l[i], ndcg_5_l[i], ndcg_l[i], ilad_l[i], err_ia_5_l[i], err_ia_l[i])
    x = [i / 10 for i in range(len(map_l))]
    plt.subplot(2, 2, 1)
    plt.plot(x, map_l, 'r-')
    plt.xlabel('auc_preference')
    plt.ylabel('map')
    plt.subplot(2, 2, 2)
    plt.plot(x, ndcg_l, 'g-')
    plt.xlabel('auc_preference')
    plt.ylabel('ndcg')
    plt.subplot(2, 2, 3)
    plt.plot(x, ilad_l, 'b-')
    plt.xlabel('auc_preference')
    plt.ylabel('ilad')
    plt.subplot(2, 2, 4)
    plt.plot(x, err_ia_l, 'y-')
    plt.xlabel('auc_preference')
    plt.ylabel('err_ia')
    plt.suptitle("{}_{}".format('APDR', 'controllable'))
    plt.legend()
    plt.show()
