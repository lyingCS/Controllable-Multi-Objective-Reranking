import os
import pickle as pkl
import random
from collections import defaultdict
import numpy as np
import math
import json
from librerank.utils import save_file


# map old feature id to new one
def convert(data, ft_map):
    res = []
    for i, v in enumerate(data):
        res.append(ft_map[i][v])
    return res


def process_data_with_hist(ad_feature, behavior, raw_sample, user_profile, store_dir, max_hist_len, list_len):

    user_tl = defaultdict(list)
    behav_tl = defaultdict(list)
    item_sft = {}
    item_dft = {}
    user_prof = {}
    last_ts = {}
    uid_set = set()
    iid_set = set()
    idx = 0
    pos = 0

    First = True
    with open(raw_sample, 'r') as r:
        for line in r:
            if First:
                First = False
                continue
            uid, ts, iid, pid, noclk, clk = line.split(',')
            d = float(ts)
            pos += int(clk)
            user_tl[int(uid)].append([d, int(iid), int(clk)])
            idx += 1
    print('num of records: ', idx, 'pos vs neg: ', pos / (idx - pos), 'user:', len(user_tl))

    removed_1, removed_2 = 0, 0
    pos = 0
    total_len, max_len, min_len = 0, 0, 1e9
    for uid in user_tl.keys():
        tl = user_tl[uid]
        if len(tl) > 3:
            tl = sorted(tl, key=lambda k: k[0], reverse=True)
            tl_len = len(tl)
            tl = np.array(tl)
            if sum(tl[:, 2]) == 0:
                removed_2 += 1
                continue

            si, ei = 0, 1
            while si < tl_len:
                while ei < tl_len and (tl[si][0] - tl[ei][0] <= 60 * 5):
                    ei += 1
                if sum(tl[si:ei][:, 2]) == 0:
                    si = ei
                    ei += 1
                else:
                    break

            if si >= tl_len:
                print('wrong', si, ei, tl[:, 2])
                exit()
            total_len += (ei - si)
            max_len = max(max_len, ei-si)
            min_len = min(min_len, ei-si)

            uid_set.add(uid)
            remain_tl = tl[si: ei][-list_len:]

            pos += sum(remain_tl[:, 2])
            remain_tl = remain_tl.tolist()
            remain_tl.reverse()
            user_tl[uid] = remain_tl
            last_ts[uid] = remain_tl[0][0]
            for v in remain_tl:
                iid_set.add(v[1])
        else:
            removed_1 += 1
    print('filter <= 4: num of user:', len(uid_set), ' num of item: ', len(iid_set), ' removed <= 3: ', removed_1,
          'remove all 0', removed_2, 'pos per list: ', pos / len(uid_set))
    print('max len:', max_len, 'min len:', min_len, 'average len:', total_len / len(uid_set))

    First = True
    with open(behavior, 'r') as r:
        for line in r:
            if First:
                First = False
                continue
            uid, ts, btag, cate, brand = line.split(',')
            # if cate == 'null' or brand == 'null':
            #     continue
            uid = int(uid)
            if uid in uid_set:
                d = float(ts)
                early_d = last_ts[uid]
                if d <= early_d:
                    behav_tl[uid].append([d, [btag, cate, brand]])
    print('finish loading history, num behavior:', len(behav_tl))
    uid_set = set(behav_tl.keys())

    for uid in uid_set:
        tl = behav_tl[uid]
        tl = sorted(tl, key=lambda k: k[0], reverse=True)
        behav_tl[uid] = tl[: max_hist_len]

    First = True
    with open(user_profile, 'r') as r:
        for line in r:
            if First:
                First = False
                continue
            values = line.split(',')
            uid = int(values[0])
            if uid in uid_set:
                user_prof[uid] = values[1:]
    print('finish loading user profile, num profile:', len(user_prof))
    # uid_set = set(user_prof.keys())

    First = True
    with open(ad_feature, 'r') as r:
        for line in r:
            if First:
                First = False
                continue
            iid, cate_id, cam_id, cust_id, brand, price = line.split(',')
            iid = int(iid)
            if iid in iid_set:
                item_sft[iid] = [iid, cate_id, cam_id, cust_id, brand]
                item_dft[iid] = [eval(price)]
    print('finish loading ad feature, num item:', len(item_sft))


    profile_fnum, itm_dens_fnum, itm_spar_fnum, hist_dens_fnum, hist_spar_fnum = 8, 1, 5, 1, 3
    print('profile fnum:', profile_fnum, 'itm sparse fnum:', itm_spar_fnum, 'itm dense fnum:', itm_dens_fnum)
    wo_profile = 0
    for uid in uid_set:
        if not user_prof.__contains__(uid):
            user_prof[uid] = ['null' for _ in range(profile_fnum)]
            wo_profile += 1
    print('user w/o profile:', wo_profile)


    uid_map, usr_ft_map, item_ft_map, hist_ft_map = {}, [{} for _ in range(profile_fnum)], [{} for _ in range(itm_spar_fnum)], \
                                       [{} for _ in range(hist_spar_fnum)]
    ft_idx = 1
    for iid in iid_set:
        itm_ft = item_sft[iid]
        for i in range(itm_spar_fnum):
            if not item_ft_map[i].__contains__(itm_ft[i]):
                item_ft_map[i][itm_ft[i]] = ft_idx
                ft_idx += 1
    for uid in uid_set:
        for hist_itm in behav_tl[uid]:
            for i in range(hist_spar_fnum):
                if not hist_ft_map[i].__contains__(hist_itm[1][i]):
                    hist_ft_map[i][hist_itm[1][i]] = ft_idx
                    ft_idx += 1
        prof = user_prof[uid]
        for i in range(profile_fnum):
            if not usr_ft_map[i].__contains__(prof[i]):
                usr_ft_map[i][prof[i]] = ft_idx
                ft_idx += 1
    # uid_idx = ft_idx
    # for uid in uid_set:
    #     uid_map[uid] = uid_idx
    #     uid_idx += 1
    print('total feature', ft_idx)
    print('user profile')
    for i in range(profile_fnum):
        print(len(usr_ft_map[i]), end='  ')
    print()
    print('item sparse feature')
    for i in range(itm_spar_fnum):
        print(len(item_ft_map[i]), end='  ')
    print()
    print('hist sparse feature')
    for i in range(hist_spar_fnum):
        print(len(hist_ft_map[i]), end='  ')

    stat = {'user_num': len(uid_set), 'item_num': len(iid_set), 'cate_num': len(item_ft_map[1]),
            'ft_num': ft_idx, 'list_num': len(uid_set), 'user_fnum': 1, 'profile_fnum': profile_fnum,
            'itm_spar_fnum': itm_spar_fnum, 'itm_dens_fnum': itm_dens_fnum,
            'hist_spar_fnum': hist_spar_fnum, 'hist_dens_fnum': hist_dens_fnum}
    with open(store_dir + 'data.stat', 'wb') as f:
        pkl.dump(stat, f)

    prop = {}
    for uid in uid_set:
        tl = user_tl[uid]
        for i, itm in enumerate(tl):
            d, iid, clk = itm
            cid = item_sft[iid][1]
            cate = item_ft_map[1][cid]
            if not prop.__contains__(cate):
                prop[cate] = [0 for i in range(list_len)]
            prop[cate][i] += clk
        prof = user_prof[uid]
        new_prof = []
        for i, v in enumerate(prof):
            new_prof.append(usr_ft_map[i][v])
        user_prof[uid] = new_prof

    _print = 5
    for k, v in prop.items():
        first_pos_click = v[0]
        for i in range(list_len):
            prop[k][i] = (v[i] + 1e-6) / (first_pos_click + 1e-6)
        if _print:
            print(k)
            print(prop[k])
            _print -= 1
    prop[0] = [1e-6 for i in range(list_len)]

    with open(store_dir + 'prop', 'wb') as f:
        pkl.dump(prop, f)

    with open(store_dir + 'user.profile', 'wb') as f:
        pkl.dump(user_prof, f)

    prop, user_prof, usr_ft_map = [], [], []
    print('save prop & user profile')

    # map old feature id to new one
    res = []
    ge5e4, ge3e4, ge1e4, ge5e3, ge1e3 = 0, 0, 0, 0, 0
    for uid in uid_set:
        ft_spar, ft_dens, label, hist_spar, hist_dens = [], [], [], [], []
        for d, iid, clk in user_tl[uid]:
            ft_spar.append(convert(item_sft[iid], item_ft_map))
            ft_dens.append(item_dft[iid])
            label.append(clk)
        for d, sft in behav_tl[uid]:
            hist_spar.append(convert(sft, hist_ft_map))
            interval = (last_ts[uid] - d) / 60
            if interval > 5e4:
                ge5e4 += 1
            if interval > 3e4:
                ge3e4 += 1
                interval = 3e4
            if interval > 1e4:
                ge1e4 += 1
            if interval > 5e3:
                ge5e3 += 1
            if interval > 1e3:
                ge1e3 += 1
            hist_dens.append([math.log2(interval + 1)])
        res.append([uid,  ft_spar, ft_dens, hist_spar, hist_dens, label])
    print('get res total:', len(res), '>5e4', ge5e4, '>3e4', ge3e4, '>1e4', ge1e4, '>5e3', ge5e3, '>1e3', ge1e3)
    user_tl, behav_tl, item_sft, item_dft, item_ft_map, hist_ft_map = [], [], [], [], [], []

    # split
    random.shuffle(res)
    num = len(res)
    train_list, val_list, test_list = res[: int(0.4 * num)], res[int(0.4 * num): int(0.9 * num)], res[int(0.9*num):]

    print('train num', len(train_list), 'val num', len(val_list), 'test num', len(test_list))
    with open(store_dir + 'data.data', 'wb') as f:
        pkl.dump([train_list, val_list, test_list], f)


def process_data_for_rerank(ad_feature, raw_sample, user_profile, store_dir, list_len):

    user_tl = defaultdict(list)
    # behav_tl = defaultdict(list)
    item_sft = {}
    item_dft = {}
    user_prof = {}
    last_ts = {}
    uid_set = set()
    iid_set = set()
    idx = 0
    pos = 0

    First = True
    with open(raw_sample, 'r') as r:
        for line in r:
            if First:
                First = False
                continue
            uid, ts, iid, pid, noclk, clk = line.split(',')
            d = float(ts)
            pos += int(clk)
            user_tl[int(uid)].append([d, int(iid), int(clk)])
            idx += 1
    print('num of records: ', idx, 'pos vs neg: ', pos / (idx - pos), 'user:', len(user_tl))

    removed_1, removed_2 = 0, 0
    pos = 0
    total_len, max_len, min_len = 0, 0, 1e9
    for uid in user_tl.keys():
        tl = user_tl[uid]
        if len(tl) > 2:
            tl = sorted(tl, key=lambda k: k[0], reverse=True)
            tl_len = len(tl)
            tl = np.array(tl)
            if sum(tl[:, 2]) == 0:
                removed_2 += 1
                continue

            si, ei = 0, 1
            while si < tl_len:
                while ei < tl_len and (tl[si][0] - tl[ei][0] <= 60 * 5):
                    ei += 1
                if sum(tl[si:ei][:, 2]) == 0:
                    si = ei
                    ei += 1
                else:
                    break

            if si >= tl_len:
                print('wrong', si, ei, tl[:, 2])
                exit()
            total_len += (ei - si)
            max_len = max(max_len, ei-si)
            min_len = min(min_len, ei-si)

            uid_set.add(uid)
            remain_tl = tl[si: ei][-list_len:]

            pos += sum(remain_tl[:, 2])
            remain_tl = remain_tl.tolist()
            remain_tl.reverse()
            user_tl[uid] = remain_tl
            last_ts[uid] = remain_tl[0][0]
            for v in remain_tl:
                iid_set.add(v[1])
        else:
            removed_1 += 1
    print('filter < 3: num of user:', len(uid_set), ' num of item: ', len(iid_set), ' removed <= 2: ', removed_1,
          'remove all 0', removed_2, 'pos per list: ', pos / len(uid_set))
    print('max len:', max_len, 'min len:', min_len, 'average len:', total_len / len(uid_set))

    First = True
    with open(user_profile, 'r') as r:
        for line in r:
            if First:
                First = False
                continue
            values = line.split(',')
            uid = int(values[0])
            if uid in uid_set:
                user_prof[uid] = values[1:]
    print('finish loading user profile, num profile:', len(user_prof))
    # uid_set = set(user_prof.keys())

    First = True
    with open(ad_feature, 'r') as r:
        for line in r:
            if First:
                First = False
                continue
            iid, cate_id, cam_id, cust_id, brand, price = line.split(',')
            iid = int(iid)
            if iid in iid_set:
                item_sft[iid] = [iid, cate_id, cam_id, cust_id, brand]
                item_dft[iid] = [eval(price)]
    print('finish loading ad feature, num item:', len(item_sft))


    profile_fnum, itm_dens_fnum, itm_spar_fnum= 8, 1, 5
    print('profile fnum:', profile_fnum, 'itm sparse fnum:', itm_spar_fnum, 'itm dense fnum:', itm_dens_fnum)
    wo_profile = 0
    for uid in uid_set:
        if not user_prof.__contains__(uid):
            user_prof[uid] = ['null' for _ in range(profile_fnum)]
            wo_profile += 1
    print('user w/o profile:', wo_profile)


    uid_map, usr_ft_map, item_ft_map = {}, [{} for _ in range(profile_fnum)], [{} for _ in range(itm_spar_fnum)]
    ft_idx = 1
    for iid in iid_set:
        itm_ft = item_sft[iid]
        for i in range(itm_spar_fnum):
            if not item_ft_map[i].__contains__(itm_ft[i]):
                item_ft_map[i][itm_ft[i]] = ft_idx
                ft_idx += 1
    for uid in uid_set:
        prof = user_prof[uid]
        for i in range(profile_fnum):
            if not usr_ft_map[i].__contains__(prof[i]):
                usr_ft_map[i][prof[i]] = ft_idx
                ft_idx += 1

    print('total feature', ft_idx)
    print('user profile')
    for i in range(profile_fnum):
        print(len(usr_ft_map[i]), end='  ')
    print()
    print('item sparse feature')
    for i in range(itm_spar_fnum):
        print(len(item_ft_map[i]), end='  ')
    print()


    # map old feature id to new one
    res = []
    for uid in uid_set:
        ft_spar, ft_dens, label, profile = [], [], [], []
        for d, iid, clk in user_tl[uid]:
            ft_spar.append(convert(item_sft[iid], item_ft_map))
            ft_dens.append(item_dft[iid])
            label.append(clk)
        profile.append(convert(user_prof[uid], usr_ft_map))

        res.append([uid, profile, ft_spar, ft_dens, label])

    cate_num = len(item_ft_map[1])
    user_tl, item_sft, item_dft, item_ft_map, usr_ft_map = [], [], [], [], []

    # split train:va;:test = 4:5:1
    random.shuffle(res)
    num = len(res)
    train_list, val_list, test_list = res[: int(0.4 * num)], res[int(0.4 * num): int(0.9 * num)], res[int(0.9*num):]

    print('train num', len(train_list), 'val num', len(val_list), 'test num', len(test_list))
    save_file(train_list, store_dir + 'data.train')
    save_file(val_list, store_dir + 'data.valid')
    save_file(test_list, store_dir + 'data.test')

    stat = {'item_num': len(iid_set), 'cate_num': cate_num,
            'ft_num': ft_idx, 'profile_fnum': profile_fnum,
            'train_num': len(train_list), 'val_num': len(val_list), 'test_num': len(test_list),
            'itm_spar_fnum': itm_spar_fnum, 'itm_dens_fnum': itm_dens_fnum}
    with open(store_dir + 'data.stat', 'w') as f:
        stat = json.dumps(stat)
        f.write(stat)


if __name__ == '__main__':
    # parameters
    random.seed(1234)
    raw_data_dir = '/'
    data_dir = '/'
    data_set_name = 'ad'
    raw_data_dir = ''
    data_set_name = '.'
    # max_hist_len = 60
    list_len = 30
    ad_feature = os.path.join(raw_data_dir, data_set_name + '/raw_data/ad_feature.csv')
    behavior = os.path.join(raw_data_dir, data_set_name + '/raw_data/behavior_log.csv')
    raw_sample = os.path.join(raw_data_dir, data_set_name + '/raw_data/raw_sample.csv')
    user_profile = os.path.join(raw_data_dir, data_set_name + '/raw_data/user_profile.csv')
    processed_dir = os.path.join(data_dir, data_set_name + '/processed/')

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    if not os.path.isfile(processed_dir + 'data.test'):
        process_data_for_rerank(ad_feature, raw_sample, user_profile, processed_dir, list_len)
