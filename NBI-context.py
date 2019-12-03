# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 09:42:53 2018

@author: icruicks
"""
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import math
import gc
import scipy as spy

class NBI(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, A):
        '''
        A is a biaprtite network of two not neccesarily equal dimensions and a numpy
        array of binary meetup
        '''
        A = np.asanyarray(A)
        k_x = np.sum(A, axis=0)  # 项目的参与数 按列加
        k_y = np.sum(A, axis=1)  # 用户的参与数 按行加
        W = np.zeros((A.shape[1], A.shape[1]))
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                if k_x[j] != 0:
                    W[i, j] = 1 / k_x[j] * np.sum(np.divide(np.multiply(A[:, i], A[:, j]), np.where(k_y > 0, k_y, 1)))
        self.W_ = W
        return self

    def predict(self, a):

        a = np.asanyarray(a)
        f_prime = np.zeros(a.shape[0])

        for j in range(f_prime.shape[0]):
            f_prime[j] = np.sum(np.multiply(self.W_[j, :], a))

        self.y_ = f_prime
        return self.y_


def getTopN(predict_mat, u_cand):
    topn_ratings = []
    for i in u_cand:
        topn_ratings.append([predict_mat[i], i])
    topn_ratings.sort(reverse=True)
    return topn_ratings


def cos(p, q):
    fenzi = np.sum(p*q)
    fenmu1 = np.sum(p)
    fenmu2 = np.sum(q)
    # print(fenmu1)
    if fenmu1 * fenmu2 != 0:
        return fenzi / np.sqrt(fenmu1 * fenmu2)
    return 0


if __name__ == '__main__':
    para = {}
    with open('NBI-context.txt') as f:  # 需要重新打开文本进行读取
        for line in f:
            content = line.rstrip()  # 删除字符串末尾的空白
            if (len(content) > 1):
                kv = content.split('=')
                para[kv[0]] = kv[1]
    dataset = para['dataset']
    ufactor = para['ufactor']
    ufactor = float(ufactor)
    lcfactor = para['lcfactor']
    lcfactor = float(lcfactor)
    gfactor = para['hfactor']
    gfactor = float(gfactor)
    time_factor = para['time_factor']
    time_factor = float(time_factor)
    top_n = para['topN']
    top_n = int(top_n)
    result_name = para['results']

    file_path = dataset+'/train.csv'
    train_file = open(file_path, 'r')
    train_data = []
    for line in train_file:
        data = line.split(',')
        uid = int(data[0])
        mid = int(data[1])
        train_data.append([uid, mid])
    train_file.close()

    file_path = dataset+'/test.csv'
    test_file = open(file_path, 'r')
    test_data = []
    cand_users = []
    cand_events = []
    for line in test_file:
        data = line.split(',')
        uid = int(data[0])
        mid = int(data[1])
        test_data.append([uid, mid])
        if uid not in cand_users:
            cand_users.append(uid)
        if mid not in cand_events:
            cand_events.append(mid)
    test_file.close()

    train_data.sort()
    train_data = np.array(train_data)
    test_data = np.array(test_data)

    u_num = max(max(train_data[:,0]), max(test_data[:,0])) + 1
    i_num = max(max(train_data[:,1]), max(test_data[:,1])) + 1

    print(u_num, i_num)

    # row_offset = np.zeros(u_num, dtype=int)
    # col_indices = []
    # value_array = []
    # for tuple in train_data:
    #     uid = tuple[0]
    #     eid = tuple[1]
    #     row_offset[uid] += 1
    #     col_indices.append(eid)
    #     value_array.append(1)
    # for i in range(1,u_num):
    #     row_offset[i] += row_offset[i-1]
    # ue = spy.sparse.csr_matrix((value_array, col_indices, row_offset), shape=(u_num, i_num))
    ue = np.zeros((u_num, i_num))
    for tuple in train_data:
        ue[tuple[0]][tuple[1]] = 1

    # 读取群组活动文件
    he = []
    eh_file = open(dataset+'/group_event.csv', 'r')
    for line in eh_file:
        data = line.split(',')
        he.append([int(data[0]),int(data[1])])
    eh_file.close()
    he = np.array(he)
    host_num = max(he[:,0]) + 1
    host_event = np.zeros((host_num, i_num))
    for line in he:
        if line[1] >= i_num:
            continue
        host_event[line[0]][line[1]] = 1
    # # 地点活动文件
    # le = []
    # et_file = open(dataset+'/location_event.csv', 'r')
    # for line in et_file:
    #     data = line.split(',')
    #     le.append([int(data[0]),int(data[1])])
    # et_file.close()
    # le = np.array(le)
    # location_num = max(le[:, 0]) + 1
    # loc_event = np.zeros((location_num, i_num))
    # for line in le:
    #     if line[1] >= i_num:
    #         continue
    #     loc_event[line[0]][line[1]] = 1
    # 时间活动文件
    time_e = []
    et_file = open(dataset + '/time_event.csv', 'r')
    for line in et_file:
        data = line.split(',')
        time_e.append([int(data[0]), int(data[1])])
    et_file.close()
    time_e = np.array(time_e)
    time_num = max(time_e[:, 0]) + 1
    time_event = np.zeros((time_num, i_num))
    for line in time_e:
        if line[1] >= i_num:
            continue
        time_event[line[0]][line[1]] = 1

    nbi = NBI()
    # nbi1 = NBI()
    # nbi2 = NBI()
    # nbi3 = NBI()
    nbi.fit(ue)
    np.savetxt(dataset+"_result/W.txt", nbi.W_)
    # nbi1.fit(loc_event)
    # np.savetxt(dataset+"_result/locW.txt", nbi1.W_)
    # nbi2.fit(host_event)
    # np.savetxt(dataset+"_result/hostW.txt", nbi2.W_)
    # nbi3.fit(time_event)
    # np.savetxt(dataset + "_result/timeW.txt", nbi3.W_)
    exit()
    input("done!")
    participant_list = []
    for i in range(i_num):
        participant_list.append(0)
    for ratingtuple in train_data:
        (i, j) = ratingtuple
        participant_list[j] += 1
    max_k = 0
    for i in range(i_num):
        max_k = max(max_k, participant_list[i])
    print(max_k)
    u_testset = []
    for u in range(u_num):
        u_testset.append([])
    for tuple in test_data:
        u_testset[tuple[0]].append(tuple[1])

    del train_data,test_data
    gc.collect()
    # W = np.loadtxt(dataset+"_result/W.txt")
    # nbi1.W_ = np.loadtxt(dataset+"_result/locW.txt")
    # nbi2.W_ = np.loadtxt(dataset+"_result/hostW.txt")
    # nbi3.W_ = np.loadtxt(dataset+"_result/timeW.txt")
    nbi.W_ = nbi.W_ - 0.75 * np.multiply(nbi.W_, nbi.W_)
    # nbi1.W_ = nbi1.W_ - 0.75 * np.multiply(nbi1.W_, nbi1.W_)
    # nbi2.W_ = nbi2.W_ - 0.75 * np.multiply(nbi2.W_, nbi2.W_)
    # nbi3.W_ = nbi3.W_ - 0.75 * np.multiply(nbi3.W_, nbi3.W_)
    # nbi.W_ = W
    # nbi1.W_ = typeW
    # nbi2.W_ = hostW
    # nbi3.W_ = timeW
    predict_ratings = np.zeros((u_num, i_num))

    recommend_list = []
    # need_novelty = True
    for u in cand_users:
        predict_ratings[u] = ufactor * nbi.predict(ue[u])
        # + lcfactor * nbi1.predict(ue[u]) + gfactor * nbi2.predict(ue[u])+time_factor * nbi3.predict(ue[u])
        # if need_novelty:
        #     for j in range(i_num):
        #         print(len(participant_list[j]))
        #         if len(participant_list[j]) > 1:
        #             predict_ratings[u][j] = predict_ratings[u][j] / math.log(len(participant_list[j]), 2)
        tmp = getTopN(predict_ratings[u], cand_events)[:top_n]
        sub_list = []
        for j in tmp:
            sub_list.append(j[1])
        recommend_list.append(sub_list)
    del cand_events, ue
    gc.collect()
    inter_div = 0
    cand_user_num = len(cand_users)
    for u, _ in enumerate(cand_users):
        for t in range(u + 1, cand_user_num):
            common_list = [i for i in recommend_list[u] if i in recommend_list[t]]
            inter_div += 1 - len(common_list) / top_n
    inter_div /= (cand_user_num * (cand_user_num - 1)) / 2

    result_set = set()
    Precision = 0
    Recall = 0
    Novelty = 0
    hits = 0
    NDCG = 0
    iDCG = 0
    DCG = 0
    for i in range(1,top_n+1):
        iDCG += math.log(i + 1,2)
    for i, u in enumerate(cand_users):
        if len(u_testset[u]) > 1:
            for j in range(top_n):
                result_set.add(recommend_list[i][j])
                if recommend_list[i][j] in u_testset[u]:
                    hits += 1
                    DCG += math.log(j + 2, 2)
                Novelty -= (math.log((1+participant_list[recommend_list[i][j]])/(1+max_k), 2)/math.log(j + 2, 2))
            Precision += top_n
            Recall += len(u_testset[u])

    Novelty = Novelty / top_n / cand_user_num
    Precision = hits / Precision
    Recall = hits / Recall
    NDCG = DCG / cand_user_num / iDCG

    print(hits)
    f1 = 2 * Precision * Recall / (Recall + Precision)
    f = open(result_name, 'w')
    f.write('precisions=' + str(Precision) + '\n')
    f.write('recall=' + str(Recall) + '\n')
    f.write('f1=' + str(f1) + '\n')
    f.write('NDCG=' + str(NDCG) + '\n')
    f.write('Coverage=' + str(len(result_set)/i_num) + '\n')
    f.write('novelty=' + str(Novelty))
    f.close()
