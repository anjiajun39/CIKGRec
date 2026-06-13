import multiprocessing
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import time

cores = max(1, multiprocessing.cpu_count() // 2)

Ks = [20, 50, 100]
max_K = max(Ks)

score_matrix = None
rated_dict = None
candidate_dict = None
user_num = None
item_num = None

def init_eval_worker(worker_score_matrix, worker_rated_dict, worker_candidate_dict, worker_user_num, worker_item_num):
    global score_matrix
    global rated_dict
    global candidate_dict
    global user_num
    global item_num

    score_matrix = worker_score_matrix
    rated_dict = worker_rated_dict
    candidate_dict = worker_candidate_dict
    user_num = worker_user_num
    item_num = worker_item_num


def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)

def dcg_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k, ground_truth):
    GT = set(ground_truth)
    if len(GT) > k :
        sent_list = [1.0] * k
    else:
        sent_list = [1.0]*len(GT) + [0.0]*(k-len(GT))
    dcg_max = dcg_at_k(sent_list, k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = r[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    r = r[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.

def test_one_user(uid):
    global score_matrix
    global rated_dict
    global candidate_dict
    global user_num
    global item_num
    score = score_matrix[uid].copy()
    #mask the train items
    try:
        rated_items = np.array(list(rated_dict[uid]))
        #sheild the train items
        score[rated_items-user_num] = -(1<<10)
    except:
        rated_items = np.array([])

    y_pred = score  # (1, item_num)
    #get true，and convert itemid to start with 0
    true_list = [x-user_num for x in candidate_dict[uid]]
    y_true = np.array([0.0]*item_num)
    y_true[true_list] = 1.0
    
    ground_true_num = len(true_list)
    precision, recall, ndcg, hit_ratio = [], [], [], []
    #Find the k-largest in O(N)
    topk_idx = np.argpartition(y_pred, -max_K)[-max_K:]
    #Sort the k-largest
    order = topk_idx[np.argsort(y_pred[topk_idx])[::-1]]
    recall_sort = np.take(y_true, order)
    for k in Ks:
        precision.append(precision_at_k(recall_sort, k))
        recall.append(recall_at_k(recall_sort, k, ground_true_num))
        ndcg.append(ndcg_at_k(recall_sort, k, true_list))
        hit_ratio.append(hit_at_k(recall_sort, k))
    
    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio)}

def eval_model(model, loader, dtype='test'):
    global score_matrix
    global rated_dict
    global candidate_dict
    global user_num
    global item_num

    start = time.time()
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks))}
    rated_dict, candidate_dict = loader.get_eval_data(dtype)
    user_num = model.users
    item_num = model.items
    ndcg = []
    recall = []
    model.eval()
    with torch.no_grad():
        score_matrix = model.get_score_matrix()
        score_matrix = score_matrix.detach().cpu().numpy()       
    pred_true_uid = []
    test_users = list(candidate_dict.keys())
    n_test_users = len(test_users)
    if multiprocessing.get_start_method(allow_none=True) == 'fork' and cores > 1:
        worker_args = (score_matrix, rated_dict, candidate_dict, user_num, item_num)
        with multiprocessing.Pool(cores, initializer=init_eval_worker, initargs=worker_args) as pool:
            batch_result = pool.map(test_one_user, test_users)
    else:
        batch_result = [test_one_user(uid) for uid in test_users]
    for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users        
    end = time.time()
    # print(f'Evaluate time: {round(end-start, 1)}s')
    return result
