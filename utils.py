import torch
import random
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.data import Data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def calculate_metrics(y_true, y_pred):
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)
    sensitivity = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    mcc = (TP*TN-FP*FN)/np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))
    F1_score = 2*(precision*sensitivity)/(precision+sensitivity + 1e-10)
    return accuracy, sensitivity, precision, specificity, F1_score, mcc


def get_data(data_ID, output_dim):
    miRNA = np.loadtxt('data/dataset1/miRNA_feature.txt')  # (541, 541)
    SM = np.loadtxt('data/dataset1/SM_feature.txt')  # (831, 831)
    association = np.loadtxt('data/dataset1/miRNA_SM_adj.txt')  # (541, 831)
    if data_ID == 2:
        miRNA = np.loadtxt('data/dataset2/miRNA_feature.txt', delimiter=',')  # (286, 286)
        SM = np.loadtxt('data/dataset2/SM_feature.txt', delimiter=',')  # (39, 39)
        association = np.loadtxt('data/dataset2/miRNA_SM_adj.txt', delimiter=',')  # (286, 39)

    m_emb = []
    for m in range(len(miRNA)):
        m_emb.append(miRNA[m].tolist())
    m_emb = [lst + [0] * (output_dim - len(m_emb[0])) for lst in m_emb]
    m_emb = torch.Tensor(m_emb)

    s_emb = []
    for s in range(len(SM)):
        s_emb.append(SM[s].tolist())
    s_emb = [lst + [0] * (output_dim - len(s_emb[0])) for lst in s_emb]
    s_emb = torch.Tensor(s_emb)

    feature = torch.cat([m_emb, s_emb])

    adj = []
    for m in range(len(miRNA)):
        for s in range(len(SM)):
            if association[m][s] == 1:
                adj.append([m, s + len(miRNA)])
    adj = torch.LongTensor(adj).T
    data = Data(x=feature, edge_index=adj).cuda()

    train_data, _, test_data = T.RandomLinkSplit(num_val=0, num_test=0.2,
                                                 is_undirected=True, split_labels=True,
                                                 add_negative_train_samples=True)(data)

    splits = dict(train=train_data, test=test_data)
    return splits


if __name__ == '__main__':
    data = get_data(2, 1024)
