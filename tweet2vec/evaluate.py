'''
For evaluating precision and recall metrics
'''
import numpy as np
import sys
import pickle as pkl
import seaborn as sns
import codecs
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

K1 = 1
K2 = 2

HIST = False
CONF_MAT = True # Show confusion matrix
EVALUATE_USER_LVL = True # show scores per user, not per tweet

def precision(p, t, k):
    '''
    Compute precision @ k for predictions p and targets t
    '''
    n = p.shape[0]
    res = np.zeros(n)
    for idx in range(n):
        index = p[idx,:k]
        for i in index:
            if i in t[idx]:
                res[idx] += 1
    return np.sum(res)/(n*k)

def recall(p, t, k):
    '''
    Compute recall @ k for predictions p and targets k
    '''
    n = p.shape[0]
    res = np.zeros(n)
    for idx,items in enumerate(t):
        index = p[idx,:k]
        for i in items:
            if i in index:
                res[idx] += 1
        res[idx] = res[idx] / len(items)
    return np.sum(res)/n

def meanrank(p, t):
    '''
    Compute mean rank of targets in the predictions
    '''
    n = p.shape[0]
    res = np.zeros(n)
    for idx, items in enumerate(t):
        ind = p[idx,:]
        minrank = p.shape[1]+1
        for i in items:
            currrank = np.where(ind==i)[0]+1
            if currrank < minrank:
                minrank = currrank
        res[idx] = minrank
    return np.mean(res), res

def readable_predictions(p, t, d, k, u, labeldict):
    out = []
    for idx, item in enumerate(d):
        preds = p[idx,:k]
        plabels = ','.join([labeldict.keys()[ii-1] if ii > 0 else '<unk>' for ii in preds])
        tlabels = ','.join([labeldict.keys()[ii-1] if ii > 0 else '<unk>' for ii in t[idx]])
        out.append('%s\t%s\t%s\t%s\n'%(tlabels, plabels, u[idx], item))
    return out

def consolidate_users(predictions, targets, users):
    unique_users, user_indices, user_cnt = np.unique(users, return_inverse=True, return_counts=True)
    p = []
    t = []
    u = []
    for i in range(len(unique_users)):
        # fetch first occurrence of user and use its values (all values per user should be identical)
        idx = np.argmax(user_indices==i)
        p.append(predictions[idx])
        t.append(targets[idx])
        u.append(unique_users[i])

    return np.array(p), t, u


def main(result_path, dict_path):
    with open('%s/predictions.npy'%result_path,'rb') as f:
        p = np.load(f)
    with open('%s/targets.pkl'%result_path,'rb') as f:
        t = pkl.load(f)
    with open('%s/data.pkl'%result_path,'rb') as f:
        d = pkl.load(f)
    with open('%s/embeddings.npy'%result_path,'rb') as f:
        e = np.load(f)
    with open('%s/label_dict.pkl'%dict_path,'rb') as f:
        labeldict = pkl.load(f)

    if EVALUATE_USER_LVL:
        with open('%s/users.pkl' % result_path, 'rb') as f:
            u = pkl.load(f)

    readable = readable_predictions(p, t, d, 10, u, labeldict)
    with codecs.open('%s/readable.txt'%result_path,'w','utf-8') as f:
        for line in readable:
            f.write(line)

    if EVALUATE_USER_LVL:
        p, t, u = consolidate_users(p, t, u)

    meanr, allr = meanrank(p,t)
    print("Precision @ {} = {}".format(K1,precision(p,t,K1)))
    print("Recall @ {} = {}".format(K2,recall(p,t,K2)))
    print("Mean rank = {}".format(meanr))

    # histogram
    if HIST:
        hist, bins = np.histogram(allr, bins=50)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        plt.show()

    if CONF_MAT:
        conf_matrix = confusion_matrix(t,p[:,0])
        sns.heatmap(conf_matrix, annot=True, xticklabels=list(labeldict), yticklabels=list(labeldict))
        plt.show()

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2])
