import sys
import random
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

ADVERTISERS = ["1458", "2259", "2261", "2821", "2997", "3358", "3386", "3427", "3476"]
chanel_index = 11

def ints(s):
    res = []
    for ss in s:
        res.append(int(ss))
    return res

def sigmoid(p):
    return 1.0 / (1.0 + math.exp(-p))

def estimator_lr(feats):
    pred = 0.0
    for feat in feats:
        if feat in featWeight:
            pred += featWeight[feat]
    pred = sigmoid(pred)
    return pred

for adv in ADVERTISERS:
    mplist = []
    y = []
    yp = []
    mplist_train = []
    y_train = []
    yp_train = []
    featWeight = {}
    exchange_train = []
    exchange = []


    #initialize the lr
    fi = open("../../make-ipinyou-data/"+adv+"/train.yzx.txt.lr.weight", 'r')
    for line in fi:
        s = line.strip().split()
        feat = int(s[0])
        weight = float(s[1])
        featWeight[feat] = weight
    fi.close()

    fi = open("../../make-ipinyou-data/"+adv+"/test.yzx.txt", 'r')
    print "read " + adv + " test yzx"
    for line in fi:
        data = ints(line.strip().replace(":1", "").split())
        clk = data[0]
        mp = data[1]
        fsid = 2 # feature start id
        feats = data[fsid:]
        pred = estimator_lr(feats)
        y.append(clk)
        yp.append(pred)
        mplist.append(mp)
    fi.close()

    fi = open("../../make-ipinyou-data/"+adv+"/train.yzx.txt", 'r')
    print "read " + adv + " train yzx"
    for line in fi:
        data = ints(line.strip().replace(":1", "").split())
        clk = data[0]
        mp = data[1]
        fsid = 2 # feature start id
        feats = data[fsid:]
        pred = estimator_lr(feats)
        y_train.append(clk)
        yp_train.append(pred)
        mplist_train.append(mp)
    fi.close()

    fi = open("../../make-ipinyou-data/"+adv+"/train.log.txt", 'r')
    print "read " + adv + " train log"
    for i, line in enumerate(fi):
        if i > 0:
            data = line.split("\t")
            exchange_train.append(data[chanel_index])
    fi.close

    fi = open("../../make-ipinyou-data/"+adv+"/test.log.txt", 'r')
    print "read " + adv + " test log"
    for i, line in enumerate(fi):
        if i > 0:
            data = line.split("\t")
            exchange.append(data[chanel_index])
    fi.close

    fo = open("../../make-ipinyou-data/"+adv+"/train.yzpc.txt", 'w')
    print "write " + adv + " train yzpc"
    for i, val in enumerate(exchange_train):
        if i == len(exchange_train) - 1:
            fo.write("%d\t%d\t%f\t%s" % (y_train[i], mplist_train[i], yp_train[i], val))
        else:
            fo.write("%d\t%d\t%f\t%s\n" % (y_train[i], mplist_train[i], yp_train[i], val))
    fo.close()

    fo = open("../../make-ipinyou-data/"+adv+"/test.yzpc.txt", 'w')
    print "write " + adv + " test yzpc"
    for i, val in enumerate(exchange):
        if i == len(exchange_train) - 1:
            fo.write("%d\t%d\t%f\t%s" % (y[i], mplist[i], yp[i], val))
        else:
            fo.write("%d\t%d\t%f\t%s\n" % (y[i], mplist[i], yp[i], val))
    fo.close()