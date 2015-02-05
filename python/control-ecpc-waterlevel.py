#!/usr/bin/python
import sys
import random
import math

advs_train_bids = {"1458": 3083056, "2259": 835556, "2261": 687617, "2821": 1322561, "2997": 312437, "3358": 1742104, "3386": 2847802, "3427": 2593765, "3476": 1970360}
advs_test_bids = {"1458": 614638, "2259": 417197, "2261": 343862, "2821": 661964, "2997": 156063, "3358": 300928, "3386": 545421, "3427": 536795, "3476": 523848}
advs_train_clicks = {"1458": 2454, "2259": 280, "2261": 207, "2821": 843, "2997": 1386, "3358": 1358, "3386": 2076, "3427": 1926, "3476": 1027}
advs_test_clicks = {"1458": 543, "2259": 131, "2261": 97, "2821": 394, "2997": 533, "3358": 339, "3386": 496, "3427": 395, "3476": 302}
advs_train_cost = {"1458": 212400000, "2259": 77754000, "2261": 61610000, "2821": 118082000, "2997": 19689000, "3358": 160943000, "3386": 219066000, "3427": 210239000, "3476": 156088000}
advs_test_cost = {"1458": 45216000, "2259": 43497000, "2261": 28795000, "2821": 68257000, "2997": 8617000, "3358": 34159000, "3386": 45715000, "3427": 46356000, "3476": 43627000}

advertiser = "1458"
mode = "single"
basebid = 69
ref = 20000
temp = 20000
print "%s\t%s\t%d\t%f" % (advertiser, mode, basebid, ref)

# parameter setting
minbid = 5
cntr_rounds = 40
para_gamma = 200
div = 1
para_gammas = range(60, 120, 5)
settle_con = 0.1
rise_con = 0.9
min_phi = 0.7
max_phi = 100
damping = 0.25
budget_train = damping * advs_train_cost[advertiser]
budget_test = damping * advs_test_cost[advertiser]
print "test-budget: " + str(budget_test)
print "train-budget: " + str(budget_train)
max_ref = ref * 1.05
min_ref = ref * 0.95

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

# bidding functions
def lin(pctr, basectr, basebid):
    return int(pctr *  basebid / basectr)

# calculate settling time
def cal_settling_time(ecpcs, ref):
    settled = False
    settling_time = 0
    for key, value in ecpcs.iteritems():
        error = ref - value
        if abs(error) / ref <= settle_con and settled == False:
            settled = True
            settling_time = key
        elif abs(error) / ref > settle_con:
            settled = False
            settling_time = cntr_rounds
    return settling_time

# # calculate steady-state error
def cal_rmse_ss(ecpcs, ref):
    settled = False
    settling_time = cal_settling_time(ecpcs, ref)
    rmse = 0.0
    if settling_time >= cntr_rounds:
        settling_time = cntr_rounds - 1
    for round in range(settling_time, cntr_rounds):
        rmse += (ecpcs[round] - ref) * (ecpcs[round] - ref)
    rmse /= (cntr_rounds - settling_time)
    rmse = math.sqrt(rmse) / ref # weinan: relative rmse
    return rmse

# # calculate steady-state standard deviation
def cal_sd_ss(ecpcs, ref):
    settled = False
    settling_time = cal_settling_time(ecpcs, ref)
    if settling_time >= cntr_rounds:
        settling_time = cntr_rounds - 1
    sum2 = 0.0
    sum = 0.0
    for round in range(settling_time, cntr_rounds):
        sum2 += ecpcs[round] * ecpcs[round]
        sum += ecpcs[round]
    n = cntr_rounds - settling_time
    mean = sum / n
    sd = math.sqrt(sum2 / n - mean * mean) / mean # weinan: relative sd
    return sd

# calculate rise time
def cal_rise_time(ecpcs, ref, rise_con):
    rise_time = 0
    for key, value in ecpcs.iteritems():
        error = ref - value
        if abs(error) / ref <= (1 - rise_con):
            rise_time = key
            break
    return rise_time

# calculate percentage overshoot
def cal_overshoot(ecpcs, ref):
    if ecpcs[0] > ref:
        min = ecpcs[0];
        for key, value in ecpcs.iteritems():
            if value <= min:
                min = value
        if min < ref:
            return (ref - min) * 100.0 / ref
        else:
            return 0.0
    elif ecpcs[0] < ref:
        max = ecpcs[0]
        for key, value in ecpcs.iteritems():
            if value >= max:
                max = value
        if max > ref:
            return (max - ref) * 100.0 / ref
        else:
            return 0.0
    else:
        max = 0
        for key, value in ecpcs.iteritems():
            if abs(value - ref) >= max:
                max = value
        return (max - ref) * 100.0 / ref

# control function
def control(cntr_rounds, ref, para_gamma, outfile):
    fo = open(outfile, 'w')
    fo.write("round\tecpc\tstage\tphi\ttotal_click\tclick_ratio\twin_ratio\ttotal_cost\tref\n")
    ecpcs = {}
    click_nums = {}
    first_round = True
    cntr_size = int(len(yp) / cntr_rounds)
    total_cost = 0.0
    total_clks = 0
    total_wins = 0
    temp_ref = {}
    temp_ref[0] = temp
    tc = {}
    for round in range(0, cntr_rounds):
        if first_round:
            phi = 0.0
            first_round = False
        else:
            temp_ref[round] = (budget_test - total_cost) / ((budget_test / ref) - ((total_cost / ecpcs[round-1])))
            if temp_ref[round] <= min_ref:
                temp_ref[round] = min_ref
            elif temp_ref[round] >= max_ref:
                temp_ref[round] = max_ref
            # error = click_nums[round-1] * 1.0 / (budget_test / temp_ref[round]) - 1.0 / cntr_rounds
            error = click_nums[round-1] * 1.0 / (budget_test / ref) - 1.0 / cntr_rounds
            phi = para_gamma * (-1) * error
        cost = 0
        clks = 0

        imp_index = ((round+1)*cntr_size)

        if round == cntr_rounds - 1:
            imp_index = imp_index + (len(yp) - cntr_size*cntr_rounds)

        # fang piao
        if phi <= min_phi:
            phi = min_phi
        elif phi >= max_phi:
            phi = max_phi

        for i in range(round*cntr_size, imp_index):
            clk = y[i]
            pctr = yp[i]
            mp = mplist[i]
            bid = max(minbid,lin(pctr, basectr, basebid) * (math.exp(phi)))
            if round == 0:
                bid = 1000.0

            if bid > mp:
                total_wins += 1
                clks += clk
                total_clks += clk
                cost += mp
                total_cost += mp
        tc[round] = total_cost
        ecpcs[round] = total_cost / (total_clks+1)
        click_nums[round] = clks
        click_ratio = total_clks * 1.0 / advs_test_clicks[advertiser]
        win_ratio = total_wins * 1.0 / advs_test_bids[advertiser]
        fo.write("%d\t%.4f\t%s\t%.4f\t%d\t%.4f\t%.4f\t%.4f\t%.1f\n" % (round, ecpcs[round], "test", phi, total_clks,  click_ratio, win_ratio, total_cost, ref))
    for round in range(0, cntr_rounds):
        fo.write("%d\t%.4f\t%s\t%.4f\t%d\t%.4f\t%.4f\t%.4f\t%.1f\n" % (round, ref, "test-ref", 0.0, 0,  0.0, 0.0, tc[round], ref))
    overshoot.append(cal_overshoot(ecpcs, ref))
    settling_time.append(cal_settling_time(ecpcs, ref))
    rise_time.append(cal_rise_time(ecpcs, ref, rise_con))
    rmse_ss.append(cal_rmse_ss(ecpcs, ref))
    sd_ss.append(cal_sd_ss(ecpcs, ref))

    # train
    ecpcs_train = {}
    click_nums_train = {}
    first_round = True
    cntr_size = int(len(yp_train) / cntr_rounds)
    total_cost = 0.0
    total_clks = 0
    total_wins = 0
    temp_ref = {}
    temp_ref[0] = temp
    tc = {}
    for round in range(0, cntr_rounds):
        if first_round:
            phi = 0.0
            first_round = False
        else:
            temp_ref[round] = (budget_train - total_cost) / ((budget_train / ref)  - ((total_cost / ecpcs_train[round-1])))
            if temp_ref[round] <= min_ref:
                temp_ref[round] = min_ref
            elif temp_ref[round] >= max_ref:
                temp_ref[round] = max_ref
            # error = click_nums_train[round-1] * 1.0 / (budget_train / temp_ref[round]) - 1.0 / cntr_rounds
            error = click_nums_train[round-1] * 1.0 / (budget_train / ref) - 1.0 / cntr_rounds
            phi = para_gamma * (-1) * error
        cost = 0
        clks = 0

        imp_index = ((round+1)*cntr_size)

        if round == cntr_rounds - 1:
            imp_index = imp_index + (len(yp_train) - cntr_size*cntr_rounds)

        # fang piao
        if phi <= min_phi:
            phi = min_phi
        elif phi >= max_phi:
            phi = max_phi

        for i in range(round*cntr_size, imp_index):
            clk = y_train[i]
            pctr = yp_train[i]
            mp = mplist_train[i]
            bid = max(minbid,lin(pctr, basectr, basebid) * (math.exp(phi)))
            if round == 0:
                bid = 1000.0

            if bid > mp:
                total_wins += 1
                clks += clk
                total_clks += clk
                cost += mp
                total_cost += mp
        tc[round] = total_cost
        ecpcs_train[round] = total_cost / (total_clks+1)
        click_nums_train[round] = clks
        click_ratio = total_clks * 1.0 / advs_train_clicks[advertiser]
        win_ratio = total_wins * 1.0 / advs_train_bids[advertiser]
        fo.write("%d\t%.4f\t%s\t%.4f\t%d\t%.4f\t%.4f\t%.4f\t%.1f\n" % (round, ecpcs_train[round], "train", phi, total_clks,  click_ratio, win_ratio, total_cost, ref))
    for round in range(0, cntr_rounds):
        fo.write("%d\t%.4f\t%s\t%.4f\t%d\t%.4f\t%.4f\t%.4f\t%.1f\n" % (round, ref, "train-ref", 0.0, 0,  0.0, 0.0, total_cost, ref))
    fo.close()

def control_test(cntr_rounds, ref, para_gamma):
    ecpcs = {}
    click_nums = {}
    first_round = True
    cntr_size = int(len(yp) / cntr_rounds)
    total_cost = 0.0
    total_clks = 0
    total_wins = 0
    temp_ref = {}
    temp_ref[0] = temp
    for round in range(0, cntr_rounds):
        if first_round:
            phi = 0.0
            first_round = False
        else:
            temp_ref[round] = (budget_test - total_cost) / ((budget_test / ref) - ((total_cost / ecpcs[round-1])))
            if temp_ref[round] <= min_ref:
                temp_ref[round] = min_ref
            elif temp_ref[round] >= max_ref:
                temp_ref[round] = max_ref
            # error = click_nums[round-1] * 1.0 / (budget_test / temp_ref[round]) - 1.0 / cntr_rounds
            error = click_nums[round-1] * 1.0 / (budget_test / ref) - 1.0 / cntr_rounds
            phi = para_gamma * (-1) * error
        cost = 0
        clks = 0

        imp_index = ((round+1)*cntr_size)

        if round == cntr_rounds - 1:
            imp_index = imp_index + (len(yp) - cntr_size*cntr_rounds)

        # fang piao
        if phi <= min_phi:
            phi = min_phi
        elif phi >= max_phi:
            phi = max_phi

        for i in range(round*cntr_size, imp_index):
            clk = y[i]
            pctr = yp[i]
            mp = mplist[i]
            bid = max(minbid,lin(pctr, basectr, basebid) * (math.exp(phi)))
            if round == 0:
                bid = 1000.0

            if bid > mp:
                total_wins += 1
                clks += clk
                total_clks += clk
                cost += mp
                total_cost += mp
        ecpcs[round] = total_cost / (total_clks+1)
        click_nums[round] = clks
        click_ratio = total_clks * 1.0 / advs_test_clicks[advertiser]
        win_ratio = total_wins * 1.0 / advs_test_bids[advertiser]
        print "%d\t%.4f\t%.4f\t%d\t%.4f\t%.4f\t%.4f\t%.1f" % (round, ecpcs[round], phi, total_clks, click_ratio, win_ratio, total_cost, ref)
    overshoot.append(cal_overshoot(ecpcs, ref))
    settling_time.append(cal_settling_time(ecpcs, ref))
    rise_time.append(cal_rise_time(ecpcs, ref, rise_con))
    rmse_ss.append(cal_rmse_ss(ecpcs, ref))
    sd_ss.append(cal_sd_ss(ecpcs, ref))

    # train
    ecpcs_train = {}
    click_nums_train = {}
    first_round = True
    cntr_size = int(len(yp_train) / cntr_rounds)
    total_cost = 0.0
    total_clks = 0
    total_wins = 0
    temp_ref = {}
    temp_ref[0] = temp
    for round in range(0, cntr_rounds):
        if first_round:
            phi = 0.0
            first_round = False
        else:
            temp_ref[round] = (budget_train - total_cost) / ((budget_train / ref)  - ((total_cost / ecpcs_train[round-1])))
            if temp_ref[round] <= min_ref:
                temp_ref[round] = min_ref
            elif temp_ref[round] >= max_ref:
                temp_ref[round] = max_ref
            # error = click_nums_train[round-1] * 1.0 / (budget_train / temp_ref[round]) - 1.0 / cntr_rounds
            error = click_nums_train[round-1] * 1.0 / (budget_train / ref) - 1.0 / cntr_rounds
            phi = para_gamma * (-1) * error
        cost = 0
        clks = 0

        imp_index = ((round+1)*cntr_size)

        if round == cntr_rounds - 1:
            imp_index = imp_index + (len(yp_train) - cntr_size*cntr_rounds)

        # fang piao
        if phi <= min_phi:
            phi = min_phi
        elif phi >= max_phi:
            phi = max_phi

        for i in range(round*cntr_size, imp_index):
            clk = y_train[i]
            pctr = yp_train[i]
            mp = mplist_train[i]
            bid = max(minbid,lin(pctr, basectr, basebid) * (math.exp(phi)))
            if round == 0:
                bid = 1000.0

            if bid > mp:
                total_wins += 1
                clks += clk
                total_clks += clk
                cost += mp
                total_cost += mp
        ecpcs_train[round] = total_cost / (total_clks+1)
        click_nums_train[round] = clks
        click_ratio = total_clks * 1.0 / advs_train_clicks[advertiser]
        win_ratio = total_wins * 1.0 / advs_train_bids[advertiser]
        print "%d\t%.4f\t%.4f\t%d\t%.4f\t%.4f\t%.4f\t%.1f" % (round, ecpcs_train[round], phi, total_clks, click_ratio, win_ratio, total_cost, ref)

random.seed(10)

# if len(sys.argv) != 3:
#     print 'campaignId mode'
#     exit(-1)

mplist = []
y = []
yp = []
mplist_train = []
y_train = []
yp_train = []
featWeight = {}


#initialize the lr
fi = open("../../make-ipinyou-data/"+advertiser+"/train.yzx.txt.lr.weight", 'r')
for line in fi:
    s = line.strip().split()
    feat = int(s[0])
    weight = float(s[1])
    featWeight[feat] = weight
fi.close()

fi = open("../../make-ipinyou-data/"+advertiser+"/test.yzx.txt", 'r')
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

fi = open("../../make-ipinyou-data/"+advertiser+"/train.yzx.txt", 'r')
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

basectr = sum(yp_train) / float(len(yp_train))

# for reporting
parameters = []
overshoot = []
settling_time = []
rise_time = []
rmse_ss = []
sd_ss = []
report_path = ""


if mode == "test": # test mode
    report_path = "../report/report-ecpc-waterlevel-test.tsv"
    parameter = ""+advertiser+"\t"+str(cntr_rounds)+"\t"+str(basebid)+"\t"+str(ref)+"\t" + \
                str(para_gamma)+"\t"+str(settle_con)+"\t"+str(rise_con)
    parameters.append(parameter)
    control_test(cntr_rounds, ref, para_gamma)
    rout = open(report_path, 'w')
    rout.write("campaign\ttotal-rounds\tbase-bid\tref\tgamma\tsettle-con\trise-con\trise-time\tsettling-time\tovershoot\trmse-ss\tsd-ss\n")
    for idx, val in enumerate(parameters):
        rout.write(val+"\t"+str(rise_time[idx])+"\t"+str(settling_time[idx])+"\t"+str(overshoot[idx])+"\t" + \
                   str(rmse_ss[idx]) + "\t" + str(sd_ss[idx]))
    rout.close()
elif mode == "batch":# batch mode
    report_path = "../report/report-ecpc-waterlevel-batch.tsv"
    for temp_gamma in para_gammas:
        para_gamma = temp_gamma * 1.0 * div
        out_path = "../exp-data/win_waterlevel_"+advertiser+"_ref="+str(ref)+"_gamma=" + \
                    str(para_gamma)+".tsv"
        control(cntr_rounds, ref, para_gamma, out_path)
        parameter = ""+advertiser+"\t"+str(cntr_rounds)+"\t"+str(basebid)+"\t"+str(ref)+"\t" + \
                     str(para_gamma)+"\t"+str(settle_con)+"\t"+str(rise_con)
        parameters.append(parameter)
    rout = open(report_path, 'w')
    rout.write("campaign\ttotal-rounds\tbase-bid\tref\tgamma\tsettle-con\trise-con\trise-time\tsettling-time\t\overshoot\trmse-ss\tsd-ss\n")
    for idx, val in enumerate(parameters):
        rout.write(val+"\t"+str(rise_time[idx])+"\t"+str(settling_time[idx])+"\t"+str(overshoot[idx])+"\t" + \
                   str(rmse_ss[idx]) + "\t" + str(sd_ss[idx]))
    rout.close()
elif mode == "single": # single mode
    out_path = "../exp-data/win_waterlevel_"+advertiser+"_ref="+str(ref)+"_p="+str(para_gamma)+".tsv"
    control(cntr_rounds, ref, para_gamma, out_path)
    report_path = "../report/report-ecpc-waterlevel-single.tsv"
    parameter = ""+advertiser+"\t"+str(cntr_rounds)+"\t"+str(basebid)+"\t"+str(ref)+"\t" + \
                str(para_gamma)+"\t"+str(settle_con)+"\t"+str(rise_con)
    parameters.append(parameter)
    rout = open(report_path, 'w')
    rout.write("campaign\ttotal-rounds\tbase-bid\tref\tgamma\tsettle-con\trise-con\trise-time\tsettling-time\tovershoot\trmse-ss\tsd-ss\n")
    for idx, val in enumerate(parameters):
        rout.write(val+"\t"+str(rise_time[idx])+"\t"+str(settling_time[idx])+"\t"+str(overshoot[idx])+"\t" + \
                   str(rmse_ss[idx]) + "\t" + str(sd_ss[idx]))
    rout.close()
else:
    print "wrong mode entered"