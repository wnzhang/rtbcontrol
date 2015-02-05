#!/usr/bin/python
import sys
import random
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

advertiser = "1458"
mode = "test"
ref = 40000
advs_test_bids = 100000
advs_train_bids = 100000
advs_train_clicks = 79
advs_test_clicks = 65
basebid = 69

print "Example of PID control eCPC."
print "Data sample from campaign 1458 from iPinYou dataset."
print "Reference eCPC: " + str(ref)

# parameter setting
minbid = 5
cntr_rounds = 40
para_p = 0.003
para_i = 0.000001
para_d = 0.0001
div = 1e-6
para_ps = range(0, 40, 5)
para_is = range(0, 25, 5)
para_ds = range(0, 25, 5)
settle_con = 0.1
rise_con = 0.9
min_phi = -2
max_phi = 5


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
def control(cntr_rounds, ref, para_p, para_i, para_d, outfile):
    fo = open(outfile, 'w')
    fo.write("round\tecpc\tstage\tphi\ttotal_click\tclick_ratio\twin_ratio\ttotal_cost\tref\n")
    ecpcs = {}
    error_sum = 0.0
    first_round = True
    sec_round = False
    cntr_size = int(len(yp) / cntr_rounds)
    total_cost = 0.0
    total_clks = 0
    total_wins = 0
    tc = {}
    for round in range(0, cntr_rounds):
        if first_round and (not sec_round):
            phi = 0.0
            first_round = False
            sec_round = True
        elif sec_round and (not first_round):
            error = ref - ecpcs[round-1]
            error_sum += error
            phi = para_p*error + para_i*error_sum
            sec_round = False
        else:
            error = ref - ecpcs[round-1]
            error_sum += error
            phi = para_p*error + para_i*error_sum + para_d*(ecpcs[round-2]-ecpcs[round-1])
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
        click_ratio = total_clks * 1.0 / advs_test_clicks
        win_ratio = total_wins * 1.0 / advs_test_bids
        fo.write("%d\t%.4f\t%s\t%.4f\t%d\t%.4f\t%.4f\t%.4f\t%.1f\n" % (round, ecpcs[round], "test", phi, total_clks,  click_ratio, win_ratio, total_cost, ref))
    for round in range(0, cntr_rounds):
        fo.write("%d\t%.4f\t%s\t%.4f\t%d\t%.4f\t%.4f\t%.4f\t%.1f\n" % (round, ref, "test-ref", 0.0, 0, 0.0, 0.0, tc[round], ref))
    overshoot.append(cal_overshoot(ecpcs, ref))
    settling_time.append(cal_settling_time(ecpcs, ref))
    rise_time.append(cal_rise_time(ecpcs, ref, rise_con))
    rmse_ss.append(cal_rmse_ss(ecpcs, ref))
    sd_ss.append(cal_sd_ss(ecpcs, ref))

    # train
    ecpcs_train = {}
    error_sum = 0.0
    first_round = True
    sec_round = False
    cntr_size = int(len(yp_train) / cntr_rounds)
    total_cost = 0.0
    total_clks = 0
    total_wins = 0
    tc_train = {}
    for round in range(0, cntr_rounds):
        if first_round and (not sec_round):
            phi = 0.0
            first_round = False
            sec_round = True
        elif sec_round and (not first_round):
            error = ref - ecpcs_train[round-1]
            error_sum += error
            phi = para_p*error + para_i*error_sum
            sec_round = False
        else:
            error = ref - ecpcs_train[round-1]
            error_sum += error
            phi = para_p*error + para_i*error_sum + para_d*(ecpcs_train[round-2]-ecpcs_train[round-1])
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
        tc_train[round] = total_cost
        ecpcs_train[round] = total_cost / (total_clks+1)
        click_ratio = total_clks * 1.0 / advs_train_clicks
        win_ratio = total_wins * 1.0 / advs_train_bids
        fo.write("%d\t%.4f\t%s\t%.4f\t%d\t%.4f\t%.4f\t%.4f\t%.1f\n" % (round, ecpcs_train[round], "train", phi, total_clks,  click_ratio, win_ratio, total_cost, ref))
    for round in range(0, cntr_rounds):
        fo.write("%d\t%.4f\t%s\t%.4f\t%d\t%.4f\t%.4f\t%.4f\t%.1f\n" % (round, ref, "train-ref", 0.0, 0,  0.0, 0.0, tc_train[round], ref))
    fo.close()

def control_test(cntr_rounds, ref, para_p, para_i , para_d):
    ecpcs = {}
    error_sum = 0.0
    first_round = True
    sec_round = False
    cntr_size = int(len(yp) / cntr_rounds)
    total_cost = 0.0
    total_clks = 0
    total_wins = 0
    print "test performance:"
    print "round\tecpc\tphi\ttotal_click\tclick_ratio\twin_ratio\ttotal_cost\tref"
    for round in range(0, cntr_rounds):
        if first_round and (not sec_round):
            phi = 0.0
            first_round = False
            sec_round = True
        elif sec_round and (not first_round):
            error = ref - ecpcs[round-1]
            error_sum += error
            phi = para_p*error + para_i*error_sum
            sec_round = False
        else:
            error = ref - ecpcs[round-1]
            error_sum += error
            phi = para_p*error + para_i*error_sum + para_d*(ecpcs[round-2]-ecpcs[round-1])
        cost = 0
        clks = 0

        imp_index = ((round+1)*cntr_size)

        if round == cntr_rounds - 1:
            imp_index = imp_index + (len(yp) - cntr_size*cntr_rounds)

        # phi bound
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
        click_ratio = total_clks * 1.0 / advs_test_clicks
        win_ratio = total_wins * 1.0 / advs_test_bids
        print "%d\t%.4f\t%.4f\t%d\t%.4f\t%.4f\t%.4f\t%.1f" % (round, ecpcs[round], phi, total_clks, click_ratio, win_ratio, total_cost, ref)
    overshoot.append(cal_overshoot(ecpcs, ref))
    settling_time.append(cal_settling_time(ecpcs, ref))
    rise_time.append(cal_rise_time(ecpcs, ref, rise_con))
    rmse_ss.append(cal_rmse_ss(ecpcs, ref))
    sd_ss.append(cal_sd_ss(ecpcs, ref))

    # train
    print "\ntrain performance:"
    print "round\tecpc\tphi\ttotal_click\tclick_ratio\twin_ratio\ttotal_cost\tref"
    ecpcs_train = {}
    error_sum = 0.0
    first_round = True
    sec_round = False
    cntr_size = int(len(yp_train) / cntr_rounds)
    total_cost = 0.0
    total_clks = 0
    total_wins = 0
    for round in range(0, cntr_rounds):
        if first_round and (not sec_round):
            phi = 0.0
            first_round = False
            sec_round = True
        elif sec_round and (not first_round):
            error = ref - ecpcs_train[round-1]
            error_sum += error
            phi = para_p*error + para_i*error_sum
            sec_round = False
        else:
            error = ref - ecpcs_train[round-1]
            error_sum += error
            phi = para_p*error + para_i*error_sum + para_d*(ecpcs_train[round-2]-ecpcs_train[round-1])
        cost = 0
        clks = 0

        imp_index = ((round+1)*cntr_size)

        if round == cntr_rounds - 1:
            imp_index = imp_index + (len(yp_train) - cntr_size*cntr_rounds)

        # phi bound
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
        click_ratio = total_clks * 1.0 / advs_train_clicks
        win_ratio = total_wins * 1.0 / advs_train_bids
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


#initialize the lr
fi = open("../exp-data/train.txt", 'r')
for line in fi:
    s = line.strip().split()
    y_train.append(int(s[0]))
    mplist_train.append(int(s[1]))
    yp_train.append(float(s[2]))
fi.close()

fi = open("../exp-data/test.txt", 'r')
for line in fi:
    s = line.strip().split()
    y.append(int(s[0]))
    mplist.append(int(s[1]))
    yp.append(float(s[2]))
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
    report_path = "../report/report-test.tsv"
    parameter = ""+advertiser+"\t"+str(cntr_rounds)+"\t"+str(basebid)+"\t"+str(ref)+"\t" + \
                str(para_p)+"\t"+str(para_i)+"\t"+str(para_d)+"\t"+str(settle_con)+"\t"+str(rise_con)
    parameters.append(parameter)
    control_test(cntr_rounds, ref, para_p, para_i, para_d)
    rout = open(report_path, 'w')
    rout.write("campaign\ttotal-rounds\tbase-bid\tref\tp\ti\td\tsettle-con\trise-con\trise-time\tsettling-time\tovershoot\trmse-ss\tsd-ss\n")
    for idx, val in enumerate(parameters):
        rout.write(val+"\t"+str(rise_time[idx])+"\t"+str(settling_time[idx])+"\t"+str(overshoot[idx])+"\t" + \
                   str(rmse_ss[idx]) + "\t" + str(sd_ss[idx]))
    rout.close()
elif mode == "batch":# batch mode
    report_path = "../report/report-batch.tsv"
    for temp_p in para_ps:
        for temp_i in para_is:
            for temp_d in para_ds:
                para_p = temp_p * 1.0 * div
                para_i = temp_i * 1.0 * div
                para_d = temp_d * 1.0 * div
                out_path = "../exp-data/"+advertiser+"_ref="+str(ref)+"_p=" + \
                           str(para_p)+"_i="+str(para_i)+"_d="+str(para_d)+".tsv"
                control(cntr_rounds, ref, para_p, para_i, para_d, out_path)
                parameter = ""+advertiser+"\t"+str(cntr_rounds)+"\t"+str(basebid)+"\t"+str(ref)+"\t" + \
                           str(para_p)+"\t"+str(para_i)+"\t"+str(para_d)+"\t"+str(settle_con)+"\t"+str(rise_con)
                parameters.append(parameter)
    rout = open(report_path, 'w')
    rout.write("campaign\ttotal-rounds\tbase-bid\tref\tp\ti\td\tsettle-con\trise-con\trise-time\tsettling-time\t\overshoot\trmse-ss\tsd-ss\n")
    for idx, val in enumerate(parameters):
        rout.write(val+"\t"+str(rise_time[idx])+"\t"+str(settling_time[idx])+"\t"+str(overshoot[idx])+"\t" + \
                   str(rmse_ss[idx]) + "\t" + str(sd_ss[idx]))
    rout.close()
elif mode == "single": # single mode
    out_path = "../exp-data/"+advertiser+"_ref="+str(ref)+"_p="+str(para_p)+"_i="+str(para_i)+"_d="+str(para_d)+".tsv"
    control(cntr_rounds, ref, para_p, para_i, para_d, out_path)
    report_path = "../exp-data/report-single.tsv"
    parameter = ""+advertiser+"\t"+str(cntr_rounds)+"\t"+str(basebid)+"\t"+str(ref)+"\t" + \
                str(para_p)+"\t"+str(para_i)+"\t"+str(para_d)+"\t"+str(settle_con)+"\t"+str(rise_con)
    parameters.append(parameter)
    rout = open(report_path, 'w')
    rout.write("campaign\ttotal-rounds\tbase-bid\tref\tp\ti\td\tsettle-con\trise-con\trise-time\tsettling-time\tovershoot\trmse-ss\tsd-ss\n")
    for idx, val in enumerate(parameters):
        rout.write(val+"\t"+str(rise_time[idx])+"\t"+str(settling_time[idx])+"\t"+str(overshoot[idx])+"\t" + \
                   str(rmse_ss[idx]) + "\t" + str(sd_ss[idx]))
    rout.close()
else:
    print "wrong mode entered"





