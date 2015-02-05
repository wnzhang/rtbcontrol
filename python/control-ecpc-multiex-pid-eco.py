#!/usr/bin/python
import sys
import os
import random
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

advs_train_bids = {"1458": 3083056, "2259": 835556, "2261": 687617, "2821": 1322561, "2997": 312437, "3358": 1742104, "3386": 2847802, "3427": 2593765, "3476": 1970360}
advs_test_bids = {"1458": 614638, "2259": 417197, "2261": 343862, "2821": 661964, "2997": 156063, "3358": 300928, "3386": 545421, "3427": 536795, "3476": 523848}
advs_train_clicks = {"1458": 2454, "2259": 280, "2261": 207, "2821": 843, "2997": 1386, "3358": 1358, "3386": 2076, "3427": 1926, "3476": 1027}
advs_test_clicks = {"1458": 543, "2259": 131, "2261": 97, "2821": 394, "2997": 533, "3358": 339, "3386": 496, "3427": 395, "3476": 302}
advs_train_ori_ecpc = {"1458": 86550.0000, "2259": 277700.0000, "2261": 297640.0000, "2821": 140070.0000, "2997": 14210.0000, "3358": 118510.0000, "3386": 105520.0000, "3427": 109160.0000, "3476": 151980.0000}
advs_test_ori_ecpc = {"1458": 83270.0000, "2259": 332040.0000, "2261": 296870.0000, "2821": 173240.0000, "2997": 16170.0000, "3358": 100770.0000, "3386": 92170.0000, "3427": 117360.0000, "3476": 144460.0000}
advs_train_cost = {"1458": 212400000, "2259": 77754000, "2261": 61610000, "2821": 118082000, "2997": 19689000, "3358": 160943000, "3386": 219066000, "3427": 210239000, "3476": 156088000}
advs_test_cost = {"1458": 45216000, "2259": 43497000, "2261": 28795000, "2821": 68257000, "2997": 8617000, "3358": 34159000, "3386": 45715000, "3427": 46356000, "3476": 43627000}
advs_base_bid = {"1458": 69, "2259": 93, "2261":90, "2821":90, "2997":63, "3358":92, "3386":77, "3427": 81, "3476": 79}

advs_test_lin_ori_cost = {"1458": 1630539.0, "2259": 23220744.0, "2261": 14531891.0, "2821": 28152710.0, "2997": 4241173.0, "3358": 2518069.0, "3386": 13076826.0, "3427": 4986476.0, "3476": 9075114.0}
budget_damping = 0.5

# pre-calculated references using economics model
advs_adex_ref = {'3386': {'1': 19.650368000000004, '3': 37.711368000000014, '2': 44.213328000000004},
                 '2997': {'null': 5.983467456521738},
                 '3476': {'1': 19.33278951724137, '3': 23.360453999999994, '2': 26.20928985365854},
                 '3427': {'1': 9.885587179487173, '3': 8.355310526315792, '2': 6.185100000000001},
                 '2821': {'1': 61.942298684210535, '3': 51.99172861445784, '2': 64.04917959183673, '4': 49.54528928571428},
                 '2259': {'1': 155.06496331797234, '3': 102.05130064516129, '2': 136.23116210526317},
                 '2261': {'1': 68.34028703703704, '3': 99.87262888198757, '2': 139.2594528301887},
                 '1458': {'1': 2.580408429906542, '3': 2.2326444905660416, '2': 1.517053307692311},
                 '3358': {'1': 8.342108526315794, '3': 1.9785028543689314, '2': 8.860251913043474}}

# control parameters
advs_optimal_p = {"1458": 1000000,   "2259": 100000, "2261":500000,  "2821":40000, "2997":1000000,  "3358":500000,  "3386":50000,  "3427": 500000, "3476": 500000}
advs_optimal_i = {"1458": 100,       "2259": 20000,  "2261":5000,   "2821":200,   "2997":1000,   "3358":100,   "3386":10000,   "3427": 1000,  "3476": 1000}
advs_optimal_d = {"1458": 100,       "2259": 10000,  "2261":50000,   "2821":100,   "2997":10000,   "3358":100,   "3386":1000,   "3427": 1000,  "3476": 10005}
advs_optimal_min_phi = {"1458": -2,  "2259": -0.6, "2261":-0.8,     "2821":-0.5,   "2997":-0.5,   "3358":-1,   "3386":-0.45,   "3427": -1,  "3476": -2}
advs_optimal_max_phi = {"1458": 5,  "2259": 4,      "2261":5,       "2821":5,       "2997":5,       "3358":2,   "3386":2,   "3427": 2,  "3476": 3}

advs_filtered_adex = {"2259":[]}

advertiser = "2259"
if len(sys.argv) > 1:
    advertiser = sys.argv[1]
mode = "batch"
basebid = advs_base_bid[advertiser]
print "%s\t%s\t%d" % (advertiser, mode, basebid)

# parameter setting
minbid = 5
cntr_rounds = 40
div = 1e-6
para_p = advs_optimal_p[advertiser] * div
para_i = advs_optimal_i[advertiser] * div
para_d = advs_optimal_d[advertiser] * div
max_p = advs_optimal_p[advertiser] * 20
min_p = advs_optimal_p[advertiser] / 5
p_num = 20
base_step_p = math.exp(math.log(max_p/min_p) / p_num)
max_i = 50000.
min_i = 25000.
i_num = 4
base_step_i = math.exp(math.log(max_i/min_i) / i_num)
max_d = 20000.
min_d = 10000.
d_num = 4
base_step_d = math.exp(math.log(max_d/min_d) / d_num)

para_ps = range(0, p_num, 1)
para_is = range(0, i_num, 1)
para_ds = range(0, d_num, 1)
settle_con = 0.1
rise_con = 0.9
min_phi = advs_optimal_min_phi[advertiser]
max_phi = advs_optimal_max_phi[advertiser]
#budget = 9999999999999.0
budget = advs_test_lin_ori_cost[advertiser] * budget_damping
lin_budget = budget

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
def cal_settling_time(ecpcs, adex_ref):
    settling_time = {}
    for ex, ex_ecpcs in ecpcs.iteritems():
        if ex == "all":
            continue
        settled = False
        settling_time[ex] = 0
        for key, value in ex_ecpcs.iteritems():
            error = adex_ref[ex] - value
            if abs(error) / adex_ref[ex] <= settle_con and (not settled):
                settled = True
                settling_time[ex] = key
            elif abs(error) / adex_ref[ex] > settle_con:
                settled = False
                settling_time[ex] = cntr_rounds
    return settling_time

# # calculate steady-state error
def cal_rmse_ss(ecpcs, adex_ref):
    settled = False
    settling_time = cal_settling_time(ecpcs, adex_ref)
    rmse = {}
    for ex, ex_ecpcs in ecpcs.iteritems():
        if ex == "all":
            continue
        rmse[ex] = 0.0
        if settling_time[ex] >= cntr_rounds:
            settling_time[ex] = cntr_rounds - 1
        for round in range(settling_time[ex], cntr_rounds):
            rmse[ex] += (ecpcs[ex][round] - adex_ref[ex]) * (ecpcs[ex][round] - adex_ref[ex]) / (adex_ref[ex] * adex_ref[ex])

        rmse[ex] /= (cntr_rounds - settling_time[ex])
        rmse[ex] = math.sqrt(rmse[ex]) # weinan: relative rmse
    return rmse

# # calculate steady-state standard deviation
def cal_sd_ss(ecpcs, adex_ref):
    settled = False
    settling_time = cal_settling_time(ecpcs, adex_ref)
    sd = {}
    for ex, ex_ecpcs in ecpcs.iteritems():
        if ex == "all":
            continue
        sd[ex] = 0.0
        if settling_time[ex] >= cntr_rounds:
            settling_time[ex] = cntr_rounds - 1
        sum2 = 0.0
        sum = 0.0
        for round in range(settling_time[ex], cntr_rounds):
            sum2 += ecpcs[ex][round] * ecpcs[ex][round]
            sum += ecpcs[ex][round]
        n = cntr_rounds - settling_time[ex]
        mean = sum / n
        sd[ex] = math.sqrt(sum2 / n - mean * mean) / mean # weinan: relative sd
    return sd

# calculate rise time
def cal_rise_time(ecpcs, adex_ref):
    rise_time = {}
    for ex, ex_ecpcs in ecpcs.iteritems():
        if ex == "all":
            continue
        for key, value in ex_ecpcs.iteritems():
            error = adex_ref[ex] - value
            if abs(error) / adex_ref[ex] <= (1 - rise_con):
                rise_time[ex] = key
                break
        try:
            is_set = rise_time[ex]
        except:
            rise_time[ex] = 0
    return rise_time

# calculate percentage overshoot
def cal_overshoot(ecpcs, adex_ref):
    overshoot = {}
    for ex, ex_ecpcs in ecpcs.iteritems():
        if ex == "all":
            continue
        max_os = 0.0
        if ex_ecpcs[0] > adex_ref[ex]:
            for key, value in ex_ecpcs.iteritems():
                if value <= adex_ref[ex]:
                    os = (adex_ref[ex] - value) * 100.0 / adex_ref[ex]
                    if os > max_os:
                        max_os = os
        elif ex_ecpcs[0] < adex_ref[ex]:
            for key, value in ex_ecpcs.iteritems():
                if value >= adex_ref[ex]:
                    os = (value - adex_ref[ex]) * 100.0 / adex_ref[ex]
                    if os > max_os:
                        max_os = os
        else:
            for key, value in ex_ecpcs.iteritems():
                os = abs(value - adex_ref[ex]) * 100.0 / adex_ref[ex]
                if os >= max_os:
                    max_os = os
        overshoot[ex] = max_os
    return overshoot

# control function
def control(para_p, para_i, para_d, outfile):
    fo = open(outfile, 'w')
    fo.write("stage\tround\texchange\tecpc\tphi\ttotal_bid_num\ttotal_wins\ttotal_clks\tclick_ratio\ttotal_cost\tbudget\tref\tori_ecpc\n")
    ecpcs = {}
    first_round = True
    sec_round = False
    cntr_size = int(len(yp) / cntr_rounds)
    adex_ref = advs_adex_ref[advertiser]
    error_sum = {}
    phi = {}
    total_cost = {}
    total_clks = {}
    total_wins = {}
    total_bid_num = {}
    lin_total_wins = 0
    lin_total_clks = 0
    lin_total_cost = 0.0
    total_wins["all"] = 0
    total_clks["all"] = 0
    total_cost["all"] = 0.0
    lin_total_bid_num = 0
    break_round = False

    for val in list(set(exchange)):
        total_clks[val] = 0
        total_cost[val] = 0.0
        total_wins[val] = 0
        total_bid_num[val] = 0

    for round in range(0, cntr_rounds):
        if first_round and (not sec_round):
            for val in list(set(exchange)):
                phi[val] = 0.0
                error_sum[val] = 0.0
                ecpcs["all"] = {}
            first_round = False
            sec_round = True
        elif sec_round and (not first_round):
            for val in list(set(exchange)):
                error = adex_ref[val] - ecpcs[val][round-1]
                error_sum[val] += error
                phi[val] = para_p*error + para_i*error_sum[val]
            sec_round = False
        else:
            for val in list(set(exchange)):
                error = adex_ref[val] - ecpcs[val][round-1]
                error_sum[val] += error
                phi[val] = para_p*error + para_i*error_sum[val] + para_d*(ecpcs[val][round-2]-ecpcs[val][round-1])

        # fang piao
        for val in list(set(exchange)):
            if phi[val] <= min_phi:
                phi[val] = min_phi
            elif phi[val] >= max_phi:
                phi[val] = max_phi

        imp_index = ((round+1)*cntr_size)

        if round == cntr_rounds - 1:
            imp_index = imp_index + (len(yp) - cntr_size*cntr_rounds)

        for i in range(round*cntr_size, imp_index):
            clk = y[i]
            pctr = yp[i]
            mp = mplist[i]

            if total_cost['all'] + mp < budget:
                if i == 0:
                    total_bid_num["all"] = 1
                else:
                    total_bid_num["all"] += 1

                if exchange[i] in total_bid_num:
                    total_bid_num[exchange[i]] += 1
                else:
                    total_bid_num[exchange[i]] = 1

            clk = y[i]
            pctr = yp[i]
            mp = mplist[i]
            bid = int(max(minbid, lin(pctr, basectr, basebid) * (math.exp(phi[exchange[i]]))))
            # bid = int(max(minbid, lin(pctr, basectr, basebid) * (math.exp(0))))

            if round == 0:
                bid = int(max(minbid, lin(pctr, basectr, basebid) * (math.exp(0))))

            if bid > mp:
                if total_cost['all'] + mp < budget:
                    total_wins["all"] += 1
                    total_clks["all"] += clk
                    total_cost["all"] += mp

                    if exchange[i] in total_wins:
                        total_wins[exchange[i]] += 1
                    else:
                        total_wins[exchange[i]] = 1

                    if exchange[i] in total_clks:
                        total_clks[exchange[i]] += clk
                    else:
                        total_clks[exchange[i]] = clk

                    if exchange[i] in total_cost:
                        total_cost[exchange[i]] += mp
                    else:
                        total_cost[exchange[i]] = mp

            lin_bid = int(max(minbid, lin(pctr, basectr, basebid)))

            if lin_total_cost + mp < lin_budget:
                lin_total_bid_num += 1

            if lin_bid > mp:
                if lin_total_cost + mp < lin_budget:
                    lin_total_clks += clk
                    lin_total_wins += 1
                    lin_total_cost += mp


        ecpcs["all"][round] = total_cost["all"] * 1.0 / total_clks["all"] / 1000.0

        for val in list(set(exchange)):
            if round == 0:
                ecpcs[val] = {}
            if total_clks[val] == 0:
                ecpcs[val][round] = total_cost[val] * 1.0 / (total_clks[val]+1) / 1000.0
            else:
                ecpcs[val][round] = total_cost[val] * 1.0 / total_clks[val] / 1000.0
            fo.write("%s\t%d\t%s\t%.4f\t%.4f\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % ("test", round, val, ecpcs[val][round], phi[val], total_bid_num[val], total_wins[val], total_clks[val], total_clks[val] * 1.0/advs_test_clicks[advertiser], total_cost[val], budget, adex_ref[val], advs_test_ori_ecpc[advertiser]))
        fo.write("%s\t%d\t%s\t%.4f\t%.4f\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % ("test", round, "all", ecpcs["all"][round], 0.0, total_bid_num["all"], total_wins["all"], total_clks["all"], total_clks["all"] * 1.0/advs_test_clicks[advertiser], total_cost["all"], budget, 0.0, advs_test_ori_ecpc[advertiser]))
        if lin_total_clks == 0:
            lin_total_clks = 1
        lin_ecpc = lin_total_cost * 1.0 /lin_total_clks / 1000.0
        fo.write("%s\t%d\t%s\t%.4f\t%.4f\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % ("test", round, "lin", lin_ecpc, 0.0, lin_total_bid_num, lin_total_wins, lin_total_clks, lin_total_clks * 1.0/advs_test_clicks[advertiser], lin_total_cost, lin_budget, 0.0, advs_test_ori_ecpc[advertiser]))

    bid_nums.append(total_bid_num)
    imp_nums.append(total_wins)
    adex_clks.append(total_clks)
    adex_costs.append(total_cost)
    adex_ecpcs.append(ecpcs)


    # print "lin total cost: " + str(lin_total_cost)

    # train
    ecpcs = {}
    first_round = True
    sec_round = False
    cntr_size = int(len(yp_train) / cntr_rounds)
    adex_ref = advs_adex_ref[advertiser]
    error_sum = {}
    phi = {}
    total_cost = {}
    total_clks = {}
    total_wins = {}
    total_bid_num = {}
    lin_total_wins = 0
    lin_total_clks = 0
    lin_total_cost = 0

    for val in list(set(exchange_train)):
        total_clks[val] = 0
        total_cost[val] = 0.0
        total_wins[val] = 0
        total_bid_num[val] = 0

    for round in range(0, cntr_rounds):
        if first_round and (not sec_round):
            for val in list(set(exchange_train)):
                phi[val] = 0.0
                error_sum[val] = 0.0
                ecpcs["all"] = {}
            first_round = False
            sec_round = True
        elif sec_round and (not first_round):
            for val in list(set(exchange_train)):
                error = adex_ref[val] - ecpcs[val][round-1]
                error_sum[val] += error
                phi[val] = para_p*error + para_i*error_sum[val]
            sec_round = False
        else:
            for val in list(set(exchange_train)):
                error = adex_ref[val] - ecpcs[val][round-1]
                error_sum[val] += error
                phi[val] = para_p*error + para_i*error_sum[val] + para_d*(ecpcs[val][round-2]-ecpcs[val][round-1])

        # fang piao
        for val in list(set(exchange_train)):
            if phi[val] <= min_phi:
                phi[val] = min_phi
            elif phi[val] >= max_phi:
                phi[val] = max_phi

        imp_index = ((round+1)*cntr_size)

        if round == cntr_rounds - 1:
            imp_index = imp_index + (len(yp) - cntr_size*cntr_rounds)

        for i in range(round*cntr_size, imp_index):
            if i == 0:
                total_bid_num["all"] = 1
            else:
                total_bid_num["all"] += 1

            if exchange_train[i] in total_bid_num:
                total_bid_num[exchange_train[i]] += 1
            else:
                total_bid_num[exchange_train[i]] = 1

            clk = y_train[i]
            pctr = yp_train[i]
            mp = mplist_train[i]
            bid = max(minbid, lin(pctr, basectr, basebid) * (math.exp(phi[exchange_train[i]])))
            # bid = int(max(minbid, lin(pctr, basectr, basebid) * (math.exp(0))))

            if round == 0:
                bid = int(max(minbid, lin(pctr, basectr, basebid) * (math.exp(0))))

            if bid > mp:
                if not "all" in total_wins:
                    total_wins["all"] = 1
                    total_clks["all"] = clk
                    total_cost["all"] = mp
                else:
                    total_wins["all"] += 1
                    total_clks["all"] += clk
                    total_cost["all"] += mp

                if exchange_train[i] in total_wins:
                    total_wins[exchange_train[i]] += 1
                else:
                    total_wins[exchange_train[i]] = 1

                if exchange_train[i] in total_clks:
                    total_clks[exchange_train[i]] += clk
                else:
                    total_clks[exchange_train[i]] = clk

                if exchange_train[i] in total_cost:
                    total_cost[exchange_train[i]] += mp
                else:
                    total_cost[exchange_train[i]] = mp

            lin_bid = int(max(minbid, lin(pctr, basectr, basebid)))

            if lin_bid > mp:
                lin_total_clks += clk
                lin_total_wins += 1
                lin_total_cost += mp


        ecpcs["all"][round] = total_cost["all"]  * 1.0 / total_clks["all"] / 1000.0

        for val in list(set(exchange_train)):
            if round == 0:
                ecpcs[val] = {}
            if total_clks[val] == 0:
                ecpcs[val][round] = total_cost[val] * 1.0 / (total_clks[val]+1) / 1000.0
            else:
                ecpcs[val][round] = total_cost[val] * 1.0 / total_clks[val] / 1000.0
            fo.write("%s\t%d\t%s\t%.4f\t%.4f\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % ("train", round, val, ecpcs[val][round], phi[val], total_bid_num[val], total_wins[val], total_clks[val], total_clks[val] * 1.0/advs_train_clicks[advertiser], total_cost[val], budget, adex_ref[val], advs_train_ori_ecpc[advertiser]))
        fo.write("%s\t%d\t%s\t%.4f\t%.4f\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % ("train", round, "all", ecpcs["all"][round], 0.0, total_bid_num["all"], total_wins["all"], total_clks["all"], total_clks["all"] * 1.0/advs_train_clicks[advertiser], total_cost["all"], budget, 0.0, advs_train_ori_ecpc[advertiser]))
        if lin_total_clks == 0:
            lin_total_clks = 1
        lin_ecpc = lin_total_cost * 1.0 /lin_total_clks / 1000.0
        fo.write("%s\t%d\t%s\t%.4f\t%.4f\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % ("train", round, "lin", lin_ecpc, 0.0, total_bid_num["all"], lin_total_wins, lin_total_clks, lin_total_clks * 1.0/advs_train_clicks[advertiser], lin_total_cost, lin_budget, 0.0, advs_train_ori_ecpc[advertiser]))

    # weinan changes the report from test to train because test has the budget, which leads no settling
    settling_time.append(cal_settling_time(ecpcs, adex_ref))
    rmse_ss.append(cal_rmse_ss(ecpcs, adex_ref))
    sd_ss.append(cal_sd_ss(ecpcs, adex_ref))
    rise_time.append(cal_rise_time(ecpcs, adex_ref))
    overshoot.append(cal_overshoot(ecpcs, adex_ref))

    fo.close()

# control function - batch mode
def control_batch(para_p, para_i, para_d, outfile, rout, parameter):
    global tmp_clks
    global tmp_cost
    # fo = open(outfile, 'w')
    # fo.write("stage\tround\texchange\tecpc\tphi\ttotal_bid_num\ttotal_wins\ttotal_clks\tclick_ratio\ttotal_cost\tbudget\tref\tori_ecpc\n")
    ecpcs = {}
    first_round = True
    sec_round = False
    cntr_size = int(len(yp) / cntr_rounds)
    adex_ref = advs_adex_ref[advertiser]
    error_sum = {}
    phi = {}
    total_cost = {}
    total_clks = {}
    total_wins = {}
    total_bid_num = {}
    lin_total_wins = 0
    lin_total_clks = 0
    lin_total_cost = 0.0
    total_wins["all"] = 0
    total_clks["all"] = 0
    total_cost["all"] = 0.0
    lin_total_bid_num = 0
    break_round = False

    for val in list(set(exchange)):
        total_clks[val] = 0
        total_cost[val] = 0.0
        total_wins[val] = 0
        total_bid_num[val] = 0

    for round in range(0, cntr_rounds):
        if first_round and (not sec_round):
            for val in list(set(exchange)):
                phi[val] = 0.0
                error_sum[val] = 0.0
                ecpcs["all"] = {}
            first_round = False
            sec_round = True
        elif sec_round and (not first_round):
            for val in list(set(exchange)):
                error = adex_ref[val] - ecpcs[val][round-1]
                error_sum[val] += error
                phi[val] = para_p*error + para_i*error_sum[val]
            sec_round = False
        else:
            for val in list(set(exchange)):
                error = adex_ref[val] - ecpcs[val][round-1]
                error_sum[val] += error
                phi[val] = para_p*error + para_i*error_sum[val] + para_d*(ecpcs[val][round-2]-ecpcs[val][round-1])

        # fang piao
        for val in list(set(exchange)):
            if phi[val] <= min_phi:
                phi[val] = min_phi
            elif phi[val] >= max_phi:
                phi[val] = max_phi

        imp_index = ((round+1)*cntr_size)

        if round == cntr_rounds - 1:
            imp_index = imp_index + (len(yp) - cntr_size*cntr_rounds)

        for i in range(round*cntr_size, imp_index):
            clk = y[i]
            pctr = yp[i]
            mp = mplist[i]

            if total_cost['all'] + mp < budget:
                if i == 0:
                    total_bid_num["all"] = 1
                else:
                    total_bid_num["all"] += 1

                if exchange[i] in total_bid_num:
                    total_bid_num[exchange[i]] += 1
                else:
                    total_bid_num[exchange[i]] = 1

            clk = y[i]
            pctr = yp[i]
            mp = mplist[i]
            bid = int(max(minbid, lin(pctr, basectr, basebid) * (math.exp(phi[exchange[i]]))))
            # bid = int(max(minbid, lin(pctr, basectr, basebid) * (math.exp(0))))

            if round == 0:
                bid = int(max(minbid, lin(pctr, basectr, basebid) * (math.exp(0))))

            if bid > mp:
                if total_cost['all'] + mp < budget:
                    total_wins["all"] += 1
                    total_clks["all"] += clk
                    total_cost["all"] += mp

                    if exchange[i] in total_wins:
                        total_wins[exchange[i]] += 1
                    else:
                        total_wins[exchange[i]] = 1

                    if exchange[i] in total_clks:
                        total_clks[exchange[i]] += clk
                    else:
                        total_clks[exchange[i]] = clk

                    if exchange[i] in total_cost:
                        total_cost[exchange[i]] += mp
                    else:
                        total_cost[exchange[i]] = mp

            lin_bid = int(max(minbid, lin(pctr, basectr, basebid)))

            if lin_total_cost + mp < lin_budget:
                lin_total_bid_num += 1

            if lin_bid > mp:
                if lin_total_cost + mp < lin_budget:
                    lin_total_clks += clk
                    lin_total_wins += 1
                    lin_total_cost += mp


        ecpcs["all"][round] = total_cost["all"] * 1.0 / total_clks["all"] / 1000.0

        for val in list(set(exchange)):
            if round == 0:
                ecpcs[val] = {}
            if total_clks[val] == 0:
                ecpcs[val][round] = total_cost[val] * 1.0 / (total_clks[val]+1) / 1000.0
            else:
                ecpcs[val][round] = total_cost[val] * 1.0 / total_clks[val] / 1000.0
            # fo.write("%s\t%d\t%s\t%.4f\t%.4f\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % ("test", round, val, ecpcs[val][round], phi[val], total_bid_num[val], total_wins[val], total_clks[val], total_clks[val] * 1.0/advs_test_clicks[advertiser], total_cost[val], budget, adex_ref[val], advs_test_ori_ecpc[advertiser]))
        # fo.write("%s\t%d\t%s\t%.4f\t%.4f\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % ("test", round, "all", ecpcs["all"][round], 0.0, total_bid_num["all"], total_wins["all"], total_clks["all"], total_clks["all"] * 1.0/advs_test_clicks[advertiser], total_cost["all"], budget, 0.0, advs_test_ori_ecpc[advertiser]))
        if lin_total_clks == 0:
            lin_total_clks = 1
        lin_ecpc = lin_total_cost * 1.0 /lin_total_clks / 1000.0
        # fo.write("%s\t%d\t%s\t%.4f\t%.4f\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % ("test", round, "lin", lin_ecpc, 0.0, lin_total_bid_num, lin_total_wins, lin_total_clks, lin_total_clks * 1.0/advs_test_clicks[advertiser], lin_total_cost, lin_budget, 0.0, advs_test_ori_ecpc[advertiser]))

    test_report_bid_num = total_bid_num
    test_report_total_wins = total_wins
    test_report_total_clks = total_clks
    test_report_total_cost = total_cost
    test_report_ecpcs = ecpcs
    tmp_clks = test_report_total_clks["all"]
    tmp_cost = test_report_total_cost["all"]
    # print "lin total cost: " + str(lin_total_cost)

    # train
    ecpcs = {}
    first_round = True
    sec_round = False
    cntr_size = int(len(yp_train) / cntr_rounds)
    adex_ref = advs_adex_ref[advertiser]
    error_sum = {}
    phi = {}
    total_cost = {}
    total_clks = {}
    total_wins = {}
    total_bid_num = {}
    lin_total_wins = 0
    lin_total_clks = 0
    lin_total_cost = 0

    for val in list(set(exchange_train)):
        total_clks[val] = 0
        total_cost[val] = 0.0
        total_wins[val] = 0
        total_bid_num[val] = 0

    for round in range(0, cntr_rounds):
        if first_round and (not sec_round):
            for val in list(set(exchange_train)):
                phi[val] = 0.0
                error_sum[val] = 0.0
                ecpcs["all"] = {}
            first_round = False
            sec_round = True
        elif sec_round and (not first_round):
            for val in list(set(exchange_train)):
                error = adex_ref[val] - ecpcs[val][round-1]
                error_sum[val] += error
                phi[val] = para_p*error + para_i*error_sum[val]
            sec_round = False
        else:
            for val in list(set(exchange_train)):
                error = adex_ref[val] - ecpcs[val][round-1]
                error_sum[val] += error
                phi[val] = para_p*error + para_i*error_sum[val] + para_d*(ecpcs[val][round-2]-ecpcs[val][round-1])

        # fang piao
        for val in list(set(exchange_train)):
            if phi[val] <= min_phi:
                phi[val] = min_phi
            elif phi[val] >= max_phi:
                phi[val] = max_phi

        imp_index = ((round+1)*cntr_size)

        if round == cntr_rounds - 1:
            imp_index = imp_index + (len(yp) - cntr_size*cntr_rounds)

        for i in range(round*cntr_size, imp_index):
            if i == 0:
                total_bid_num["all"] = 1
            else:
                total_bid_num["all"] += 1

            if exchange_train[i] in total_bid_num:
                total_bid_num[exchange_train[i]] += 1
            else:
                total_bid_num[exchange_train[i]] = 1

            clk = y_train[i]
            pctr = yp_train[i]
            mp = mplist_train[i]
            bid = max(minbid, lin(pctr, basectr, basebid) * (math.exp(phi[exchange_train[i]])))
            # bid = int(max(minbid, lin(pctr, basectr, basebid) * (math.exp(0))))

            if round == 0:
                bid = int(max(minbid, lin(pctr, basectr, basebid) * (math.exp(0))))

            if bid > mp:
                if not "all" in total_wins:
                    total_wins["all"] = 1
                    total_clks["all"] = clk
                    total_cost["all"] = mp
                else:
                    total_wins["all"] += 1
                    total_clks["all"] += clk
                    total_cost["all"] += mp

                if exchange_train[i] in total_wins:
                    total_wins[exchange_train[i]] += 1
                else:
                    total_wins[exchange_train[i]] = 1

                if exchange_train[i] in total_clks:
                    total_clks[exchange_train[i]] += clk
                else:
                    total_clks[exchange_train[i]] = clk

                if exchange_train[i] in total_cost:
                    total_cost[exchange_train[i]] += mp
                else:
                    total_cost[exchange_train[i]] = mp

            lin_bid = int(max(minbid, lin(pctr, basectr, basebid)))

            if lin_bid > mp:
                lin_total_clks += clk
                lin_total_wins += 1
                lin_total_cost += mp


        ecpcs["all"][round] = total_cost["all"]  * 1.0 / total_clks["all"] / 1000.0

        for val in list(set(exchange_train)):
            if round == 0:
                ecpcs[val] = {}
            if total_clks[val] == 0:
                ecpcs[val][round] = total_cost[val] * 1.0 / (total_clks[val]+1) / 1000.0
            else:
                ecpcs[val][round] = total_cost[val] * 1.0 / total_clks[val] / 1000.0
            # fo.write("%s\t%d\t%s\t%.4f\t%.4f\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % ("train", round, val, ecpcs[val][round], phi[val], total_bid_num[val], total_wins[val], total_clks[val], total_clks[val] * 1.0/advs_train_clicks[advertiser], total_cost[val], budget, adex_ref[val], advs_train_ori_ecpc[advertiser]))
        # fo.write("%s\t%d\t%s\t%.4f\t%.4f\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % ("train", round, "all", ecpcs["all"][round], 0.0, total_bid_num["all"], total_wins["all"], total_clks["all"], total_clks["all"] * 1.0/advs_train_clicks[advertiser], total_cost["all"], budget, 0.0, advs_train_ori_ecpc[advertiser]))
        if lin_total_clks == 0:
            lin_total_clks = 1
        lin_ecpc = lin_total_cost * 1.0 /lin_total_clks / 1000.0
        # fo.write("%s\t%d\t%s\t%.4f\t%.4f\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % ("train", round, "lin", lin_ecpc, 0.0, total_bid_num["all"], lin_total_wins, lin_total_clks, lin_total_clks * 1.0/advs_train_clicks[advertiser], lin_total_cost, lin_budget, 0.0, advs_train_ori_ecpc[advertiser]))

    # weinan changes the report from test to train because test has the budget, which leads no settling
    settling_time = cal_settling_time(ecpcs, adex_ref)
    rmse_ss = cal_rmse_ss(ecpcs, adex_ref)
    sd_ss = cal_sd_ss(ecpcs, adex_ref)
    rise_time = cal_rise_time(ecpcs, adex_ref)
    overshoot = cal_overshoot(ecpcs, adex_ref)

    for val in list(set(exchange)):
        rout.write(parameter+"\t"+str(rise_time[val])+"\t"+str(settling_time[val])+"\t"+str(overshoot[val])+"\t" + \
               str(rmse_ss[val]) + "\t" + str(sd_ss[val]) + "\t" + str(test_report_bid_num[val]) + "\t" + str(test_report_total_wins[val]) +
                   "\t" + str(test_report_total_clks[val]) + "\t" + str(test_report_total_cost[val]) + "\t" + str(test_report_ecpcs[val][39]) + "\t" + str(budget) + "\t" + val +"\n")
    rout.write(parameter+"\t0.0\t0.0\t0.0\t0.0\t0.0\t" + str(test_report_bid_num["all"]) + "\t" + str(test_report_total_wins["all"]) +
           "\t" + str(test_report_total_clks["all"]) + "\t" + str(test_report_total_cost["all"]) + "\t" + str(test_report_ecpcs["all"][39]) + "\t" + str(budget) + "\t" + "all" +"\n")
    rout.flush()
    # fo.close()

random.seed(10)

# if len(sys.argv) != 3:
#     print 'campaignId mode'
#     exit(-1)

tmp_clks = 0
tmp_cost = 0

mplist = []
y = []
yp = []
mplist_train = []
y_train = []
yp_train = []
featWeight = {}
exchange = []
exchange_train = []

fi = open("../../make-ipinyou-data/"+advertiser+"/test.yzpc.txt", 'r')
for line in fi:
    data = line.strip().split("\t")
    if advertiser in advs_filtered_adex and data[3] in advs_filtered_adex[advertiser]:
        continue  # filtered ad exchange for this advertiser
    y.append(int(data[0]))
    mplist.append(int(data[1]))
    yp.append(float(data[2]))
    exchange.append(data[3].replace("\n", ""))
fi.close()

fi = open("../../make-ipinyou-data/"+advertiser+"/train.yzpc.txt", 'r')
for line in fi:
    data = line.strip().split("\t")
    if advertiser in advs_filtered_adex and data[3] in advs_filtered_adex[advertiser]:
        continue  # filtered ad exchange for this advertiser
    y_train.append(int(data[0]))
    mplist_train.append(int(data[1]))
    yp_train.append(float(data[2]))
    exchange_train.append(data[3].replace("\n", ""))
fi.close()

basectr = sum(yp_train) / float(len(yp_train))

# for reporting
parameters = []
overshoot = []
settling_time = []
rise_time = []
rmse_ss = []
sd_ss = []
bid_nums = []
imp_nums = []
adex_clks = []
adex_costs = []
adex_ecpcs = []

report_path = ""


if mode == "single": # single mode
    out_path = "../exp-data/"+advertiser+"_p="+str(para_p)+"_i="+str(para_i)+"_d="+str(para_d)+"-eco.tsv"
    control(para_p, para_i, para_d, out_path)
    report_path = "../report/report-single-eco.tsv"
    parameter = ""+advertiser+"\t"+str(cntr_rounds)+"\t"+str(basebid)+"\t" + \
                str(para_p)+"\t"+str(para_i)+"\t"+str(para_d)+"\t"+str(settle_con)+"\t"+str(rise_con)
    parameters.append(parameter)
    rout = open(report_path, 'w')
    rout.write("campaign\ttotal-rounds\tbase-bid\tp\ti\td\tsettle-con\trise-con\trise-time\tsettling-time\tovershoot\trmse-ss\tsd-ss\texchange\n")
    for val in list(set(exchange)):
        for idx, value in enumerate(parameters):
            rout.write(value+"\t"+str(rise_time[idx][val])+"\t"+str(settling_time[idx][val])+"\t"+str(overshoot[idx][val])+"\t" + \
                   str(rmse_ss[idx][val]) + "\t" + str(sd_ss[idx][val]) + "\t" + val +"\n")
    rout.close()
elif mode == "batch":
    count = 0
    report_path = "../report/"+ advertiser + "-report-batch-eco.tsv"
    rout = open(report_path, 'w')
    rout.write("campaign\ttotal-rounds\tbase-bid\tp\ti\td\tsettle-con\trise-con\trise-time\tsettling-time\tovershoot\trmse-ss\tsd-ss\tbid_num\timp_num\tclk\tcost\tecpc\tbudget\texchange\n")

    for i in para_is:
        if len(para_is) > 1:
            para_i = min_i * pow(base_step_i, i) * div
        for d in para_ds:
            if len(para_ds) > 1:
                para_d = min_d * pow(base_step_d, d) * div
            for p in para_ps:
                if len(para_ps) > 1:
                    para_p = min_p * pow(base_step_p, p) * div
                count += 1
                directory = "../batch-exp-data/"+advertiser
                if not os.path.exists(directory):
                    os.makedirs(directory)
                out_path = directory+"/"+advertiser+"_p="+str(para_p)+"_i="+str(para_i)+"_d="+str(para_d)+"-eco.tsv"


                parameter = ""+advertiser+"\t"+str(cntr_rounds)+"\t"+str(basebid)+"\t" + \
                            str(para_p)+"\t"+str(para_i)+"\t"+str(para_d)+"\t"+str(settle_con)+"\t"+str(rise_con)
                # parameters.append(parameter)

                control_batch(para_p, para_i, para_d, out_path, rout, parameter)
                print advertiser + "\t" + str(count) + "\t" + str(tmp_clks) + "\t" + str(tmp_cost) + "\t" + str(budget) + "\t" +str(para_p) + "\t" +str(para_i) + "\t" +str(para_d)
    rout.close()
else:
    print "wrong mode entered"





