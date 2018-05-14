import numpy as np
import matplotlib.pyplot as plt
import time

from aes import AES
from lracpa import *
from condaveraes import *

# traceset, number of traces, and S-box to attack
trace_set_filename = "../traces/swaes_atmega_power.npz"
sample_range = (800, 1500)  # range of samples to attack, in the format (low, high)
N = 200  # number of traces to attack (less or equal to the amount of traces in the file)
offset = 0  # trace number to start from
evolution_step = 200  # step for intermediate reports
num_sboxes_to_attack = 16  # bytes

# Leakage model
# (these parameters correspond to function names in lracpa module)
intermediateFunction = sBoxOut  # for CPA and LRA
leakageFunction = leakageModelHW  # for CPA
basisFunctionsModel = basisModelSingleBits  # for LRA

# Known key for ranking
known_key_str_encoded = "2B7E151628AED2A6ABF7158809CF4F3C"
knownKeyStr = known_key_str_encoded.decode("hex")  # the correct key
encrypt = True  # to avoid selective commenting in the following lines below

if encrypt: # for encryption, the first round key is as is
    known_key = np.array(map(ord, knownKeyStr), dtype="uint8")
else:       # for decryption, need to run key expansion
    expandedKnownKey = AES().expandKey(map(ord, knownKeyStr), 16, 16 * 11)  # this returs a list
    known_key = np.array(expandedKnownKey[176 - 16:177], dtype="uint8")

data = []
traces = []
trace_length = 0
log_intermediate_results = False


def perform_attack():
    print "---\nAttack"
    t0 = time.clock()
    lra_key = []
    cpa_key = []
    cond_aver = []
    for i in range(num_sboxes_to_attack):
        cond_aver.append(ConditionalAveragerAesSbox(256, trace_length))

    # allocate arrays for storing key rank evolution
    num_steps = int(np.ceil(N / np.double(evolution_step)))
    key_rank_evolution_cpa = np.zeros(num_steps)
    key_rank_evolution_lra = np.zeros(num_steps)

    # the incremental loop
    traces_to_skip = 20  # warm-up to avoid numerical problems for small evolution step
    for i in range(0, traces_to_skip - 1):
        for j in range(num_sboxes_to_attack):
            cond_aver[j].addTrace(data[j][i], traces[i])
    for i in range(traces_to_skip - 1, N):
        for j in range(num_sboxes_to_attack):
            cond_aver[j].addTrace(data[j][i], traces[i])

        if ((i + 1) % evolution_step == 0) or ((i + 1) == N):
            for j in range(num_sboxes_to_attack):
                print "Handling Sbox %d results..." % j
                (avdata, avtraces) = cond_aver[j].getSnapshot()

                corr_traces = cpaAES(avdata, avtraces, intermediateFunction, leakageFunction)
                R2, coefs = lraAES(avdata, avtraces, intermediateFunction, basisFunctionsModel)

                if log_intermediate_results:
                    print "---\nResults after %d traces" % (i + 1)
                    print "CPA"
                corr_peaks = np.max(np.abs(corr_traces), axis=1)  # global maximization, absolute value!
                cpa_winning_candidate = np.argmax(corr_peaks)
                cpa_winning_candidate_peak = np.max(corr_peaks)
                cpa_correct_candidate_rank = np.count_nonzero(corr_peaks >= corr_peaks[known_key[j]])
                cpa_correct_candidate_peak = corr_peaks[known_key[j]]
                if log_intermediate_results:
                    print "Winning candidate: 0x%02x, peak magnitude %f" % (cpa_winning_candidate, cpa_winning_candidate_peak)
                    print "Correct candidate: 0x%02x, peak magnitude %f, rank %d" % (
                        known_key[j], cpa_correct_candidate_peak, cpa_correct_candidate_rank)
                cpa_key.append(cpa_winning_candidate)

                if log_intermediate_results:
                    print "LRA"
                r2_peaks = np.max(R2, axis=1)  # global maximization
                lra_winning_candidate = np.argmax(r2_peaks)
                lra_winning_candidate_peak = np.max(r2_peaks)
                lra_correct_candidate_rank = np.count_nonzero(r2_peaks >= r2_peaks[known_key[j]])
                lra_correct_candidate_peak = r2_peaks[known_key[j]]
                if log_intermediate_results:
                    print "Winning candidate: 0x%02x, peak magnitude %f" % (lra_winning_candidate, lra_winning_candidate_peak)
                    print "Correct candidate: 0x%02x, peak magnitude %f, rank %d" % (
                        known_key[j], lra_correct_candidate_peak, lra_correct_candidate_rank)
                lra_key.append(cpa_winning_candidate)

                step_count = int(np.floor(i / np.double(evolution_step)))
                key_rank_evolution_cpa[step_count] = cpa_correct_candidate_rank
                key_rank_evolution_lra[step_count] = lra_correct_candidate_rank

    t1 = time.clock()
    time_all = t1 - t0

    print "---\nCumulative timing"
    print "%0.2f s" % time_all
    print "---\nCPA key approximation: %s" % ''.join('{:02x}'.format(x) for x in cpa_key)
    print "---\nLRA key approximation: %s" % ''.join('{:02x}'.format(x) for x in lra_key)
    print "---\nKnown key: %s" % known_key_str_encoded


def load_data():
    global data
    global traces
    global trace_length
    if log_intermediate_results:
        print "---\nLoading " + trace_set_filename
    t0 = time.clock()
    npzfile = np.load(trace_set_filename)

    for i in range(num_sboxes_to_attack):
        data.append(npzfile['data'][offset:offset + N, i])  # selecting only the required byte
        traces = npzfile['traces'][offset:offset + N, sample_range[0]:sample_range[1]]
    t1 = time.clock()
    time_load = t1 - t0

    (numTraces, trace_length) = traces.shape
    if log_intermediate_results:
        print "Number of traces loaded :", numTraces
        print "Trace length            :", trace_length
        print "Loading time            : %0.2f s" % time_load


def log_parameters():
    if log_intermediate_results:
        print "---\nAttack parameters"
        print "Intermediate function        :", intermediateFunction.__name__
        print "CPA leakage function         :", leakageFunction.__name__
        print "LRA basis functions          :", basisFunctionsModel.__name__
        print "Encryption                   :", encrypt
        print "Number of sboxes to attack   :", num_sboxes_to_attack
        print "Known roundkey               : 0x%s" % str(bytearray(known_key)).encode("hex")
        print "Known key                    : 0x" + knownKeyStr.encode("hex")


if __name__ == "__main__":
    log_parameters()
    load_data()
    perform_attack()
