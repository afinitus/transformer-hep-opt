import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_eval_helpers import read_file, extract_value
from sklearn.metrics import roc_curve, roc_auc_score
from decimal import Decimal

def get_all_matching_directories_with_file(base_dir, pattern, filename):
    dirs = [d for d in os.listdir(base_dir) if pattern in d]
    dirs = [os.path.join(base_dir, d) for d in dirs]
    return [dir for dir in dirs if os.path.isfile(os.path.join(dir, filename))]

def GetDataFromQCDT(file_dir, file_name_qcd, file_name_top):
    file_qcd = os.path.join(file_dir, file_name_qcd)
    evalprob_qcdfromqcd = np.load(file_qcd)
    
    file_top = os.path.join(file_dir, file_name_top)
    evalprob_topfromqcd = np.load(file_top)

    return evalprob_qcdfromqcd, evalprob_topfromqcd

def GetDataFromTopT(file_dir, file_name_top, file_name_qcd):
    file_top = os.path.join(file_dir, file_name_top)
    evalprob_topfromtop = np.load(file_top)
    
    file_qcd = os.path.join(file_dir, file_name_qcd)
    evalprob_qcdfromtop = np.load(file_qcd)
    
    return evalprob_topfromtop, evalprob_qcdfromtop

def ExpectedProb(evalprob):
    exp_logp = np.log(1/(39402)) * evalprob['n_const']
    return exp_logp

def ComputeLLR(evalprobT, evalprobF, type):
    s = evalprobT['probs'] - evalprobF['probs']
    return s

def PlotLLR(s_qcd, s_top, path_to_plots, plot_title):
    bins = np.linspace(-340, 340, 40)
    plt.hist(s_qcd, histtype='step', bins=bins, density=True, color='blue', label='QCD')
    plt.hist(s_top, histtype='step', bins=bins, density=True, color='black', label='Top')
    plt.legend()
    plt.title(plot_title, loc='left')
    plt.savefig(os.path.join(path_to_plots, 'plot_LLR_test_1.pdf'))
    plt.close()

def ROCcurve(s_qcd, s_top, path_to_plots, plot_title):
    fpr, tpr, _ = roc_curve(np.append(np.zeros(len(s_qcd)), np.ones(len(s_top))), np.append(s_qcd, s_top))
    auc_score = roc_auc_score(np.append(np.zeros(len(s_qcd)), np.ones(len(s_top))), np.append(s_qcd, s_top))

    plt.plot(tpr, 1/fpr, label="LLR Test AUC=" + str(truncate_float(auc_score, 5)), c="blue")
    plt.ylim(1, 1e8)
    plt.xlim(0, 1)
    plt.xlabel(r"$\epsilon_{\rm{top}}$")
    plt.ylabel(r"$1 / \epsilon_{\rm{QCD}}$")
    plt.yscale('log')
    plt.title(plot_title + ' -- auc=' + str(auc_score), loc='left')
    plt.savefig(os.path.join(path_to_plots, 'plot_ROC2_1.pdf'))
    # plt.close()

def ROCcurveTrain(s_qcd, s_top, path_to_plots, plot_title):
    fpr, tpr, _ = roc_curve(np.append(np.zeros(len(s_qcd)), np.ones(len(s_top))), np.append(s_qcd, s_top))
    auc_score = roc_auc_score(np.append(np.zeros(len(s_qcd)), np.ones(len(s_top))), np.append(s_qcd, s_top))

    plt.plot(tpr, 1/fpr, label="LLR Train AUC=" + str(truncate_float(auc_score, 5)), c="black")
    plt.ylim(1, 1e8)
    plt.xlim(0, 1)
    plt.xlabel(r"$\epsilon_{\rm{top}}$")
    plt.ylabel(r"$1 / $\epsilon_{\rm{QCD}}$")
    plt.yscale('log')
    plt.title(plot_title + ' -- auc=' + str(auc_score), loc='left')
    plt.savefig(os.path.join(path_to_plots, 'plot_ROC2_1.pdf'))
    # plt.close()

def truncate_float(float_number, decimal_places):
    multiplier = 10 ** decimal_places
    return int(float_number * multiplier) / multiplier

def plot_roc_curve(predictions, n_train, classifier_name, plotted_curves):
    # Compute ROC curve and ROC area for each class
    str_ntrain = '%.0E' % Decimal(str(n_train))
    
    if str_ntrain in plotted_curves:
        return
    
    fpr, tpr, _ = roc_curve(predictions['labels'], predictions['predictions'])
    roc_auc = roc_auc_score(predictions['labels'], predictions['predictions'])
    
    plt.plot(tpr, 1/fpr, label=classifier_name + ' AUC=' + str(truncate_float(roc_auc, 5)), linestyle='--')
    plotted_curves.add(str_ntrain)

def GetPredictions(classifier_dir):
    return np.load(os.path.join(classifier_dir, 'predictions_test.npz'))

test_results_dir = '/pscratch/sd/n/nishank/humberto/FirstTime_topvsqcd_100const/densities'

qcd_file_dir = os.path.join(test_results_dir, 'QCD-transformer/')
qcd_file_name_qcd = 'results_test_eval_optclass_testset_nsamples200000.npz'
qcd_file_name_top = 'results_test_eval_optclass_testset_other_nsamples200000.npz'

top_file_dir = os.path.join(test_results_dir, 'Top-transformer/')
top_file_name_top = 'results_test_eval_optclass_testset_nsamples200000.npz'
top_file_name_qcd = 'results_test_eval_optclass_testset_other_nsamples200000.npz'

plot_title = 'T-Classifiers vs LLR'
path_to_plots = os.path.join(test_results_dir, 'test_model_a_ttbar_qcd_200k_wall/')
os.makedirs(path_to_plots, exist_ok=True)

evalprob_qcdfromqcd, evalprob_topfromqcd = GetDataFromQCDT(qcd_file_dir, qcd_file_name_qcd, qcd_file_name_top)
evalprob_topfromtop, evalprob_qcdfromtop = GetDataFromTopT(top_file_dir, top_file_name_top, top_file_name_qcd)

s_top = ComputeLLR(evalprob_topfromtop, evalprob_topfromqcd, 'qcd')
s_qcd = ComputeLLR(evalprob_qcdfromtop, evalprob_qcdfromqcd, 'qcd')

ROCcurve(s_qcd, s_top, path_to_plots, plot_title)

qcd_file_dir = os.path.join(test_results_dir, 'QCD-transformer/')
qcd_file_name_qcd = 'results_test_eval_optclass__nsamples1000000.npz'
qcd_file_name_top = 'results_test_eval_optclass__other_nsamples1000000.npz'

top_file_dir = os.path.join(test_results_dir, 'Top-transformer/')
top_file_name_top = 'results_test_eval_optclass_nsamples1000000.npz'
top_file_name_qcd = 'results_test_eval_optclass_other_nsamples1000000.npz'

evalprob_qcdfromqcd, evalprob_topfromqcd = GetDataFromQCDT(qcd_file_dir, qcd_file_name_qcd, qcd_file_name_top)
evalprob_topfromtop, evalprob_qcdfromtop = GetDataFromTopT(top_file_dir, top_file_name_top, top_file_name_qcd)

s_top = ComputeLLR(evalprob_topfromtop, evalprob_topfromqcd, 'qcd')
s_qcd = ComputeLLR(evalprob_qcdfromtop, evalprob_qcdfromqcd, 'qcd')

ROCcurveTrain(s_qcd, s_top, path_to_plots, plot_title)

dir_classifier_results = '/pscratch/sd/n/nishank/humberto/log_dir'
classifier_dirs = get_all_matching_directories_with_file(dir_classifier_results, 'top_vs_qcd_transformerdata_classifier_test_2', 'predictions_test.npz')

plotted_curves = set()
for classifier_dir in classifier_dirs:
    predicitions_test = GetPredictions(classifier_dir)
    lines = read_file(classifier_dir + '/arguments.txt')
    n_train = extract_value('num_events', lines)
    classifier_name = 'T-' + '%.0E' % Decimal(str(n_train))
    plot_roc_curve(predicitions_test, n_train, classifier_name, plotted_curves)

roc_data_dir = '../roc_info'
roc_files = [
    ('dsa_roc_data_test.npz', 'DSA'),
    ('dsapre_roc_data_test.npz', 'DSA Pre'),
    ('efn_roc_data_test.npz', 'EFN'),
    ('efnpre_roc_data_test.npz', 'EFN Pre'),
    ('pfn_roc_data_test.npz', 'PFN'),
    ('pfnpre_roc_data_test.npz', 'PFN Pre'),
]

for roc_file, model_name in roc_files:
    data = np.load(os.path.join(roc_data_dir, roc_file))
    fpr = data['fpr']
    tpr = data['tpr']
    roc_auc = data['roc_auc']

    plt.plot(tpr, 1/fpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.5f})')

plt.ylim(1, 2e6)
plt.xlim(0, 1)
plt.xlabel('TPR')
plt.ylabel('1/FPR')
plt.yscale('log')
plt.legend(loc='upper right')
plt.title(plot_title, loc='left')
plt.savefig(os.path.join(path_to_plots, 'plot_ROCcurve_all.pdf'))
plt.close()

exit()

