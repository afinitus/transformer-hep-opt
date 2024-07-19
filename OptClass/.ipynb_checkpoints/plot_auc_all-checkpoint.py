import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_eval_helpers import read_file,extract_value
from sklearn.metrics import roc_curve, roc_auc_score

from decimal import Decimal



def GetDataFromQCDT(file_dir,file_name_qcd,file_name_top):


    file_qcd=file_dir+'/'+file_name_qcd
    evalprob_qcdfromqcd = np.load(file_qcd)
    
    file_top=file_dir+'/'+file_name_top
    evalprob_topfromqcd = np.load(file_top)



    return evalprob_qcdfromqcd,evalprob_topfromqcd


def GetDataFromTopT(file_dir,file_name_top,file_name_qcd):

    

    file_top=file_dir+'/'+file_name_top
    evalprob_topfromtop = np.load(file_top)
    
    file_qcd=file_dir+'/'+file_name_qcd
    evalprob_qcdfromtop = np.load(file_qcd)
    
    return evalprob_topfromtop,evalprob_qcdfromtop


def ExpectedProb(evalprob):
    exp_logp=np.log(1/(39402))*evalprob['n_const']
    return exp_logp

def ComputeLLR(evalprobT,evalprobF,type):

    s=evalprobT['probs']-evalprobF['probs']

    return s

def PlotLLR(s_qcd,s_top,path_to_plots,plot_title):
    bins = np.linspace(-340, 340, 40)
    plt.hist(s_qcd,histtype='step',bins=bins,density=True,color='blue',label='QCD')
    plt.hist(s_top,histtype='step',bins=bins,density=True,color='black',label='Top')
    plt.legend()
    plt.title(plot_title,loc='left')
    plt.savefig(path_to_plots+'/plot_LLR_test_1.pdf')
    plt.close()
    return


def ROCcurve(s_qcd,s_top,path_to_plots,plot_title):
    
    fpr, tpr, _ = roc_curve(np.append(np.zeros(len(s_qcd)), np.ones(len(s_top))), np.append(s_qcd, s_top))
    #fpr, tpr, _ = roc_curve(np.append(np.zeros(len(s_top)), np.ones(len(s_qcd))), np.append(s_top, s_qcd))
    #fpr, tpr, _ = roc_curve(np.append(np.ones(len(s_qcd)), np.zeros(len(s_top))), np.append(s_qcd, s_top))
    print(roc_auc_score(np.append(np.zeros(len(s_qcd)), np.ones(len(s_top))), np.append(s_qcd, s_top)))
    auc_score=roc_auc_score(np.append(np.zeros(len(s_qcd)), np.ones(len(s_top))), np.append(s_qcd, s_top))

    plt.plot(tpr,1/fpr, label="LLR Test AUC="+str(truncate_float(auc_score,5)), c="blue")
    plt.ylim(1, 1e8)
    plt.xlim(0, 1)
    plt.xlabel(r"$\epsilon_{\rm{top}}$")
    plt.ylabel(r"$1 / \epsilon_{\rm{QCD}}$")
    plt.yscale('log')
    #plt.xscale('log')
   
    plt.title(plot_title+' -- auc='+str(auc_score),loc='left')
    plt.savefig(path_to_plots+'/plot_ROC2_1.pdf')
    #plt.close()
    return


def ROCcurveTrain(s_qcd,s_top,path_to_plots,plot_title):
    
    fpr, tpr, _ = roc_curve(np.append(np.zeros(len(s_qcd)), np.ones(len(s_top))), np.append(s_qcd, s_top))
    #fpr, tpr, _ = roc_curve(np.append(np.zeros(len(s_top)), np.ones(len(s_qcd))), np.append(s_top, s_qcd))
    #fpr, tpr, _ = roc_curve(np.append(np.ones(len(s_qcd)), np.zeros(len(s_top))), np.append(s_qcd, s_top))
    print(roc_auc_score(np.append(np.zeros(len(s_qcd)), np.ones(len(s_top))), np.append(s_qcd, s_top)))
    auc_score=roc_auc_score(np.append(np.zeros(len(s_qcd)), np.ones(len(s_top))), np.append(s_qcd, s_top))

    plt.plot(tpr,1/fpr, label="LLR Train AUC="+str(truncate_float(auc_score,5)), c="black")
    plt.ylim(1, 1e8)
    plt.xlim(0, 1)
    plt.xlabel(r"$\epsilon_{\rm{top}}$")
    plt.ylabel(r"$1 / \epsilon_{\rm{QCD}}$")
    plt.yscale('log')
    #plt.xscale('log')
   
    plt.title(plot_title+' -- auc='+str(auc_score),loc='left')
    plt.savefig(path_to_plots+'/plot_ROC2_1.pdf')
    #plt.close()
    return

def truncate_float(float_number, decimal_places):
    multiplier = 10 ** decimal_places
    return int(float_number * multiplier) / multiplier

def plot_roc_curve(predictions,n_train):
    # Compute ROC curve and ROC area for each class
    
    str_ntrain= '%.0E' % Decimal(str(n_train))
    
    fpr, tpr, _ = roc_curve(predictions['labels'], predictions['predictions'])
    #fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(predictions['labels'],predictions['predictions'])
    
    plt.plot(tpr,1/fpr, label='T-'+str_ntrain+'. AUC='+str(truncate_float(roc_auc,5)),linestyle='--')
  

    return


def plot_probs(evalprob,path_to_plots,plot_title):

     plt.hist(evalprob['probs'],histtype='step',bins=30,density=True,color='blue')

     exp_logp=ExpectedProb(evalprob)

     plt.hist(exp_logp,histtype='step',bins=30,linestyle='--',density=True,color='blue')
     plt.title(plot_title,loc='left')
     plt.savefig(path_to_plots+'/plot_probs_test_1.pdf')
     plt.close()

     return



def plot_probs(evalprob_qcdfromqcd,evalprob_topfromqcd,evalprob_topfromtop,evalprob_qcdfromtop,path_to_plots,plot_title):
         plt.hist(evalprob_qcdfromqcd['probs'],histtype='step',bins=30,density=True,color='blue',label='QCD')
         plt.hist(evalprob_qcdfromtop['probs'],histtype='step',bins=30,density=True,color='blue',linestyle='--')

         plt.hist(evalprob_topfromtop['probs'],histtype='step',bins=30,density=True,color='black',label='Top')
         plt.hist(evalprob_topfromqcd['probs'],histtype='step',bins=30,density=True,color='black',linestyle='--')
         plt.xlabel('log(p)')
         plt.legend()
         plt.title(plot_title,loc='left')
         plt.savefig(path_to_plots+'/plot_probs_test_1.pdf')
         plt.close()

         return

def GetPredictions(classifier_dir):

    predicitions_test = np.load(classifier_dir+'/predictions_test.npz')
    return predicitions_test


def plot_color(evalprob,path_to_plots,plot_title):

     plt.plot(evalprob['probs'],evalprob['n_const'],'.')
     plt.xlabel('n_{const}')
     plt.ylabel('log(p)')
     plt.title(plot_title,loc='left')
     plt.savefig(path_to_plots+'/plot_plot_test_1.pdf')
     plt.close()

     return
 
 
def plot_contour(evalprob_qcdfromqcd,evalprob_topfromtop,path_to_plots,plot_title):

     sns.kdeplot(data=evalprob_qcdfromqcd,x='probs',y='n_const',levels=4,color='blue', fill=True,alpha=.5)
     sns.kdeplot(data=evalprob_topfromtop,x='probs',y='n_const',levels=4,color='black', fill=True,alpha=.5)
     plt.plot(np.log(1. / 39402) * np.linspace(0, 100, 100), np.linspace(0, 100, 100), linestyle="--", color='grey')
     plt.ylabel('$n_{const}$')
     plt.xlabel('log(p)')
     plt.title(plot_title,loc='left')
     plt.savefig(path_to_plots+'/plot_contour_test_1.png')
     plt.close()

     return

test_results_dir='/pscratch/sd/n/nishank/humberto/FirstTime_topvsqcd_100const/densities'


qcd_file_dir=test_results_dir+'/QCD-transformer/'
qcd_file_name_qcd='results_test_eval_optclass_testset_nsamples200000.npz'
qcd_file_name_top='results_test_eval_optclass_testset_other_nsamples200000.npz'


top_file_dir=test_results_dir+'/Top-transformer/'
top_file_name_top='results_test_eval_optclass_testset_nsamples200000.npz'
top_file_name_qcd='results_test_eval_optclass_testset_other_nsamples200000.npz'
    
plot_title='T-Classifiers vs LLR'
#plot_title='JetClass test events'

path_to_plots=test_results_dir+'/test_model_a_ttbar_qcd_200k_wall/'
os.makedirs(path_to_plots,exist_ok=True)


evalprob_qcdfromqcd,evalprob_topfromqcd=GetDataFromQCDT(qcd_file_dir,qcd_file_name_qcd,qcd_file_name_top)
evalprob_topfromtop,evalprob_qcdfromtop=GetDataFromTopT(top_file_dir,top_file_name_top,top_file_name_qcd)

s_top=ComputeLLR(evalprob_topfromtop,evalprob_topfromqcd,'qcd')
s_qcd=ComputeLLR(evalprob_qcdfromtop,evalprob_qcdfromqcd,'qcd')

ROCcurve(s_qcd,s_top,path_to_plots,plot_title)

qcd_file_dir=test_results_dir+'/QCD-transformer/'
qcd_file_name_qcd='results_test_eval_optclass__nsamples1000000.npz'
qcd_file_name_top='results_test_eval_optclass__other_nsamples1000000.npz'


top_file_dir=test_results_dir+'/Top-transformer/'
top_file_name_top='results_test_eval_optclass_nsamples1000000.npz'
top_file_name_qcd='results_test_eval_optclass_other_nsamples1000000.npz'
    


evalprob_qcdfromqcd,evalprob_topfromqcd=GetDataFromQCDT(qcd_file_dir,qcd_file_name_qcd,qcd_file_name_top)
evalprob_topfromtop,evalprob_qcdfromtop=GetDataFromTopT(top_file_dir,top_file_name_top,top_file_name_qcd)

s_top=ComputeLLR(evalprob_topfromtop,evalprob_topfromqcd,'qcd')
s_qcd=ComputeLLR(evalprob_qcdfromtop,evalprob_qcdfromqcd,'qcd')

ROCcurveTrain(s_qcd,s_top,path_to_plots,plot_title)




dir_classifier_results='/pscratch/sd/n/nishank/humberto/log_dir'

classifiers_dirs=os.listdir(dir_classifier_results)


for classifier_dir in classifiers_dirs:
    if 'test_2' not in classifier_dir:
        continue
    classifier_dir=dir_classifier_results+'/'+classifier_dir
    
    predicitions_test=GetPredictions(classifier_dir)
    lines=read_file(classifier_dir+'/arguments.txt')
    n_train=extract_value('num_events',lines)
    plot_roc_curve(predicitions_test,n_train)



plt.ylim(1, 2e6)
plt.xlim(0, 1)
#plt.xlabel(r"$\epsilon_{\rm{top}}$")
#plt.ylabel(r"$1 / \epsilon_{\rm{QCD}}$")
plt.xlabel('TPR')
plt.ylabel('1/FPR')
plt.yscale('log')
#plt.xscale('log')
plt.legend(loc='upper right')
plt.title(plot_title,loc='left')
plt.savefig(path_to_plots+'/plot_ROCcurve_all.pdf')
plt.close()

exit()

