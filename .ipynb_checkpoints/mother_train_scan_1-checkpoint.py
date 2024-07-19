import os
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import random
import string
###make sufix a random number

def make_continues(jets, mask,pt_bins,eta_bins,phi_bins, noise=False):


    pt_disc = jets[:, :, 0]
    eta_disc = jets[:, :, 1]
    phi_disc = jets[:, :, 2]

    if noise:
        pt_con = (pt_disc - np.random.uniform(0.0, 1.0, size=pt_disc.shape)) * (
            pt_bins[1] - pt_bins[0]
        ) + pt_bins[0]
        eta_con = (eta_disc - np.random.uniform(0.0, 1.0, size=eta_disc.shape)) * (
            eta_bins[1] - eta_bins[0]
        ) + eta_bins[0]
        phi_con = (phi_disc - np.random.uniform(0.0, 1.0, size=phi_disc.shape)) * (
            phi_bins[1] - phi_bins[0]
        ) + phi_bins[0]
    else:
        pt_con = (pt_disc - 0.5) * (pt_bins[1] - pt_bins[0]) + pt_bins[0]
        eta_con = (eta_disc - 0.5) * (eta_bins[1] - eta_bins[0]) + eta_bins[0]
        phi_con = (phi_disc - 0.5) * (phi_bins[1] - phi_bins[0]) + phi_bins[0]


    pt_con = np.exp(pt_con)
    pt_con[mask] = 0.0
    eta_con[mask] = 0.0
    phi_con[mask] = 0.0
    
    pxs = np.cos(phi_con) * pt_con
    pys = np.sin(phi_con) * pt_con
    pzs = np.sinh(eta_con) * pt_con
    es = (pxs ** 2 + pys ** 2 + pzs ** 2) ** (1. / 2)

    pxj = np.sum(pxs, -1)
    pyj = np.sum(pys, -1)
    pzj = np.sum(pzs, -1)
    ej = np.sum(es, -1)
    
    ptj = np.sqrt(pxj**2 + pyj**2)
    mj = (ej ** 2 - pxj ** 2 - pyj ** 2 - pzj ** 2) ** (1. / 2)

    continues_jets = np.stack((pt_con, eta_con, phi_con), -1)

    return continues_jets, ptj, mj

def Make_Plots(jets,pt_bins,eta_bins,phi_bins,mj,jets_true,ptj_true,mj_true,path_to_plots):

    plt.hist(np.log(jets[:,:,0]).flatten(), bins=pt_bins, color='blue',histtype='step',density=True)
    plt.hist(np.log(jets_true[:,:,0]).flatten(), bins=pt_bins, color='red',histtype='step',density=True)
    plt.xlabel('$\log (p_T)$')
    
    plt.savefig(path_to_plots+'/plot_pt_trans.png')
    plt.close()
    
    plt.hist(jets[:,:,1].flatten(), bins=eta_bins, color='blue',histtype='step',density=True)
    plt.hist(jets_true[:,:,1].flatten(), bins=eta_bins, color='red',histtype='step',density=True)
    plt.xlabel('$\Delta\eta$')
    plt.savefig(path_to_plots+'/plot_eta_trans.png')
    plt.close()
    
    plt.hist(jets[:,:,2].flatten(), bins=phi_bins, color='blue',histtype='step',density=True)
    plt.hist(jets_true[:,:,2].flatten(), bins=phi_bins, color='red',histtype='step',density=True)
    plt.xlabel('$\Delta\phi$')
    plt.savefig(path_to_plots+'/plot_phi_trans.png')
    plt.close()
    
    mask = jets[:, :, 0] != 0
    plt.hist(np.sum(mask, axis=1), bins=np.linspace(-0.5, 120.5, 102),color='blue',histtype='step',density=True)
    mask = jets_true[:, :, 0] != 0
    plt.hist(np.sum(mask, axis=1), bins=np.linspace(-0.5, 120.5, 102),color='red',histtype='step',density=True)
    plt.xlabel('Multiplicity')
    plt.savefig(path_to_plots+'/plot_mul_trans.png')
    plt.close()
    
    
    mj_bins = np.linspace(0, 1, 100)
    plt.hist(np.clip(mj, mj_bins[0], mj_bins[-1]), bins=mj_bins,color='blue',histtype='step',density=True)
    plt.hist(np.clip(mj_true, mj_bins[0], mj_bins[-1]), bins=mj_bins,color='red',histtype='step',density=True)
    plt.xlabel('$m_{jet}$')
    plt.savefig(path_to_plots+'/plot_mj_trans.png')
    plt.close()
    return


def LoadTrue(discrete_truedata_filename,n_samples):


    tmp = pd.read_hdf(discrete_truedata_filename, key="discretized", stop=None)
    print(tmp)
    print(tmp.shape)
    tmp=tmp.sample(n_samples)
    tmp = tmp.to_numpy()[:, :300].reshape(len(tmp), -1, 3)
    print(tmp)
    print(tmp.shape)
    print('hello')
    tmp=tmp[:,:,:]
    print(tmp.shape)
    print('hello')
 

    mask = tmp[:, :, 0] == -1
    print(mask)
    jets_true,ptj_true,mj_true = make_continues(tmp, mask,pt_bins,eta_bins,phi_bins, noise=False)

    return jets_true,ptj_true,mj_true

def random_string():
    # initializing size of string
    N = 7
 
    # using random.choices()
    # generating random strings
    res = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=N))
    # print result
    print("The generated random string : " + str(res))
    return str(res)
####Trainning parameters

#list_of_jets=['TTBar','ZJetsToNuNu','HToBB','HToCC','HToGG','HToWW2Q1L','HToWW4Q','TTBarLep','WToQ','ZToQQ']
list_of_jets=['TTBar']
num_const_list=[20,40,70]
num_epochs_list=[30]
lr_list=[.001]
lr_decay_list=[.000001]
num_events_list=[1000000]
dropout_list=[0]
num_heads_list=[4]
num_layers_list=[8]
num_bins_list=["41 31 31"]
weight_decay_list=[0.00001]
hidden_dim_list=[256]
batch_size_list=[100]
num_events_val=500000
###Sampling parameters
num_samples_test_list=[200000]
train_batch_size=100
num_const_test=120
trunc_test_list=[5000]
model_name='model_best.pt'

main_dir='/net/data_t2k/transformers-hep/JetClass/'
bins_path_prefix='preprocessing_bins/'
###Training stage
for jet in list_of_jets:

    mother_dir='/net/data_t2k/transformers-hep/JetClass/'+jet+'_models/'
    tag_oftrain=jet+'_run_scan_'
    data_path='/net/data_t2k/transformers-hep/JetClass/discretized/'+jet+'_train___10M_'+jet+'.h5'
    model_path=mother_dir+'/'+tag_oftrain
    log_dir='/net/data_t2k/transformers-hep/JetClass/'+jet+'_models/'+tag_oftrain
    output='linear'
    
    ###for sampling
    

    test_dataset='/net/data_t2k/transformers-hep/JetClass/discretized/'+jet+'_test___10M_'+jet+'.h5'
    
    
    ###for plotting samples
    bin_tag='10M_'+str(jet)
    pt_bins = np.load(bins_path_prefix+'pt_bins_'+bin_tag+'.npy')
    eta_bins = np.load(bins_path_prefix+'eta_bins_'+bin_tag+'.npy')
    phi_bins = np.load(bins_path_prefix+'phi_bins_'+bin_tag+'.npy')

    n_samples=200000
    test_data_name='/discretized/'+jet+'_test___10M_'+jet+'.h5'
    discrete_truedata_filename=main_dir+test_data_name
    jets_true,ptj_true,mj_true=LoadTrue(discrete_truedata_filename,n_samples)
    print(mj_true)


    for num_events  in num_events_list:
        for num_const in num_const_list:
            for batch_size in batch_size_list:
                for num_bins in num_bins_list:
                    for num_epochs in num_epochs_list:
                        for lr_decay in lr_decay_list:
                            for weight_decay in weight_decay_list:
                                for num_layers in num_layers_list:
                                    for dropout in dropout_list:
                                        for num_heads in num_heads_list:
                                            for lr in lr_list:
                                                for hidden_dim in hidden_dim_list:
                                                
                                                
                                                    name_sufix=random_string()
                                                
                                                    os.system('python train_1.py --data_path '+str(data_path)+' --model_path '+str(model_path)+' --log_dir '+str(log_dir)+'  --output '+str(output)+' --num_const '+str(num_const)+' --num_epochs '+str(num_epochs)+'  --lr '+str(lr)+' --lr_decay '+str(lr_decay)+' --batch_size '+str(batch_size)+' --num_events '+str(num_events)+' --dropout '+str(dropout)+' --num_heads '+str(num_heads)+' --num_layers '+str(num_layers)+' --num_bins '+str(num_bins)+' --weight_decay '+str(weight_decay)+' --hidden_dim '+str(hidden_dim)+' --end_token --start_token '+' --name_sufix '+str(name_sufix)+' --num_events_val '+str(num_events_val)+' --checkpoint_steps 1200000')
                                                    
                                                    


                                                    model_path_curr=model_path+'_'+name_sufix
                                                    for num_samples_test in num_samples_test_list:
                                                    
                                                        tag_foreval='test_eval_nsamples'+str(num_samples_test)
                                                        command_eval='python evaluate_probabilities.py --model '+str(model_path_curr)+'/'+str(model_name)+' --data '+str(test_dataset)+' --tag '+tag_foreval+' --num_const '+str(num_const_test)+' --num_events '+str(num_samples_test)
        
        
                                                        os.system(command_eval)
                                                    
                                                        for trunc in trunc_test_list:
                                                            
                                                            tag_forsample='_nsamples'+str(num_samples_test)+'_trunc_'+str(trunc)
                                                            command_sample= 'python sample_jets_1.py --model_dir '+model_path_curr+' --savetag '+str(tag_forsample)+' --num_samples '+str(num_samples_test)+' --num_const '+str(num_const_test)+' --trunc '+str(trunc)+' --batchsize '+str(train_batch_size)+' --model_name '+model_name
                                                            print(command_sample)
                                                            os.system(command_sample)
                                                    
                                                    
                                                    
                                                            ####plotting
                                                            filename=model_path_curr+'/samples_'+tag_forsample+'.h5'

                                                            path_to_plots=model_path_curr+'/'+tag_forsample
                                                            os.makedirs(path_to_plots,exist_ok=True)

                                                            #filename = test_results_dir+"/ttbar_run_b_2_6/samples_test_sample_200k.h5"
                                                            tmp = pd.read_hdf(filename, key="discretized", stop=None)
                                                            tmp = tmp.to_numpy()[:, :300].reshape(len(tmp), -1, 3)
                                                            print(tmp.shape)

                                                            mask = tmp[:, :, 0] == -1
                                                            jets,ptj,mj = make_continues(tmp, mask,pt_bins,eta_bins,phi_bins, noise=False)

                                                            Make_Plots(jets,pt_bins,eta_bins,phi_bins,mj,jets_true,ptj_true,mj_true,path_to_plots)
