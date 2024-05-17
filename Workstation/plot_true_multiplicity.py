import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os
from data_eval_helpers import make_continues,LoadTrue

import matplotlib.colors as mcolors




def PlotMultiplicity(jets,color):

    mask = jets[:, :, 0] != 0
    print(np.shape(mask))
    plt.hist(np.sum(mask, axis=1), bins=np.linspace(-0.5, 200.5, 102),color=color,histtype='step',density=True)

    return



discrete_truedata_filename='/net/data_t2k/transformers-hep/JetClass/discretized/TTBar_train___10M_TTBar.h5'

bins_path_prefix='../preprocessing_bins/'
bin_tag='10M_TTBar'
pt_bins = np.load(bins_path_prefix+'pt_bins_'+bin_tag+'.npy')
eta_bins = np.load(bins_path_prefix+'eta_bins_'+bin_tag+'.npy')
phi_bins = np.load(bins_path_prefix+'phi_bins_'+bin_tag+'.npy')

n_test_samples=9000



list_of_jets=['TTBar','ZJetsToNuNu','HToBB','HToCC','HToGG','HToWW2Q1L','HToWW4Q','TTBarLep','WToQ','ZToQQ']
color_list=list(mcolors.TABLEAU_COLORS.values())




for j in range(len(list_of_jets)):

    jet=list_of_jets[j]
    print(jet)
    discrete_truedata_filename='/net/data_t2k/transformers-hep/JetClass/discretized/'+jet+'_train___10M_'+jet+'.h5'
    
    jets_true,ptj_true,mj_true=LoadTrue(discrete_truedata_filename,n_test_samples,pt_bins,eta_bins,phi_bins)
    
    PlotMultiplicity(jets_true,color_list[j])


plt.xlabel('Multiplicity')
plt.legend()
plt.savefig('/plot_mul_all.png')
plt.close()