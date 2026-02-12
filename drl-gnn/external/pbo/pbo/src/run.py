# Generic imports
import os
import sys
import json
import time
import collections
import numpy             as np
import matplotlib.pyplot as plt

# Custom imports
from pbo.src.train import *

########################
# Parameters decoder to collect json file
########################
def params_decoder(p_dict):

    return collections.namedtuple('X', p_dict.keys())(*p_dict.values())

########################
# Average training over multiple runs
########################
def run():

    # Check command-line input for json file
    if (len(sys.argv) == 2):
        json_file = sys.argv[1]
        env_path  = os.path.dirname(os.path.abspath(json_file))
    else:
        print('Command line error, please use as follows:')
        print('pbo path/to/my_env.json')

    # Read json parameter file
    with open(json_file, "r") as f:
        params = json.load(f, object_hook=params_decoder)

    # Storage arrays
    res_path  = 'results'
    gen       = np.zeros((              params.n_gen), dtype=int)
    rwd       = np.zeros((params.n_avg, params.n_gen), dtype=float)
    avg_rwd   = np.zeros((params.n_avg, params.n_gen), dtype=float)
    stdp_rwd  = np.zeros((              params.n_gen), dtype=float)
    stdm_rwd  = np.zeros((              params.n_gen), dtype=float)

    # Open storage repositories
    if (not os.path.exists(res_path)): os.makedirs(res_path)

    t           = time.localtime()
    path_time   = time.strftime("%m-%d", t)
    output_path = res_path+'/'+params.env_name+'_'+str(path_time)
    version = 1
    while os.path.exists(output_path):
        if version == 1:
            output_path = res_path+'/'+params.env_name+'_'+str(path_time)+'_v1'
        else:
            # Remove previous version and add new one
            base_path = res_path+'/'+params.env_name+'_'+str(path_time)
            output_path = base_path+'_v'+str(version)
        version += 1
    os.makedirs(output_path, exist_ok=True)

    for i in range(params.n_avg):
        print('### Avg run #'+str(i))
        start_time = time.time()
        train(params, output_path, env_path, i)
        dt = time.time() - start_time
        print('#   Elapsed time: {:.3f} seconds'.format(dt))

        f            = np.loadtxt(output_path+'/pbo_bst_'+str(i))
        gen          = f[:params.n_gen,0]
        rwd[i,:]     = f[:params.n_gen,2]

        f_avg        = np.loadtxt(output_path+'/pbo_avg_'+str(i))
        avg_rwd[i,:] = f_avg[:params.n_gen,1]

    # Write to file
    f     = output_path+'/pbo_avg.dat'
    array = np.vstack(gen)
    avg   = np.mean(rwd[:,:], axis=0)
    std   = np.std (rwd[:,:], axis=0)

    avg_avg   = np.mean(avg_rwd[:,:], axis=0)
    avg_std   = np.std (avg_rwd[:,:], axis=0)

    if (params.avg_type == "log"):
        log_avg = np.log(avg)
        log_std = 0.434*std/avg
        log_p   = log_avg + log_std
        log_m   = log_avg - log_std
        p       = np.exp(log_p)
        m       = np.exp(log_m)

        # for average rwd
        log_avg = np.log(avg_avg)
        log_std = 0.434*avg_std/avg_avg
        log_p   = log_avg + log_std
        log_m   = log_avg - log_std
        p_bis       = np.exp(log_p)
        m_bis       = np.exp(log_m)

    else:
        p       = avg + std
        m       = avg - std
        p_bis   = avg_avg + avg_std
        m_bis   = avg_avg - avg_std

    array = np.hstack((array,np.vstack(avg)))
    array = np.hstack((array,np.vstack(p)))
    array = np.hstack((array,np.vstack(m)))

    np.savetxt(f, array, fmt='%.5e')

    # Plot avg and std
    plt.rcParams['font.family'] = "serif"  # Use a serif font
    plt.rcParams['font.serif'] = ["Computer Modern Serif", "DejaVu Serif"]
    plt.rcParams['mathtext.fontset'] = "cm"
    plt.rcParams['mathtext.rm'] = "serif"
    plt.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['figure.titleweight'] = 'bold'

    plt.title('Reward - '+params.env_name)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.yscale(params.avg_type)
    plt.plot(-avg,
             color='darkblue',
             label='best')
    plt.fill_between(gen, -p, -m,
                     alpha=0.4,
                     color='darkblue')

    plt.plot(-avg_avg,
             color='darkred',
             label='avg')
    plt.fill_between(gen, -p_bis, -m_bis,
                     alpha=0.4,
                     color='darkred')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_path, 'pbo.png'), bbox_inches='tight')
