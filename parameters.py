### Authors: Nicolas Y. Masse, Gregory D. Grant

import numpy as np
from itertools import product, chain

print("\n--> Loading parameters...")

##############################
### Independent parameters ###
##############################

global par
par = {
    # Setup parameters
    'save_dir'              : './savedir/',
    'stabilization'         : 'pathint',    # 'EWC' (Kirkpatrick method) or 'pathint' (Zenke method)
    'save_analysis'         : False,
    'reset_weights'         : False,        # reset weights between tasks

    # Network configuration
    'synapse_config'        : 'std_stf',     # Full is 'std_stf'
    'exc_inh_prop'          : 0.8,          # Literature 0.8, for EI off 1
    'var_delay'             : False,
    'training_method'       : 'RL',         # 'SL', 'RL'
    'architecture'          : 'LSTM',       # 'BIO', 'LSTM'

    # Network shape
    'num_motion_tuned'      : 48,
    'num_fix_tuned'         : 4,
    'num_rule_tuned'        : 0,
    'n_val'                 : 1,
    'include_rule_signal'   : True,
    'n_hidden'              : [32, 32],

    # Timings and rates
    'dt'                    : 20,
    'learning_rate'         : 4e-3,
    'membrane_time_constant': 100,
    'connection_prob'       : 1.0,
    'discount_rate'         : 0.,

    # Variance values
    'clip_max_grad_val'     : 1.0,
    'input_mean'            : 0.0,
    'noise_in_sd'           : 0.0,
    'noise_rnn_sd'          : 0.05,

    # Task specs
    'task'                  : 'multistim',
    'n_tasks'               : 20,
    'multistim_trial_length': 2000,
    'mask_duration'         : 0,
    'dead_time'             : 200,

    # Tuning function data
    'num_motion_dirs'       : 8,
    'tuning_height'         : 4.0,        # magnitude scaling factor for von Mises

    # Cost values
    'spike_cost'            : 0.,
    'weight_cost'           : 0.,
    'entropy_cost'          : 0.0001,
    'val_cost'              : 0.01,
    'error_cost'            : 0.1,

    # Synaptic plasticity specs
    'tau_fast'              : 200,
    'tau_slow'              : 1500,
    'U_stf'                 : 0.15,
    'U_std'                 : 0.45,

    # Training specs
    'batch_size'            : 256,
    'n_train_batches'       : 5000, #50000,

    # Omega parameters
    'omega_c'               : 2.,
    'omega_xi'              : 0.001,
    'EWC_fisher_num_batches': 16,   # number of batches when calculating EWC

    # Gating parameters
    'gating_type'           : None, # 'XdG', 'partial', 'split', None
    'gate_pct'              : 0.8,  # Num. gated hidden units for 'XdG' only
    'n_subnetworks'         : 4,    # Num. subnetworks for 'split' only

    # Stimulus parameters
    'fix_break_penalty'     : -1.,
    'wrong_choice_penalty'  : -0.01,
    'correct_choice_reward' : 1.,

    # Save paths
    'save_fn'               : 'model_results.pkl',
    'ckpt_save_fn'          : 'model.ckpt',
    'ckpt_load_fn'          : 'model.ckpt',

}


############################
### Dependent parameters ###
############################


def update_parameters(updates):
    """
    Takes a list of strings and values for updating parameters in the parameter dictionary
    Example: updates = [(key, val), (key, val)]
    """
    for (key, val) in updates.items():
        par[key] = val
        print('Updating : ', key, ' -> ', val)
    update_dependencies()


def update_dependencies():
    """ Updates all parameter dependencies """

    ###
    ### Putting together network structure
    ###

    print('Using LSTM networks; setting to EI to False')
    par['EI'] = False
    par['exc_inh_prop'] = 1.
    par['synapse_config'] = None
    par['spike_cost'] = 0.

    # Number of output neurons
    par['n_output'] = par['num_motion_dirs'] + 1
    par['n_pol'] = par['num_motion_dirs'] + 1

    # Number of input neurons
    par['num_pred_cells'] = len(par['n_hidden'])
    par['n_cell_input'] = []
    par['n_LSTM_input'] = []
    # input into predictive cell will consist of feedforward input plus "dopamine" reward signal
    par['n_cell_input'].append((par['num_motion_tuned'] + par['num_fix_tuned'] + par['num_rule_tuned'] + 1))
    for i in range(1, par['num_pred_cells']):
        par['n_cell_input'].append((par['n_hidden'][i-1] + 1))

    for i in range(par['num_pred_cells']-1):
        par['n_LSTM_input'].append(2*par['n_cell_input'][i] + par['n_hidden'][i-1])
    par['n_LSTM_input'].append(2*par['n_cell_input'][-1])

    # Specify time step in seconds and neuron time constant
    par['dt_sec'] = par['dt']/1000
    par['alpha_neuron'] = np.float32(par['dt'])/par['membrane_time_constant']

    # Generate noise deviations
    par['noise_rnn'] = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
    par['noise_in'] = np.sqrt(2/par['alpha_neuron'])*par['noise_in_sd']

    # Set trial step length
    par['num_time_steps'] = par['multistim_trial_length']//par['dt']

    # Set up gating vectors for hidden layer
    #gen_gating()

    ###
    ### Setting up weights, biases, masks, etc.
    ###

    # Specify initial RNN state
    par['h_init'] = []
    for i in range(par['num_pred_cells']):
        par['h_init'].append(0.1*np.ones((par['batch_size'], par['n_hidden'][i]), dtype=np.float32))

    # Initialize weights
    c = 0.05

    # Initialize RL-specific weights
    par['W_pol_out_init'] = np.float32(np.random.uniform(-c, c, size = [par['n_hidden'][-1], par['n_pol']]))
    par['b_pol_out_init'] = np.zeros((1,par['n_pol']), dtype = np.float32)

    par['W_val_out_init'] = np.float32(np.random.uniform(-c, c, size = [par['n_hidden'][-1], par['n_val']]))
    par['b_val_out_init'] = np.zeros((1,par['n_val']), dtype = np.float32)

    ### Setting up LSTM weights and biases
    c = 0.05
    LSTM_var_names = ['Wf', 'Wi', 'Wo', 'Wc', 'W_pred', 'Uf', 'Ui', 'Uo', 'Uc', 'bf', 'bi', 'bo', 'bc', 'b_pred']
    for name in LSTM_var_names:
        par[name + '_init'] = []
        for i in range(par['num_pred_cells']):
            if name == 'W_pred':
                par[name + '_init'].append(np.float32(np.random.uniform(-c, c, size = [par['n_hidden'][i], par['n_cell_input'][i]])))
            elif name == 'b_pred':
                par[name + '_init'].append(np.float32(np.random.uniform(-c, c, size = [1, par['n_cell_input'][i]])))
            elif name.startswith('W'):
                par[name + '_init'].append(np.float32(np.random.uniform(-c, c, size = [par['n_LSTM_input'][i], par['n_hidden'][i]])))
            elif name.startswith('U'):
                par[name + '_init'].append(np.float32(np.random.uniform(-c, c, size = [par['n_hidden'][i], par['n_hidden'][i]])))
            elif name.startswith('b'):
                par[name + '_init'].append(np.float32(np.random.uniform(-c, c, size = [1, par['n_hidden'][i]])))


def gen_gating():
    """
    Generate the gating signal to applied to all hidden units
    """
    par['gating'] = []

    for t in range(par['n_tasks']):
        gating_task = np.zeros(par['n_hidden'], dtype=np.float32)
        for i in range(par['n_hidden']):

            if par['gating_type'] == 'XdG':
                if np.random.rand() < 1-par['gate_pct']:
                    gating_task[i] = 1

            elif par['gating_type'] == 'split':
                if t%par['n_subnetworks'] == i%par['n_subnetworks']:
                    gating_layer[i] = 1

            elif par['gating_type'] is None:
                gating_task[i] = 1

        par['gating'].append(gating_task)


def initialize_weight(dims, connection_prob):
    w = np.random.gamma(shape=0.25, scale=1.0, size=dims)
    w *= (np.random.rand(*dims) < connection_prob)
    return np.float32(w)


update_dependencies()
print("--> Parameters successfully loaded.\n")
