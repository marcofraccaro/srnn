"""
Implementation of the Stochastic RNN (SRNN) for speech modelling (TIMIT data)
"""
from __future__ import print_function, division
import time, datetime
from ml.data.timit_data import TimitData
from ml.models.SRNN_timit import SRNN_timit
from ml.training.train_srnn_timit import TrainSRNN_timit
from ml.training.decay import *
import ml.settings
import os.path
import cPickle as pkl

srnn_folder = os.path.dirname(os.path.realpath(__file__))  # folder in which this file is contained
data_folder = os.path.join(srnn_folder, 'data')  # subfolder with data


def run(nonlinearity_encoder='clipped_very_leaky_rectify', nonlinearity_decoder='clipped_very_leaky_rectify',
        range_nonlin=3.0, num_hidden_mlp=512, num_layers_mlp=2, latent_size_d=1024, latent_size_z=256,
        latent_size_a=1024, sequence_length=40, output_dim=200, smoothing=1, use_mu_residual=1,
        p_d_drop=0.0, p_z_drop=0.0, p_emb_u_drop=0.0, p_emb_x_drop=0.0, optimizer='adam', batch_size=64,
        log10_lr=-3.0, momentum=0.9, decay_type='exponential', decay=1.1, scale_decay=1.0, no_decay_epochs=15,
        init_mlp='normal', init_rnn='orthogonal', init_range=0.001, tempKL_type='linear', tempKL_start=0.0,
        tempKL_epochs=20, tempKL_decay=1.02, max_grad_norm=3000.0, clip_gradients=1000.0, max_num_epochs=50,
        eval_epoch=1, random_seed=1234, batch_size_test=32, unroll_scan=False, tolerance_softmax=1e-10, cons=-6.0,
        output_folder=None, run_name=None, writeLog=True, load_pickled_file=None):
    """
    Function to define and train the SRNN on Timit Data
    """

    # Output folder for log file and pickle
    if output_folder is None:
        output_folder = os.path.join(srnn_folder, 'output')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Name of this simulation (for log file and pickle)
    if run_name is None:
        # Str containing the current time in the format YYYYMMDD_HHMMSS e.g. 20160112_154201
        str_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
        run_name = "SRNNtimit_" + str_time

    # Path for pickle file, set to None if you don't want to pickle
    pickle_path = os.path.join(output_folder, run_name + ".pickle")
    plot_path = os.path.join(output_folder, run_name + "-plot")

    if load_pickled_file is None:
        settings = Settings_SRNN(nonlinearity_encoder=nonlinearity_encoder,
                                 nonlinearity_decoder=nonlinearity_decoder,
                                 range_nonlin=range_nonlin,
                                 num_hidden_mlp=num_hidden_mlp, num_layers_mlp=num_layers_mlp,
                                 latent_size_d=latent_size_d,
                                 latent_size_z=latent_size_z,
                                 latent_size_a=latent_size_a,
                                 sequence_length=sequence_length, output_dim=output_dim,
                                 smoothing=smoothing, use_mu_residual=use_mu_residual,
                                 p_d_drop=p_d_drop, p_z_drop=p_z_drop,
                                 p_emb_u_drop=p_emb_u_drop, p_emb_x_drop=p_emb_x_drop,
                                 optimizer=optimizer,
                                 batch_size=batch_size, log10_lr=log10_lr, momentum=momentum, decay_type=decay_type,
                                 decay=decay, scale_decay=scale_decay, no_decay_epochs=no_decay_epochs,
                                 init_rnn=init_rnn, init_range=init_range,
                                 init_mlp=init_mlp,
                                 tempKL_type=tempKL_type,
                                 tempKL_start=tempKL_start, tempKL_epochs=tempKL_epochs, tempKL_decay=tempKL_decay,
                                 max_grad_norm=max_grad_norm, clip_gradients=clip_gradients,
                                 max_num_epochs=max_num_epochs, eval_epoch=eval_epoch, random_seed=random_seed,
                                 batch_size_test=batch_size_test, unroll_scan=unroll_scan,
                                 tolerance_softmax=tolerance_softmax, cons=cons)

        # Initialize training object
        train = TrainSRNN_timit()

    else:
        # Load pickled settings. The model weights are not loaded now as we first need the pickled settings to construct
        # it. Once the model is defined, we can assign the pickled weights (this is done in the training file).
        with open(load_pickled_file, 'rb') as f:
            print("Loading pickled settings and train from %s" % load_pickled_file)
            _ = pkl.load(f)
            settings = Settings_SRNN.load_settings(f)
            train = TrainSRNN_timit.load_train(f)

    np.random.seed(settings.random_seed)
    # Get data object
    timit_data = TimitData(os.path.join(data_folder, 'timit_raw_batchsize%i_seqlen40.npz') % settings.batch_size,
                           batch_size=settings.batch_size)

    # Print data and settings info
    settings.settings_info()
    settings.print_csv()

    # Initialize SRNN model
    srnn_timit = SRNN_timit(settings)
    srnn_timit.model_info()

    # Compile training/evaluation functions
    f_train, f_valid = srnn_timit.initialize_computation_graph(timit_data, settings)

    # Choose learning rate decay schedule
    if settings.decay_type.lower() == 'power':
        decay_learning_rate = PowerDecaySchedule(settings.decay, settings.scale_decay, settings.max_num_epochs,
                                                 settings.no_decay_epochs)
    elif settings.decay_type.lower() == 'exponential':
        decay_learning_rate = ExponentialDecaySchedule(settings.decay, settings.max_num_epochs,
                                                       settings.no_decay_epochs)
    else:
        raise ValueError('Invalid decay_type \'' + settings.decay_type + '\'')

    # Choose temperature schedule for the KL term
    # We change the KL divergence slightly after every batch, e.g. with temperature linearly increasing from 0.2 to 1
    n_batches_train = timit_data.n_train // settings.batch_size
    max_decay_iters_KL = np.inf
    max_num_iters_KL = settings.max_num_epochs * n_batches_train
    no_decay_iters_KL = max_num_iters_KL - settings.tempKL_epochs * n_batches_train
    y_range_KL = (float(settings.tempKL_start), 1.0)
    reverse_KL = True
    if settings.tempKL_type.lower() == 'power':
        temperature_KL = PowerDecaySchedule(settings.tempKL_decay, scale_decay=1.0, max_num_epochs=max_num_iters_KL,
                                            no_decay_epochs=no_decay_iters_KL, max_decay_epochs=max_decay_iters_KL,
                                            reverse=reverse_KL, y_range=y_range_KL)
    elif settings.tempKL_type.lower() == 'exponential':
        temperature_KL = ExponentialDecaySchedule(settings.tempKL_decay, max_num_epochs=max_num_iters_KL,
                                                  no_decay_epochs=no_decay_iters_KL,
                                                  max_decay_epochs=max_decay_iters_KL, reverse=reverse_KL,
                                                  y_range=y_range_KL)
    elif settings.tempKL_type.lower() == 'linear':
        # in this case settings.tempKL_decay is useless as we are also passing y_range_KL
        temperature_KL = LinearDecaySchedule(settings.tempKL_decay, max_num_epochs=max_num_iters_KL,
                                             no_decay_epochs=no_decay_iters_KL, max_decay_epochs=max_decay_iters_KL,
                                             reverse=reverse_KL, y_range=y_range_KL)
    else:
        raise ValueError('Invalid tempKL_type \'' + settings.tempKL_type + '\'')

    # Train the model
    elbo_valid = train.train_model(timit_data, srnn_timit, f_train, f_valid,
                                   decay_learning_rate, temperature_KL, settings, pickle_path,
                                   load_pickled_file, plot_path)

    neg_elbo_valid = -elbo_valid
    print("Negative ELBO valid: %s" % neg_elbo_valid)

    return neg_elbo_valid


class Settings_SRNN(ml.settings.Settings):
    def __init__(self, nonlinearity_encoder='clipped_very_leaky_rectify',
                 nonlinearity_decoder='clipped_very_leaky_rectify',
                 range_nonlin=3.0, num_hidden_mlp=512, num_layers_mlp=2, latent_size_d=1024, latent_size_z=256,
                 latent_size_a=1024, sequence_length=40, output_dim=200, smoothing=1, use_mu_residual=1,
                 p_d_drop=0.0, p_z_drop=0.0, p_emb_u_drop=0.0, p_emb_x_drop=0.0, optimizer='adam', batch_size=64,
                 log10_lr=-3.0, momentum=0.9, decay_type='exponential', decay=1.1, scale_decay=1.0, no_decay_epochs=15,
                 init_mlp='normal', init_rnn='orthogonal', init_range=0.001, tempKL_type='linear', tempKL_start=0.0,
                 tempKL_epochs=20, tempKL_decay=1.02, max_grad_norm=3000.0, clip_gradients=1000.0, max_num_epochs=50,
                 eval_epoch=1, random_seed=1234, batch_size_test=32, unroll_scan=False, tolerance_softmax=1e-10,
                 cons=-6.0):

        # MODEL SETTINGS
        self.nonlinearity_encoder = nonlinearity_encoder  # Nonlinearities to be used in the encoding networks
        self.nonlinearity_decoder = nonlinearity_decoder  # Nonlinearities to be used in the decoding networks
        self.range_nonlin = range_nonlin  # Range when clipping the nonlinearity
        self.latent_size_d = latent_size_d  # Size of the deterministic latent vectors
        self.latent_size_z = latent_size_z  # Size of the stochastic latent vectors
        self.num_hidden_mlp = num_hidden_mlp  # Number of neurons in each layer of the neural networks
        self.num_layers_mlp = num_layers_mlp  # Number of layers of the neural networks
        self.latent_size_a = latent_size_a  # Size of the deterministic latent vectors

        self.smoothing = smoothing
        self.use_mu_residual = use_mu_residual

        self.sequence_length = sequence_length  # how many steps to unroll
        self.output_dim = output_dim

        self.p_d_drop = p_d_drop  # Dropout probability fot the deterministic states in the non-recurrent connections
        self.p_z_drop = p_z_drop  # Dropout probability fot the stochastic states in the non-recurrent connections
        self.p_emb_u_drop = p_emb_u_drop  # Dropout probability for the embeddings of u
        self.p_emb_x_drop = p_emb_x_drop  # Dropout probability for the embeddings of x

        #  TRAINING SETTINGS
        self.optimizer = optimizer  # Optimizer to be used
        self.batch_size = batch_size  # batch size
        self.log10_lr = log10_lr  # learning rate
        self.momentum = momentum  # for nesterov_momentum

        self.decay_type = decay_type  # decay type
        self.decay = decay  # decay factor
        self.scale_decay = scale_decay  # scale for the decay factor, only used with PowerDecaySchedule
        self.no_decay_epochs = no_decay_epochs  # run this many epochs before first decay

        self.init_mlp = init_mlp  # Weight initializer (uniform or normal)
        self.init_rnn = init_rnn  # Weight initializer for rnn (uniform or orthogonal)
        self.init_range = init_range  # Parameters are initialized as Unif(-init_range,init_range)

        self.tempKL_type = tempKL_type  # schedule for the KL term
        self.tempKL_start = tempKL_start  # starting point for the temperature (set to 1 no to use temperature)
        self.tempKL_epochs = tempKL_epochs  # Number  of epochs to arrive to temperature 1
        self.tempKL_decay = tempKL_decay  # decay factor for the chosen schedule

        self.max_grad_norm = max_grad_norm  # scale steps if norm is above this value
        self.clip_gradients = clip_gradients  # Gradient clipping
        self.max_num_epochs = max_num_epochs  # Number of epochs to run
        self.batch_size_test = batch_size_test  # batch size for validation/test set
        self.eval_epoch = eval_epoch  # epochs between evaluation of test performance
        self.unroll_scan = unroll_scan
        self.tolerance_softmax = tolerance_softmax  # numerial stability for softmax
        self.cons = cons  # numerial stability for lognormal, kl
        self.random_seed = random_seed  # random seed for numpy


if __name__ == '__main__':
    lower_bound_test = run()
