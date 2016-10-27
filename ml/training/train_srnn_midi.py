from base import Train
import numpy as np
import time
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import theano


class TrainSRNN_midi(Train):
    """
    The :class:'TrainSRNN_midi' class represents a training loop for SRNNs
    """

    def __init__(self):
        super(TrainSRNN_midi, self).__init__()

        # Lists to store the ELBO results of all the epochs
        self.lower_bound_train_all = []
        self.lower_bound_train_all_std = []
        self.lower_bound_valid_all = []
        self.lower_bound_test_all = []

        # Monitor norm of the updates
        self.mean_norm_all = []
        self.std_norm_all = []

    def train_model(self, data, model, f_train, f_valid, decay_learning_rate, temperature_KL, \
                    settings, pickle_path=None, load_pickled_file=None, plot_path=None, ylim=None):

        print "Training model..."

        # Load the pickled weights of the model if needed
        if not load_pickled_file is None:
            with open(load_pickled_file, 'rb') as f:
                print "Loading pickled model from %s" % load_pickled_file
                model.load_model(f)

        # Helper function for testing
        def test_epoch(dataset):

            rep_test_elbo = []
            rep_test_elbo_seq = []

            for jj in range(1):

                if dataset == 'valid':
                    u, x, mask = data.get_validdata()
                    n_batches = data.n_valid // settings.batch_size
                elif dataset == 'test':
                    u, x, mask = data.get_testdata()
                    n_batches = data.n_test // settings.batch_size
                else:
                    raise ValueError()

                # Reset to 0 the initial hidden states
                model.reset_state(settings, settings.batch_size)

                l_elbo = []
                for i in range(n_batches):
                    u_batch = u[i * settings.batch_size:(i + 1) * settings.batch_size]
                    x_batch = x[i * settings.batch_size:(i + 1) * settings.batch_size]
                    mask_batch = mask[i * settings.batch_size:(i + 1) * settings.batch_size]
                    elbo = f_valid(u_batch, x_batch, mask_batch)

                    l_elbo.append(elbo)

                # l_elbo has size batch_size, and it is summed over the sequence_length (not the mean)
                elbo = np.hstack(l_elbo).flatten()
                assert len(elbo) == u.shape[0]

                # get non padded samples
                n_not_padded = np.sum(mask[:, 0])
                assert n_not_padded <= u.shape[0]
                elbo = elbo[:n_not_padded]  # Remove samples taht are all padded

                test_elbo = np.mean(elbo)
                test_elbo_seq = test_elbo * settings.sequence_length

                rep_test_elbo.append(test_elbo)
                rep_test_elbo_seq.append(test_elbo_seq)

            return np.mean(rep_test_elbo), np.mean(rep_test_elbo_seq)

        # Training loop
        # Compute number of batches
        n_batches_train = data.n_train // settings.batch_size

        range_epochs = range(self.epoch, settings.max_num_epochs)  # to restart traning from a pickled version
        for self.epoch in range_epochs:
            batch_time = time.time()

            # Reset to 0 the initial hidden states
            model.reset_state(settings, settings.batch_size)

            # Initialize lists for training stats
            lower_bound_batch, norm_batch = [], []

            # Training. Loop through all the batches
            for i in range(n_batches_train):
                temp_KL = np.asarray(temperature_KL.get_decay(self.epoch * n_batches_train + i),
                                     dtype=theano.config.floatX)
                x_batch, y_batch, mask_batch = data.get_train_batch()
                lower_bound, norm = f_train(x_batch, y_batch, mask_batch, temp_KL)

                # The KL bound and its separate terms
                lower_bound_batch.append(lower_bound)

                # Norm of the updates
                norm_batch.append(norm)

            time_epoch = time.time() - batch_time

            self.lower_bound_train_all.append(np.mean(lower_bound_batch))
            self.lower_bound_train_all_std.append(np.std(lower_bound_batch))

            self.mean_norm_all.append(np.mean(norm_batch))
            self.std_norm_all.append(np.std(norm_batch))
            self.time_epoch_all.append(time_epoch)

            if not np.isfinite(self.lower_bound_train_all[-1]):
                print "NaN or inf in training lower bound! Stopping job.."
                return np.nan

            # Decay of the learning rate
            model.set_learning_rate(decay_learning_rate.get_decay(self.epoch))

            if self.epoch % settings.eval_epoch == 0:
                self.epochs_eval.append(self.epoch)

                # Evaluate test performance
                time_test = time.time()

                lower_bound_valid, elbo_seq_valid = test_epoch('valid')
                lower_bound_test, elbo_seq_test = test_epoch('test')

                # Below this point a lot of bookkeeping and plotting
                self.lower_bound_valid_all.append(lower_bound_valid)
                self.lower_bound_test_all.append(lower_bound_test)

                # Print info
                self.print_training_info()
                print "Time test:    %s" % (time.time() - time_test)
                print "Leraning rate: %s" % model.get_learning_rate()
                print "Temperature KL: %s" % (temp_KL)
                print("")

                # Plots of errors/norm vs epochs
                if not plot_path is None:
                    self.plot_results(plot_path, ylim)

                # Pickle
                if not pickle_path is None:
                    with open(pickle_path, 'wb') as f:
                        model.dump_model(f)
                        settings.dump_settings(f)
                        self.epoch = self.epoch + 1  # If we continue training from this pickled file we want to restart
                        # from the next epoch (see definition of range_epochs)
                        self.dump_train(f)

        return self.lower_bound_valid_all[-1]

    def print_training_info(self):
        print "Epoch      : %s" % self.epoch
        print "ELBO train: %s   (std: %s)" % (self.lower_bound_train_all[-1], self.lower_bound_train_all_std[-1])
        print "ELBO valid: %s " % self.lower_bound_valid_all[-1]
        print "ELBO test: %s " % self.lower_bound_test_all[-1]
        print "Mean norm:   %s   (std: %s)" % (self.mean_norm_all[-1], self.std_norm_all[-1])
        print "Time train (s):    %s" % self.time_epoch_all[-1]

    def plot_results(self, plot_path=None, ylim=None):

        # Plotting parameters
        label_size = 18
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size
        plot_params = dict()
        plot_params['ms'] = 10
        plot_params['linewidth'] = 3

        # Plot training bound on the perplexity
        f = plt.figure(figsize=[12, 12])
        plt.errorbar(self.epochs_eval, [self.lower_bound_train_all[i] for i in self.epochs_eval],
                     [self.lower_bound_train_all_std[i] for i in self.epochs_eval], marker='d', color='b',
                     label='Train', **plot_params)
        plt.plot(self.epochs_eval, self.lower_bound_valid_all, "-rh", label="Valid", **plot_params)
        plt.plot(self.epochs_eval, self.lower_bound_test_all, "-k^", label="Test", **plot_params)
        plt.xlabel('Epochs', fontsize=20)
        plt.grid('on')
        plt.title('ELBO', fontsize=24, y=1.01)
        plt.legend(loc="lower right", handlelength=3, fontsize=20)
        if ylim is not None:
            plt.ylim(ylim)
        if plot_path is not None:
            plt.savefig(plot_path + "_epochs.png", format='png', bbox_inches='tight', dpi=200)
        plt.close(f)

        # Plot norm of the updates
        f = plt.figure(figsize=[12, 12])
        plt.errorbar(self.epochs_eval, [self.mean_norm_all[i] for i in self.epochs_eval],
                     [self.std_norm_all[i] for i in self.epochs_eval], marker='d', color='m', **plot_params)
        plt.grid('on')
        plt.title('Norm of the updates', fontsize=24, y=1.01)
        if plot_path is not None:
            plt.savefig(plot_path + "_norm.png", format='png', bbox_inches='tight', dpi=200)
        plt.close(f)
