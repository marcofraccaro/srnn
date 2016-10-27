import theano
import theano.tensor as T
import lasagne
import parmesan
from base import Model
import numpy as np
from ml.lasagne_extensions.stochastic_recurrent_layer import StochsticRecurrentLayer
from parmesan.layers import ListIndexLayer
import math


def kl_normal2_normal2(mean1, log_var1, mean2, log_var2):
    return 0.5 * log_var2 - 0.5 * log_var1 + (T.exp(log_var1) + (mean1 - mean2) ** 2) / (2 * T.exp(log_var2)) - 0.5


c = - 0.5 * math.log(2 * math.pi)


def log_normal2(x, mean, log_var):
    return c - log_var / 2 - (x - mean) ** 2 / (2 * T.exp(log_var))


class SRNN_timit(Model):
    """
    The :class:'SRNN_timit' class represents the implementation of Stochastic RNN
    """

    def __init__(self, settings):

        # Call constructor of base model
        super(SRNN_timit, self).__init__()

        # Define initializers for the parameters
        if settings.init_rnn == 'uniform':
            init_rnn = lasagne.init.Uniform(range=settings.init_range)
        elif settings.init_rnn == 'orthogonal':
            init_rnn = lasagne.init.Orthogonal()
        else:
            raise ValueError('Invalid initializer \'' + settings.init_rnn + '\'')

        if settings.init_mlp == 'uniform':
            init_mlp = lasagne.init.GlorotUniform()
        elif settings.init_mlp == 'normal':
            init_mlp = lasagne.init.GlorotNormal()
        else:
            raise ValueError('Invalid initializer \'' + settings.init_mlp + '\'')

        # For stability
        init_last_layer_mlp = lasagne.init.Uniform(range=settings.init_range)

        def dense_layer(l, num_units, nonlinearity, name, W=init_mlp, b=lasagne.init.Constant(0.)):
            l = lasagne.layers.DenseLayer(l, num_units=num_units, name=name + "-dense", W=W, b=b,
                                          nonlinearity=nonlinearity)
            return l

        # Define MLP to be used in the encoding and decoding networks
        def mlp(input_layer, num_units, nonlinearity, name, num_mlp_layers=1, W=init_mlp, b=lasagne.init.Constant(0.)):
            output_layer = input_layer
            for i in range(num_mlp_layers):
                output_layer = dense_layer(output_layer, num_units=num_units, name=name + '_' + str(i + 1),
                                           nonlinearity=nonlinearity, W=W, b=b)
            return output_layer

        # Define nonlinearities for encoder and decoder
        def clipped_very_leaky_rectify(x):
            return T.clip(theano.tensor.nnet.relu(x, 1. / 3), -settings.range_nonlin, settings.range_nonlin)

        def get_nonlinearity(nonlin):
            if nonlin == 'rectify':
                return lasagne.nonlinearities.rectify
            elif nonlin == 'very_leaky_rectify':
                return lasagne.nonlinearities.very_leaky_rectify
            elif nonlin == 'tanh':
                return lasagne.nonlinearities.tanh
            elif nonlin == 'clipped_very_leaky_rectify':
                return clipped_very_leaky_rectify
            else:
                raise ValueError('Invalid non-linearity \'' + nonlin + '\'')

        nonlin_encoder = get_nonlinearity(settings.nonlinearity_encoder)
        nonlin_decoder = get_nonlinearity(settings.nonlinearity_decoder)

        ## INPUTS
        self.u_sym = T.tensor3()
        self.u_sym.tag.test_value = np.random.randn(settings.batch_size, settings.sequence_length,
                                                    settings.output_dim).astype('float32')

        self.x_sym = T.tensor3()
        self.x_sym.tag.test_value = np.random.randn(settings.batch_size, settings.sequence_length,
                                                    settings.output_dim).astype('float32')

        # To handle sequences of different lengths
        self.sym_mask = T.matrix()
        self.sym_mask.tag.test_value = np.ones((settings.batch_size, settings.sequence_length)).astype('float32')

        # Input layer for the inputs of the GRU network
        self.input_layer_u = lasagne.layers.InputLayer((settings.batch_size, None, settings.output_dim),
                                                       self.u_sym, name="input_layer_u")

        self.input_layer_mask = lasagne.layers.InputLayer((settings.batch_size, None), self.sym_mask,
                                                          name="input_layer_mask")

        input_layer_u_flat = lasagne.layers.ReshapeLayer(self.input_layer_u, (-1, settings.output_dim),
                                                         name="input_layer_u_flat")

        u_dense1_flat = lasagne.layers.DenseLayer(input_layer_u_flat, num_units=settings.num_hidden_mlp,
                                                  nonlinearity=nonlin_decoder, name="u_dense1_flat")
        u_dense2_flat = lasagne.layers.DenseLayer(u_dense1_flat, num_units=settings.num_hidden_mlp,
                                                  nonlinearity=nonlin_decoder, name="u_dense2_flat")

        u_emb_dropout_flat = lasagne.layers.DropoutLayer(u_dense2_flat, p=settings.p_emb_u_drop,
                                                         name="u_mlp_dropout_flat")

        u_emb_dropout = lasagne.layers.ReshapeLayer(u_emb_dropout_flat, (settings.batch_size, -1,
                                                                         settings.num_hidden_mlp))

        ## MODEL SETUP
        # We first initialize the shared variables for the initial deterministic hidden stated (initialized to 0).  Due
        # to the way we have divided the data in batches we can use the last hidden state of the current batch to
        # initialize the hidden state of the following batch.
        self.d_init_sh = theano.shared(np.zeros((settings.batch_size, settings.latent_size_d),
                                                dtype=theano.config.floatX))

        self.input_layer_d_tm1 = lasagne.layers.InputLayer((None, settings.latent_size_d), name="input_layer_d_tm1")
        # We first compute the output from the RNN (deterministic hidden state)
        # First GRU layer
        # Inputs: a (batch_size x sequence_length x hidden_layer_dim) matrix coming from the dropout layer
        #         and a (batch_size x hidden_layer_dim) initial hidden state
        # Output: a (batch_size x sequence_length x latent_size_d) matrix
        self.d_layer = lasagne.layers.GRULayer(u_emb_dropout,
                                               num_units=settings.latent_size_d,
                                               resetgate=lasagne.layers.Gate(W_in=init_rnn, W_hid=init_rnn,
                                                                             W_cell=None),
                                               updategate=lasagne.layers.Gate(W_in=init_rnn, W_hid=init_rnn,
                                                                              W_cell=None),
                                               hidden_update=lasagne.layers.Gate(W_in=init_rnn, W_hid=init_rnn,
                                                                                 W_cell=None,
                                                                                 nonlinearity=lasagne.nonlinearities.tanh),
                                               learn_init=False,
                                               hid_init=self.input_layer_d_tm1,
                                               mask_input=self.input_layer_mask,
                                               unroll_scan=settings.unroll_scan,
                                               name="d_layer")

        # We add dropout to all non-recurrent connections
        self.d_dropout_layer = lasagne.layers.DropoutLayer(self.d_layer, p=settings.p_d_drop, name="d_dropout_layer")

        # Define x inputs to the encoder
        self.input_layer_x = lasagne.layers.InputLayer((settings.batch_size, None, settings.output_dim),
                                                       self.x_sym, name="input_layer_x")

        input_layer_x_flat = lasagne.layers.ReshapeLayer(self.input_layer_x, (-1, settings.output_dim),
                                                         name="input_layer_x_flat")

        # I share the parameters (also nonlinearities) with the mlp after u, these are like feature extractors
        x_dense1_flat = lasagne.layers.DenseLayer(input_layer_x_flat, num_units=settings.num_hidden_mlp,
                                                  nonlinearity=nonlin_decoder, name="x_dense1_flat",
                                                  W=u_dense1_flat.W, b=u_dense1_flat.b)
        x_dense2_flat = lasagne.layers.DenseLayer(x_dense1_flat, num_units=settings.num_hidden_mlp,
                                                  nonlinearity=nonlin_decoder, name="x_dense2_flat",
                                                  W=u_dense2_flat.W, b=u_dense2_flat.b)

        x_emb_dropout_flat = lasagne.layers.DropoutLayer(x_dense2_flat, p=settings.p_emb_x_drop,
                                                         name="x_mlp_dropout_flat")

        x_emb_dropout = lasagne.layers.ReshapeLayer(x_emb_dropout_flat, (settings.batch_size, -1,
                                                                         settings.num_hidden_mlp),
                                                    name='x_emb_dropout')

        input_a_layer = lasagne.layers.ConcatLayer([self.d_dropout_layer, x_emb_dropout], axis=2,
                                                   name="input_a_layer")

        if settings.smoothing:
            print "Doing smoothing"
            # The hidden state is intialized with zeros
            a_layer = lasagne.layers.GRULayer(input_a_layer,
                                              num_units=settings.latent_size_a,
                                              resetgate=lasagne.layers.Gate(W_in=init_rnn, W_hid=init_rnn,
                                                                            W_cell=None),
                                              updategate=lasagne.layers.Gate(W_in=init_rnn, W_hid=init_rnn,
                                                                             W_cell=None),
                                              hidden_update=lasagne.layers.Gate(W_in=init_rnn, W_hid=init_rnn,
                                                                                W_cell=None,
                                                                                nonlinearity=lasagne.nonlinearities.tanh),
                                              learn_init=False,
                                              backwards=True,
                                              unroll_scan=settings.unroll_scan,
                                              mask_input=self.input_layer_mask,
                                              name="a_layer")

        else:  # We only do filtering
            print "Doing filtering"
            input_a_layer_flat = lasagne.layers.ReshapeLayer(input_a_layer, (-1, [2]),
                                                             name="input_a_layer_flat")
            a_layer_flat = mlp(input_a_layer_flat, settings.latent_size_a, nonlin_encoder, "a_layer_flat",
                               num_mlp_layers=settings.num_layers_mlp)

            a_layer = lasagne.layers.ReshapeLayer(a_layer_flat, (settings.batch_size, -1, settings.latent_size_a))

        # Define shared variables for quantities to be updated across batches (truncated BPTT)
        self.z_init_sh = theano.shared(np.zeros((settings.batch_size, settings.latent_size_z),
                                                dtype=theano.config.floatX))

        self.input_layer_z_tm1 = lasagne.layers.InputLayer((None, settings.latent_size_z),
                                                           name="input_layer_z_tm1")

        self.mean_prior_init_sh = theano.shared(np.zeros((settings.batch_size, settings.latent_size_z),
                                                         dtype=theano.config.floatX))

        self.input_layer_mean_prior_tm1 = lasagne.layers.InputLayer((None, settings.latent_size_z),
                                                                    name="input_layer_mean_prior_tm1")

        self.log_var_prior_init_sh = theano.shared(np.zeros((settings.batch_size, settings.latent_size_z),
                                                            dtype=theano.config.floatX))

        self.input_layer_log_var_prior_tm1 = lasagne.layers.InputLayer((None, settings.latent_size_z),
                                                                       name="input_layer_log_var_prior_tm1")

        # Define MLPs to be used in StochsticRecurrentLayer
        mlp_prior_input_dim = settings.latent_size_d + settings.latent_size_z

        input_prior_mlp = lasagne.layers.InputLayer((None, mlp_prior_input_dim))
        mean_prior_dense1 = lasagne.layers.DenseLayer(input_prior_mlp, num_units=settings.num_hidden_mlp,
                                                      nonlinearity=nonlin_decoder, name="mean_prior_dense1")
        mean_prior_dense2 = lasagne.layers.DenseLayer(mean_prior_dense1, num_units=settings.latent_size_z,
                                                      W=init_last_layer_mlp,
                                                      nonlinearity=None, name="mean_prior_dense2")
        log_var_prior_dense1 = lasagne.layers.DenseLayer(input_prior_mlp, num_units=settings.num_hidden_mlp,
                                                         nonlinearity=nonlin_decoder, name="log_var_prior_dense1")
        log_var_prior_dense2 = lasagne.layers.DenseLayer(log_var_prior_dense1, num_units=settings.latent_size_z,
                                                         W=init_last_layer_mlp,
                                                         nonlinearity=None, name="log_var_prior_dense2")

        mlp_q_input_dim = settings.latent_size_a + settings.latent_size_z  # [input_q_n, z_previous]

        input_q_mlp = lasagne.layers.InputLayer((None, mlp_q_input_dim))
        mean_q_dense1 = lasagne.layers.DenseLayer(input_q_mlp, num_units=settings.num_hidden_mlp,
                                                  nonlinearity=nonlin_encoder, name="mean_q_dense1")
        mean_q_dense2 = lasagne.layers.DenseLayer(mean_q_dense1, num_units=settings.latent_size_z,
                                                  W=init_last_layer_mlp,
                                                  nonlinearity=None, name="mean_q_dense2")
        log_var_q_dense1 = lasagne.layers.DenseLayer(input_q_mlp, num_units=settings.num_hidden_mlp,
                                                     nonlinearity=nonlin_encoder, name="log_var_q_dense1")
        log_var_q_dense2 = lasagne.layers.DenseLayer(log_var_q_dense1, num_units=settings.latent_size_z,
                                                     W=init_last_layer_mlp,
                                                     nonlinearity=None, name="log_var_q_dense2")

        if settings.cons == 0:
            cons = 0
        elif settings.cons < 0:
            cons = 10 ** (settings.cons)
        else:
            raise ValueError()

        stochastic_recurrent_layer = StochsticRecurrentLayer(input_p=self.d_dropout_layer,
                                                             input_q=a_layer,
                                                             mu_p_mlp=mean_prior_dense2,
                                                             logvar_p_mlp=log_var_prior_dense2,
                                                             q_mu_mlp=mean_q_dense2,
                                                             q_logvar_mlp=log_var_q_dense2,
                                                             num_units=settings.latent_size_z,
                                                             unroll_scan=settings.unroll_scan,
                                                             use_mu_residual_q=settings.use_mu_residual,
                                                             z_init=self.input_layer_z_tm1,
                                                             mu_p_init=self.input_layer_mean_prior_tm1,
                                                             mask_input=self.input_layer_mask,
                                                             cons=cons,
                                                             name='stochastic_recurrent_layer')

        # ListIndexLayer is needed after a Layer that returns multiple outputs
        self.z_layer = ListIndexLayer(stochastic_recurrent_layer, index=0, name='z_layer')
        self.mean_prior_layer = ListIndexLayer(stochastic_recurrent_layer, index=1, name='mean_prior_layer')
        self.log_var_prior_layer = ListIndexLayer(stochastic_recurrent_layer, index=2, name='log_var_prior_layer')
        self.mean_q_layer = ListIndexLayer(stochastic_recurrent_layer, index=3, name='mean_q_layer')
        self.log_var_q_layer = ListIndexLayer(stochastic_recurrent_layer, index=4, name='log_var_q_layer')

        # Finish the generative model
        self.z_dropout_layer = lasagne.layers.DropoutLayer(self.z_layer, p=settings.p_z_drop,
                                                           name="z_dropout_layer")

        # The softmax mlp needs 2d tensors, hence we reshape here, add the mlp and reshape again
        d_layer_reshaped = lasagne.layers.ReshapeLayer(self.d_dropout_layer, (-1, settings.latent_size_d),
                                                       name="d_layer_reshaped")

        z_layer_reshaped = lasagne.layers.ReshapeLayer(self.z_dropout_layer, (-1, settings.latent_size_z),
                                                       name="z_layer_reshaped")

        input_generative_mlp = lasagne.layers.ConcatLayer([d_layer_reshaped, z_layer_reshaped], axis=1,
                                                          name="input_generative_mlp")
        generative_mlp = mlp(input_generative_mlp, settings.num_hidden_mlp, nonlin_decoder, "generative_mlp",
                             num_mlp_layers=settings.num_layers_mlp)

        # Compute the softmax output and reshape
        mean_gauss_output_reshaped = lasagne.layers.DenseLayer(generative_mlp, num_units=settings.output_dim,
                                                               nonlinearity=lasagne.nonlinearities.identity,
                                                               W=init_last_layer_mlp,
                                                               name="mean_gauss_output_reshaped")

        log_var_output_reshaped = lasagne.layers.DenseLayer(generative_mlp, num_units=settings.output_dim,
                                                            nonlinearity=lasagne.nonlinearities.identity,
                                                            W=init_last_layer_mlp,
                                                            name="log_var_output_reshaped")

        self.mean_gauss_output_layer = lasagne.layers.ReshapeLayer(mean_gauss_output_reshaped,
                                                                   (settings.batch_size, -1, settings.output_dim),
                                                                   name="mean_gauss_output_layer")
        self.log_var_gauss_output_layer = lasagne.layers.ReshapeLayer(log_var_output_reshaped,
                                                                      (settings.batch_size, -1, settings.output_dim),
                                                                      name="log_var_output_layer")

        # List of all layers that we need to pass for pickle/model_info (see base model)
        self.output_layer = [self.z_layer, self.mean_prior_layer, self.log_var_prior_layer,
                             self.mean_gauss_output_layer, self.log_var_gauss_output_layer]
        # Get a list of all parameters in the network.
        self.model_params = lasagne.layers.get_all_params(self.output_layer)
        # Get a list of all trainable parameters in the network.
        self.model_params_trainable = lasagne.layers.get_all_params(self.output_layer, trainable=True)

    def initialize_computation_graph(self, data, settings):
        """
        Compile training/evaluation functions
        """

        ###############################################################################################################
        # Define training function
        ###############################################################################################################
        d, z, mean_q, log_var_q, mean_prior, log_var_prior, mean_gauss_output, log_var_gauss_output = \
            lasagne.layers.get_output([self.d_layer, self.z_layer, self.mean_q_layer, self.log_var_q_layer,
                                       self.mean_prior_layer, self.log_var_prior_layer,
                                       self.mean_gauss_output_layer, self.log_var_gauss_output_layer],
                                      inputs={self.input_layer_u: self.u_sym,
                                              self.input_layer_x: self.x_sym,
                                              self.input_layer_d_tm1: self.d_init_sh,
                                              self.input_layer_z_tm1: self.z_init_sh,
                                              self.input_layer_mean_prior_tm1: self.mean_prior_init_sh,
                                              self.input_layer_log_var_prior_tm1: self.log_var_prior_init_sh,
                                              self.input_layer_mask: self.sym_mask},
                                      deterministic=False)

        temperature_KL_sym = T.scalar('temperature_KL')
        temperature_KL_sym.tag.test_value = 1.0

        # Compute the lower bound to the average frame (the vector of size output_dim) log-likelihood, meaning that we
        # take a mean over both batch_size and sequence_length
        def elbo_h_gaussian_x_gaussian(d, z, mean_q, log_var_q, mean_prior, log_var_prior, mean_gauss_output,
                                       log_var_gauss_output, x, mask, settings, temperature_KL=1.0,
                                       test=False):

            if not test:
                mask_sum = T.sum(mask, axis=1)
                # mean_gauss_output has size (batch_size, seq_length, output_dim)
                # We some over output_dim and we take the mean over batch_size and sequence_length
                log_p_x_given_h_tot = log_normal2(x, mean=mean_gauss_output, log_var=log_var_gauss_output,
                                                  ) * mask.dimshuffle(0, 1, 'x')
                log_p_x_given_h_tot = log_p_x_given_h_tot.sum(axis=(1, 2)) / mask_sum
                log_p_x_given_h_tot = log_p_x_given_h_tot.mean()

                kl_divergence = kl_normal2_normal2(mean_q, log_var_q, mean_prior, log_var_prior)

                # kl_divergence has size (batch_size, sequence_length, output_dim)
                kl_divergence_tmp = kl_divergence * mask.dimshuffle(0, 1, 'x')
                kl_divergence_tmp = kl_divergence_tmp.sum(axis=(1, 2)) / mask_sum
                kl_divergence_tot = T.mean(kl_divergence_tmp)

                lower_bound = log_p_x_given_h_tot - temperature_KL * kl_divergence_tot

                return lower_bound

            else:  # For test we do not mean over the batches and divide by mask_sum

                # mean_gauss_output has size (batch_size, seq_length, output_dim)
                # We some over output_dim and we take the mean over batch_size and sequence_length
                log_p_x_given_h_tot = log_normal2(x, mean=mean_gauss_output, log_var=log_var_gauss_output,
                                                  ) * mask.dimshuffle(0, 1, 'x')
                log_p_x_given_h_tot = log_p_x_given_h_tot.sum(axis=(1, 2))

                kl_divergence = kl_normal2_normal2(mean_q, log_var_q, mean_prior, log_var_prior)

                # kl_divergence has size (batch_size, sequence_length, output_dim)

                kl_divergence_seq = T.reshape(kl_divergence, (settings.batch_size, -1, settings.latent_size_z))
                kl_divergence_seq = kl_divergence_seq * mask.dimshuffle(0, 1, 'x')
                kl_divergence_tot = kl_divergence_seq.sum(axis=(1, 2))

                lower_bound = log_p_x_given_h_tot - temperature_KL * kl_divergence_tot

                return lower_bound

        lower_bound_train = elbo_h_gaussian_x_gaussian(d, z, mean_q, log_var_q, mean_prior, log_var_prior,
                                                       mean_gauss_output, log_var_gauss_output,
                                                       self.x_sym, self.sym_mask, settings,
                                                       temperature_KL=temperature_KL_sym)

        # Calculate symbolic gradients w.r.t lower bound. Note the minus as we want to do a minimization problem
        all_grads = T.grad(-lower_bound_train, self.model_params_trainable)
        all_grads, norm = lasagne.updates.total_norm_constraint(all_grads, settings.max_grad_norm, return_norm=True)
        # Clip the gradients
        all_grads = [T.clip(g, -settings.clip_gradients, settings.clip_gradients) for g in all_grads]

        # Use shared variable for learning rate. Allows us to change the learning rate during training.
        self.learning_rate_init = lasagne.utils.floatX(10 ** settings.log10_lr)
        self.learning_rate_sh = theano.shared(self.learning_rate_init)

        # Gradient updates
        if settings.optimizer.lower() == 'adam':
            updates = lasagne.updates.adam(all_grads, self.model_params_trainable, beta1=0.9, beta2=0.999,
                                           epsilon=1e-4, learning_rate=self.learning_rate_sh)
        elif settings.optimizer.lower() == 'rmsprop':
            updates = lasagne.updates.rmsprop(all_grads, self.model_params_trainable,
                                              learning_rate=self.learning_rate_sh)
        elif settings.optimizer.lower() == 'nesterov_momentum':
            updates = lasagne.updates.nesterov_momentum(all_grads, self.model_params_trainable,
                                                        learning_rate=self.learning_rate_sh, momentum=settings.momentum)
        elif settings.optimizer.lower() == 'sgd':
            updates = lasagne.updates.sgd(all_grads, self.model_params_trainable, learning_rate=self.learning_rate_sh)
        else:
            raise ValueError('Unknown optimizer in settings.optimizer')

        # We add two updates to update the shared variables for the initial hidden state of the next batch to be
        # the last hidden state of the current batch
        updates[self.d_init_sh] = d[:, -1, :]
        updates[self.z_init_sh] = z[:, -1, :]
        updates[self.mean_prior_init_sh] = mean_prior[:, -1, :]
        updates[self.log_var_prior_init_sh] = log_var_prior[:, -1, :]

        # Compile training function
        print("compiling f_train...")
        f_train = theano.function(inputs=[self.u_sym, self.x_sym, self.sym_mask, temperature_KL_sym],
                                  outputs=[lower_bound_train, norm],
                                  updates=updates)

        ###############################################################################################################
        # Define test function
        ###############################################################################################################
        d_eval, z_eval, mean_q_eval, log_var_q_eval, mean_prior_eval, log_var_prior_eval, mean_gauss_output_eval, \
        log_var_gauss_output_eval = \
            lasagne.layers.get_output([self.d_layer, self.z_layer, self.mean_q_layer, self.log_var_q_layer,
                                       self.mean_prior_layer, self.log_var_prior_layer,
                                       self.mean_gauss_output_layer, self.log_var_gauss_output_layer],
                                      inputs={self.input_layer_u: self.u_sym,
                                              self.input_layer_x: self.x_sym,
                                              self.input_layer_d_tm1: self.d_init_sh,
                                              self.input_layer_z_tm1: self.z_init_sh,
                                              self.input_layer_mean_prior_tm1: self.mean_prior_init_sh,
                                              self.input_layer_log_var_prior_tm1: self.log_var_prior_init_sh,
                                              self.input_layer_mask: self.sym_mask},
                                      deterministic=True)

        lower_bound_eval = elbo_h_gaussian_x_gaussian(d_eval, z_eval,
                                                      mean_q_eval, log_var_q_eval, mean_prior_eval,
                                                      log_var_prior_eval, mean_gauss_output_eval,
                                                      log_var_gauss_output_eval, self.x_sym,
                                                      self.sym_mask,
                                                      settings,
                                                      test=True)

        # Compile the function to compute the cost on a batch of the validation set and of the test set
        print("compiling f_valid...")
        f_valid = theano.function(inputs=[self.u_sym, self.x_sym, self.sym_mask],
                                  outputs=[lower_bound_eval])

        return f_train, f_valid

    def reset_state(self, settings, n_data_points):
        """
        Resets the hidden states to their default values.
        """
        self.d_init_sh.set_value(
            np.zeros((n_data_points, settings.latent_size_d), dtype=theano.config.floatX))
        self.z_init_sh.set_value(
            np.zeros((n_data_points, settings.latent_size_z), dtype=theano.config.floatX))
        self.mean_prior_init_sh.set_value(
            np.zeros((n_data_points, settings.latent_size_z), dtype=theano.config.floatX))
        self.log_var_prior_init_sh.set_value(
            np.zeros((n_data_points, settings.latent_size_z), dtype=theano.config.floatX))
