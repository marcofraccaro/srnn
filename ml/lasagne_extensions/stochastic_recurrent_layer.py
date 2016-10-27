import theano
import theano.tensor as T
from lasagne.utils import unroll_scan
from lasagne.layers import MergeLayer, helper, get_output
from lasagne.random import get_rng
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class StochsticRecurrentLayer(MergeLayer):
    def __init__(self, input_p, input_q, num_units,
                 mu_p_mlp,
                 logvar_p_mlp,
                 q_mu_mlp,
                 q_logvar_mlp,
                 z_init,
                 mu_p_init,
                 use_mu_residual_q=True,
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 cons = 0.0,
                 use_lik_q = 0,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have three
        # inputs - the layer input, the mask and the initial hidden state.  We
        # will just provide the layer input as incomings, unless a mask input
        # or initial hidden state was provided.
        incomings = [input_p, input_q,
                     z_init,
                     mu_p_init
                     ]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1

        # Initialize parent layer
        super(StochsticRecurrentLayer, self).__init__(incomings, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.learn_init = learn_init
        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final
        self.logvar_p_mlp = logvar_p_mlp
        self.q_mu_mlp = q_mu_mlp
        self.q_logvar_mlp = q_logvar_mlp
        self.mu_p_mlp = mu_p_mlp
        self.use_mu_residual_q = use_mu_residual_q
        self.cons = cons
        self.use_lik_q = use_lik_q

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")


    def get_params(self, **tags):
        # Get all parameters from this layer, the master layer
        params = super(StochsticRecurrentLayer, self).get_params(**tags)
        if self.logvar_p_mlp is not None:
            params += helper.get_all_params(self.logvar_p_mlp, **tags)
            params += helper.get_all_params(self.q_mu_mlp, **tags)
            params += helper.get_all_params(self.q_logvar_mlp, **tags)
            params += helper.get_all_params(self.mu_p_mlp, **tags)
        return params

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable

        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input_p = inputs[0]
        input_q = inputs[1]
        z_init = inputs[2]
        mu_p_init = inputs[3]

        # Retrieve the mask when it is supplied
        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input_p = input_p.dimshuffle(1, 0, 2)
        input_q = input_q.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input_p.shape

        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        def log_sum_exp(a, b):
            return T.log(T.exp(a) + T.exp(b))

        def step(noise_n, input_p_n, input_q_n,
                 z_previous,
                 mu_p_previous, logvar_p_previous,
                 mu_q_previous, logvar_q_previous, *args):

            input_p = T.concatenate([input_p_n, z_previous], axis=1)
            mu_p = get_output(self.mu_p_mlp, input_p)

            logvar_p = get_output(self.logvar_p_mlp, input_p)
            logvar_p = T.log(T.exp(logvar_p) + self.cons)

            q_input_n = T.concatenate([input_q_n, z_previous], axis=1)

            mu_q = get_output(self.q_mu_mlp, q_input_n)
            if self.use_mu_residual_q:
                print "Using residuals for mean_q"
                mu_q += mu_p

            logvar_q = get_output(self.q_logvar_mlp, q_input_n)

            # Numerical stability
            logvar_q = T.log(T.exp(logvar_q) + self.cons)

            z_n = mu_q + T.exp(0.5*logvar_q) * noise_n

            return z_n, mu_p, logvar_p, mu_q, logvar_q


        def step_masked(noise_n, input_p_n, input_q_n, mask_n,
                 z_previous,
                 mu_p_previous, logvar_p_previous,
                 mu_q_previous, logvar_q_previous, *args):

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.

            z_n, mu_p, logvar_p, mu_q, logvar_q = step(
                noise_n, input_p_n, input_q_n,
                z_previous, mu_p_previous, logvar_p_previous,
                mu_q_previous, logvar_q_previous, *args)

            z_n = T.switch(mask_n, z_n, z_previous)
            mu_p = T.switch(mask_n, mu_p, mu_p_previous)
            logvar_p = T.switch(mask_n, logvar_p, logvar_p_previous)
            mu_q = T.switch(mask_n, mu_q, mu_q_previous)
            logvar_q = T.switch(mask_n, logvar_q, logvar_q_previous)

            return z_n, mu_p, logvar_p, mu_q, logvar_q

        eps = self._srng.normal(
            size=(seq_len, num_batch, self.num_units), avg=0.0, std=1.0)
        logvar_init = T.zeros((num_batch, self.num_units))
        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [eps, input_p, input_q, mask]
            step_fun = step_masked
        else:
            sequences = [eps, input_p, input_q]
            step_fun = step

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = helper.get_all_params(self.logvar_p_mlp)
        non_seqs += helper.get_all_params(self.mu_p_mlp)
        non_seqs += helper.get_all_params(self.q_mu_mlp)
        non_seqs += helper.get_all_params(self.q_logvar_mlp)


        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            scan_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[z_init, mu_p_init, logvar_init, mu_p_init, logvar_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            scan_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[z_init, mu_p_init, logvar_init, mu_p_init, logvar_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        z, mu_p, logvar_p, mu_q, logvar_q = scan_out

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            assert False
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            z = z.dimshuffle(1, 0, 2)
            mu_p = mu_p.dimshuffle(1, 0, 2)
            logvar_p = logvar_p.dimshuffle(1, 0, 2)
            mu_q = mu_q.dimshuffle(1, 0, 2)
            logvar_q = logvar_q.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                z = z[:, ::-1]
                mu_p = mu_p[:, ::-1]
                logvar_p = logvar_p[:, ::-1]
                mu_q = mu_q[:, ::-1]
                logvar_q = logvar_q[:, ::-1]

        return z, mu_p, logvar_p, mu_q, logvar_q