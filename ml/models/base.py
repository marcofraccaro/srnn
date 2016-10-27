import cPickle as pkl
import numpy as np
import theano
import lasagne


class Model(object):
    """
    The :class:'Model' class represents a model following the basic deep learning priciples.
    It should be subclassed when implementing new types of models as it contains all the common code.
    """

    def __init__(self):
        """
        Initialisation of the basic architecture and programmatic settings of any model.
        This method should be called from any subsequent inheriting model.
        """
        self.model_params = None
        self.output_layer = None

        # Model state serialisation and logging variables.
        self.model_name = self.__class__.__name__

        # Initial value for the learning rate
        self.learning_rate_init = None

        # Use shared variable for learning rate. Allows us to change the learning rate during
        # training.
        self.learning_rate_sh = None

    def initialize_computation_graph(self):
        """
        Building the graph should be done prior to training. It will implement the training, testing and validation
        functions.
        As this function is model-dependent it must be implemented in subclasses
        """
        raise NotImplementedError

    def set_learning_rate(self, decay_factor):
        # Computes the learning rate for the model according to the decay schedule
        learning_rate_new = np.asarray(self.learning_rate_init * decay_factor, dtype=theano.config.floatX)
        self.learning_rate_sh.set_value(learning_rate_new)

    def get_learning_rate(self):
        return self.learning_rate_sh.get_value()

    def visualize_training(self, settings):
        """
        Visualize the progress of the training.  If not implemented this function doesn't do anything
        (and doesn't throw an error)..
        """
        pass

    def dump_model(self, f):
        """
        Dump the model into a pickled version in the model path formulated in the initialisation method.
        """
        if self.model_params is None:
            raise "Model params are not set and can therefore not be pickled."
        model_params = [param.get_value() for param in self.model_params]
        pkl.dump(model_params, f, protocol=pkl.HIGHEST_PROTOCOL)

    def load_model(self, f):
        """
        Load the pickled version of the model into a 'new' model instance.
        :param id: The model ID is constructed from the timestamp when the model was defined.
        """
        model_params = pkl.load(f)
        for i in range(len(self.model_params)):
            self.model_params[i].set_value(np.asarray(model_params[i], dtype=theano.config.floatX), borrow=True)

    def model_info(self):
        """
        Return the layers of the model and their output shapes.
        """
        print "*** " + self.model_name + " ***"
        model_layers = lasagne.layers.get_all_layers(self.output_layer)
        model_shapes = lasagne.layers.get_output_shape(model_layers)
        n_params = lasagne.layers.count_params(self.output_layer)
        print "Number of parameters: %s" % n_params

        for i, layer in enumerate(model_layers):
            class_name = layer.__class__.__name__
            name = layer.name if not layer.name is None else "no_name"
            shape = model_shapes[i]
            print class_name + ", " + name + ", " + str(shape)
        print("-" * 80)
