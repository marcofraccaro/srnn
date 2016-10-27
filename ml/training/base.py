import cPickle as pkl


class Train(object):
    """
    The :class:'Train' class represents a general training loop
    It should be subclassed when implementing new types of training.
    """

    def __init__(self):
        """
        Initialisation of the basic architecture and programmatic settings of any training procedure.
        This method should be called from any subsequent inheriting training procedure.
        :param model: The model to train on.

        """
        self.epoch = 0 # Current epoch
        self.epochs_eval = [] # List containing the epochs in which we run the evaluation (test). Used for plotting
        self.time_epoch_all = []

    def train_model(self, *args):
        """
        This is where the training of the model is performed.
        """
        raise NotImplementedError

    def print_training_info(self):
        raise NotImplementedError

    def dump_train(self, f):
        pkl.dump(self, f, pkl.HIGHEST_PROTOCOL)

    @staticmethod
    # Load a pickled instance of Train
    def load_train(f):
        return pkl.load(f)

    def plot_results(self, plot_path=None, ylim=None):
        raise NotImplementedError