import matplotlib.pyplot as plt
import numpy as np
import warnings

class DecaySchedule(object):
    """
    The :class:'DecaySchedule' specifies a decay schedule to be used during training (e.g. learning rate)
    By default it returns a decreasing schedule, that starts constant at 1.0 for no_decay_epochs, applies the desired
    schedule up to max_decay_epochs and then stays constant up to max_num_epochs.

    To be able to use this also as an annealing temperature it is the possible to make it an increasing schedule by
    setting increasing=True, and to flip it left-right by using reverse=True. It can finally be scaled using e.g.
    y_range=(0,1)

    The schedule can be easily plotted using plot_decay_schedule()

    """

    def __init__(self, max_num_epochs=np.inf, no_decay_epochs=0, max_decay_epochs=np.inf, reverse=False,
                 increasing=False, y_range=None):
        """
        Initialization of the parameters of the decay schedule
        """
        self.no_decay_epochs = no_decay_epochs # Number of epochs before we start with the decay
        self.max_decay_epochs = max_decay_epochs # Number of epochs before we start with the decay
        self.max_num_epochs = max_num_epochs # Max number of epochs
        self.decay_schedule = self.compute_decay_schedule() # The actual decay schedule we will use (plot it!)
        if reverse:
            self.decay_schedule = self.decay_schedule[::-1]
        if increasing:
            self.decay_schedule = [-x for x in self.decay_schedule]
        # Scale between y_range[0] and y_range[1]
        if y_range is not None:
            tmp = self.decay_schedule
            if np.min(tmp) < np.max(tmp):
                tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
                self.decay_schedule = tmp * (y_range[1] - y_range[0]) + y_range[0]
            else:
                #TODO: in this case we could want decay_schedule to be either y_range[1] or y_range[0]
                warnings.warn('The decay_schedule is constant: not using y_range')

    def get_decay(self, epoch):
        return self.decay_schedule[epoch]

    def compute_decay_factor(self, t):
        raise NotImplementedError

    def compute_decay_schedule(self):
        decay_schedule = []
        decay_factor = 1.0
        for i in range(self.max_num_epochs):
            if i >= self.no_decay_epochs and i <= self.max_decay_epochs:  # epoch starts from 0
                t = float(i - self.no_decay_epochs + 1)  # t>=1
                decay_factor = self.compute_decay_factor(t)
            decay_schedule.append(decay_factor)
        return decay_schedule

    def plot_decay_schedule(self, marker = '-b.'):
        plt.plot(self.decay_schedule, marker)


class LinearDecaySchedule(DecaySchedule):
    """
    See "AN EMPIRICAL STUDY OF LEARNING RATES IN DEEP NEURAL NETWORKS FOR SPEECH RECOGNITION"
    http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/40808.pdf
    Power schedule:
        -1/decay*t + 1.0
    """

    def __init__(self, decay, max_num_epochs=np.inf, no_decay_epochs=0, max_decay_epochs=np.inf, reverse=False,
                 increasing=False, y_range=None):
        self.decay = float(decay)
        super(LinearDecaySchedule, self).__init__(max_num_epochs, no_decay_epochs, max_decay_epochs, reverse,
                                                  increasing, y_range)

    def compute_decay_factor(self, t):
        decay_factor = -(1.0 / self.decay) * t + 1.0
        return decay_factor


class ExponentialDecaySchedule(DecaySchedule):
    """
    See "AN EMPIRICAL STUDY OF LEARNING RATES IN DEEP NEURAL NETWORKS FOR SPEECH RECOGNITION"
    http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/40808.pdf
    Exponential schedule:
        1/(decay^t)
    """

    def __init__(self, decay, max_num_epochs=np.inf, no_decay_epochs=0, max_decay_epochs=np.inf, reverse=False,
                 increasing=False, y_range=None):
        self.decay = float(decay)
        super(ExponentialDecaySchedule, self).__init__(max_num_epochs, no_decay_epochs, max_decay_epochs, reverse,
                                                       increasing, y_range)

    def compute_decay_factor(self, t):
        decay_factor = (1.0 / self.decay) ** t
        return decay_factor


class PowerDecaySchedule(DecaySchedule):
    """
    See "AN EMPIRICAL STUDY OF LEARNING RATES IN DEEP NEURAL NETWORKS FOR SPEECH RECOGNITION"
    http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/40808.pdf
    Power schedule:
        (1+t/scale)^-decay
    """

    def __init__(self, decay, scale_decay = 1.0, max_num_epochs=np.inf, no_decay_epochs=0, max_decay_epochs=np.inf,
                 reverse=False, increasing=False, y_range=None):
        self.decay = float(decay)
        self.scale_decay = float(scale_decay)
        super(PowerDecaySchedule, self).__init__(max_num_epochs, no_decay_epochs, max_decay_epochs, reverse,
                                                 increasing, y_range)

    def compute_decay_factor(self, t):
        decay_factor = (1.0 + t / self.scale_decay) ** -self.decay
        return decay_factor