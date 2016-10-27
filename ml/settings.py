import cPickle as pkl


class Settings(object):
    def __init__(self):
        pass
        #raise NotImplementedError

    def settings_info(self):
        # Print settings
        print("*** Settings ***")
        dict_settings = self.__dict__

        for keys, values in dict_settings.items():
            print "%s: %s" % (keys, values)

        print("-" * 80)

    def dump_settings(self, f):
        pkl.dump(self, f, pkl.HIGHEST_PROTOCOL)

    def print_csv(self):
        dict_settings = self.__dict__
        str_names = ''
        str_values = ''
        for keys, values in dict_settings.items():
            str_names = str_names + "%s, " % keys
            str_values = str_values + "%s, " % values

        print str_names
        print str_values

    @staticmethod
    # Load a pickled instance of Settings
    def load_settings(f):
        return pkl.load(f)