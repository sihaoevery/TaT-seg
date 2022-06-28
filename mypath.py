import os

class Path(object):

    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/path/to/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/benchmark_RELEASE/'    
        elif dataset == 'cocostuff10k':
            return '/path/to/cocostuff-10k/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
