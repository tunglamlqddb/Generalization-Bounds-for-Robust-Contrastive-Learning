from importlib import import_module

def get_dataset(dataset, **kwargs):
    '''
    Return the number of classes, image size, and the dataloader.
    '''
    dataset = import_module(f'datasets.{dataset}')
    return dataset.num_classes, dataset.size, dataset.load(**kwargs)
