'''
Created on Mar 7, 2017

@author: julien
'''
import logging


def cpu_device():
    return '/cpu:0'


def get_available_gpus():
    try:
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        print(local_device_protos)
        return [x.name for x in local_device_protos if x.device_type == 'CPU']
    except Exception as ex:
        logging.info(
            'Error while trying to list available GPUs: %s' % str(ex))
        return list()


def default_device():
    gpus = get_available_gpus()
    if len(gpus) > 0:
        return gpus[1]
    return cpu_device()


def get_logical_device(physical_device):
    if is_gpu_device(physical_device):
        return '/gpu:0'
    return physical_device


def is_cpu_device(device):
    return device\
        and isinstance(device, str)\
        and device.startswith('/cpu')


def is_gpu_device(device):
    return device\
        and isinstance(device, str)\
        and device.startswith('/gpu')


def get_device_idx(device):
    return device.split(':')[1]
def setup_tf_session(device):
    import tensorflow as tf
    from keras import backend
