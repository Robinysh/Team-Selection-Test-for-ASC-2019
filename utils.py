import configparser
import os

def get_args():
    config = configparser.ConfigParser()
    if not os.path.exists('config.ini'):
        raise IOError('config.ini not found.')
    config.read('config.ini')
    return config

def normalize(images):
    return (images-images.mean(axis=0))/images.std(axis=0)


