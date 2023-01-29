import argparse

from .log import defaults as log_defaults
from .metadata import defaults as meta_defaults

def defaults():
    return [
            log_defaults(),
            meta_defaults()
            ]

class ArgumentParser(argparse.ArgumentParser):

    def __init__(self, *args, **kwargs):
        k = {**kwargs}
        if 'parents' in k:
            k['parents'] += defaults()
        else:
            k['parents'] = defaults()
        super().__init__(*args, **k)

