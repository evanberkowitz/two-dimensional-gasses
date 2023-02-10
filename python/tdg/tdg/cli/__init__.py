import argparse

from .log import defaults as log_defaults
from .metadata import defaults as meta_defaults

def defaults():
    r'''
    Provides a list of standard-library ``ArgumentParser`` objects.

    Currently provides defaults from

    * :func:`tdg.cli.log.defaults`
    * :func:`tdg.cli.metadata.defaults`
    '''
    return [
            log_defaults(),
            meta_defaults()
            ]

class ArgumentParser(argparse.ArgumentParser):
    r'''
    Forwards all arguments, except that it adds :func:`~.cli.defaults` to the `parents`_ option.

    Parameters
    ----------
        *args:
            Forwarded to the standard library's ``ArgumentParser``.
        *kwargs:
            Forwarded to the standard library's ``ArgumentParser``.

    .. _parents: https://docs.python.org/3/library/argparse.html#parents
    '''
    def __init__(self, *args, **kwargs):
        k = {**kwargs}
        if 'parents' in k:
            k['parents'] += defaults()
        else:
            k['parents'] = defaults()
        super().__init__(*args, **k)

