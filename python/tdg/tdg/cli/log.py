#!/usr/bin/env python

import logging
logger = logging.getLogger(__name__)

import argparse

def LogAction(action):
    '''Construct an argparse.Action on the fly with init and call determined by the passed action.'''

    class anonymous(argparse.Action):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            logging.basicConfig()
            action.method(action['default'])

        def __call__(self, parser, namespace, values, option_string, ):
            action.method(values)

    return anonymous

class StarStarSugar:

    def __init__(self, **kwargs):
        self.parameters.update({**kwargs})
        if 'default' in self.parameters:
            self.parameters['help']+=f' Default is {str(self.parameters["default"]).replace("%","%%")}.'
        self.parameters['action'] = LogAction(self)

    def keys(self):
        return self.parameters.keys()

    def __getitem__(self, key):
        return self.parameters[key]

class LogLevel(StarStarSugar):

    parameters = {
            'default': 'WARNING',
            'help':    'Log level; one of DEBUG, INFO, WARNING, ERROR, CRITICAL, FATAL.',
            'type':    str
            }

    levels = {
            'DEBUG':    logging.DEBUG,
            'INFO':     logging.INFO,
            'WARNING':  logging.WARNING,
            'ERROR':    logging.ERROR,
            'CRITICAL': logging.CRITICAL,
            }

    def method(self, values):
        try:
            logging.getLogger().setLevel(self.levels[values])
        except:
            raise ValueError(f'Must be one of {self.levels.keys()}')

class LogFormat(StarStarSugar):

    parameters = {
            'type': str,
            'default': '%(asctime)s %(name)s %(levelname)10s %(message)s',
            'help': 'Log format.  See https://docs.python.org/3/library/logging.html#logrecord-attributes for details.',
            }

    def method(self, fmt):
        formatter= logging.Formatter(fmt)
        for handler in logging.getLogger().handlers:
            handler.setFormatter(formatter)

def defaults():
    log_arguments = argparse.ArgumentParser(add_help=False)
    log_arguments.add_argument('--log-level',  **LogLevel())
    log_arguments.add_argument('--log-format', **LogFormat())
    return log_arguments

if __name__ == '__main__':

    parser = argparse.ArgumentParser(parents=[defaults(), ]) 
    args = parser.parse_args()
    print(args)

    logger.debug    ("This is DEBUG")
    logger.info     ("This is INFO")
    logger.warning  ("This is a WARNING")
    logger.error    ("This is an ERROR")
    logger.critical ("This is CRITICAL")
