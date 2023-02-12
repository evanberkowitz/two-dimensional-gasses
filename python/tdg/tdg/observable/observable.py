from inspect import signature
import tdg.ensemble

import logging
logger = logging.getLogger(__name__)

####
#### Observable interface
####

class Observable:

    def __init_subclass__(cls, name=''):
        if name == '':
            cls.name = cls.__name__
        else:
            cls.name = name
        logger.info(f'Observable registered: {cls.name}')
        setattr(tdg.ensemble.GrandCanonical, cls.name, cls())

    def __set_name__(self, owner, name):
        self.name  = name

    def __get__(self, obj, objtype=None):
        # Cache:
        if self.name in obj.__dict__:
            return obj.__dict__[self.name]

        if objtype is tdg.ensemble.GrandCanonical:
            result = self.measure(obj)
            obj.__dict__[self.name] = result
            return result

        raise NotImplemented()

    def __set__(self, obj, value):
        setattr(obj, self.name, value)

####
#### Callable observable interface
####

class CallableObservable:

    def __init_subclass__(cls, name=''):
        if name == '':
            cls.name = cls.__name__
        else:
            cls.name = name
        logger.info(f'Observable registered: {cls.name}')
        setattr(tdg.ensemble.GrandCanonical, cls.name, cls())

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if objtype is tdg.ensemble.GrandCanonical:
            def curried(*args, **kwargs):
                return self.measure(obj, *args, **kwargs)
            return curried
        raise NotImplemented()

####
#### The Decorator
####

def observable(func):
    sig = signature(func)
    parameters = len(sig.parameters)

    if parameters == 0:
        raise
    elif parameters == 1:
        class anonymous(Observable, name=func.__name__):
            
            def measure(self, ensemble):
                return func(ensemble)
    elif parameters > 1:
        class anonymous(CallableObservable, name=func.__name__):

            def measure(self, ensemble, *args, **kwargs):
                return func(ensemble, *args, **kwargs)

    return func # This is a hack to get sphinx to document observables sensibly.


