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

def observable(func):
    
    class anonymous(Observable, name=func.__name__):
        
        def measure(self, ensemble):
            return func(ensemble)

    return func # This is a hack to get sphinx to document observables sensibly.

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

def callable_observable(func):

    class anonymous(CallableObservable, name=func.__name__):

        def measure(self, ensemble, *args, **kwargs):
            return func(ensemble, *args, **kwargs)

    return func # This is a hack to get sphinx to document observables sensibly.

import tdg.observable.field
import tdg.observable.action
import tdg.observable.UU
import tdg.observable.number
import tdg.observable.spin
import tdg.observable.contact
