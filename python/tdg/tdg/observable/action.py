import functorch
from tdg.observable import observable

@observable
def S(ensemble):
    r'''
    The ``Action`` evaluated on the ensemble.
    '''
    return functorch.vmap(ensemble.Action)(ensemble.configurations)

