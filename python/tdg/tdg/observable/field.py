from tdg.observable import observable

@observable
def average_field(ensemble):
    r'''
    The average auxiliary field, one per configuration.
    '''
    return ensemble.configurations.mean((1,2))

