import torch

r'''
These results are from Ref.~:cite:`bertaina2013`.
'''

import logging
logger = logging.getLogger(__name__)

name  = 'Bertaina, EPJ Special Topics 217, 153-162 (2013)'
bibtex = '''@article{bertaina2013,
    title={Two-dimensional short-range interacting attractive and repulsive Fermi gases at zero temperature},
    author={Bertaina, Gianluca},
    journal={The European Physical Journal Special Topics},
    volume={217},
    number={1},
    pages={153--162},
    year={2013},
    publisher={Springer}
}'''

_cited = False
def _cite():
    global _cited
    if not _cited:
        logger.info(f'Using a result from {name}.')
        logger.debug(bibtex)
        _cited = True
    return

def _contact_by_kF4():

    _cite()
    return torch.tensor([
        [0.561347, 0.070021],
        [0.473085, 0.051458],
        [0.382930, 0.034817],
        [0.291280, 0.020615],
        [0.195786, 0.009559],
        [0.098755, 0.002458],
        [-0.622725, 0.067143],
        [-0.490006, 0.047254],
        [-0.395163, 0.032882],
        [-0.309663, 0.019606],
        [-0.255783, 0.014222],
        [-0.234334, 0.012622]
        ])

def contact_comparison(ax, **kwargs):
    r'''

    '''
    
    _cite()

    c_by_kF4 = _contact_by_kF4()
    positive = (c_by_kF4[:,0] > 0)
    negative =  (c_by_kF4[:,0] < 0)

    ax.plot(
            c_by_kF4[negative,0],
            c_by_kF4[negative,1],
            color='gray', marker='o', linestyle='none',
            label='Square Well JS-DMC [Bertaina (2013)]',
            )

    ax.plot(
            c_by_kF4[positive,0],
            c_by_kF4[positive,1],
            color='blue', marker='v', linestyle='none',
            label='Hard Disk JS-DMC [Bertaina (2013)]',
            )

