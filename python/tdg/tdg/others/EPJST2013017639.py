import torch

from tdg.others.citation import Citation
citation = Citation(
    'Bertaina, EPJ Special Topics 217, 153-162 (2013)',
    '''@article{bertaina2013,
    title={Two-dimensional short-range interacting attractive and repulsive Fermi gases at zero temperature},
    author={Bertaina, Gianluca},
    journal={The European Physical Journal Special Topics},
    volume={217},
    number={1},
    pages={153--162},
    year={2013},
    publisher={Springer}
}''')

def contact_by_kF4():
    r'''
    Returns a tensor whose rows are :math:`\alpha` and :math:`c/kF^4`.
    '''

    citation()
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
        ]).T

def contact_comparison(ax, **kwargs):
    r'''
    Plots the data in :func:`contact_by_kF4` as points in a style that matches :cite:`Beane:2022wcn`.
    '''

    alpha, c_by_kF4 = contact_by_kF4()
    positive = (alpha > 0)
    negative =  (alpha < 0)

    ax.plot(
            alpha[negative],
            c_by_kF4[negative],
            color='gray', marker='o', linestyle='none',
            label='Square Well JS-DMC [Bertaina (2013)]',
            )

    ax.plot(
            alpha[positive],
            c_by_kF4[positive],
            color='blue', marker='v', linestyle='none',
            label='Hard Disk JS-DMC [Bertaina (2013)]',
            )

