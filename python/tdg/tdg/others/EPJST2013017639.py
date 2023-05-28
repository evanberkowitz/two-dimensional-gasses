import torch
import tdg.conventions

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

def figure_2():
    r'''
    Figure 2 provides the normalized, mean-field subtracted energy normalized directly but as a function of their :math:`g=-1/2 \log(k_F b)` (just after (2)) where :math:`b` sets the binding energy as :math:`-1/mb^2` (just below Figure 1).

    Therefore, :math:`b` is the scattering length in our convention.  However, :math:`g=\alpha/2`.

    Returns THREE tensors, data representing their green squares, blue triangles, and gray circles, respectively.
    '''

    citation()

    # All are (alpha, (E_ - E_MF)/E_FG)
    greenSquares = torch.tensor([
        [ +0.40, +0.00 ],
        [ +0.36, +0.02 ],
        [ +0.32, +0.03 ],
        [ +0.27, +0.03 ],
        [ +0.23, +0.03 ],
        [ +0.19, +0.02 ],
        [ +0.14, +0.02 ],
        [ +0.09, +0.01 ],
        [ +0.05, +0.00 ],
        ]).T

    blueTriangles = torch.tensor([
        [ +0.36, -0.01 ],
        [ +0.32, -0.00 ],
        [ +0.28, -0.00 ],
        [ +0.23, +0.00 ],
        [ +0.19, +0.00 ],
        [ +0.14, +0.00 ],
        [ +0.09, +0.00 ],
        [ +0.05, +0.00 ],
        ]).T

    grayCircles = torch.tensor([
        [ -0.37, +0.11 ],
        [ -0.32, +0.08 ],
        [ -0.26, +0.05 ],
        [ -0.20, +0.03 ],
        [ -0.15, +0.01 ],
        [ -0.10, +0.00 ],
        [ -0.07, +0.00 ],
        [ -0.04, +0.00 ],
        [ -0.02, +0.00 ],
        ]).T

    return greenSquares, blueTriangles, grayCircles

def conventional_figure_2():
    r'''
    Rather than :math:`g`, gives data as a function of :math:`\alpha`.
    '''
    green, blue, gray = figure_2()

    green[0] = 2*green[0]
    blue [0] = 2*blue [0]
    gray [0] = 2*gray [0]

    return green, blue, gray

def energy_comparison(ax, **kwargs):

    green, blue, gray = conventional_figure_2()

    ax.plot(
            green[0],
            green[1],
            color='green', marker='s', linestyle='none',
            label="SW UB-VMC [Bertaina (2013)]"
            )
    ax.plot(
            blue [0],
            blue [1],
            color='blue', marker='v', linestyle='none',
            label='HD JS-DMC [Bertaina (2013)]'
            )
    ax.plot(
            gray [0],
            gray [1],
            color='gray', marker='o', linestyle='none',
            label='SW JS-DMC [Bertaina (2013)]'
            )

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

