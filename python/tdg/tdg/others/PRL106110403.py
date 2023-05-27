import torch
import tdg.conventions

from tdg.others.citation import Citation
citation = Citation(
    'Bertaina and Giorgini, Phys. Rev. Lett. 106, 110403 (2011)',
    '''@article{PhysRevLett.106.110403,
    author = {Bertaina, G. and Giorgini, S.},
    doi = {10.1103/PhysRevLett.106.110403},
    issue = {11},
    journal = {Phys. Rev. Lett.},
    month = {Mar},
    numpages = {4},
    pages = {110403},
    publisher = {American Physical Society},
    title = {BCS-BEC Crossover in a Two-Dimensional Fermi Gas},
    url = {https://link.aps.org/doi/10.1103/PhysRevLett.106.110403},
    volume = {106},
    year = {2011}
}''')

def table_I():
    r'''
    Returns
    -------
        torch.tensor: Each row is are :math:`ln(k_F a_{2D})`, :math:`E/E_{FG}`, and the uncertainty on :math:`E/E_{FG}`.
    '''

    citation()

    return torch.tensor([
    #   From Table I
    [-2.00, -137.761,   0.007,  -137.832],
    [-1.50, -50.593,    0.004,  -50.675 ],
    [-1.00, -18.532,    0.004,  -18.637 ],
    [-0.50, -6.714,     0.004,  -6.856  ],
    [+0.00, -2.318,     0.002,  -2.522  ],
    [+0.25, -1.283,     0.012,  -1.530  ],
    [+0.50, -0.638,     0.010,  -0.928  ],
    [+0.75, -0.201,     0.012,  -0.563  ],
    [+1.44, +0.349,     0.006,  -0.143  ],
    [+1.72, +0.459,     0.016,  -0.080  ],
    [+2.15, +0.552,     0.002,  -0.034  ],
    [+2.64, +0.634,     0.004,  -0.013  ],
    [+3.34, +0.706,     0.002,  -0.003  ],
    [+4.03, +0.755,     0.004,  +0.000  ],
    [+4.37, +0.775,     0.001,  +0.000  ],
    [+5.18, +0.821,     0.007,  +0.000  ],
    ]).T

def conventional_table_I():
    r'''
    The same as :func:`table_I` but converted from the geometric convention.
    '''
    logkFa, E_by_FG, dE_by_FG, half_Eb = table_I()
    
    # They use the geometric a.  Therefore we need to convert.
    logkFa = tdg.conventions.from_geometric.log_ka(logkFa)

    return torch.stack([logkFa, E_by_FG, dE_by_FG, half_Eb])

def figure_4():
    r'''
    Figure 4 of Ref. :cite:`PhysRevLett.106.110403` provides measurements of :func:`~.contact.contact_by_kF4` as a function of :math:`\log k_F a_{2D}`.

    Returns the data from Figure 4, lifted by hand, in TWO tensors, the first with BCS data and the other with JS data.  Each tensor has :math:`\log k_F a_{2D}` in the first row and :math:`c/k_F^4` in the second.
    '''

    citation()

    BCS = torch.tensor([
        [ +1.00, +0.24 ],
        [ +0.74, +0.35 ],
        [ +0.49, +0.52 ],
        [ +0.24, +0.81 ],
        [ -0.00, +1.29 ],
        [ -0.50, +3.45 ],
        ]).T

    JS = torch.tensor([
        [ +4.37, +0.01 ],
        [ +4.02, +0.01 ],
        [ +3.34, +0.01 ],
        [ +2.64, +0.03 ],
        [ +2.15, +0.04 ],
        [ +1.72, +0.07 ],
        [ +1.43, +0.09 ],
        [ +1.15, +0.15 ],
        ]).T

    return BCS, JS

def conventional_figure_4():
    r'''
    The same data as :func:`figure_4` but with :math:`\log k_F a` rather than :math:`\log k_F a_{2D}`.
    '''

    BCS, JS = figure_4()

    BCS[0] = tdg.conventions.from_geometric.log_ka(BCS[0])
    JS [0] = tdg.conventions.from_geometric.log_ka(JS [0])

    return BCS, JS

def energy_comparison(ax, **kwargs):
    r'''
    The first column of Table I gives :math:`E/E_{FG}` .  To get a fair comparison we need to subtract :math:`E_{MF}/E_{FG} = (1+\alpha)`.
    '''

    logkFa, E_by_FG, dE_by_FG, half_Eb = conventional_table_I()

    alpha = -1. / logkFa
    difference = E_by_FG - (1+alpha)
    error = dE_by_FG

    ax.errorbar(
            alpha,
            difference,
            yerr = error,
            color='black',
            marker='s',
            linestyle='none',
            label='Square Well JS-DMC [Bertaina and Giorgini (2011)]',
            )

def contact_comparison(ax, include_all=False, **kwargs):
    r'''
    Many of the points are off-scale compared to the contact comparison in Ref. :cite:`Beane:2022wcn`.

    Plots the contact from Ref. :cite:`PhysRevLett.106.110403` on the axis ``ax`` only if ``include_all``.
    '''


    if ('include_all' not in kwargs) or (not kwargs['include_all']):
        return

    BCS, JS = conventional_figure_4()

    alpha_BCS = -1. / BCS[0]
    alpha_JS  = -1. / JS [0]

    ax.plot(
            alpha_BCS,
            BCS[1],
            color='black',
            marker='s',
            linestyle='none',
            label='Square Well BCS-DMC [Bertaina and Giorgini (2011)]',
            )

    ax.plot(
            alpha_JS,
            JS[1],
            color='gray',
            marker='s',
            linestyle='none',
            label='Square Well JS-DMC [Bertaina and Giorgini (2011)]',
            )
