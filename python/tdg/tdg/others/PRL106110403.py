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
