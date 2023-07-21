import torch
import tdg.conventions

import logging
logger = logging.getLogger(__name__)

from tdg.references.citation import Citation
citation = Citation(
    'Shi, Chiesa, and Zhang, Phys. Rev. A 92, 033603 (2015)',
    'PhysRevA.92.033603')

def table_II():
    r'''
    Returns
    -------
        torch.tensor whose rows are :math:`ln(k_F a_{2D})`, :math:`E/E_{FG}`, and the uncertainty on :math:`E/E_{FG}`.
    '''

    citation('Data from Table II')

    return torch.tensor([
    #   From Table II
    #   ln(a kF)    EQMC / EFG  Error bar  
    [   -0.403426,  -5.512634,  +0.000619,  ],
    [   -0.153426,  -3.262997,  +0.000487,  ],
    [   +0.096574,  -1.884889,  +0.000325,  ],
    [   +0.346574,  -1.027841,  +0.000453,  ],
    [   +0.596574,  -0.487667,  +0.000335,  ],
    [   +0.846574,  -0.137058,  +0.000272,  ],
    [   +1.096574,  +0.096228,  +0.000203,  ],
    [   +1.346574,  +0.256943,  +0.000167,  ],
    [   +1.596574,  +0.371799,  +0.000162,  ],
    [   +1.846574,  +0.456471,  +0.000141,  ],
    [   +2.096574,  +0.521804,  +0.000173,  ],
    [   +2.346574,  +0.572904,  +0.000111,  ],
    [   +2.846574,  +0.647340,  +0.000103,  ],
    [   +3.346574,  +0.700067,  +0.000067,  ],
    [   +3.846574,  +0.737144,  +0.000128,  ],
    [   +4.346574,  +0.767283,  +0.000099,  ],
    [   +4.846574,  +0.793547,  +0.000068,  ],
    [   +5.346574,  +0.813073,  +0.000053,  ],
    [   +6.346574,  +0.842689,  +0.000036,  ],
    ]).T

def conventional_table_II():
    r'''
    Converts the first row of :func:`table_II` from the geometric :math:`\log k_F a_{2D}` to :math:`\log k_F a`.
    '''

    logkFa, E_by_FG, dE_by_FG = table_II()
    
    # They use the geometric a.  Therefore we need to convert.
    logkFa = tdg.conventions.from_geometric.log_ka(logkFa)

    logger.warning('While the data from Shi, Chiesa, and Zhang (2015) is lifted from their Table II, the resulting plot differs from Beane et al. (2023)')

    return torch.stack([logkFa, E_by_FG, dE_by_FG])

def energy_comparison(ax, **kwargs):
    r'''

    '''

    logkFa, E_by_FG, dE_by_FG = conventional_table_II()

    alpha = -1. / logkFa
    difference = E_by_FG - (1+alpha)
    error = dE_by_FG

    ax.errorbar(
            alpha.clone().detach().cpu().numpy(),
            difference.clone().detach().cpu().numpy(),
            yerr = error.clone().detach().cpu().numpy(),
            color='green',
            marker='D',
            linestyle='none',
            label='Lattice AFQMC [Shi, Chiesa, Zhang (2015)]',
            )

def contact_comparison(ax, **kwargs):
    pass
