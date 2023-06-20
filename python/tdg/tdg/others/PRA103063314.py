import torch
import tdg.conventions

from tdg.others.citation import Citation
citation = Citation(
    'Pilati, Orso, and Bertaina, Phys. Rev. A, 103:063314, (2021)',
    'Pilati:2021')

def figure_2_P_0_and_N_98():
    r'''
    Figure 2 shows the energy per particle normalized by the ideal fermi gas :math:`e/e_{FG}` as a function of population imbalance for a variety of choices for :math:`k_F a_{2D}` and system sizes.

    This function returns a tensor with :math:`k_Fa_{2D}` in the first row, :math:`e=E/N` in the second row, and its uncertainty in the third for the population-balanced :math:`P=0` with :math:`N=98`.

    The energies per particle already include a finite-size effect using an effective mass from second-order perturbation theory.  The uncertainties include the statistical fluctuations and 50% of the difference between first- and second-order perturbative evaluations of the effective mass as an estimate of the remaining finite-size effects, summed in quadrature.

    The data were provided by Gianluca Bertaina.
    '''

    citation('Data provided by Gianluca Bertaina')

    hard_disks = torch.tensor([
        # The hard disks have diameter R = a_2D
        # kFa2D       E/N         delta
        [0.000221557, 1.11899,  0.00036 ],
        [0.00110778,  1.14714,  0.00052 ],
        [0.00221557,  1.16341,  0.00061 ],
        [0.0110778,   1.2216,   0.00091 ],
        [0.0221557,   1.2603,   0.0011  ],
        [0.0553892,   1.3392,   0.0014  ],
        [0.110778,    1.4386,   0.0018  ],
        [0.221557,    1.6141,   0.0020  ],
        [0.332335,    1.7901,   0.0022  ],
        [0.443113,    1.9726,   0.0023  ],
        [0.487425,    2.0461,   0.0024  ],
        ]).T

    soft_disks = torch.tensor([
        # The soft disks have diameter R = 2*a2D, twice as big as the hard disks.
        # kFa2D       E/N      delta
        [0.0553892,   1.3397,  0.0014   ],
        [0.110778,    1.4391,  0.0017   ],
        [0.221557,    1.6138,  0.0022   ],
        [0.332335,    1.7867,  0.0025   ],
        [0.443113,    1.9585,  0.0024   ],
        [0.487425,    2.0254,  0.0024   ],
        ]).T

    return hard_disks, soft_disks

def conventional_figure_2_P_0_and_N_98():
    r'''
    Converts :math:`k_Fa_{2D}` in the first row of :func:`figure_2_P_0_and_N_98` to :math:`\alpha`.
    '''

    hard, soft = figure_2_P_0_and_N_98()

    hard[0] = -1./tdg.conventions.from_geometric.log_ka(hard[0].log())
    soft[0] = -1./tdg.conventions.from_geometric.log_ka(soft[0].log())

    return hard, soft

def energy_comparison(ax, **kwargs):

    hard, soft = conventional_figure_2_P_0_and_N_98()

    ax.errorbar(hard[0], hard[1]-(1+hard[0]), hard[2],
            marker='o', color='orange', markerfacecolor='none', linestyle='none',
            label='HD DMC [Pilati (2021)]')

    ax.errorbar(soft[0], soft[1]-(1+soft[0]), soft[2],
            marker='s', color='lightsteelblue', linestyle='none',
            label='SD DMC [Pilati (2021)]')

def contact_comparison(ax, **kwargs):
    pass
