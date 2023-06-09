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
    Note that Fig. 2 shows 18 points but Table I gives only 16 rows.

    Returns
    -------
        torch.tensor: The rows are :math:`\ln(k_F a_{2D})`, :math:`E/E_{FG}`, the uncertainty on :math:`E/E_{FG}`, and half the binding energy :math:`\mathcal{E}_B`.
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
    # Attractive
    [+1.44, +0.349,     0.006,  -0.143  ],
    [+1.72, +0.459,     0.016,  -0.080  ],
    [+2.15, +0.552,     0.002,  -0.034  ],
    [+2.64, +0.634,     0.004,  -0.013  ],
    [+3.34, +0.706,     0.002,  -0.003  ],
    [+4.03, +0.755,     0.004,  +0.000  ],
    [+4.37, +0.775,     0.002,  +0.000  ],
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

    Returns the data from Figure 4, in TWO tensors, the first with BCS data and the other with JS data.  Each tensor has :math:`\log k_F a_{2D}` in the first row, :math:`c/k_F^4` in the second, and the uncertainty on :math:`c/k_F^4` in the third.

    The data were provided by Gianluca Bertaina.
    '''

    citation()

    BCS = torch.tensor([
        [ +1.00, +0.241, 0.003 ],
        [ +0.75, +0.357, 0.003 ],
        [ +0.50, +0.518, 0.004 ],
        [ +0.25, +0.815, 0.004 ],
        [ -0.00, +1.290, 0.007 ],
        [ -0.50, +3.456, 0.006 ],
        ]).T

    JS = torch.tensor([
        [ +4.3729,  +0.0126, 0.0004 ],
        [ +4.02627, +0.0143, 0.0006 ],
        [ +3.34079, +0.0199, 0.0004 ],
        [ +2.64491, +0.0337, 0.0004 ],
        [ +2.15031, +0.0495, 0.0006 ],
        [ +1.72414, +0.0725, 0.002  ],
        [ +1.43603, +0.1015, 0.002  ],
        [ +1.15825, +0.1451, 0.0015 ],
        ]).T

    return BCS, JS

def c0_by_kF4(eta):
    r'''
    Explained just above Figure 3, :math:`c_0` is 'the contribution to the contact from the molecular state'.

    Parameters
    ----------
    eta: torch.tensor
        :math:`\eta = \log k_F a_{2D}`

    Returns
    -------
    torch.tensor:
        The two-body contribution to the contact at :math:`\eta`.
    '''

    return 4*torch.exp(-2*tdg.conventions.Euler_Mascheroni)*torch.exp(-2*eta)

def figure_4_fit(eta):
    r'''
    Figure 4 of Ref. :cite:`PhysRevLett.106.110403` shows a fit of :func:`~.contact.contact_by_kF4` as a function of :math:`\log k_F a_{2D}`.

    The fit parameters were provided by Gianluca Bertaina.

    Parameters
    ----------
    eta: torch.tensor
        :math:`\eta = \log k_F a_{2D}`

    Returns
    -------
    torch.tensor
        :math:`c/kF^4` at :math:`\eta`.
    '''

    citation()

    CC      =-torch.tensor(2.).log() + 2. * tdg.conventions.Euler_Mascheroni
    eta0    = 0.424695699160316
    alpha   = 3.14630241190657
    a0      = 0.266210848587747
    a1      =-1.57111547468969
    c0      = 0.266210848587747
    c1      = 0.433271186790472
    c2      = 0.2632418553941

    return torch.where(eta < eta0,
        c0_by_kF4(eta) +(1.0/4)*(4*a0**2 + a1*(1 + 4*a1*(eta - eta0)**2) + 2*a0*a1*(-alpha + CC + 4*eta - 2*eta0))/(1 - 4*a0*(eta - eta0) + 2*a1*(alpha - CC - 2*eta)*(eta - eta0))**2,
        c0_by_kF4(eta) +(1.0/4)*(c1 - c0*c1 + c2*(c0*(-1 - 2*eta + 2*eta0) + (eta - eta0)*(2 + c2*eta - c2*eta0)))/(1 + c1*(eta - eta0) + c2*(eta + eta**2 - 2*eta*eta0 + (-1 + eta0)*eta0))**2
        )

def figure_4_reproduction(ax):
    r'''
    Reproduce Figure 4 of Ref. :cite:`PhysRevLett.106.110403` relying on data in :func:`figure_4` and the :func:`figure_4_fit`.

    .. plot::
       :include-source:

       import matplotlib.pyplot as plt
       fig, ax = plt.subplots(1,1, figsize=(6,4))

       from tdg.others import PRL106110403
       ax, inset = PRL106110403.figure_4_reproduction(ax)

    Parameters
    ----------
    ax: matplotlib.Axes
        Where to draw the figure.

    Returns
    -------
    ax: matplotlib.Axes
        The same handle given, but with art.
    inset:
        A handle for the inset.
    '''

    citation()

    bcs, js = figure_4()

    ax.errorbar(bcs[0], bcs[1], yerr=bcs[2], marker='s', color='black', linestyle='none', label=r'$g_{\uparrow\downarrow}$ DMC BCS')
    ax.errorbar(js [0], js [1], yerr=js [2], marker='o', color='gray',  linestyle='none', label=r'$g_{\uparrow\downarrow}$ DMC  JS')

    eta = torch.linspace(-1, 5, 1000)
    ax.plot(eta, figure_4_fit(eta), color='red', label='Eq. of state')

    ax.set_xlim([-1, 5])
    ax.set_ylim([-0.1, 4])
    ax.set_xlabel(r'$\eta=\log(k_F/a_{2D})$')
    ax.set_ylabel(r'$c/k_F^4$')
    ax.legend()

    inset = ax.inset_axes([1.66, 1.66, 3, 2.16], transform=ax.transData)
    inset.errorbar(bcs[0], bcs[1] - c0_by_kF4(bcs[0]), yerr=bcs[2], marker='s', color='black', linestyle='none')
    inset.errorbar( js[0],  js[1] - c0_by_kF4( js[0]), yerr= js[2], marker='o', color='gray', linestyle='none')
    inset.plot(eta, figure_4_fit(eta) - c0_by_kF4(eta), color='red')

    inset.set_xlim([-1, 5])
    inset.set_ylim([0., 0.09])
    inset.set_ylabel(r'$(c-c_0)/k_F^4$')

    return ax, inset


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


    if not include_all:
        return

    BCS, JS = conventional_figure_4()

    alpha_BCS = -1. / BCS[0]
    alpha_JS  = -1. / JS [0]

    ax.errorbar(
            alpha_BCS,
            BCS[1],
            yerr=BCS[2],
            color='black',
            marker='s',
            linestyle='none',
            label='Square Well BCS-DMC [Bertaina and Giorgini (2011)]',
            )

    ax.errorbar(
            alpha_JS,
            JS[1],
            yerr=JS[2],
            color='gray',
            marker='s',
            linestyle='none',
            label='Square Well JS-DMC [Bertaina and Giorgini (2011)]',
            )
