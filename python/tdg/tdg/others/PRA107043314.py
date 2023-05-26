import torch

r'''
These results are from Ref.~:cite:`Beane:2022wcn`.
'''

import logging
logger = logging.getLogger(__name__)

name  = 'Beane et al. Phys. Rev. A 107, 043314 (2023)'
bibtex = '''@article{Beane:2022wcn,
    author = "Beane, Silas R. and Bertaina, Gianluca and Farrell, Roland C. and Marshall, William R.",
    title = "{Toward precision Fermi liquid theory in flatland}",
    eprint = "2212.05177",
    archivePrefix = "arXiv",
    primaryClass = "cond-mat.quant-gas",
    reportNumber = "NT@UW-22-19",
    journal = {Phys. Rev. A},
    volume = {107},
    issue = {4},
    pages = {043314},
    numpages = {17},
    year = {2023},
    month = {Apr},
    publisher = {American Physical Society},
    doi = {10.1103/PhysRevA.107.043314},
    url = {https://link.aps.org/doi/10.1103/PhysRevA.107.043314}
}'''

label = r'Fermi liquid $\mathcal{O}(\alpha^3)$ [Beane et al. (2023)]'

_cited = False
def _cite():
    global _cited
    if not _cited:
        logger.info(f'Using a result from {name}.')
        logger.debug(bibtex)
        _cited = True
    return


def contact_by_kF4(alpha):
    r'''
    The contact density normalized by :math:`k_F^4`, an intensive and dimensionless quantity,

    .. math::
        \begin{align}
            \frac{c}{k_F^4} &= \frac{1}{4} \alpha^2 \left[
            1
            + \left(\frac{3}{2} - \ln 4\right) \alpha
            + 3 \left[0.16079 - (\ln 4 - 1) \right] \alpha^2
            + \mathcal{O}(\alpha^3)
            \right]
            &&(98)
        \end{align}

    where :math:`c` is the contact density so that this may be compared to :func:`~.contact.contact_by_kF4`.

    Parameters
    ----------
        alpha: torch.tensor
            The natural expansion parameters in the EFT.

    Returns
    -------
        torch.tensor:
            :math:`c/k_F^4` given above evaluated for `alpha`
    '''

    _cite()

    log4 = torch.log(torch.tensor(4.))
    return 0.25 * alpha**2 * (
                    1
                    + (1.5 - log4) * alpha
                    + 3*(0.16079 - (log4-1))*alpha**2
                    # + O(alpha^3)
    )

def Fermi_Liquid_Energy_by_Fermi_Gas_Energy(alpha):
    r'''
    Using the energy-per-particle definitions from Ref. :cite:`Beane:2022wcn`,

    .. math::
        \begin{align}
            \frac{E_{FG}}{N} &= \varepsilon_{FG} = \frac{k_F^2}{2M} &&(30)
            \\
            \frac{E_{FL}}{N} &= \varepsilon_{FG} \left[
                1
                +   \alpha
                +   \alpha^2    (0.05685)
                -   \alpha^3    (0.22550)
                +   \mathcal{O}(\alpha^4)
                \right]
                && (102)
        \end{align}

    this returns function returns the dimensionless ratio :math:`E_{FG} / E_{FL}`.

    In Figs. 15-18 Ref. :cite:`Beane:2022wcn` plots the difference of this ratio and :func:`Mean_Field_Energy_by_Fermi_Gas_Energy`.
    '''

    _cite()

    return (
            1
            + alpha
            + 0.05685 * alpha**2
            - 0.22550 * alpha**3
            )

def Mean_Field_Energy_by_Fermi_Gas_Energy(alpha):
    r'''

    Ref. :cite:`Beane:2022wcn` gives the mean-field energy per particle

    .. math::
        \begin{align}
            \frac{E_{MF}}{N} &= \varepsilon_{FG} \left[ 1 + \alpha \right]
            && \text{just after (102) and}
            \\
            \frac{E_{FG}}{N} &= \varepsilon_{FG} = \frac{k_F^2}{2M}
            &&(30)
        \end{align}

    In Figs. 15-18 Ref. :cite:`Beane:2022wcn` plots the difference of this ratio and :func:`Fermi_Liquid_Energy_by_Fermi_Gas_Energy`.
    '''

    _cite()

    return 1+alpha

def contact_comparison(ax, *, alpha, cutoff_variation=0.05):
    r'''

    Plots :func:`contact_by_kF4` as a function of alpha.

    Error bars are produced by varying the cutoff; see Ref. :cite:`Beane:2022wcn` Fig. 12.
    '''
    
    ax.plot(alpha,
            contact_by_kF4(alpha),
            color='black',
            label=label,
            )
    ax.fill_between(
            alpha,
            contact_by_kF4((1-cutoff_variation)*alpha),
            contact_by_kF4((1+cutoff_variation)*alpha),
            color='gray',
            alpha=0.2
            )

def energy_comparison(ax, *, alpha, cutoff_variation=0.05):
    r'''

    Plots the difference between :func:`Fermi_Liquid_Energy_by_Fermi_Gas_Energy` and :func:`Mean_Field_Energy_by_Fermi_Gas_Energy` as a function of alpha.

    Error bars are produced by varying the cutoff; see Ref. :cite:`Beane:2022wcn` Fig. 15.
    Note that for α>0 they include hard-disk effective-range effects.
    '''

    ax.plot(alpha,
            Fermi_Liquid_Energy_by_Fermi_Gas_Energy(alpha) - Mean_Field_Energy_by_Fermi_Gas_Energy(alpha),
            color='black',
            label=label,
            )
    ax.fill_between(
            alpha,
            Fermi_Liquid_Energy_by_Fermi_Gas_Energy((1-cutoff_variation)*alpha) - Mean_Field_Energy_by_Fermi_Gas_Energy((1-cutoff_variation)*alpha),
            Fermi_Liquid_Energy_by_Fermi_Gas_Energy((1+cutoff_variation)*alpha) - Mean_Field_Energy_by_Fermi_Gas_Energy((1+cutoff_variation)*alpha),
            color='gray',
            alpha=0.2
            )


