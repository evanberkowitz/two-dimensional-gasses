import tdg.others.PRL106110403 as PRL106110403
import tdg.others.PRA107043314 as PRA107043314
import tdg.others.EPJST2013017639 as EPJST2013017639
import tdg.others.PRA93023602 as PRA93023602
import tdg.others.PRA92033603 as PRA92033603
import tdg.others.PRA103063314 as PRA103063314

import logging
logger = logging.getLogger(__name__)

REFERENCES = {
        PRL106110403,
        PRA107043314,
        EPJST2013017639,
        PRA93023602,
        PRA92033603,
        PRA103063314,
}
r'''

The ``references`` parameters in the functions below expects a set of modules; each module represents a different reference.

This variable contains all available references.
'''

def contact_comparison(ax, *, references=REFERENCES, **kwargs):
    r'''contact_comparison(ax, *, references=REFERENCES, **kwargs)

    Plots the values of :math:`c/k_F^4` provided by different references for comparison with :func:`~.contact.contact_by_kF4`.

    Parameters
    ----------
        ax: matplotlib axis
            Where to plot the references' results.
        alpha: torch.tensor
            If a reference provides a function (rather than just points), pass alpha to it.
            Also sets the limits for the x-axis of the plot.

    .. plot:: examples/plot/contact-comparison.py
       :include-source:

    '''

    for ref in references:
        try:
            ref.contact_comparison(ax, **kwargs)
        except Exception as e:
            logger.debug(str(e))

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$c/k_F^4$')

def energy_comparison(ax, *, references=REFERENCES, **kwargs):
    r'''energy_comparison(ax, *, references=REFERENCES, **kwargs)

    Plots the difference between the Fermi Liquid and Mean Field energies normalized by the energy of the free Fermi Gas, :math:`(E_{FL}-E_{MF})/E_{FG}`.

    Parameters
    ----------
        ax: matplotlib axis
            Where to plot the references' results.
        alpha: torch.tensor
            If a reference provides a function (rather than just points), pass alpha to it.
            Also sets the limits for the x-axis of the plot.


    .. plot:: examples/plot/energy-comparison.py
       :include-source:

    '''

    for ref in references:
        try:
            ref.energy_comparison(ax, **kwargs)
        except Exception as e:
            logger.debug(str(e))

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$(E_{FL}-E_{MF})/E_{FG}$')

