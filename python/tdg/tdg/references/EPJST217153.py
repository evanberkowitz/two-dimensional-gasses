import torch
import tdg.conventions

from tdg.references.citation import Citation
citation = Citation(
    'Bertaina, EPJ Special Topics 217, 153-162 (2013)',
    'bertaina2013')

def figure_2():
    r'''
    Returns two tensors, data representing their blue triangles (HD JS-DMC) and gray circles (SW JS-DMC), respectively.

    The rows are particle density times the scattering length squared :math:`na_{2D}^2`, the uncorrected energy normalized by the ideal Fermi gas :math:`E/E_{FG}`, its uncertainty, the finite-size-corrected energy normalized by the ideal Fermi gas, and its uncertainty.

    The finite-size correction is described at the end of Section 2; it amounts to using the energy difference for the infinite and finite-volume noninteracting case.

    The data were provided by Gianluca Bertaina.
    '''

    citation('Data provided by Gianluca Bertaina')


    blueTriangles =torch.tensor([
        # na_{2D}^2,                E/E_FG, ±,      corrected   ±
        [3.28043e-10,               1.1045, 0.0002, 1.0996,     0.0012],
        [7.22562e-6,                1.2033, 0.001,  1.1984,     0.002 ],
        [0.00020254596022905975,    1.2994, 0.001,  1.2945,     0.002 ],
        [0.0010723775711956546,     1.3923, 0.0016, 1.3874,     0.0026],
        [0.0029150244650281935,     1.4806, 0.001,  1.4757,     0.002 ],
        [0.0056776923810426105,     1.5653, 0.0016, 1.5604,     0.0026],
        [0.009140685251156135,      1.6442, 0.001,  1.6393,     0.002 ],
        [0.013064233284684921,      1.7183, 0.001,  1.7134,     0.002 ],
        [0.017247306568862027,      1.7850, 0.002,  1.7801,     0.003 ],
        [0.021539279301848634,      1.8495, 0.0022, 1.8446,     0.0032]
        ]).T

    grayCircles = torch.tensor([
        # na_{2D}^2,      E/E_FG, ±,      corrected ±
        [3.74627E16,      0.9546, 0.001,  0.9497,   0.002  ],
        [7.72164E7,       0.9050, 0.001,  0.9001,   0.002  ],
        [98268.25037,     0.8541, 0.0012, 0.8492,   0.0022 ],
        [3505.62091,      0.8056, 0.001,  0.8007,   0.002  ],
        [125.059497024,   0.7096, 0.001,  0.7047,   0.002  ],
        [23.620687891,    0.6170, 0.0015, 0.6121,   0.0025 ],
        [8.6895654614478, 0.5243, 0.0017, 0.5194,   0.0027 ],
        [4.4613716648616, 0.4400, 0.004,  0.4351,   0.005  ],
        [2.771159405951,  0.3577, 0.004,  0.3528,   0.005  ],
        [1.938904133033,  0.2641, 0.009,  0.2592,   0.010  ],
        ]).T

    return blueTriangles, grayCircles

def conventional_figure_2():
    r'''
    Rather than :math:`na_{2D}^2`, gives data as a function of :math:`\alpha`.
    '''
    blue, gray = figure_2()

    blue[0] = tdg.conventions.from_geometric.n_asquared_to_alpha(blue[0])
    gray[0] = tdg.conventions.from_geometric.n_asquared_to_alpha(gray[0])

    return blue, gray

def energy_comparison(ax, **kwargs):
    r'''

    .. plot::
       :include-source:

       import tdg, torch
       import matplotlib.pyplot as plt

       fig, ax = plt.subplots(1,1, figsize=(8,6))

       tdg.references.EPJST217153.energy_comparison(ax)

       alpha = torch.linspace(-0.9, 0.9, 1000)
       # Show just the mean-field subtracted piece of eq (2)
       ax.plot(alpha, (3-4 *torch.tensor(2.).log())*(alpha/2)**2, color='orange', label='eq. (2)')
       ax.set_xlim([-0.9, 0.9])
       ax.set_ylim([-0.025, 0.15])
       ax.set_xlabel('α = 2g')
       ax.set_ylabel('(E/N - MF)/E_FG')

       inset = ax.inset_axes([0, 0.06, 0.8, 0.08], transform=ax.transData)
       blue, gray = tdg.references.EPJST217153.conventional_figure_2()
       inset.errorbar(blue[0], blue[3], yerr=blue[4], color='blue', marker='v', linestyle='none')
       inset.errorbar(gray[0], gray[3], yerr=gray[4], color='gray', marker='o', linestyle='none')
       inset.set_xlim([-0.9, 0.9])
       inset.set_ylabel('E/N / E_FG')
       # Show just the whole of eq (2)
       inset.plot(alpha, 1+alpha + (3-4 *torch.tensor(2.).log())*(alpha/2)**2, color='orange')

       ax.legend()
    '''
    blue, gray = conventional_figure_2()
    # To plot we need to subtract the mean-field (1-alpha)

    ax.errorbar(
            blue[0].clone().detach().cpu().numpy(),
            (blue[3]-(1+blue[0])).clone().detach().cpu().numpy(),
            yerr=blue[4].clone().detach().cpu().numpy(),
            color='blue', marker='v', linestyle='none',
            label='HD JS-DMC [Bertaina (2013)]'
            )
    ax.errorbar(
            gray[0].clone().detach().cpu().numpy(),
            (gray[3]-(1+gray[0])).clone().detach().cpu().numpy(),
            yerr=gray[4].clone().detach().cpu().numpy(),
            color='gray', marker='o', linestyle='none',
            label='SW JS-DMC [Bertaina (2013)]'
            )

def contact_by_kF4():
    r'''
    Returns a tensor whose rows are :math:`\alpha` and :math:`c/kF^4`.
    '''

    citation('Data provided by Gianluca Bertaina')

    return torch.tensor([
        # Repulsive results from hard-disks
        [0.647457, 0.0886,  1.5e-04],
        [0.560979, 0.0700,  1.5e-04],
        [0.472605, 0.0515,   1.e-04],
        [0.382273, 0.0348,   1.e-04],
        [0.289917, 0.02050,  7.e-05],
        [0.195468, 0.00955,  3.e-05],
        [0.098854, 0.002461, 7.e-06],
        ]).T

def contact_comparison(ax, **kwargs):
    r'''
    Plots the data in :func:`contact_by_kF4` as points in a style that matches :cite:`Beane:2022wcn`.
    '''

    alpha, c_by_kF4, dc_by_kF4 = contact_by_kF4()

    ax.errorbar(
            alpha.clone().detach().cpu().numpy(),
            c_by_kF4.clone().detach().cpu().numpy(),
            yerr=dc_by_kF4.clone().detach().cpu().numpy(),
            color='blue', marker='v', linestyle='none',
            label='Hard Disk JS-DMC [Bertaina (2013)]',
            )

