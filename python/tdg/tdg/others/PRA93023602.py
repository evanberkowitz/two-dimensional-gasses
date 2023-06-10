import torch
import tdg.conventions

from tdg.others.citation import Citation
citation = Citation(
    'Galea, Dawkins, Gandolfi, and Gezerlis, Phys. Rev. A 93, 023602 (2016)',
    '''@article{Galea:2015vdy,
    author = "Galea, Alexander and Dawkins, Hillary and Gandolfi, Stefano and Gezerlis, Alexandros",
    title = "{Diffusion Monte Carlo study of strongly interacting two-dimensional Fermi gases}",
    eprint = "1511.05123",
    archivePrefix = "arXiv",
    primaryClass = "cond-mat.quant-gas",
    reportNumber = "LA-UR-15-28921",
    doi = "10.1103/PhysRevA.93.023602",
    journal = "Phys. Rev. A",
    volume = "93",
    number = "2",
    pages = "023602",
    year = "2016"
}''')

def figure_6():
    r'''
    Figure 6 shows :math:`E/N - \epsilon_b/2` in units of :math:`E_{FG}` as a function of :math:`\log k_F a_{2D}`.


    This returns two tensors, the first for the blue circles and the latter for the purple triangles.  Each tensor is two rows; the first row being the x-coordinate and the second row the y.
    '''
    citation()

    blue_circles = torch.tensor([
        [ +3.00, +0.71 ],
        [ +2.15, +0.60 ],
        [ +1.99, +0.57 ],
        [ +1.49, +0.48 ],
        [ +1.44, +0.46 ],
        [ +1.00, +0.37 ],
        [ +0.74, +0.31 ],
        [ +0.49, +0.26 ],
        [ +0.00, +0.18 ],
        [ -0.50, +0.13 ],
        [ -1.00, +0.10 ],
        ]).T

    purple_triangles = torch.tensor([
        [ +3.00, +0.71 ],
        [ +2.15, +0.62 ],
        [ +1.44, +0.52 ],
        [ +0.74, +0.45 ],
        [ +0.00, +0.36 ],
        [ -0.50, +0.34 ],
        [ -0.99, +0.33 ],
        ]).T

    return blue_circles, purple_triangles

def conventional_figure_6():
    r'''
    The authors only imply the meaning of :math:`\epsilon_b` at the beginning of section VI, where they state the mean-field BCS result :math:`E_{BCS} = E_{FG} + \epsilon_b/2`.
    If we use (99) of Ref. :cite:`Beane:2022wcn`, presumably :math:`\epsilon_b/2E_{FG} = -\alpha`.
    '''
    blue_circles, purple_triangles = figure_6()

    blue_circles[0] = -1. / tdg.conventions.from_geometric.log_ka(blue_circles[0])
    purple_triangles[0] = -1. / tdg.conventions.from_geometric.log_ka(purple_triangles[0])

    return blue_circles, purple_triangles

def energy_comparison(ax, **kwargs):

    blue_circles, purple_triangles = conventional_figure_6()

    ax.plot(
            blue_circles[0],
            blue_circles[1],
            marker='o', color='blue', linestyle='none',
            label='Optimized Jastrow-BCS [Galea (2016)]'
            )
    ax.plot(
            purple_triangles[0],
            purple_triangles[1],
            marker='^', color='purple', linestyle='none',
            label='Jastrow-BCS [Galea (2016)]'
            )

def contact_comparison(ax, **kwargs):
    pass
