import torch

Euler_Mascheroni = torch.tensor(0.5772156649015328)

class from_geometric:

    @staticmethod
    def scattering_length(geometric):
        r'''
        To match the log terms we must identify

        .. math::
           
           a_{2} = \frac{e^{\gamma}}{2} a_{2D} = (0.8905362090\cdots) a_{2D}.

        '''
        return Euler_Mascheroni.exp() * geometric / 2

    @staticmethod
    def log_ka(geometric):
        r'''
        To match the logs we must identify

        .. math::

           \log k a_2 = \log k a_{2D} + \gamma - \log 2.
        '''
        return geometric + Euler_Mascheroni - torch.tensor(2.).log()

    @staticmethod
    def alpha(geometric):

        logkFa = from_geometric.log_ka(-1./geometric)
        return -1./logkFa

    @staticmethod
    def n_asquared_to_alpha(geometric):
        r'''
        Some references quote the dimensionless combination of the number density and the square of the geometric-convention scattering length :math:`na_{2D}^2`.

        This can be converted to :math:`\alpha`
        '''

        return -1. / ((2*torch.pi*geometric).sqrt() * torch.exp(Euler_Mascheroni)/2).log()
