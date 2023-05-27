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

