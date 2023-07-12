import torch
import tdg

_thermodynamic_tolerance=1e-10

class Thermodynamic:
    r'''
    You may specify any combination of parameters as long as they give enough information.  If you do not provide enough information, raises a ValueError.
    
    In the parameters below, you must provide at least one of the first three, at least one of the second three, and at least one of the third three.

    Because you can provide redundant information, consistency is enforced; two quantities that must match are checked for equality up to a specified tolerance.
    If any of these checks fail, raises a ValueError.
    No guarantee about which quantity 'takes precedence'.

    Parameters
    ----------
        aspect_ratio: torch.tensor
            :math:`\tilde{\beta}/\tilde{a}^2 = 1/4\pi^2 M a^2 T`
        deBroglie_by_scattering_length: torch.tensor
            :math:`\lambda_{dB}/a = \sqrt{2\pi / Ma^2 T}`
        beta_binding_energy: torch.tensor
            :math:`\beta \epsilon_B = - 1/Ma^2 T`; since :math:`a>0`, must be negative.

        beta_mu: torch.tensor
            :math:`\beta \mu = \tilde{\beta} \tilde{\mu}`
        log_z: torch.tensor
            :math:`\log z = \beta \mu`
        z: torch.tensor
            :math:`z = \exp \beta \mu`.

        beta_h: torch.tensor
            :math:`\beta \vec{h} = \tilde{\beta} \vec{\tilde{h}}`
        log_zh: torch.tensor
            :math:`\vec{\log z_h} = \beta \vec{h}`
        zh: torch.tensor
            :math:`z_h = \exp \beta \vec{h}` elementwise

        tolerance: float
            Strictness of consistency when redundant information is provided.
    '''
    def __init__(self, *,
        aspect_ratio=None, deBroglie_by_scattering_length=None, beta_binding_energy=None,
        beta_mu=None, log_z=None, z=None,
        beta_h=None, log_zh=None, zh=None,

        tolerance=_thermodynamic_tolerance
        ):
        
        self._init_aspect_ratio(
            aspect_ratio                   = aspect_ratio, 
            deBroglie_by_scattering_length = deBroglie_by_scattering_length, 
            beta_binding_energy            = beta_binding_energy,
            tolerance                      = tolerance
        )
        self._init_beta_mu(beta_mu=beta_mu, log_z =log_z,  z=z,   tolerance=tolerance)
        self._init_beta_h (beta_h =beta_h,  log_zh=log_zh, zh=zh, tolerance=tolerance)

    # Now we need to initialize each combination of parameters.
    # We start with perhaps the more annoying.
    def _init_aspect_ratio(self, *, aspect_ratio, deBroglie_by_scattering_length, beta_binding_energy, tolerance):
        if beta_binding_energy and (beta_binding_energy >= 0):
            raise ValueError('You specified a non-negative {beta_binding_energy=}.')
            
        # To ensure consistency we need to be prepared for the user to give any possible combination of parameters.
        # So we just do the dumbest thing and do a bunch of case analysis.  With 3 parameters there are 3 possible comparisons.
        if aspect_ratio is not None:
            # Assuming one parameter is provided, check the provision and consistency of another.
            if deBroglie_by_scattering_length is not None and ((2 * torch.pi)**3 * aspect_ratio - deBroglie_by_scattering_length**2).abs() > tolerance:
                raise ValueError(f'You specified both {aspect_ratio=} and {deBroglie_by_scattering_length=} but they are incompatible.')
            # and again
            if beta_binding_energy is not None  and ((2 * torch.pi)**2 * aspect_ratio + beta_binding_energy).abs() > tolerance:
                raise ValueError(f'You specified both {aspect_ratio=} and {beta_binding_energy=} but they are incompatible.')

            # If everything is consistent, make a particular choice of assignment.
            # Another possibility is to make assignments directly based on the values from the user.
            # This is an implementation detail (although I suppose if you wanted to do backprop you might disagree).
            # Anyway, no promises about how the assignments work..
            self.aspect_ratio                   = aspect_ratio
            self.deBroglie_by_scattering_length = ((2 * torch.pi)**3 * aspect_ratio).sqrt()
            self.beta_binding_energy            = - (2 * torch.pi)**2 * aspect_ratio

        # Now we know that the aspect_ratio is not provided, so we need fewer consistency checks.
        elif deBroglie_by_scattering_length is not None:
            
            if beta_binding_energy is not None and (deBroglie_by_scattering_length**2 + 2*torch.pi*beta_binding_energy).abs() > tolerance:
                raise ValueError(f'You specified both {deBroglie_by_scattering_length=} and {beta_binding_energy=} but they are incompatible.')

            self.aspect_ratio                   = deBroglie_by_scattering_length**2 / (2*torch.pi)**3
            self.deBroglie_by_scattering_length = deBroglie_by_scattering_length
            self.beta_binding_energy            = - deBroglie_by_scattering_length**2 / (2*torch.pi)
            
        # Now we know that neither the aspect_ratio nor the deBroglie_by_scattering_length were provided, so no checks are possible.
        elif beta_binding_energy is not None:
            
            self.aspect_ratio                   = - beta_binding_energy / (2 * torch.pi)**2
            self.deBroglie_by_scattering_length = (- 2 * torch.pi * beta_binding_energy).sqrt()
            self.beta_binding_energy            = beta_binding_energy

            
        else:
            raise ValueError('You must specify at least one of aspect_ratio, deBroglie_by_scattering_length, or beta_binding_energy')

    # The other two initialization routines follow similar logic.
    def _init_beta_mu(self, *, beta_mu, log_z, z, tolerance):
        
        if z and (z <= 0):
            raise ValueError('You specified a non-positive {z=}.')
        
        if beta_mu is not None:
            
            if log_z and not (beta_mu - log_z).abs() < tolerance:
                raise ValueError(f'You specified both {beta_mu=} and {log_z=} but they are incompatible.')

            if z and not (z.log() - beta_mu).abs() < tolerance:
                raise ValueError(f'You specified both {beta_mu=} and {z=} but they are incompatible.')

            self.beta_mu = beta_mu
            self.log_z = beta_mu
            self.z = beta_mu.exp()
            
        elif log_z is not None:
            
            if log_z and not ((z.log() - log_z).abs() < tolerance):
                raise ValueError(f'You specified both {log_z=} and {z=} but they are incompatible.')

            self.beta_mu = log_z
            self.log_z   = log_z            
            self.z       = log_z.exp()
            
        elif z is not None:
            
            self.beta_mu = z.log()
            self.log_z   = z.log()
            self.z       = z
            
        else:
            raise ValueError('You must specify at least one of beta_mu, log_z, or z')
            
    def _init_beta_h(self, *, beta_h, log_zh, zh, tolerance):
        
        if beta_h is not None:
            
            if log_zh is not None and not ((beta_h - log_zh).abs() < tolerance).all():
                raise ValueError(f'You specified both {beta_h=} and {log_zh=} but they are incompatible.')
                
            if zh is not None and not ((beta_h - zh.log()).abs() < tolerance).all():
                raise ValueError(f'You specified both {beta_h=} and {zh=} but they are incompatible.')
                
            self.beta_h = beta_h
            self.log_zh = beta_h
            self.zh     = beta_h.exp()
            
        elif log_zh is not None:
            
            if zh is not None and not ((log_zh - zh.log()).abs() < tolerance).all():
                raise ValueError(f'You specified both {log_zh=} and {zh=} but they are incompatible.')
            
            self.beta_h = log_zh
            self.log_zh = log_zh
            self.zh     = log_zh.exp()
            
        elif zh is not None:
            
            self.beta_h = zh.log()
            self.log_zh = zh.log()
            self.zh     = zh
        
        else:
            raise ValueError('You must specify at least one of beta_h, log_zh, or zh')

    def __str__(self):
        
        return (    f'Thermodynamic(aspect_ratio={self.aspect_ratio}, deBroglie_by_scattering_length={self.deBroglie_by_scattering_length}, '
                    f'beta_binding_energy={self.beta_binding_energy}, beta_mu={self.beta_mu}, log_z={self.log_z}, z={self.z}, '
                    f'beta_h={self.beta_h}, log_zh={self.log_zh}, zh={self.zh}'
                    ')'
               )
    
    def FiniteVolume(self, *, beta_tilde=None, a_tilde=None, mu_tilde=None, h_tilde=None, tolerance=_thermodynamic_tolerance):
        r'''
        The point is that any of the "bare" tilded parameters require L and therefore in combination with the L-independent thermodynamic parameters totally fixes the finite-volume physics.

        You can specify any single finite-volume parameter, or any combination, which will be checked for consistency up to the provided tolerance.
        If any consistency checks fail, raise a ValueError.

        Parameters
        ----------
            beta_tilde: torch.tensor
                The dimensionless temperature :math:`\tilde{\beta} = \beta / ML^2`.
            a_tilde: torch.tensor
                The dimensionless scattering ratio :math:`\tilde{a} = 2\pi a / L`.
            mu_tilde: torch.tensor
                The dimensionless chemical potential :math:`\tilde{\mu} = ML^2 \mu`.
            h_tilde: torch.tensor
                The dimensionless external field :math:`\tilde{h} = ML^2 h`.
            tolerance: float
                Strictness of consistency when redundant information is provided.

        Returns
        -------
            tdg.parameters.FiniteVolume
        '''
        
        if (beta_tilde is not None):
            if (a_tilde is not None) and (self.aspect_ratio - beta_tilde / a_tilde**2).abs() > tolerance:
                raise ValueError(f'You specified both {beta_tilde=} and {a_tilde=} but they are incompatible with the thermodynamic aspect_ratio={self.aspect_ratio}.')
            if (mu_tilde is not None) and (self.beta_mu - beta_tilde * mu_tilde).abs() > tolerance:
                raise ValueError(f'You specified both {beta_tilde=} and {mu_tilde=} but they are incompatible with the thermodynamic beta_mu={self.beta_mu}.')
            if (h_tilde is not None) and ((self.beta_h - beta_tilde * h_tilde).abs() > tolerance).any():
                raise ValueError(f'You specified both {beta_tilde=} and {h_tilde=} but they are incompatible with the thermodynamic beta_h={self.beta_h}.')

            return FiniteVolume(beta_tilde, (beta_tilde/self.aspect_ratio).sqrt(), self.beta_mu / beta_tilde, self.beta_h / beta_tilde)
            
        # beta_tilde is None!
        if (a_tilde is not None):
            return self.FiniteVolume(
                beta_tilde = self.aspect_ratio * a_tilde**2,
                a_tilde    = a_tilde,
                mu_tilde   = mu_tilde,
                h_tilde    = h_tilde,
                tolerance  = tolerance)
        
        # a_tilde is None!
        if (mu_tilde is not None):
            return self.FiniteVolume(
                beta_tilde = self.beta_mu / mu_tilde,
                a_tilde    = a_tilde,
                mu_tilde   = mu_tilde,
                h_tilde    = h_tilde,
                tolerance  = tolerance)
        
        # mu_tilde is None!
        if (h_tilde is not None):
            
            ratio = self.beta_h / h_tilde
            # Don't have to check that the ratio is consistent; that will be caught by the thermodynamic check above.
            
            return self.FiniteVolume(
                beta_tilde = ratio[0],
                a_tilde    = a_tilde,
                mu_tilde   = mu_tilde,
                h_tilde    = h_tilde,
                tolerance  = tolerance)

class FiniteVolume:
    r'''
    Parameters
    ----------
        beta_tilde: torch.tensor
            The dimensionless temperature :math:`\tilde{\beta} = \beta / ML^2`.
        a_tilde: torch.tensor
            The dimensionless scattering ratio :math:`\tilde{a} = 2\pi a / L`.
        mu_tilde: torch.tensor
            The dimensionless chemical potential :math:`\tilde{\mu} = ML^2 \mu`.
        h_tilde: torch.tensor
            The dimensionless external field :math:`\tilde{h} = ML^2 h`.
    '''
    def __init__(self, beta_tilde, a_tilde, mu_tilde, h_tilde):
        
        self.beta_tilde = beta_tilde
        self.a_tilde    = a_tilde
        self.mu_tilde   = mu_tilde
        self.h_tilde    = h_tilde
        
        self.Thermodynamic = Thermodynamic(
            aspect_ratio = beta_tilde / a_tilde**2,
            beta_mu      = beta_tilde * mu_tilde,
            beta_h       = beta_tilde * h_tilde
        )
        r'''
        A :class:`tdg.parameters.Thermodynamic` object which has thermodynamic parameters corresponding to these finite-volume parameters.
        '''

    def __str__(self):

        return f'FiniteVolume(β̃={self.beta_tilde}, ã={self.a_tilde}, µ̃={self.mu_tilde}, h̃={self.h_tilde})'
        
    def Action(self, nt, lattice):
        r'''
        Of course, this is the whole point!  Given the finite-volume parameters and a specification of the discretization we have enough to construct a :class:`~.Action`.

        Parameters
        ----------
            nt: int
                Number of time slices.
            lattice: tdg.Lattice
                The spatial discretization.

        Returns
        -------
            :class:`~.Action` constructed from :math:`\tilde{\beta}`, :math:`\tilde{a}`, :math:`\tilde{\mu}`, and :math:`\tilde{h}` and the provided discretization.
        '''
        ere = tdg.EffectiveRangeExpansion(torch.tensor([self.a_tilde]))
        return tdg.AnalyticTuning(ere, lattice).Action(nt, self.beta_tilde, self.mu_tilde, self.h_tilde)

    def scale_L(self, scale):
        r'''We can scale L without needing to know what it is.  If we want to send :math:`L \rightarrow \texttt{scale} L` then we can replace

        .. math::
           \begin{align}
           \tilde{\beta} &\rightarrow \tilde{\beta} / \texttt{scale}^2      \\
           \tilde{a}     &\rightarrow \tilde{a}     / \texttt{scale}        \\
           \tilde{\mu}   &\rightarrow \tilde{\mu}   * \texttt{scale}^2      \\
           \tilde{h}     &\rightarrow \tilde{h}     * \texttt{scale}^2      
           \end{align}

        Parameters
        ----------
            scale: torch.tensor
                How to scale L

        Returns
        -------
            tdg.parameters.FiniteVolume with the rescaled parameters.
        '''
        return FiniteVolume(
            self.beta_tilde / scale**2,
            self.a_tilde    / scale,
            self.mu_tilde   * scale**2,
            self.h_tilde    * scale**2
        )
    
    def scale_beta(self, scale):
        r'''
        Parameters
        ----------
            scale: torch.tensor
                How to scale :math:`\beta`

        Returns
        -------
            tdg.parameters.FiniteVolume with the rescaled parameters.
        '''
        
        return FiniteVolume(
            self.beta_tilde * scale,
            self.a_tilde,
            self.mu_tilde,
            self.h_tilde
        )
    
    def scale_mu(self, scale):
        r'''
        Parameters
        ----------
            scale: torch.tensor
                How to scale :math:`\mu`

        Returns
        -------
            tdg.parameters.FiniteVolume with the rescaled parameters.
        '''
        
        return FiniteVolume(
            self.beta_tilde,
            self.a_tilde,
            self.mu_tilde * scale,
            self.h_tilde
        )
    
    def scale_h(self, scale):
        r'''
        Parameters
        ----------
            scale: torch.tensor
                How to scale :math:`h`

        Returns
        -------
            tdg.parameters.FiniteVolume with the rescaled parameters.
        '''
        
        return FiniteVolume(
            self.beta_tilde,
            self.a_tilde,
            self.mu_tilde,
            self.h_tilde * scale
        )
    
        

class Dimensionful:
    r'''
    All of the parameters must be one consistent system of units.
    For example, we could do
    
    .. code-block:: python

       from tdg.units import eV, µm, nK, Mass
       Dimensionful(Mass["6Li"], 250*nK, -500*nK, 0.25*µm ).FiniteVolume(2*µm)

    Parameters
    ----------
        M: torch.tensor
            Dimensionful mass.
        T: torch.tensor
            Dimensionful temperature.
        mu: torch.tensor
            Dimensionful chemical potential.
        a: torch.tensor
            Dimensionful scattering length.
        h: torch.tensor
            Dimensionful external field.
    '''

    def __init__(self, M, T, mu, a, h=torch.tensor([0.,0.,0.])):
        
        self.M = M
        self.T = T
        self.mu = mu
        self.a = a
        self.h = h
        

    def Thermodynamic(self):
        r'''
        A :class:`tdg.parameters.Thermodynamic` built from the dimensionful parameters.
        '''
        return Thermodynamic(
            beta_binding_energy = - 1./(self.M*self.a**2 * self.T),
            beta_mu = self.mu/self.T,
            beta_h  = self.h /self.T
        )

    def FiniteVolume(self, L):
        r'''
        A :class:`tdg.parameters.FiniteVolume` built from the dimensionful parameters and the dimensionful L (in the same system of units).
        '''
        beta_tilde = 1. / (self.T * self.M * L**2)
        return self.Thermodynamic().FiniteVolume(beta_tilde = beta_tilde)
