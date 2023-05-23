#!/usr/bin/env python3

import torch
import pandas as pd
import numpy as np
import scipy.optimize, scipy.stats

from tdg import _no_op
from tdg.h5 import H5able
import tdg.ensemble
from collections import deque

import warnings
import logging
logger = logging.getLogger(__name__)

r'''

HMC is an importance-sampling algorithm. 

'''

class Hamiltonian(H5able):
    r"""The HMC *Hamiltonian* for a given action :math:`S` is
    
    .. math::
        \mathcal{H} = \frac{1}{2} p M^{-1} p + S(x)
    
    which has the standard non-relativistic kinetic energy and a potential energy given by the action.
    
    An HMC Hamiltonian serves two purposes:
    
    * The first is to draw momenta and evaluate the starting energy, and to do accept/reject according to the final energy.
      For this purpose Hamiltonians are callable.
    * The second is to help make a proposal to consider in the first place.  To do this we start with a position and momentum pair and
      integrate Hamilton's equations of motion,

    .. math::
            \begin{aligned}
                \frac{dx}{d\tau} &= + \frac{\partial \mathcal{H}}{\partial p}
                &
                \frac{dp}{d\tau} &= - \frac{\partial \mathcal{H}}{\partial x}
            \end{aligned}

    Of course, these two applications are closely related, and in 'normal' circumstances we use the same Hamiltonian for both purposes.
    """

    def __init__(self, S):

        self.V = S

        # The kinetic piece could be 1/2 (p M^-1 p) for an arbitrary mass matrix.
        # A generic M was supported through 1dce9a4 but the simplification M=1 keeps things simple and
        # allows us to avoid constructing unpicklable lambdas that allow us to write an HMC Hamiltonian to_h5.

    def __call__(self, x, p):
        r"""

        Parameters
        ----------
            x:  torch.tensor
                a configuration compatible with the Hamiltonian
            p:  torch.tensor
                a momentum of the same shape

        Returns
        -------
            torch.tensor:
                :math:`\mathcal{H}` for the given ``x`` and ``p``.
                
        """
        return self.T(p) + self.V(x)

    def T(self, p):
        return torch.sum(p*p)/2

    def velocity(self, p):
        r"""
        The velocity is needed to update the positions in Hamilton's equations of motions.

        .. math::
            \texttt{velocity(p)} = \left.\frac{\partial \mathcal{H}}{\partial p}\right|_p
        """
        return p

    def force(self, x):
        r"""
        The force is needed to update the momenta in Hamilton's equations of motions.

        .. math::
            \texttt{force(x)} = \left.-\frac{\partial \mathcal{H}}{\partial x}\right|_x
        """
        grad, = torch.autograd.grad(self.V(x), x)
        return -grad

class MarkovChain(H5able):
    r"""
    The HMC algorithm for updating an initial configuration :math:`x_i` goes as follows:

    #.  Draw a sample momentum :math:`p_i` from the gaussian distribution given by the kinetic piece of the Hamiltonian.
    #.  Calculate :math:`\mathcal{H}` for the given :math:`x_i` and drawn :math:`p_i`.
    #.  Integrate Hamilton's equations of motion to generate a proposal for a new :math:`x_f` and :math:`p_f`.
    #.  Accept the proposal according to :math:`\exp[-(\Delta\mathcal{H} = \mathcal{H}_f - \mathcal{H}_i)]`
    """
    def __init__(self, H, integrator):
        """
        H is used for accept-reject
        integrator is used for the molecular dynamics and need not use the same H.
        """
        self.H = H
        
        self.refresh_momentum = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(H.V.Spacetime.sites), 
            torch.eye(H.V.Spacetime.sites), 
        )
        
        self.integrator = integrator
        
        # And we do accept/reject sampling
        self.metropolis_hastings = torch.distributions.uniform.Uniform(0,1)
        
        self.steps = 0
        self.accepted = 0
        self.rejected = 0
        self.dH = deque()
        self.acceptance_probability = deque()
        
    def step(self, x):
        r"""

        Parameters
        ----------
            x:  torch.tensor
                a configuration compatible with the Hamiltonian and integrator.

        Returns
        -------
            torch.tensor:
                a similar configuration; a new configuration if the proposal was accepted, or the original if the proposal is rejected.
                
            jacobian:
                The Jacobian of the integration.  Currently hard-coded to 1, but may be needed for other integration schemes.
        """
        p_i = self.refresh_momentum.sample().reshape(*x.shape).requires_grad_(True)
        x_i = x.clone().requires_grad_(True)
        
        H_i = self.H(x_i,p_i)
        
        x_f, p_f = self.integrator(x_i, p_i)
        
        H_f = self.H(x_f,p_f)
        dH = (H_f - H_i).detach()

        if dH.isnan():
            raise ValueError('HMC energy change is NaN.  {H_i=} {H_f=}')

        acceptance_probability = torch.exp(-dH.real).clamp(max=1)
        accept = (acceptance_probability > self.metropolis_hastings.sample())

        self.dH.append(dH.cpu())
        self.acceptance_probability.append(acceptance_probability.clone().detach().cpu())

        logger.info(f'HMC proposal {"accepted" if accept else "rejected"} with dH={dH.real.cpu().detach().numpy():+} acceptance_probability={acceptance_probability.cpu().detach().numpy()}')

        self.steps += 1
        if accept:
            self.accepted += 1
            return x_f, 1.
        else:
            self.rejected += 1
            return x_i, 1.

class LeapFrog(H5able):
    r"""The LeapFrog integrator integrates Hamilton's equations of motion for a total of :math:`\tau` molecular dynamics time `md_time` in a reversible, symplectic way.
    
    It discretizes :math:`\tau` into `md_steps` steps of :math:`d\tau` and then uses simple finite-differencing.

    One step of :math:`d\tau` integration is accomplished by

    #. updating the coordinates by :math:`d\tau/2`
    #. updating the momenta by :math:`d\tau`
    #. updating the coordinates by :math:`d\tau/2`.

    However, if the number of steps is more than 1 the trailing half-step coordinate update from one step is combined with the leading half-step coordinate update from the next.
    """
 
    def __init__(self, H, md_steps, md_time=1):
        self.H = H
        self.md_time  = md_time
        self.md_steps = md_steps
        self.md_dt    = self.md_time / self.md_steps
        
    def __str__(self):
        return f'LeapFrog(H, md_steps={self.md_steps}, md_time={self.md_time})'

    def integrate(self, x_i, p_i):
        r"""Integrate an initial position and momentum.

        Parameters
        ----------
            x_i:    torch.tensor
                    a tensor of positions
            p_i:    torch.tensor
                    a tensor of momenta

        Returns
        -------
            x_f:    torch.tensor
                    a tensor of positions,
            p_f:    torch.tensor
                    a tensor of momenta

        """
        
        # Take an initial half-step of the coordinates
        x = x_i + self.H.velocity(p_i) * self.md_dt / 2
        
        # do the initial momentum update,
        p = p_i + self.H.force(x) * self.md_dt
        
        # Now do whole-dt coordinate AND momentum updates
        for md_step in range(self.md_steps-1):
            x = x + self.H.velocity(p) * self.md_dt
            p = p + self.H.force(x) * self.md_dt
            
        # Take a final half-step of coordinates
        x = x + self.H.velocity(p) * self.md_dt / 2

        return x, p
    
    def __call__(self, x_i, p_i):
        '''
        Forwards to ``integrate``.
        '''
        return self.integrate(x_i, p_i)

class Omelyan(H5able):
    r"""
    The Omelyan integrator is a second-order integrator which integrates Hamilton's equations of motion for a total of :math:`\tau` molecular dynamics time `md_time` in a reversible, symplectic way.

    It discretizes :math:`\tau` into `md_steps` steps of :math:`d\tau` and given :math:`0\leq\zeta\leq 0.5` applies the following integration scheme:

    #. Update the coordinates by :math:`\zeta\;d\tau`,
    #. update the momenta by :math:`\frac{1}{2}\; d\tau`,
    #. update the coordinates by :math:`(1-2\zeta)\;d\tau`,
    #. update the momenta by :math:`\frac{1}{2}\; d\tau`,
    #. update the coordinates by :math:`\zeta\;d\tau`.

    However, if the number of steps is more than 1 the trailing coordinate update from one step is combined with the leading coordinate update from the next.

    If nothing is known about the structure of the potential, the :math:`h^3` errors are minimized when :math:`\zeta \approx 0.193` :cite:`PhysRevE.65.056706`.

    When :math:`\zeta=0` this reproduces the momentum-first leapfrog; when :math:`\zeta=0.5` it reproduces the position-first LeapFrog.
    """

    def __init__(self, H, md_steps, md_time=1, zeta=0.193):
        self.H = H
        self.md_time  = md_time
        self.md_steps = md_steps
        self.md_dt    = self.md_time / self.md_steps
        self.zeta     = zeta

        if (zeta < 0) or (0.5 < zeta):
            raise ValueError("Second-order integrators need 0 <= zeta <= 0.5 for any hope of improvement over LeapFrog.")

    def __str__(self):
        return f'Omelyan(H, md_steps={self.md_steps}, md_time={self.md_time}, zeta={self.zeta})'

    def integrate(self, x_i, p_i):
        r"""Integrate an initial position and momentum.

        Parameters
        ----------
            x_i:    torch.tensor
                    a tensor of positions
            p_i:    torch.tensor
                    a tensor of momenta

        Returns
        -------
            x_f:    torch.tensor
                    a tensor of positions,
            p_f:    torch.tensor
                    a tensor of momenta

        """
        # Take an initial zeta-step of the coordinates
        x = x_i + self.H.velocity(p_i) * (self.zeta * self.md_dt)
        
        # do the initial half-step of momentum update,
        p = p_i + self.H.force(x) * (self.md_dt / 2)
        

        # Now do whole-dt coordinate AND momentum updates
        for md_step in range(self.md_steps-1):
            x = x + self.H.velocity(p) * ((1-2*self.zeta) * self.md_dt)
            p = p + self.H.force(x) * (self.md_dt / 2)
            x = x + self.H.velocity(p) * (2 * self.zeta * self.md_dt)
            p = p + self.H.force(x) * (self.md_dt / 2)

        # do the middle coordinate step of (1-2zeta)
        x = x + self.H.velocity(p) * ((1-2 * self.zeta) * self.md_dt)

        # do the final half-step of momentum integration,
        p = p + self.H.force(x) * (self.md_dt / 2)

         # take a final coordinate zeta-step
        x = x + self.H.velocity(p) * (self.zeta * self.md_dt)

        return x, p
    
    def __call__(self, x_i, p_i):
        '''
        Forwards to ``integrate``.
        '''
        return self.integrate(x_i, p_i)

class Autotuner(H5able):
    r'''
    An Autotuner seeks for a molecular dynamics discretization, targeting a given acceptance rate.
    Based on the simple ideas explained in Reference :cite:`Krieg:2018pqh`.

    Parameters
    ----------
        HMC_Hamiltonian: tdg.HMC.Hamiltonian
            The HMC_Hamiltonian is used for molecular dynamics integration and accept/reject.
            In principle these can be different, but that is not implemented here.

        integrator: a molecular dynamics integrator
            Could be tdg.HMC.LeapFrog, tdg.HMC.Omelyan, or another integrator
            which just has one molecular dynamics discretization scale and is constructed via
            ``integrator(HMC_Hamiltonian, md_steps, md_time)`` where a good value for md_steps is what we seek.

        cfgs_per_estimate: int
            How many trajectories are needed get a good estimate of the acceptance rate?
            Reference :cite:`Krieg:2018pqh` found that (for the Hubbard model on the honeycomb lattice, but for very large lattices near the continuum limit)
            as few as 5 could produce a good estimate, while they never needed more than about 30.
    '''

    def __init__(self, HMC_Hamiltonian, integrator, *, cfgs_per_estimate=5, _bootstrap_resamples=100):

        self.H = HMC_Hamiltonian
        self.I = integrator
        self.cfgs_per_estimate = cfgs_per_estimate

        self._bootstraps = _bootstrap_resamples

        self.measurements = pd.DataFrame(data=None, columns=('md_steps', 'target', 'dH', 'acceptance', '<acc>', 'd<acc>', 'fit', 'prediction', 'N_bosonic'))
        self.summary = pd.DataFrame(data=None, columns=('md_steps', '<acc>', 'd<acc>'))
        self._fit = None

    def target(self, target_acceptance=0.75, start='hot', md_time=1., starting_md_steps=20, min_md_steps=1, progress=_no_op):
        '''
        Runs the autotuning process to achieve a target acceptance rate.

        The autotuning process seeks to adjust the number of molecular dynamics steps per trajectory to achieve a
        specific acceptance rate (target_acceptance). It iteratively computes trajectories with different discretizations until
        the acceptance rate meets the target. The autotuning stops when either the target acceptance rate is achieved with some confidence
        or the discretization cannot be shrunk further without falling below the target.

        Parameters
        ----------
            target_acceptance : float
                The desired acceptance rate to achieve.
                If the acceptance rate depends very sharply on the number of molecular dynamics steps, there may be no choice which
                achieves the target.  In that case a finer molecular dynamics integration will be picked.
            start : str, optional
                The starting condition for the autotuning process.  Must be understood by :meth:`~.GrandCanonical.generate`.
            md_time : float
                The time duration for each molecular dynamics trajectory.
            starting_md_steps : int
                The initial number of molecular dynamics steps per trajectory.
            min_md_steps : int
                The minimum number of molecular dynamics steps allowed. Default is 1.
            progress : callable
                A callback function like `tqdm`_ that can be used to track HMC evolution.

        Returns
        -------
            integrator : molecular dynamics integrator
                An integrator object constructed with the molecular dynamics discretization tuned to the target acceptance.
            configuration: torch.tensor
                The final configuration resulting from all the trajectories accepted during the tuning process, which can be used in subsequent HMC.

        .. _tqdm: https://pypi.org/project/tqdm/
        '''
        self.md_steps = starting_md_steps
        self.md_steps_min = min_md_steps

        logger.info(f'Tuning to {target_acceptance=} starting with {self.md_steps} MD steps.')

        while (
            (not (self.measurements['md_steps']==self.md_steps).any()) or
            ((self.summary['<acc>']-self.summary['d<acc>']).loc[self.summary['md_steps']==self.md_steps]
             < target_acceptance).all()
        ):

            logger.info(f'Computing trajectories with {self.md_steps} molecular dynamics steps per trajectory...')
            start = self._step(self.md_steps, start=start, md_time=md_time, progress=progress)
            self.measurements['target'].iloc[-1] = target_acceptance
            self.md_steps = self._predict(target_acceptance)

        integrator = self.I(self.H, self.md_steps, md_time=md_time)
        integrator.Autotuner = self

        start = self._latest_ensemble.configurations[-1].clone().detach()
        del self._latest_ensemble # To avoid writing the ensemble to disk.

        return integrator, start

    def plot_history(self, ax, **kwargs):
        '''
        This method plots the acceptance probabilities over the course of the autotuning process. The acceptance probabilities
        are shown for each trajectory, with the x-axis representing the trajectory index and the y-axis representing the
        acceptance probability. The plot allows visualizing the change in acceptance probabilities over molecular dynamics time.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to plot the history.
        **kwargs
            Additional keyword arguments to customize the plot appearance.  Applied to the trace for every tuning step.
        '''

        ax.set_ylim([-0.05, 1.1])

        step = 0
        for key, series in self.measurements.iterrows():
            y = series['acc']
            n = len(y)
            x = step + np.arange(n)
            ax.plot(x, y, **kwargs)
            ax.text(x.mean(), 1.05, f'{series["md_steps"]}', ha='center', va='center')
            step += n

        ax.set_xlabel('Trajectory')
        ax.set_ylabel('Acceptance probability')

    def plot_models(self, ax):
        '''
        Plots the evolution of acceptance probability models with autotuning.
        Also shows, as points with error bars, the final estimated acceptance rates.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to plot the acceptance probability models.

        '''


        x=np.linspace(0,1.1*max(self.summary['md_steps']),1000)

        # An alternate way of showing the target acceptance:
        #
        # if (self.measurements['target'] == self.measurements['target'][0]).all():
        #     ax.axhline(self.measurements['target'][0], zorder=-1, color='black')

        # This shows how to go from the prediction to the next md_step evaluation.
        for (k1, s1), (k2, s2) in zip(self.measurements.iloc[:-1].iterrows(), self.measurements.iloc[1:].iterrows()):
            trace = ax.plot(x, scipy.stats.gamma.cdf(x, *s1['fit']), label=s1['md_steps'])
            color = trace[0].get_color()
            ax.plot(
                (s1['prediction'],np.ceil(s1['prediction']),np.ceil(s1['prediction'])),
                (s1['target'],s1['target'], s2['<acc>']),
                color=color, linestyle=':',
                zorder=100+k1
            )

            ax.errorbar((s1['md_steps'],), (s1['<acc>'],), yerr=(s1['d<acc>'],), color=color)
            ax.plot((0.2+s1['md_steps'],), (s1['<acc>'],), color=color, marker='<')

        # but because it's two offset copies zipped, we still need to visualize the final measurement.
        s1 = self.measurements.iloc[-1]
        trace = ax.plot(x, scipy.stats.gamma.cdf(x, *s1['fit']), label=s1['md_steps'])
        color = trace[0].get_color()
        ax.errorbar((s1['md_steps'],), (s1['<acc>'],), yerr=(s1['d<acc>'],), color=color)
        ax.plot((0.2+s1['md_steps'],), (s1['<acc>'],), color=color, marker='<')

        ax.set_xlabel('Molecular dynamics steps')
        ax.set_xticks(self.summary['md_steps'])
        ax.set_ylabel('Acceptance probability')

    def _step(self, md_steps, *, md_time=1, start, progress=_no_op):
        '''
        This method performs a single step of the autotuning process with a specific number of molecular dynamics steps (md_steps).
        It initializes the molecular dynamics integrator and uses it for HMC evolution.  It evaluates acceptance rates,
        and updates the internal measurements and summary.

        Parameters
        ----------
        md_steps : int
            The number of molecular dynamics steps per trajectory for the current step.
        md_time : int, optional
            The time duration for each molecular dynamics trajectory.
        start : str
            The starting condition for the molecular dynamics trajectories.  Must be understood by :func:`~.GrandCanonical.generate`.
        progress : callable, optional
            A callback function like tqdm that can be used to track HMC evolution.

        Returns
        -------
        torch.Tensor
            The last configuration from the ensemble after running the trajectory.

        '''
        integrator = self.I(self.H, md_steps, md_time)
        hmc = MarkovChain(self.H, integrator)

        self._latest_ensemble = tdg.ensemble.GrandCanonical(self.H.V).generate(self.cfgs_per_estimate, hmc, start=start, progress=progress)

        dH = torch.tensor(hmc.dH)

        this_step = pd.Series({
            'md_steps':   md_steps,
            'dH':         dH.cpu().clone().detach().numpy(),
            'acceptance': hmc.accepted / hmc.steps,
            # We calculate N_bosonic specifically because it's already simple sums.
            # Anything else requires the propagator, which we don't want in this loop.
            'N_bosonic':  self._latest_ensemble.N_bosonic.cpu().clone().detach().numpy().real,
        })
        this_step['acc'] = torch.exp(-dH.real).clip(max=1).cpu().clone().detach().numpy()
        this_step['<acc>'] = this_step['acc'].mean()

        self.measurements = pd.concat((self.measurements, this_step.to_frame().transpose()), axis=0, ignore_index=True)

        return self._latest_ensemble.configurations[-1]


    def _predict(self, target_acceptance=0.75, continuum=1000, maxfev=10000, uncertainty_floor=0.02):
        '''
        Given all previous HMC steps, estimates the acceptance rate for every tried molecular dynamics discretization and
        fits a model to try and predict the number of molecular dynamics steps required to achieve the target acceptance rate.

        Parameters
        ----------
        target_acceptance : float
            The desired target acceptance rate.
        continuum : int
            The number of steps in the continuum for fitting the acceptance probabilities.
        maxfev : int
            The maximum number of function evaluations for the curve fitting optimization.

        Returns
        -------
        int
            The predicted number of molecular dynamics steps to achieve the target acceptance rate.

        .. note::

            This method uses a bootstrap estimator to estimate acceptance probabilities,
            which is a gross simplification since we know the acceptance probabilities are in [0,1].

            It uses the gamma distribution to model the estimated acceptance rates.
        '''
        # Based on
        #
        # Krieg, Luu, Ostmeyer, Papaphilippou, Urbach
        #     Comput.Phys.Commun. 236 (2019) 15-25 10.1016/j.cpc.2018.10.008 1804.07195
        # Accelerating Hybrid Monte Carlo simulations of the Hubbard model on the hexagonal lattice

        points = deque()

        for (md_steps, rows) in self.measurements.groupby('md_steps'):
            # A simple bootstrap estimator for a very naive uncertainty on the acceptance probability:
            acc=np.concatenate(rows['acc'].values)
            boot=acc[torch.randint(0, len(acc), (self._bootstraps, len(acc))).cpu()].mean(axis=1)
            points.append(pd.Series({'md_steps': md_steps, '<acc>': boot.mean(), 'd<acc>': boot.std()}))

        self.summary = pd.DataFrame(points)

        points.append(pd.Series({'md_steps': 0, '<acc>': 0., 'd<acc>': 0.01}))
        if len(points) < 3:
            # Add fictitious point to stabilize the fit when there have been too few iterations.
            points.append(pd.Series({'md_steps': continuum, '<acc>': 1., 'd<acc>': uncertainty_floor}))

        points = pd.DataFrame(points)

        most_recent = points[points['md_steps'] == self.md_steps].iloc[0]

        self.measurements.iloc[-1]['<acc>'] = most_recent['<acc>']
        self.measurements.iloc[-1]['d<acc>'] = most_recent['d<acc>']

        # The above reference fits estimated acceptance rates to the CDF of a skew normal distribution, which is currently not available in torch.
        # I searched for other similar-looking CDFs and opted for the gamma distribution
        #     https://pytorch.org/docs/stable/distributions.html#gamma
        # but remarkably, even though https://github.com/pytorch/pytorch/issues/41637 is closed the distribution still does not have a cdf method!
        # However, a comment there suggested trying torch.special.{gammainc,gammaincc}.
        # That function is not differentiable, at least in the distribution parameters.  So it is not fittable using iterative torch back-prop methods.
        # Annoying.
        #   [
        #       Actually, the issue has been reopened and is available in torch 2.0.0.
        #       However, it is still not differentiable wrt the non-scale parameters, as this involves MeijerG
        #   ]
        #
        # Instead, I fell back on to scipy for fitting.  Oh well!
        # TODO: keep an eye on the availability of differentiable / fittable SkewNormal or Gamma distributions.

        # a, loc, beta
        previous = self._fit if self._fit is not None else np.array((1.0, 0.0, 1.0))

        # We relax the bootstrap errors if they're too small for fit flexibility.
        points['d<acc>'] = points['d<acc>'].clip(lower=uncertainty_floor)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=scipy.optimize.OptimizeWarning)
            # If the fit errors we want to know, but often the fitter cannot determine the covariance, warning
            # OptimizeWarning: Covariance of the parameters could not be estimated
            # which we do not care about.
            self._fit, cov = scipy.optimize.curve_fit(
                scipy.stats.gamma.cdf,
                points['md_steps'].values, points['<acc>'].values,
                p0=previous,
                sigma=points['d<acc>'].values,
                maxfev=maxfev,
            )

        self.measurements['fit'].iloc[-1] = self._fit
        target = scipy.stats.gamma.ppf(target_acceptance, *self._fit)
        self.measurements['prediction'].iloc[-1] = target
        logger.info(f'The best estimate is that for {target_acceptance=} we should target {target} molecular dynamics steps.')

        return max(self.md_steps_min, int(np.ceil(target)))

