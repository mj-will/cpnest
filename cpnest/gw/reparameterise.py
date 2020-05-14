
import numpy as np
import scipy.stats as stats

fuzz = 1e-3

def angle_to_cartesian(alpha, r=None, scale=1.0):
    """
    Decompose an angle into a real and imaginary part
    """
    rescaled_alpha = alpha * scale
    if r is None:
        r = stats.chi.rvs(2, size=alpha.shape[0])
    x = r * np.cos(rescaled_alpha)
    y = r * np.sin(rescaled_alpha)
    return x, y

def cartesian_to_angle(x, y, scale=1.0):
    """
    Reconstruct an angle given the real and imaginary part
    """
    radius = np.sqrt(x ** 2. + y** 2.)
    angle = (np.arctan2(y, x) + 2. * np.pi) % (2. * np.pi) / scale
    return angle, radius

def sky_to_cartesian(ra, dec, dL=None):
    """
    Decompose an angle into a real and imaginary part
    """
    N = ra.shape[0]
    if dL is None:
        dL = stats.chi.rvs(3, size=N)
    rescaled_ra = ra
    rescaled_dec = dec
    # amplitudes
    x = dL * np.cos(rescaled_dec) * np.cos(rescaled_ra)
    y = dL * np.cos(rescaled_dec) * np.sin(rescaled_ra)
    z = dL * np.sin(rescaled_dec)
    return x, y, z

def cartesian_to_sky(x, y, z):
    """
    Reconstruct an angle given the real and imaginary part
    """
    dL = np.sqrt(np.sum([x **2., y**2., z**2.], axis=0))
    dec = np.arctan2(z, np.sqrt(x ** 2. + y ** 2.0))
    ra = (np.arctan2(y, x) + 2. * np.pi) % (2. * np.pi)
    return ra, dec, dL

def rescale_0_to_1(x, xmin, xmax):
    """
    Rescale a value to 0 to 1
    """
    return (x - xmin ) / (xmax - xmin)

def inverse_rescale_0_to_1(y, xmin, xmax):
    """
    Rescale from 0 to 1 to xmin to xmax
    """
    return (xmax - xmin) * y + xmin

def rescale_minus_one_to_one(x, xmin, xmax):
    """
    Rescale a value to -1 to 1
    """
    return (2. * (x - xmin ) / (xmax - xmin)) - 1

def inverse_rescale_minus_one_to_one(y, xmin, xmax):
    """
    Rescale from -1 to 1 to xmin to xmax
    """
    return (xmax - xmin) * ((y + 1) / 2.) + xmin

def logit(x, xmin=0.0, xmax=1.0):
    """
    Apply the logit
    """
    xmin -= fuzz
    xmax += fuzz
    return np.log(x - xmin) - np.log(xmax - x)

def inverse_logit(y, xmin=0.0, xmax=1.0):
    """
    Apply the inverse logit
    """
    xmin -= fuzz
    xmax += fuzz
    return ((xmax - xmin) / (1 + np.exp(-y))) + xmin

def log_jacobian_inverse_logit(x):
    """
    Log Jacobian for the inverse logit
    """
    return x - 2 * np.log(1. + np.exp(x))

class GWReparam:

    def __init__(self, parameters=None, logit_parameters=[], q_inversion=False, **kwargs):
        super(GWReparam, self).__init__(**kwargs)
        self.parameters = parameters.copy()
        self.re_parameters = parameters.copy()
        self.base_dim = len(self.parameters)
        self.reparam_dim = len(self.parameters)
        self.parameter_indices = False
        # possible reparameterisations
        self.reparameterisations = []
        self.psi = False
        self.psi_scale = 2.0
        self.phase = False
        self.phase_scale = 1.0
        self.ra = False
        self.dec = False
        self.sky_radial = False
        # logit parameters
        self.logit_parameters = logit_parameters
        print('logit parameters:', logit_parameters)
        self.q_inversion = q_inversion
        if self.q_inversion:
            print('Using q inversion')
        self.prior_min = None
        self.prior_max = None

        self._angles = ['phi_12', 'phi_jl']
        self.angle_indices = []
        self.angle_scales = []


    @property
    def extrinsic_parameters(self):
        """Return a list of extrinsic parameters"""
        return ['phase', 'psi', 'ra', 'dec', 'theta_jn', 'cos_theta_jn',
                'luminosity_distance', 'sky_x', 'sky_y', 'sky_z',
                'q_0', 'q_1', 'q_2', 'q_3' ]

    @property
    def intrincsic_parameters(self):
        """Return a list of intrinsic parameters"""
        return ['mass_1', 'mass_2', 'chirp_mass', 'mass_ratio']

    @property
    def source_angles(self):
        """Return source angles that should NOT be use with GWFlow"""
        return ['iota', 'theta_jn', 'cos_theta_jn', 'psi', 'phase']

    @property
    def angles(self):
        """Return a list of angular parameters"""
        return self._angles

    @angles.setter
    def angles(self, angles):
        if not isinstance(angles, list):
            angles = list(angles)
        for a in angles:
            if a not in self._angles:
                self._angles.append(a)
        return self._angles

    def setup_angle(self, name, radial_name=False, scale=1.0):
        """Add an angular parameter to the list of reparameterisations"""
        a = self.parameters.index(name)
        self.defaults.remove(a)
        self.re_parameters[a] = name + '_x'
        if radial_name:
            r = self.parameters.index(radial_name)
            self.defaults.remove(r)
            self.re_parameters[r] = name + '_y'
        else:
            r = self.reparam_dim
            self.reparam_dim += 1
            self.parameters.append(name + '_radial')
            self.re_parameters.append(name + '_y')

        indices = (a, r, radial_name)
        self.angle_indices.append(indices)
        self.angle_scales.append(scale)
        # remove from normalisation
        # add to labels
        self.reparameterisations.append(name)

    def setup_parameter_indices(self):
        """
        Setup indices for the parameters
        """
        print('Setting up GW parameters')
        self.defaults = list(range(self.base_dim))
        self.re_parameters = self.parameters.copy()
        self.reparameterisations = []

        for s in self.source_angles:
            if s in self.parameters:
                raise RuntimeError(f'{s} should NOT be used with GWFlows')

        if all(p in self.parameters for p in['ra', 'dec']):
            self.ra = self.parameters.index('ra')
            self.defaults.remove(self.ra)
            self.dec= self.parameters.index('dec')
            self.defaults.remove(self.dec)
            if 'luminosity_distance' in self.parameters:
                self.dL = self.parameters.index('luminosity_distance')
                self.reparameterisations.append('luminosity_distance')
            self.sky_x = self.ra
            self.re_parameters[self.sky_x] = 'sky_x'
            self.sky_y = self.dec
            self.re_parameters[self.sky_y] = 'sky_y'
            if 'luminosity_distance' in self.reparameterisations:
                self.sky_z = self.dL
                self.re_parameters[self.sky_z] = 'sky_z'
                self.defaults.remove(self.dL)
                self.sky_radial = False                  # for proposal check
            else:
                self.sky_z = self.reparam_dim
                self.re_parameters.append('sky_z')
                self.sky_radial = self.reparam_dim
                self.parameters.append('sky_radial')
                self.reparam_dim += 1
            self.reparameterisations.append('sky')

        self.quaternions = []
        if 'q_0' in self.parameters:
            print('Removing quaternions from normalisation')
            for i in range(4):
                j = self.parameters.index(f'q_{i}')
                self.quaternions.append(j)
                self.defaults.remove(j)

        # setup tilt angles and spin magnitudes if present
        for i in [1, 2]:
            if f'tilt_{i}' in self.parameters:
                if f'a_{i}' in self.parameters:
                    self.setup_angle(f'tilt_{i}', f'a_{i}', scale=2.0)
                else:
                    self.setup_angle(f'tilt_{i}', scale=2.0)

        for a in ['phi_jn', 'phi_12']:
            if a in self.parameters:
                self.setup_angle(a, scale=1.0)

        if 'geocent_time' in self.parameters:
            # TODO: add catch for other time
            # geocent time is handled different to other parameters,
            # we leave it in the defaults and change the prior bounds
            # we then only need to subtract the offset if it present
            self.time = self.parameters.index('geocent_time')
            # set offset as the midpoint of the prior
            # the bounds will then be +/- duration/2
            self.time_offset = self.prior_bounds[self.time, 0] \
                    + np.ptp(self.prior_bounds[self.time]) / 2
            self.prior_bounds[self.time] -= self.time_offset
            print(self.prior_bounds[self.time])
            self.reparameterisations.append('time')
            self.defaults.remove(self.time)

        # mass ratio is last because of relation to other parameters
        if 'mass_ratio' in self.parameters:
            if 'mass_ratio' in self.logit_parameters:
                self.mass_ratio = self.parameters.index('mass_ratio')
                self.mass_ratio_logit = self.mass_ratio
                self.defaults.remove(self.mass_ratio)
                self.re_parameters[self.mass_ratio] = 'mass_ratio_logit'
                self.reparameterisations.append('mass_ratio')
            elif self.q_inversion:
                self.mass_ratio = self.parameters.index('mass_ratio')
                self.defaults.remove(self.mass_ratio)
                self.re_parameters[self.mass_ratio] = 'mass_ratio_inv'
                self.reparameterisations.append('mass_ratio')
                if 'phase' in self.parameters:
                    raise RuntimeError('You should avoid phase, use quaternions!')
                if 'tilt_1' in self.parameters:
                    raise RuntimeError('Flow proposal not implemented for spins')

        self.parameter_indices = True

    def normalise_defaults(self, x):
        """
        Normalise a set of samples to [-1, 1]
        """
        return 2. * ((x - self._prior_min) / (self._prior_max - self._prior_min)) - 1

    def inverse_normalise_defaults(self, x):
        """
        Apply the inverse of the normalisation
        """
        return (self._prior_max - self._prior_min) * ((x + 1) / 2.) + self._prior_min

    def reparameterise(self, samples):
        """
        Reparameterise the samples
        """
        # make sure if using one sample it's shape [1, N]
        if samples.ndim == 1:
            samples = samples[np.newaxis, :]

        if not self.parameter_indices:
            self.setup_parameter_indices()

        samples_out = np.empty([samples.shape[0], self.reparam_dim])
        # rescale 'default' parameters to [-1, 1]
        samples_out[:, self.defaults] = self.normalise_defaults(
                samples[:, self.defaults])

        if 'time' in self.reparameterisations:
            t = samples[:, self.time] - self.time_offset
            samples_out[:, self.time] = rescale_minus_one_to_one(t,
                    self.prior_bounds[self.time, 0], self.prior_bounds[self.time, 1])

        if self.quaternions:
            samples_out[:, self.quaternions] = samples[:, self.quaternions]

        if 'sky' in self.reparameterisations:
            if 'luminosity_distance' in self.reparameterisations:
                dL_rescaled = rescale_0_to_1(samples[:, self.dL],
                        xmin=self.prior_bounds[self.dL, 0],
                        xmax=self.prior_bounds[self.dL, 1])
                x, y, z = sky_to_cartesian(samples[:, self.ra],
                        samples[:,self.dec], dL_rescaled)
            else:
                x, y, z = sky_to_cartesian(samples[:, self.ra],
                        samples[:, self.dec])
            samples_out[:, self.sky_x] = x
            samples_out[:, self.sky_y] = y
            samples_out[:, self.sky_z] = z

        # deal spin angles
        if self.angle_indices:
            for i, s in zip(self.angle_indices, self.angle_scales):
                # if the boolean i[2] is true use radial component
                # else draw from chi with 2 dof
                r = samples[:, i[1]] if i[2] else None
                x, y = angle_to_cartesian(samples[:, i[0]], r=r, scale=s)
                samples_out[:, i[0]] = x
                samples_out[:, i[1]] = y

        if 'mass_ratio' in self.reparameterisations:
            if 'mass_ratio' in self.logit_parameters:
                samples_out[:, self.mass_ratio_logit] = logit(
                        samples[:, self.mass_ratio],
                        xmin=self.prior_bounds[self.mass_ratio, 0],
                        xmax=self.prior_bounds[self.mass_ratio, 1])
            elif self.q_inversion:
                if not samples.shape[0] == 1:
                    #q_dup = np.concatenate([samples[:, self.mass_ratio],
                    #                  1 / samples[:, self.mass_ratio]], axis=0)

                    #q_dup_rescale = 2 * ((q_dup - self.prior_bounds[self.mass_ratio, 0]) \
                    #        / ((1 / self.prior_bounds[self.mass_ratio, 0]) \
                    #        - self.prior_bounds[self.mass_ratio, 0])) - 1.
                    q_log = np.concatenate([np.log(samples[:, self.mass_ratio]),
                        -np.log(samples[:, self.mass_ratio])], axis=0)
                    samples_out = np.concatenate([samples_out, samples_out], axis=0)
                    #samples_out[:, self.mass_ratio] = q_dup_rescale
                    samples_out[:, self.mass_ratio] = q_log
                    #samples_out[:, self.mass_ratio] = 1. / samples[:, self.mass_ratio]

                    if self.quaternions:
                        n = q_log.shape[0] // 2
                        samples_out[n:, self.quaternions[0]] = -samples[:, self.quaternions[3]]
                        samples_out[n:, self.quaternions[1]] = samples[:, self.quaternions[2]]
                        samples_out[n:, self.quaternions[2]] = -samples[:, self.quaternions[1]]
                        samples_out[n:, self.quaternions[3]] = samples[:, self.quaternions[0]]

                else:
                    # if only given one sample, do not duplicate it
                    # TODO: this is not general, but only applies to generating
                    # radius and will need to be made more general
                    #samples_out[:, self.mass_ratio] = 1. / samples[:, self.mass_ratio]
                    samples_out[:, self.mass_ratio]= \
                            2 * ((samples[:, self.mass_ratio] \
                            - self.prior_bounds[self.mass_ratio, 0]) \
                            / ((1 / self.prior_bounds[self.mass_ratio, 0]) \
                            - self.prior_bounds[self.mass_ratio, 0])) - 1.


        return samples_out

    def inverse_reparameterise(self, samples):
        """
        Convert from reparameterised space to sampling space
        """
        # make sure if using one sample it's shape [1, N]
        if samples.ndim == 1:
            samples = samples[np.newaxis, :]

        if not self.parameter_indices:
            self.setup_parameter_indices()

        samples_out = np.empty([samples.shape[0], self.reparam_dim])

        samples_out[:, self.defaults] = self.inverse_normalise_defaults(
                samples[:, self.defaults])

        if 'time' in self.reparameterisations:
            t = inverse_rescale_minus_one_to_one(samples[:, self.time],
                    self.prior_bounds[self.time, 0], self.prior_bounds[self.time, 1])
            samples_out[:, self.time] = np.float64(t) + np.float64(self.time_offset)

        if self.quaternions:
            samples_out[:, self.quaternions] = samples[:, self.quaternions]

        if 'sky' in self.reparameterisations:
            ra, dec, dL = cartesian_to_sky(samples[:, self.sky_x],
                    samples[:,self.sky_y], samples[:, self.sky_z])
            samples_out[:, self.ra] = ra
            samples_out[:, self.dec] = dec

            if 'luminosity_distance' in self.reparameterisations:
                samples_out[:, self.dL] = inverse_rescale_0_to_1(dL,
                        xmin=self.prior_bounds[self.dL, 0],
                        xmax=self.prior_bounds[self.dL, 1])
            else:
                samples_out[:, self.sky_radial] = dL

        if self.angle_indices:
            for i, s in zip(self.angle_indices, self.angle_scales):
                a, r = cartesian_to_angle(samples[:, i[0]],
                        samples[:, i[1]], scale=s)
                samples_out[:, i[0]] = a
                samples_out[:, i[1]] = r

        if 'mass_ratio' in self.reparameterisations:
            if 'mass_ratio' in self.logit_parameters:
                samples_out[:, self.mass_ratio] = inverse_logit(
                        samples[:, self.mass_ratio_logit],
                        xmin=self.prior_bounds[self.mass_ratio, 0],
                        xmax=self.prior_bounds[self.mass_ratio, 1])
            else:
                q = samples[:, self.mass_ratio]
                #q = (( 1 / self.prior_bounds[self.mass_ratio, 0]) \
                #        - self.prior_bounds[self.mass_ratio, 0]) * (0.5 *(q + 1)) \
                #        + self.prior_bounds[self.mass_ratio, 0]
                # flip mass ratio above 1 to 1/1
                # log(q) -> [-inf, 0]
                # these just need exp, for q>0 need exp(-q)
                i = q <= 0.
                #q[i] = 1 / q[i]
                samples_out[i, self.mass_ratio] = np.exp(q[i])
                samples_out[~i, self.mass_ratio] = np.exp(-q[~i])
                #samples_out[:, self.mass_ratio] = 1 / samples[:, self.mass_ratio]

                if self.quaternions:
                    samples_out[~i, self.quaternions[0]] = samples[~i, self.quaternions[3]]
                    samples_out[~i, self.quaternions[1]] = -samples[~i, self.quaternions[2]]
                    samples_out[~i, self.quaternions[2]] = samples[~i, self.quaternions[1]]
                    samples_out[~i, self.quaternions[3]] = -samples[~i, self.quaternions[0]]

        return samples_out

    def compute_log_jacobian(self, samples):
        """
        Compute the log jacobian for reparameterised space to sampling space
        given samples in reparameterised space
        """
        if samples.shape[-1] != self.reparam_dim:
            raise RuntimeError('Cannot compute jacobian given samples from the sampling space')
        log_J = np.zeros(samples.shape[0])
        if 'psi' in self.reparameterisations:
            log_J += np.log(np.sqrt(samples[:, self.psi_x] ** 2.
                + samples[:, self.psi_y] ** 2.))
        if 'phase' in self.reparameterisations:
            log_J += np.log(np.sqrt(samples[:, self.phase_x] ** 2.
                + samples[:, self.phase_y] ** 2.))
        if 'sky' in self.reparameterisations:
            # compute 1 / (sqrt(rxy) sqrt(rxyz))
            log_J += 0.5 * np.log(samples[:, self.sky_x] ** 2 \
                    + samples[:, self.sky_y] ** 2)
            log_J += 0.5 * np.log(samples[:, self.sky_x] ** 2 \
                    + samples[:, self.sky_y] ** 2 \
                    + samples[:, self.sky_z] ** 2)

        if self.angle_indices:
            for i in self.angle_indices:
                log_J += 0.5 * np.log(samples[:, i[0]] **2. \
                        + samples[:, i[1]] ** 2.)

        if 'mass_ratio' in self.reparameterisations:
            # minus sig
            if 'mass_ratio' in self.logit_parameters:
                log_J -= log_jacobian_inverse_logit(
                        samples[:, self.mass_ratio_logit])
            elif self.q_inversion:
                # when using log division between q and 1/q is 0
                # for <0 conversion is exp(q) for >0 exp(-q)
                # so Jacobian is log(+/-exp(+/-q))
                # but we need inverse
                i = samples[:, self.mass_ratio] <= 0.
                log_J[i] -= samples[i, self.mass_ratio]
                log_J[~i] += samples[~i, self.mass_ratio]
                # 1 for log(1/q) and log(q), 2 for 1/q
                #log_J -= 1 * np.log(samples[j, self.mass_ratio])
                #log_J -= 1 * np.log(np.abs(samples[:, self.mass_ratio]))
                #log_J += 2. * np.log(samples[:, self.mass_ratio])

        return log_J

    def get_mask(self, mask):
        """
        Get a mask for the normalising flows. Either construct a mass from a
        list of parameters or a name.
        """
        mask_array = np.zeros(self.reparam_dim)
        if isinstance(mask, list):
            for m in mask:
               i = self.re_parameters.index(m)
               mask_array[i] = 1.

        elif mask == 'intrinsic':
            i = [j for j, p in enumerate(self.re_parameters) if p in self.intrinsic_parameters]
            mask_array[i] = 1.

        elif mask == 'extrinsic':
            i = [j for j, p in enumerate(self.re_parameters) if p in self.extrinsic_parameters]
            mask_array[i] = 1.

        else:
            mask_array = None
        print(f'Using mask: {mask_array} (parameters: {self.re_parameters})')
        return mask_array
