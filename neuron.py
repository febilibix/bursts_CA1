import numpy as np


CONFIGS = {
    '2D': {
        'pb': {"E_L": -65, "R": 10, "v_th": -45, "tau": 0.25},
        'pa': {"E_L": -65, "R": 10, "v_th": -45, "tau": 1}, 
        'pib': {"E_L": -65, "R": 10, "v_th": -45, "tau": .1}, 
        'pia': {"E_L": -65, "R": 10, "v_th": -45, "tau": 0.1}, 
        'w_ip_a': 5000, 'w_ip_b': 750, 'w_pi_a': 75, 'w_pi_b': 100,
        'tau_inh': 20, 'ma_pc': 360, 'mb_pc': 32, 
        'alpha': 0.25, 'sigma_pf': 6, 'eta': 25, 'eta_inh': 50, 'beta': 0.01, 
    },
    '1D': {
        'pb': {"E_L": -65, "R": 10, "v_th": -45, "tau": 0.25},
        'pa': {"E_L": -65, "R": 10, "v_th": -45, "tau": 1}, 
        'pib': {"E_L": -65, "R": 10, "v_th": -45, "tau": .1}, 
        'pia': {"E_L": -65, "R": 10, "v_th": -45, "tau": 0.1}, 
        'w_ip_a': 5000, 'w_ip_b': 750, 'w_pi_a': 75, 'w_pi_b': 100,
        'tau_inh': 20, 'ma_pc': 240, 'mb_pc': 32,
        'alpha': 0.75, 'sigma_pf': 6, 'eta': 15, 'eta_inh': 10, 'beta': 0.01, 
    },
}


def runge_kutta(t, y_values, f, dt):
    """Advance state using 2nd-order Runge–Kutta integration.

    Args:
        t: Current time (s).
        y_values: List of past states; last element used as current state.
        f: Callable(t, y) returning derivative dy/dt.
        dt: Time step (s).

    Returns:
        y_new: Updated state after one step.
        t_new: Updated time (t + dt).
    """

    y = y_values[-1]

    k1 = dt * f(t, y)
    k2 = dt * f(t + dt, y + k1)

    y_new = y + 0.5 * (k1 + k2)
    t_new = t + dt

    return np.array(y_new), t_new


def gauss(x, mu, sig, L = None):

    """Compute Gaussians over 1D or 2D space, for 2D with torus wrapping.

    Args:
        x: Positions of evaluation points, shape (d, n_points) or (n_points,).
        mu: Centers of Gaussian fields, shape (d, n_cells) or (n_cells,).
        sig: Standard deviation (same units as x).
        L: If provided, environment length (for wrapping differences).

    Returns:
        Array of Gaussian activations (n_cells, n_points).
    """

    if x.ndim == 1 and mu.ndim == 1:
        return np.exp(-0.5 * ((mu[:, None] - x[None, :])**2) / sig**2)
    
    elif x.ndim == 2 and mu.ndim == 2:
        if L is None:
            ## If edge length is not given, torus distance cannot be computed
            ## It will default to standard distance
            diff = mu[:, :, None] - x[:, None, :]
        else:
            raw_diff = mu[:, :, None] - x[:, None, :]
            # Wrap differences for each dimension
            for dim in range(x.shape[0]):
                raw = raw_diff[dim]
                raw_diff[dim] = np.where(
                    np.abs(raw) > L/2,
                    raw - np.sign(raw) * L,
                    raw
                )
            diff = raw_diff
        squared_dist = np.sum(diff**2, axis=0)
        return np.exp(-0.5 * squared_dist / sig**2)
    
    else:
        raise ValueError(f"Incompatible shapes: x.shape = {x.shape}, mu.shape = {mu.shape}")
    


class PyramidalCells():

    """Two-compartment pyramidal cell network with CA3 and EC top-down input and interneurons.

    Supports 1D/2D environments, place-field generation, inhibitory plasticity,
    and recurrent dynamics with Runge-Kutta integration.
    """

    def __init__(self, n_cells, len_edge, dt=0.01, n_env=2, n_dim=1,
                 inh_plasticity=True, seed=42):
        """Initialize pyramidal + interneuron populations, weights, and fields.

        Args:
            n_cells: Dict with counts {'pyramidal', 'inter_a', 'inter_b', 'CA3'}.
            len_edge: Environment size (cm).
            dt: Time step (s).
            n_env: Number of environments for place-field generation.
            n_dim: Environment dimensionality (1 or 2).
            inh_plasticity: Enable inhibitory plasticity if True.
            seed: RNG seed.
        """
        
        rng = np.random.default_rng(seed=seed)
        
        self.dt = dt
        self.n_dim = n_dim
        self.n_cells = n_cells
        self.len_edge = len_edge

        config = CONFIGS['1D' if n_dim == 1 else '2D']
        for key, value in config.items():
            setattr(self, key, value)
        
        n_int_a, n_int_b, n_pyr = n_cells['inter_a'], n_cells['inter_b'], n_cells['pyramidal']

        self.W_ip_a = self.w_ip_a*np.ones((n_int_a, n_pyr))/(n_pyr)
        self.W_pi_a = self.w_pi_a*np.ones((n_pyr, n_int_a))/n_int_a

        if inh_plasticity:
            self.W_ip_b = self.w_ip_b*rng.random((n_int_b, n_pyr))/(n_pyr)
            self.W_pi_b = self.w_pi_b*rng.random((n_pyr, n_int_b))/(n_int_b)
        else:
            self.W_ip_b = self.w_ip_b*np.ones((n_int_b, n_pyr))/(n_pyr)
            self.W_pi_b = self.w_pi_b*np.ones((n_pyr, n_int_b))/n_int_b

        self.W_CA3 = rng.random((n_cells['pyramidal'], n_cells['CA3']))
        W_CA3_norm = np.sum(self.W_CA3, axis=1)
        self.W_CA3 = self.W_CA3 / W_CA3_norm[:, np.newaxis] 
        self.W_CA3 = np.where(self.W_CA3 < 0, 0, self.W_CA3)

        self.inter_spikes_a, self.inter_spikes_b = np.zeros(n_cells['inter_a']), np.zeros(n_cells['inter_b'])
        
        self.plast_count = 0

        self.dynamics_values = {
            'v_b'    : (self.dynamics_basal,         [np.ones(n_cells['pyramidal'])*self.pb['E_L']]),
            'v_a'    : (self.dynamics_apical,        [np.ones(n_cells['pyramidal'])*self.pa['E_L']]),
            'v_i_a'  : (self.dynamics_interneuron_a, [np.ones(n_cells['inter_a'])*self.pia['E_L']]),  
            'v_i_b'  : (self.dynamics_interneuron_b, [np.ones(n_cells['inter_b'])*self.pib['E_L']]),
            }
                
        self.all_m_CA3 = self.generate_place_fields(len_edge, n_env, n_dim, rng, 'CA3')
        self.all_m_EC  = self.generate_place_fields(len_edge, n_env, n_dim, rng, 'pyramidal')

        self.m_CA3, self.m_EC = self.all_m_CA3[0], self.all_m_EC[0]  # Initial place field for CA3

        self.inh_plasticity = inh_plasticity  # Enable or disable inhibitory plasticity

        if inh_plasticity:

            # Initialize traces and accumulated inh weight change
            self.f_E_trace = np.zeros(n_cells['pyramidal']) 
            self.f_I_trace_b = np.zeros(n_cells['inter_b']) 

            self.accumulated_delta_W_ip_b = np.zeros(self.W_ip_b.shape)
            self.accumulated_delta_W_pi_b = np.zeros(self.W_pi_b.shape)

    
    def generate_place_fields(self, len_edge, n_env, n_dim, rng, area):
        """Generate shuffled 1D or 2D place fields for given area (CA3/EC).

        Args:
            len_edge: Environment size (cm).
            n_env: Number of environments to create.
            n_dim: 1 or 2.
            rng: Numpy Generator.
            area: 'CA3' or 'pyramidal'.

        Returns:
            List of arrays with place-field centers per environment.
        """

        fields = []
        n_cells = self.n_cells[area]

        for _ in range(n_env):
            if n_dim == 1:
                if area == 'CA3':
                    left, right = -len_edge/10, len_edge + len_edge/10
                else:
                    left, right = 0, len_edge
                field = np.linspace(left, right, n_cells)
                rng.shuffle(field)
            elif n_dim == 2:
                n_side = int(np.sqrt(n_cells))
                xs = np.linspace(0, len_edge, n_side, endpoint=False)
                x, y = np.meshgrid(xs, xs, indexing='ij')
                field = np.vstack((x.flatten(), y.flatten()))
                field = rng.permutation(field, axis=1)
            else:
                raise ValueError("Only n_dim=1 or 2 supported.")
            fields.append(field)
        
        return fields

        
    def dynamics_basal(self, t, v):
        """Differential equation for basal compartment voltage dynamics."""
        R_b, E_L_b, tau_b = self.pb['R'], self.pb['E_L'], self.pb['tau']
        v_dot = 1/tau_b * (E_L_b - v + R_b * (self.W_CA3@self.I_b[:, int(t/self.dt)] - self.W_pi_b@self.inter_spikes_b))
        return v_dot

  
    def dynamics_apical(self, t, v):
        """Differential equation for apical compartment voltage dynamics."""
        R_a, E_L_a, tau_a = self.pa['R'], self.pa['E_L'], self.pa['tau']
        v_dot = 1/tau_a * (E_L_a - v + R_a * (self.I_a[:, int(t/self.dt)] - self.W_pi_a@self.inter_spikes_a))
        return v_dot
    

    def dynamics_interneuron_a(self, t, v):
        """Dynamics of SST interneuron population A (driven by pyramidal spiking)."""
        R_i, E_L_i, tau_i = self.pia['R'], self.pia['E_L'], self.pia['tau']
        v_dot = 1/tau_i * (E_L_i - v + R_i * (self.W_ip_a @ self.spiking))
        return v_dot
    

    def dynamics_interneuron_b(self, t, v):
        """Dynamics of PV interneuron population B (driven by pyramidal spiking)."""
        R_i, E_L_i, tau_i = self.pib['R'], self.pib['E_L'], self.pib['tau']
        v_dot = 1/tau_i * (E_L_i - v + R_i * (self.W_ip_b @ self.spiking)) 
        return v_dot
    

    def learn_place_cells(self, t_run, x_run, t_per_epoch, top_down = True):
        """Wrapper for retrieve_place_cells with plasticity enabled."""
        return self.retrieve_place_cells(t_run, x_run, t_per_epoch=t_per_epoch, top_down=top_down, plasticity=True)


    def create_activity_pc(self, x, len_track, n_cells, m, m_cells, m_cells_new, a): 
        """Create input activity for a cell population based on Gaussian fields.

        Args:
            x: Positions visited (1D or 2D array).
            len_track: Track length or environment size (cm).
            n_cells: Number of cells in population.
            m: Scaling factor (peak rate).
            m_cells: Original field centers.
            m_cells_new: New environment field centers.
            a: Mixing factor between old and new fields (0=old, 1=new).

        Returns:
            Array (n_cells, n_timepoints+5) with activity over time.
        """

        sigma_pf = self.sigma_pf  

        activity_dim2 = x.shape[0] if x.ndim == 1 else x.shape[1]

        activity = np.zeros((int(n_cells), activity_dim2 + 5))
        activity[:, :-5] = m * ((1-a) * gauss(x, m_cells_new, sigma_pf, self.len_edge) + a * gauss(x, m_cells, sigma_pf, self.len_edge))
    
        return activity 
    

    def run_one_epoch(self, t_epoch, plasticity=True):
        """Simulate one epoch of activity with optional plasticity update.

        Updates spiking, bursting, interneuron dynamics, and accumulates plasticity terms.

        Args:
            t_epoch: Duration of epoch (s).
            plasticity: If True, apply plasticity_step at the end.
        """

        dt = self.dt

        self.bursting = np.zeros(self.n_cells['pyramidal'])
        self.spiking = np.zeros(self.n_cells['pyramidal'])

        self.spike_count = np.zeros((int(round(t_epoch / dt)), self.n_cells['pyramidal']))
        self.burst_count = np.zeros((int(round(t_epoch / dt)), self.n_cells['pyramidal']))
        ## These are commented out to save memory but can be used for diagnostic purposes:
        # self.spike_count_int_a = np.zeros((int(round(t_epoch / dt)), self.n_cells['inter_a']))
        # self.spike_count_int_b = np.zeros((int(round(t_epoch / dt)), self.n_cells['inter_b']))

        values = {k: v[1] for k, v in self.dynamics_values.items()}

        values_new = {}
        self.t_epoch = t_epoch

        t, t_old = 0, 0

        if self.inh_plasticity:

            # Reset accumulated delta W and STDP traces before each epoch:
            self.accumulated_delta_W_ip_b = np.zeros(self.W_ip_b.shape)
            self.accumulated_delta_W_pi_b = np.zeros(self.W_pi_b.shape)

            self.f_E_trace = np.zeros(self.n_cells['pyramidal'])
            self.f_I_trace_b = np.zeros(self.n_cells['inter_b'])

        for t in range(1, int(round(t_epoch / dt)) + 1):
             
            for value_name, (dynamics, value) in self.dynamics_values.items():
                value_new, t_new = runge_kutta(
                    t_old, value, dynamics, dt)
                values_new[value_name] = np.array(value_new)

            self.spiking = (values_new['v_b'] > self.pb['v_th']).astype(int)
            self.bursting = ((values_new['v_a'] > self.pa['v_th']) & self.spiking).astype(int)
            
            self.inter_spikes_a = (values_new['v_i_a'] > self.pia['v_th']).astype(int)
            self.inter_spikes_b = (values_new['v_i_b'] > self.pib['v_th']).astype(int)
            
            values_new['v_a'] = np.where(self.spiking, self.pa['E_L'], values_new['v_a'])
            values_new['v_b'] = np.where(self.spiking, self.pb['E_L'], values_new['v_b'])

            values_new['v_i_b'] = np.where(self.inter_spikes_b, self.pib['E_L'], values_new['v_i_b'])
            values_new['v_i_a'] = np.where(self.inter_spikes_a, self.pia['E_L'], values_new['v_i_a'])
            
            [values[i].append(values_new[i]) for i in list(self.dynamics_values.keys())]

            self.burst_count[t-1, :] = self.bursting
            self.spike_count[t-1, :] = self.spiking
            ## These are commented out to save memory but can be used for diagnostic purposes:
            # self.spike_count_int_a[t-1, :] = self.inter_spikes_a
            # self.spike_count_int_b[t-1, :] = self.inter_spikes_b

            if self.inh_plasticity:
                # Update traces and accumulate inh weight change:

                self.f_E_trace = self.f_E_trace * np.exp(-self.dt / self.tau_inh) + self.spiking
                self.f_I_trace_b = self.f_I_trace_b * np.exp(-self.dt / self.tau_inh) + self.inter_spikes_b

                term1 = np.outer(self.spiking, self.f_I_trace_b) 
                term2 = np.outer(self.f_E_trace, self.inter_spikes_b) # Shape (n_pyr, n_int_b)

                instantaneous_delta_W_ip_b = self.eta_inh * (term2 - term1) 
                instantaneous_delta_W_pi_b = self.eta_inh * (term1 + term2 - self.beta)

                self.accumulated_delta_W_ip_b += instantaneous_delta_W_ip_b.T * self.dt
                self.accumulated_delta_W_pi_b += instantaneous_delta_W_pi_b * self.dt

            t_old = t_new

        if plasticity:
            self.plasticity_step()
        self.values = values_new
        self.all_values = values


    def retrieve_place_cells(self, t_run, x_run, new_env=False, a=0,
                             t_per_epoch=None, top_down=False,
                             plasticity=True, len_track=None):
        """Simulate full run across epochs and return spike/burst counts.

        Args:
            t_run: Time vector of trajectory (s).
            x_run: Trajectory positions (1D or 2D).
            new_env: If True, switch to new environment fields.
            a: Mixing factor between old/new fields.
            t_per_epoch: Duration per epoch (s).
            top_down: If True, provide EC input (apical drive).
            plasticity: If True, apply plasticity during simulation.
            len_track: Track length; overrides len_edge.

        Returns:
            full_spike_count: (T, n_pyramidal).
            full_burst_count: (T, n_pyramidal).
        """

        dt = self.dt
        m_a, m_b = self.ma_pc, self.mb_pc

        tn = t_run[-1]
        len_track = np.max(x_run)
        
        n_epochs = int(round(tn/t_per_epoch))

        # TODO: Here i will need to see which ones i will need to save and which ones i don't because of memory

        full_spike_count = np.zeros((int(round(tn/dt)+1), self.n_cells['pyramidal']))
        full_burst_count = np.zeros((int(round(tn/dt)+1), self.n_cells['pyramidal']))

        ## These are commented out to save memory but can be used for diagnostic purposes:
        # self.full_spike_count_int_b = np.zeros((int(round(tn/dt)+1), self.n_cells['inter_b']))
        # self.full_spike_count_int_a = np.zeros((int(round(tn/dt)+1), self.n_cells['inter_a']))

        for j in range(n_epochs):
            t0_epoch = j*t_per_epoch

            if self.n_dim == 1:
                x = x_run[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt)]
            else:
                x = x_run[:, int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt)]

            self.I_b = self.create_activity_pc(x, len_track, self.n_cells['CA3'], m_b, self.m_CA3, self.all_m_CA3[int(new_env)], a)
            
            if top_down:
                self.I_a = self.create_activity_pc(x, len_track, self.n_cells['pyramidal'], m_a, self.m_EC, self.all_m_EC[int(new_env)], a=0)
            else: 
                self.I_a = np.zeros((self.n_cells['pyramidal'], int(t_per_epoch/dt+5))) 

            n_tpoints = x.shape[0] if self.n_dim == 1 else x.shape[1]
            self.Ib_trace = self.I_b[:, :n_tpoints].T
            self.run_one_epoch(t_per_epoch, plasticity)

            full_spike_count[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt), :] = self.spike_count
            full_burst_count[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt), :] = self.burst_count

            # These are commented out to save memory but can be used for diagnostic purposes:
            # self.full_spike_count_int_b[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt), :] = self.spike_count_int_b
            # self.full_spike_count_int_a[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt), :] = self.spike_count_int_a

        return full_spike_count, full_burst_count


    def plasticity_step(self):
        """Apply Hebbian and burst-driven CA3→CA1 plasticity plus inhibitory updates.

        Updates W_CA3 using burst- and spike-driven rules.
        If enabled, applies accumulated inhibitory weight changes.
        """

        mean_events = np.mean(self.spike_count, axis=0)
        mean_bursts = np.mean(self.burst_count, axis=0)
        mean_Ib = np.mean(self.Ib_trace, axis=0)/(self.n_cells['pyramidal'])

        delta_W = self.eta*np.outer(mean_bursts, mean_Ib) + self.alpha*np.outer(mean_events, mean_Ib)

        self.W_CA3 += delta_W   
        self.W_CA3 = np.where(self.W_CA3 < 0, 0, self.W_CA3)

        W_CA3_norm = np.sum(self.W_CA3, axis=1)
        self.W_CA3 = self.W_CA3 / W_CA3_norm[:, np.newaxis] 
        self.W_CA3 = np.where(self.W_CA3 < 0, 0, self.W_CA3)

        if self.inh_plasticity:

            self.W_ip_b += self.accumulated_delta_W_ip_b
            self.W_pi_b += self.accumulated_delta_W_pi_b

            self.W_ip_b = np.where(self.W_ip_b < 0, 0, self.W_ip_b)
            self.W_pi_b = np.where(self.W_pi_b < 0, 0, self.W_pi_b)

        self.plast_count += 1


    def get_input_map(self, area='CA3', env=0, a=0, n_bins=256):
        """ Generate activation map for CA3 or pyramidal (EC) inputs.
            This function is mainly used for visualization purposes.

        Args:
            area: 'CA3' or 'EC'.
            env: Environment index.
            a: Mixing factor between old/new fields.
            n_bins: Number of bins (1D) or grid size^2 (2D).

        Returns:
            Array (n_cells, n_bins) activation map.
        """

        area = 'pyramidal' if area == 'EC' else area  
        m = self.ma_pc if area == 'pyramidal' else self.mb_pc
        m_cells = self.m_EC if area == 'pyramidal' else self.m_CA3
        m_cells_new = self.all_m_EC[env] if area == 'pyramidal' else self.all_m_CA3[env]

        if self.n_dim == 1:
            x = np.linspace(0, self.len_edge, n_bins)
            act = self.create_activity_pc(x, self.len_edge, self.n_cells[area], m, m_cells, m_cells_new, a)
            return act[:, :-5] 
        
        elif self.n_dim == 2:

            step = n_bins * 1j
            x, y = np.mgrid[0:self.len_edge:step, 0:self.len_edge:step] 
            x = np.vstack((x.flatten(), y.flatten()))

            act = self.create_activity_pc(x, self.len_edge, self.n_cells[area], m, m_cells, m_cells_new, a)
            return act[:, :-5]
        
        else:
            raise ValueError("Only n_dim=1 or 2 supported.")
        