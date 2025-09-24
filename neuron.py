import numpy as np


CONFIGS = {
    # TODO: I will need to add inh plast parameters here

    '2D': {
        'pb'  : {"E_L": -65, "R": 10, "v_th": -45, "tau": 0.25},
        'pa'  : {"E_L": -65, "R": 10, "v_th": -45, "tau": 1}, 
        'pib' : {"E_L": -65, "R": 10, "v_th": -45, "tau": .1}, 
        'pia' : {"E_L": -65, "R": 10, "v_th": -45, "tau": 0.1}, 
        'w_ip_a' : 5000, 
        'w_ip_b' : 750,
        'w_pi_a' : 75,
        'w_pi_b' : 100,
        'ma_pc' : 360, # 160, 
        'mb_pc': 32, # 26,
        'alpha' : 0.25, # 0.075, 
        'sigma_pf' : 6, 
        'eta' : 25, # 25, 
        'eta_inh': 50, # 15
        'beta' : 0.01, 
    },

    # '2D': {.      ###### THIS IS THE ONE WHICH GIVES NICE RESULTS FOR BIOLOGICALLY PLAUSIBLE STDP TAU
    #     'pb'  : {"E_L": -65, "R": 10, "v_th": -45, "tau": 0.25},
    #     'pa'  : {"E_L": -65, "R": 10, "v_th": -45, "tau": 1}, 
    #     'pib' : {"E_L": -65, "R": 10, "v_th": -45, "tau": .1}, 
    #     'pia' : {"E_L": -65, "R": 10, "v_th": -45, "tau": 0.1}, 
    #     'w_ip_a' : 5000, 
    #     'w_ip_b' : 750,
    #     'w_pi_a' : 75,
    #     'w_pi_b' : 100,
    #     'ma_pc' : 180, 
    #     'mb_pc': 100,
    #     'alpha' : 0.1, 
    #     'sigma_pf' : 6, 
    #     'eta' : 1, 
    #     'eta_inh': 25,
    #     'beta' : 0.01, 
    # },


    '1D': {
        'pb'  : {"E_L": -65, "R": 10, "v_th": -45, "tau": 0.25},
        'pa'  : {"E_L": -65, "R": 10, "v_th": -45, "tau": 1}, 
        'pib' : {"E_L": -65, "R": 10, "v_th": -45, "tau": .1}, 
        'pia' : {"E_L": -65, "R": 10, "v_th": -45, "tau": 0.1}, 
        'w_ip_a' : 5000, 
        'w_ip_b' : 750,
        'w_pi_a' : 75,
        'w_pi_b' : 100,
        'ma_pc' : 240, # 160, 
        'mb_pc': 32, # 26,
        'alpha' : 0.75, # 0.075, 
        'sigma_pf' : 6, 
        'eta' : 15, # 25, 
        'eta_inh': 10, # 15
        'beta' : 0.01, 
    },
}


def runge_kutta(t, y_values, f, dt):

    y = y_values[-1]

    k1 = dt * f(t, y)
    k2 = dt * f(t + dt, y + k1)

    y_new = y + 0.5 * (k1 + k2)
    t_new = t + dt

    return np.array(y_new), t_new


def gauss(x, mu, sig, L = None):

    if x.ndim == 1 and mu.ndim == 1:
        return np.exp(-0.5 * ((mu[:, None] - x[None, :])**2) / sig**2)
    
    elif x.ndim == 2 and mu.ndim == 2:
        if L is None:
            diff = mu[:, :, None] - x[:, None, :]
        ### For a torus:
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

    def __init__(
            self, 
            n_cells,
            len_edge,
            # learning_rate = 0.05,
            dt = 0.01,
            n_env = 2,
            n_dim = 1,
            inh_plasticity = True,
            seed = 42
            ): 
        
        rng = np.random.default_rng(seed=seed)
        
        self.dt = dt
        self.n_dim = n_dim
        self.n_cells = n_cells
        # self.eta = learning_rate 
        # self.eta_inh = learning_rate  # Learning rate for inhibitory plasticity
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
        self.sigma_pf = self.sigma_pf
        # TODO: This part of the code will need to be cleaned still  

        if self.inh_plasticity:

            # --- NEW: Parameters for STDP traces (f_E and f_I) ---
            # Time constant for post-synaptic excitatory trace (pyramidal)
            self.tau_fE = 0.1 # s - Updated to biological range (0.01-0.2 seconds)
            # Amplitude increment for f_E upon a spike
            self.A_fE = 1.0  ## TODO: Delete this variables i guess if they're 1 anyway. In case i decide to change them, also change in report
            # Time constant for pre-synaptic inhibitory trace (interneuron B)
            self.tau_fI = 0.1 # s - Updated to biological range (0.01-0.2 seconds)
            # Amplitude increment for f_I upon a spike
            self.A_fI = 1.0  
            # The constant offset from the formula, causing baseline depression (less inhibition)
            self.constant_depression_term = self.beta 

            # Initialize traces (will be updated at each time step within run_one_epoch)
            # Trace for post-synaptic excitatory (pyramidal) neurons
            self.f_E_trace = np.zeros(n_cells['pyramidal']) 
            # Trace for pre-synaptic inhibitory (interneuron B) neurons
            self.f_I_trace_b = np.zeros(n_cells['inter_b']) 

            # Initialize accumulated delta W for W_ip_b (I->E weights)
            # This will store the total weight change over an epoch before applying it.
            self.accumulated_delta_W_ip_b = np.zeros(self.W_ip_b.shape)
            self.accumulated_delta_W_pi_b = np.zeros(self.W_pi_b.shape)

            self.W_max_ip_b = 5000/(n_cells['pyramidal'])  # Maximum weight for I->E connections
            self.W_max_pi_b = 25/(n_cells['inter_b'])  # Maximum weight for I->I connections  



        ### store updates for batch plasticity ## TODO: DELETE
        # self.W_CA3_a = np.zeros(self.W_CA3.shape)  # Store updates for CA3 weights
        # self.W_pi_b_a = np.zeros(self.W_pi_b.shape)  # Store updates for pyramidal to interneuron B weights
        # self.W_ip_b_a = np.zeros(self.W_ip_b.shape)

    
    def generate_place_fields(self, len_edge, n_env, n_dim, rng, area):
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
        R_b, E_L_b, tau_b = self.pb['R'], self.pb['E_L'], self.pb['tau']
        v_dot = 1/tau_b * (E_L_b - v + R_b * (self.W_CA3@self.I_b[:, int(t/self.dt)] - self.W_pi_b@self.inter_spikes_b))
        return v_dot

  
    def dynamics_apical(self, t, v):
        R_a, E_L_a, tau_a = self.pa['R'], self.pa['E_L'], self.pa['tau']
        v_dot = 1/tau_a * (E_L_a - v + R_a * (self.I_a[:, int(t/self.dt)] - self.W_pi_a@self.inter_spikes_a))
        return v_dot
    

    def dynamics_interneuron_a(self, t, v):
        R_i, E_L_i, tau_i = self.pia['R'], self.pia['E_L'], self.pia['tau']
        v_dot = 1/tau_i * (E_L_i - v + R_i * (self.W_ip_a @ self.spiking))
        return v_dot
    

    def dynamics_interneuron_b(self, t, v):
        R_i, E_L_i, tau_i = self.pib['R'], self.pib['E_L'], self.pib['tau']
        v_dot = 1/tau_i * (E_L_i - v + R_i * (self.W_ip_b @ self.spiking)) 
        return v_dot
    

    def learn_place_cells(self, t_run, x_run, t_per_epoch, top_down = True):
        return self.retrieve_place_cells(t_run, x_run, t_per_epoch=t_per_epoch, top_down=top_down, plasticity=True)


    def create_activity_pc(self, x, len_track, n_cells, m, m_cells, m_cells_new, a): 
        sigma_pf = self.sigma_pf  # len_track/16 # len_track/8

        activity_dim2 = x.shape[0] if x.ndim == 1 else x.shape[1]

        activity = np.zeros((int(n_cells), activity_dim2 + 5))
        activity[:, :-5] = m * ((1-a) * gauss(x, m_cells_new, sigma_pf, self.len_edge) + a * gauss(x, m_cells, sigma_pf, self.len_edge))
    
        return activity 
    

    def run_one_epoch(self, t_epoch, plasticity = True): 
        dt = self.dt

        self.bursting = np.zeros(self.n_cells['pyramidal'])
        self.spiking = np.zeros(self.n_cells['pyramidal'])

        # TODO: Here i will need to see which ones i will need to save and which ones i don't because of memory

        self.spike_count = np.zeros((int(round(t_epoch / dt)), self.n_cells['pyramidal']))
        self.burst_count = np.zeros((int(round(t_epoch / dt)), self.n_cells['pyramidal']))
        # self.spike_count_int_a = np.zeros((int(round(t_epoch / dt)), self.n_cells['inter_a']))
        # self.spike_count_int_b = np.zeros((int(round(t_epoch / dt)), self.n_cells['inter_b']))

        values = {k: v[1] for k, v in self.dynamics_values.items()}

        values_new = {}
        self.t_epoch = t_epoch

        t, t_old = 0, 0

        if self.inh_plasticity:

            # Reset accumulated delta W for W_ip_b at the beginning of each epoch
            self.accumulated_delta_W_ip_b = np.zeros(self.W_ip_b.shape)
            self.accumulated_delta_W_pi_b = np.zeros(self.W_pi_b.shape)

            # Reset STDP traces at the beginning of each epoch to ensure fresh calculation
            self.f_E_trace = np.zeros(self.n_cells['pyramidal'])
            self.f_I_trace_b = np.zeros(self.n_cells['inter_b'])

        for t in range(1, int(round(t_epoch / dt)) + 1):
             
            for value_name, (dynamics, value) in self.dynamics_values.items():
                value_new, t_new = runge_kutta(
                    t_old, value, dynamics, dt)
                values_new[value_name] = np.array(value_new)

            self.spiking = (values_new['v_b'] > self.pb['v_th']).astype(int)
            self.bursting = ((values_new['v_a'] > self.pa['v_th']) & self.spiking).astype(int)
            # print(self.spiking.sum(), self.bursting.sum())
            
            self.inter_spikes_a = (values_new['v_i_a'] > self.pia['v_th']).astype(int)
            self.inter_spikes_b = (values_new['v_i_b'] > self.pib['v_th']).astype(int)

            # print(self.inter_spikes_b.sum(), self.inter_spikes_a.sum())
            
            values_new['v_a'] = np.where(self.spiking, self.pa['E_L'], values_new['v_a'])
            values_new['v_b'] = np.where(self.spiking, self.pb['E_L'], values_new['v_b'])

            values_new['v_i_b'] = np.where(self.inter_spikes_b, self.pib['E_L'], values_new['v_i_b'])
            values_new['v_i_a'] = np.where(self.inter_spikes_a, self.pia['E_L'], values_new['v_i_a'])
            
            [values[i].append(values_new[i]) for i in list(self.dynamics_values.keys())]

            self.burst_count[t-1, :] = self.bursting
            self.spike_count[t-1, :] = self.spiking
            # self.spike_count_int_b[t-1, :] = self.inter_spikes_b

            if self.inh_plasticity:

                self.f_E_trace = self.f_E_trace * np.exp(-self.dt / self.tau_fE) + self.A_fE * self.spiking

                # Update pre-synaptic inhibitory trace (f_I for interneuron B)
                # f_I(t+dt) = f_I(t) * exp(-dt / tau_fI) + A_fI * S_I(t)
                self.f_I_trace_b = self.f_I_trace_b * np.exp(-self.dt / self.tau_fI) + self.A_fI * self.inter_spikes_b
                term1 = np.outer(self.spiking, self.f_I_trace_b) # Shape (n_pyr, n_int_b)

                # Term 2: f_E^i * V_I^j (Pre-synaptic I spike, modulated by post-synaptic E trace)
                # f_E^i is self.f_E_trace (n_pyr,)
                # V_I^j is self.inter_spikes_b (n_int_b,)
                # So, term2[i,j] = f_E[i] * V_I[j]
                term2 = np.outer(self.f_E_trace, self.inter_spikes_b) # Shape (n_pyr, n_int_b)

                # Combine terms and apply constant depression
                # Note: The constant_depression_term is applied per connection, so it's a scalar.
                # The result `instantaneous_delta_W` has shape (n_pyr, n_int_b)
                instantaneous_delta_W_ip_b = self.eta_inh * (term2 - term1) 
                instantaneous_delta_W_pi_b = self.eta_inh * (term1 + term2 - self.constant_depression_term)


                # Accumulate the weight change over the epoch (multiply by dt as it's a rate)
                # We need to transpose `instantaneous_delta_W` to match `self.W_ip_b`'s shape (n_int_b, n_pyr)
                self.accumulated_delta_W_ip_b += instantaneous_delta_W_ip_b.T * self.dt
                self.accumulated_delta_W_pi_b += instantaneous_delta_W_pi_b * self.dt

            t_old = t_new

        if plasticity:
            self.plasticity_step()
        self.values = values_new
        self.all_values = values


    def retrieve_place_cells(self, t_run, x_run, new_env = False, a = 0, t_per_epoch = None, top_down = False, plasticity = True, len_track = None):
        dt = self.dt
        m_a, m_b = self.ma_pc, self.mb_pc

        tn = t_run[-1]
        len_track = np.max(x_run)
        
        n_epochs = int(round(tn/t_per_epoch))

        # TODO: Here i will need to see which ones i will need to save and which ones i don't because of memory

        full_spike_count = np.zeros((int(round(tn/dt)+1), self.n_cells['pyramidal']))
        full_burst_count = np.zeros((int(round(tn/dt)+1), self.n_cells['pyramidal']))
        # self.full_spike_count_int_b = np.zeros((int(round(tn/dt)+1), self.n_cells['inter_b']))
        # self.full_CA3_activities = np.zeros((int(round(tn/dt)+1), self.n_cells['CA3']))
        # self.full_EC_activities = np.zeros((int(round(tn/dt)+1), self.n_cells['pyramidal']))

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
            #self.full_spike_count_int_b[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt), :] = self.spike_count_int_b
            # self.full_CA3_activities[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt), :] = self.I_b.T[:int(t_per_epoch/dt), :]
            # self.full_EC_activities[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt), :] = self.I_a.T[:int(t_per_epoch/dt), :]

        return full_spike_count, full_burst_count
    

    def get_input_map(self, area='CA3', env=0, a=0, n_bins = 256):
        """
        Get the CA3 activation map based on the CA3 activities and the original CA3 place field.
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


    def plasticity_step(self): 

        mean_events = np.mean(self.spike_count, axis=0)
        mean_bursts = np.mean(self.burst_count, axis=0)

        # self.burst_count.sum(axis=0).mean()
        # print('fr exc:', self.spike_count.sum(axis=0).mean(), 'br exc:',  self.burst_count.sum(axis=0).mean(), 'fr inh:', self.spike_count_int_b.sum(axis=0).mean())


        mean_Ib = np.mean(self.Ib_trace, axis=0)/(self.n_cells['pyramidal'])

        ### OLD: delta_W = self.eta * (np.outer(mean_bursts + self.alpha*(mean_events), mean_Ib))

        delta_W = self.eta*np.outer(mean_bursts, mean_Ib) + self.alpha*np.outer(mean_events, mean_Ib)
        ## then alpha_new = eta*alpha
        self.W_CA3 += delta_W   
        self.W_CA3 = np.where(self.W_CA3 < 0, 0, self.W_CA3)

        if np.sum(self.W_CA3) != 0:    
            W_CA3_norm = np.sum(self.W_CA3, axis=1)
            self.W_CA3 = self.W_CA3 / W_CA3_norm[:, np.newaxis] 
            self.W_CA3 = np.where(self.W_CA3 < 0, 0, self.W_CA3)
        else:
            print('Weights are zero')
            print(delta_W, self.W_CA3)
            quit()

        if self.inh_plasticity:

            
            self.W_ip_b += self.accumulated_delta_W_ip_b
            self.W_pi_b += self.accumulated_delta_W_pi_b

            self.W_ip_b = np.where(self.W_ip_b < 0, 0, self.W_ip_b)
            self.W_pi_b = np.where(self.W_pi_b < 0, 0, self.W_pi_b)


            # TODO (?) Optional: Add an upper bound if Wmax is desired, similar to the original text
            # self.W_ip_b = np.where(self.W_ip_b > self.W_max_ip_b, self.W_max_ip_b, self.W_ip_b) 
            # self.W_pi_b = np.where(self.W_pi_b > self.W_max_pi_b, self.W_max_pi_b, self.W_pi_b)

        self.plast_count += 1
