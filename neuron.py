import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr



def runge_kutta(t_values, y_values, f, dt):
    t = t_values[-1]
    y = y_values[-1]

    k1 = dt * f(t, y)
    k2 = dt * f(t + dt, y + k1)

    y_new = y + 0.5 * (k1 + k2)
    t_new = t + dt

    return np.array(y_new), t_new


def sparsify_matrix(W, p = 0.7):
    mask = np.random.choice([0, 1], size=W.shape, p=[p, 1-p])
    return W * mask


class PyramidalCells():

    def __init__(
            self, 
            n_cells,
            weights = None,
            learning_rate = 0.05
            ): 
        
        self.t_values = [0.]
        
        self.pb  = {"E_L": -65, "R": 10, "v_th": -50, "tau": 10}
        self.pa = {"E_L": -65, "R": 10, "v_th": -50, "tau": 5 }
        self.pi  = {"E_L": -65, "R": 10, "v_th": -50, "tau": 20}

        self.eta = learning_rate 
        
        self.v_reset_b, self.v_th_b = self.pb['E_L'], self.pb['v_th']
        self.v_reset_a, self.v_th_a = self.pa['E_L'], self.pa['v_th']
        self.v_reset_i, self.v_th_i = self.pi['E_L'], self.pi['v_th']
        self.tau_i = self.pi['tau']

        n_int_a, n_int_b, n_pyr = n_cells['inter_a'], n_cells['inter_b'], n_cells['pyramidal']

        self.W_ip_a = 2000*np.ones((n_int_a, n_pyr))/np.sqrt(n_pyr)
        self.W_ip_b = 1000*np.ones((n_int_b, n_pyr))/np.sqrt(n_pyr)
        self.W_pi_a = 1000*np.ones((n_pyr, n_int_a))/np.sqrt(n_pyr)
        self.W_pi_b = 1000*np.ones((n_pyr, n_int_b))/np.sqrt(n_pyr)
        self.W_pp_a = np.zeros((n_int_a, n_int_a))
        self.W_pp_b = np.zeros((n_int_b, n_int_b))
        
        # initialize weights
        if 'CA3' in n_cells and ('CA3' not in weights or weights['CA3'] is None):
            self.W_CA3 = np.random.rand(n_cells['pyramidal'], n_cells['CA3'])             
            self.W_CA3 = self.W_CA3 / np.sum(self.W_CA3)
        elif 'CA3' in n_cells:
            self.W_CA3 = weights['CA3']
        else:
            self.W_CA3 = np.eye(n_cells['pyramidal'])

        self.inter_spikes_a, self.inter_spikes_b = np.zeros(n_cells['inter_a']), np.zeros(n_cells['inter_b'])
        self.n_cells = n_cells
        self.plast_count = 0

        self.dynamics_values = {
            'v_b'   : (self.dynamics_basal,         [self.pb['E_L']]),
            'v_a'   : (self.dynamics_apical,        [self.pa['E_L']]),
            'v_i_a' : (self.dynamics_interneuron_a, [self.pi['E_L']]),  
            'v_i_b' : (self.dynamics_interneuron_b, [self.pi['E_L']]),
            }
        
        self.v0 = np.array([self.pb['E_L'], self.pa['E_L'], self.pi['E_L'], self.pi['E_L']])
        
        
    def dynamics_basal(self, t, v):
        R_b, E_L_b, tau_b = self.pb['R'], self.pb['E_L'], self.pb['tau']
        v_dot = 1/tau_b * (E_L_b - v + R_b * (self.W_CA3@self.I_b(t) - self.W_pi_b@self.inter_spikes_b))
        return v_dot

  
    def dynamics_apical(self, t, v):
        R_a, E_L_a, tau_a = self.pa['R'], self.pa['E_L'], self.pa['tau']
        v_dot = 1/tau_a * (E_L_a - v + R_a * (self.I_a(t) - self.W_pi_a@self.inter_spikes_a))
        return v_dot
    

    def dynamics_interneuron_a(self, t, v):
        R_i, E_L_i, tau_i = self.pi['R'], self.pi['E_L'], self.pi['tau']
        v_dot = 1/tau_i * (E_L_i - v + R_i * self.W_ip_a @ self.spiking)
        return v_dot
    

    def dynamics_interneuron_b(self, t, v):
        R_i, E_L_i, tau_i = self.pi['R'], self.pi['E_L'], self.pi['tau']
        v_dot = 1/tau_i * (E_L_i - v + R_i * self.W_ip_b @ self.spiking)
        return v_dot
    
    
    def create_I_CA3(self, pattern, tn, dt, t0_epoch):
    
        m_b = 8*self.n_cells['pyramidal']
        sigma = 2*6.5/1000
        I = np.zeros((int(tn/dt + 10), self.n_cells['CA3'])) # +10 to avoid index out of bounds

        for i in range(int(tn/dt + 10)):
            I[i, :] = m_b*pattern + np.random.normal(0, sigma, self.n_cells['CA3']) # sqrt(sigma) i had before

        self.I_b = lambda t: I[int((t-t0_epoch)/dt), :] 


    def create_I_EC(self, pattern, tn, dt, t0_epoch):
    
        m_a = 6.5
        sigma = 2*6.5/1000
        I = np.zeros((int(tn/dt + 10), self.n_cells['pyramidal']))

        for i in range(int(tn/dt + 10)):
            I[i, :] = m_a*pattern + np.random.normal(0, sigma, self.n_cells['pyramidal']) 

        self.I_a = lambda t: I[int((t-t0_epoch)/dt), :] 
    

    def learn_patterns(self, patterns, top_down, n_presentations, t_per_pattern, dt=.01):
        
        n_patterns = patterns.shape[1]
        tn = n_presentations * n_patterns * t_per_pattern

        self.spike_count = np.zeros((int(round(tn / dt)), self.n_cells['pyramidal']))
        self.burst_count = np.zeros((int(round(tn / dt)), self.n_cells['pyramidal']))

        n_epochs = n_patterns * n_presentations
        
        for i in range(n_presentations):
            for j in range(n_patterns):
                t_epoch = self.t_values[-1] + tn//n_epochs
                t0_epoch = self.t_values[-1]
                self.create_I_CA3(patterns[:,j], t_per_pattern, dt, t0_epoch)
                self.create_I_EC(top_down[:,j], t_per_pattern, dt, t0_epoch)
                self.run_one_epoch(t_epoch, dt)
            

    def learn_place_cells(self, t_run, x_run, t_per_epoch, dt):

        m_a, m_b = 2*6.5, 8*self.n_cells['pyramidal']*2

        tn = t_run[-1]
        len_track = np.max(x_run)
        n_epochs = int(tn//t_per_epoch)

        self.spike_count = np.zeros((int(round(tn / dt +10)), self.n_cells['pyramidal']))
        self.burst_count = np.zeros((int(round(tn / dt +10)), self.n_cells['pyramidal']))

        self.I_b, self.m_CA3, self.CA3_act = self.create_activity_pc(x_run, len_track, dt, tn, self.n_cells['CA3'], m_b)
        self.I_a, self.m_EC, _ = self.create_activity_pc(x_run, len_track, dt, tn, self.n_cells['pyramidal'], m_a)
        
        for j in range(n_epochs):
            t_epoch = self.t_values[-1] + t_per_epoch
            print('Epoch', t_epoch)
            self.run_one_epoch(t_epoch, dt)


    def create_activity_pc(self, x, len_track, dt, tn, n_cells, m):
        sigma_pf = len_track/8
        m_cells = np.arange(0, len_track, len_track/n_cells) 
        np.random.shuffle(m_cells)

        activity = np.zeros((n_cells, int(tn/dt + 10)))
        for i in range(int(tn/dt)):
            activity[:, i] = np.exp(-0.5 * ((m_cells - x[i])**2) / sigma_pf**2)

        active_cells = np.random.choice([0, 1], size=(n_cells,), p=[0, 1]) # TODO: CHANGE THIS TO A PROBABILITY
        activity = m * activity * active_cells[:, np.newaxis]

        return lambda t: activity[:, int((t)/dt)], m_cells, activity
    

    def run_one_epoch(self, t_epoch, dt):
            
        self.bursting = np.zeros(self.n_cells['pyramidal'])
        self.spiking = np.zeros(self.n_cells['pyramidal'])

        values = {k: v[1] for k, v in self.dynamics_values.items()}

        self.Ib_trace = []
        values_new = {}
        t0 = self.t_values[-1]

        while round(self.t_values[-1]) < t_epoch:
            
            for value_name, (dynamics, value) in self.dynamics_values.items():
                value_new, t_new = runge_kutta(
                    self.t_values, value, dynamics, dt)
                values_new[value_name] = np.array(value_new)

            # print(values_new['v_b'])
            self.spiking = (values_new['v_b'] > self.v_th_b).astype(int)
            self.bursting = ((values_new['v_a'] > self.v_th_a) & self.spiking).astype(int)
            
            self.inter_spikes_a = (values_new['v_i_a'] > self.v_th_i).astype(int)
            self.inter_spikes_b = (values_new['v_i_b'] > self.v_th_i).astype(int)
            
            values_new['v_a'] = np.where(self.spiking, self.v_reset_a, values_new['v_a'])
            values_new['v_b'] = np.where(self.spiking, self.v_reset_b, values_new['v_b'])

            values_new['v_i_b'] = np.where(self.inter_spikes_b, self.v_reset_i, values_new['v_i_b'])
            values_new['v_i_a'] = np.where(self.inter_spikes_a, self.v_reset_i, values_new['v_i_a'])
            
            [values[i].append(values_new[i]) for i in ['v_a', 'v_b', 'v_i_a', 'v_i_b']]

            t = int(round(t_new/dt))
            self.burst_count[t-1, :] = self.bursting
            self.spike_count[t-1, :] = self.spiking

            self.Ib_trace.append(self.I_b(t_new))
            self.t_values.append(t_new)

        t_epoch = self.t_values[-1]
        self.plasticity_step(t0, t_epoch, dt)
        self.values = values_new
    

    def pattern_retrieval(self, patterns, top_down, t_per_pattern, dt=.01):
        self.cosine_distances = []
        
        n_patterns = patterns.shape[1]
        tn = n_patterns * t_per_pattern

        self.spike_count = np.concatenate([self.spike_count, np.zeros((int(round(tn / dt)), self.n_cells['pyramidal']))])
        self.burst_count = np.concatenate([self.burst_count, np.zeros((int(round(tn / dt)), self.n_cells['pyramidal']))])

        n_epochs = n_patterns 
        zero_top_down = np.zeros((int(tn/dt + 10), self.n_cells['pyramidal']))
        
        for j in range(n_patterns):
            t_epoch = self.t_values[-1] + tn//n_epochs
            t0_epoch = self.t_values[-1]
            self.create_I_CA3(patterns[:,j], t_per_pattern, dt, t0_epoch)
            self.I_a = lambda t: zero_top_down[int((t-t0_epoch)/dt), :]
            self.run_one_epoch(t_epoch, dt)
            firing_rate = self.spike_count[int(t0_epoch/dt):int(t_epoch/dt), :].mean(axis=0) 
            cosine_distance = 1-cosine(firing_rate, top_down[:,j])
            self.cosine_distances.append(cosine_distance)
    

    def plasticity_step(self, t0, tn, dt):
        
        last_events = self.spike_count[int(t0/dt):int(tn/dt), :]
        mean_events = np.mean(last_events, axis=0)
        last_bursts = self.burst_count[int(t0/dt):int(tn/dt), :]
        mean_bursts = np.mean(last_bursts, axis=0)
 
        Ib_arr = np.array(self.Ib_trace)
        mean_Ib = np.mean(Ib_arr, axis=0)
        
        delta_W = self.eta * (np.outer(mean_bursts - 0.1*mean_events, mean_Ib))
        
        self.W_CA3 += delta_W    
        if np.sum(self.W_CA3) != 0:       
            self.W_CA3 = self.W_CA3 / (np.sum(self.W_CA3, axis=1, keepdims=True)*self.n_cells['pyramidal'])
            self.W_CA3 = np.where(self.W_CA3 < 0, 0, self.W_CA3)
        else:
            print('Weights are zero')
            print(delta_W, self.W_CA3)
            quit()
        print('Plasticity', self.plast_count, np.sum(delta_W))

        self.Ib_trace = []
        self.plast_count += 1
