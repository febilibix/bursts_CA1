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
            params_basal,
            input_basal,
            params_apical,
            input_apical,
            params_inter,
            n_cells,
            weights = None,
            plasticity = False,
            learning_rate = 0.05
            ): 
        
        params_basal  = {"E_L": -65, "R": 10, "v_th": -50, "tau": 10}
        params_apical = {"E_L": -65, "R": 10, "v_th": -50, "tau": 5 }
        params_inter  = {"E_L": -65, "R": 10, "v_th": -50, "tau": 10}

        self.plasticity = plasticity
        self.eta = learning_rate # learning rate, TODO: make this a parameter

        self.pb, self.pa, self.pi = params_basal, params_apical, params_inter
        
        self.v_reset_b = params_basal['E_L']
        self.v_th_b = params_basal['v_th']

        self.v_reset_a = params_apical['E_L']
        self.v_th_a = params_apical['v_th']

        self.v_reset_i = params_inter['E_L']
        self.v_th_i = params_inter['v_th']

        self.I_b = input_basal
        self.I_a = input_apical

        self.tau_i = 20
            
        if weights is not None:
            self.W_pi_a = weights['pi_a']
            self.W_pi_b = weights['pi_b']
            self.W_ip_a = weights['ip_a']
            self.W_ip_b = weights['ip_b']
            self.W_pp_a = weights['pp_a']
            self.W_pp_b = weights['pp_b']

        if 'CA3' in n_cells and ('CA3' not in weights or weights['CA3'] is None):
            self.W_CA3 = np.random.rand(n_cells['pyramidal'], n_cells['CA3'])
            # self.W_CA3 = self.W_CA3 / np.sum(self.W_CA3)
             
            self.W_CA3 = self.W_CA3 / np.sum(self.W_CA3)

        elif 'CA3' in n_cells:
            self.W_CA3 = weights['CA3']

        else:
            self.W_CA3 = np.eye(n_cells['pyramidal'])

        # self.events, self.bursts = [np.zeros(n_cells['pyramidal'])], [np.zeros(n_cells['pyramidal'])]
        self.inter_spikes_a, self.inter_spikes_b = np.zeros(n_cells['inter_a']), np.zeros(n_cells['inter_b'])

        self.n_cells = n_cells
        self.pattern_count = 0 
        self.plast_count = 0
        

        
    def dynamics_basal(self, t, v):
        R_b, E_L_b, tau_b = self.pb['R'], self.pb['E_L'], self.pb['tau']
        v_dot = 1/tau_b * (E_L_b - v + R_b * (self.W_CA3@self.I_b(t) - self.W_pi_b@self.inter_spikes_b))

        return v_dot

  
    def dynamics_apical(self, t, v):
        R_a, E_L_a, tau_a = self.pa['R'], self.pa['E_L'], self.pa['tau']
        v_dot = 1/tau_a * (E_L_a - v + R_a * (self.I_a(t) - self.W_pi_a@self.inter_spikes_a))

        return v_dot
    

    def dynamics_interneuron_a(self, t, v):
        R_i, E_L_i = self.pi['R'], self.pi['E_L']

        v_dot = 1/self.tau_i * (E_L_i - v + R_i * self.W_ip_a @ self.spiking)

        return v_dot
    

    def dynamics_interneuron_b(self, t, v):
        R_i, E_L_i, tau_i = self.pi['R'], self.pi['E_L'], self.pi['tau']
        v_dot = 1/self.tau_i * (E_L_i - v + R_i * self.W_ip_b @ self.spiking)

        return v_dot
    
    ## TODO : Two things that could be improved: 
    ## 2. Have them both in the same function (then i guess i would only need one W_pi and W_ip)


    def run_simulation(self, v0, t0, tn, dt, event_plot = True, n_patterns = 2, n_presentations = 10, selected_neurons = None):

        self.bursting = np.zeros(len(v0['basal']))
        self.spiking = np.zeros(len(v0['basal']))
        spike_count = np.zeros((int(round(tn / dt)), len(v0['basal'])))
        burst_count = np.zeros((int(round(tn / dt)), len(v0['basal'])))

        t_values = [t0]

        dynamics_values = {
            'v_b'   : (self.dynamics_basal,         [v0['basal']]  ),
            'v_a'   : (self.dynamics_apical,        [v0['apical']] ),
            'v_i_a' : (self.dynamics_interneuron_a, [v0['inter_a']]),     # TODO : Possibly use different initial values for apical and basal interneurons
            'v_i_b' : (self.dynamics_interneuron_b, [v0['inter_b']]),
            }
        
        values = {k: v[1] for k, v in dynamics_values.items()}

        Ib_trace = []

        plast_step = int(tn/(2*n_patterns*n_presentations))
        next_plast_update = plast_step
        t_new = 0
        self.cosine_distances = np.zeros((n_presentations, n_patterns))
        self.n_patterns = n_patterns
        
        while round(t_values[-1]) < tn:
            
            t_old = t_new
            values_new = {}

            for value_name, (dynamics, value) in dynamics_values.items():
                value_new, t_new = runge_kutta(
                    t_values, value, dynamics, dt)
                values_new[value_name] = np.array(value_new)

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
            burst_count[t-1, :] = self.bursting
            spike_count[t-1, :] = self.spiking

            Ib_trace.append(self.I_b(t_new))
            t_values.append(t_new)

            ##### plasticity   #####
            if self.plasticity == False:
                continue

            if not (t_old < next_plast_update and t_new > next_plast_update):
                continue
            next_plast_update += plast_step
            # print('Plasticity', t_new, plast_step/dt )
            self._plasticity(t, plast_step, spike_count, burst_count, Ib_trace, dt, selected_neurons)

        # if self.plasticity:
# 
        #     self._plasticity(t, plast_step, spike_count, burst_count, Ib_trace, dt, selected_neurons)
 
        v_values = values['v_b'], values['v_a'], values['v_i_a'], values['v_i_b']

        # TODO : get rid of third return value
        if event_plot:
            events, bursts = self._get_events_bursts(spike_count, burst_count, tn, dt)
        else:
            events, bursts = None, None

        return t_values, v_values, v_values, spike_count, burst_count, events, bursts
    

    def _get_events_bursts(self, spike_count, burst_count, tn, dt):
        
        events = [[] for _ in range(self.n_cells['pyramidal'])] 
        bursts = [[] for _ in range(self.n_cells['pyramidal'])]

        for i in range(spike_count.shape[0]):

            t_new = i * dt
            [events[l[0]].append(t_new) for l in np.argwhere(spike_count[i, :])]
            [bursts[l[0]].append(t_new) for l in np.argwhere(burst_count[i, :])]

        return events, bursts
    

    def _plasticity(self, t, plast_step, spike_count, burst_count, Ib_trace, dt, selected_neurons):

        last_events = spike_count[int(t-(plast_step/dt)):t, :].copy()
        firing_rate = np.sum(last_events, axis = 0)/ plast_step
        mean_events = np.mean(last_events, axis=0)
        last_bursts = burst_count[int(t-(plast_step/dt)):t, :].copy()
        mean_bursts = np.mean(last_bursts, axis=0)

        self.pattern_count += 1

        pattern_index = (self.pattern_count+1) % (2 * self.n_patterns)
        if pattern_index < self.n_patterns:
            # print(1- cosine(firing_rate, selected_neurons[pattern_index]))
            self.cosine_distances[int(self.pattern_count // (2 *  self.n_patterns)), pattern_index] = 1 - cosine(firing_rate, selected_neurons[pattern_index])

        Ib_arr = np.array(Ib_trace)
        mean_Ib = np.mean(Ib_arr, axis=0)

        Ib_trace = []

        self.plast_count += 1
        
        delta_W = self.eta * (np.outer(mean_bursts - 0.1*mean_events, mean_Ib)) # 
        # print('Plasticity', self.plast_count, np.sum(delta_W))
        self.W_CA3 += delta_W    
        if np.sum(self.W_CA3) != 0:       
            self.W_CA3 = self.W_CA3 / (np.sum(self.W_CA3, axis=1, keepdims=True)*self.n_cells['pyramidal'])
            self.W_CA3 = np.where(self.W_CA3 < 0, 0, self.W_CA3)
        else:
            print('Weights are zero')
            print(delta_W, self.W_CA3)
            quit()

        
    def _get_active_neurons(self, event_count, n_patterns, dt, tn):
    
        active_neurons = np.zeros((n_patterns, event_count.shape[1]))
        time_step = tn/n_patterns/dt
        
        for i in range(n_patterns):
            activity = np.sum(event_count[int((i+1)*time_step - time_step/2): int((i+1)*time_step), :], axis = 0) / time_step
            active_neurons[i, :] = activity 

        return active_neurons