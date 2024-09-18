import numpy as np


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
            W_pi = None,     
            W_ip_a = None,  
            W_ip_b = None,     
            W_pp = None
            ): 
        
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
            
        ## TODO: No idea how this should be initialized
        # self.W_pi = 200*sparsify_matrix(np.random.rand(n_cells['pyramidal'],n_cells['inter']))
        # self.W_ip_a = 200*sparsify_matrix(np.random.rand(n_cells['inter'],n_cells['pyramidal']))
        # self.W_ip_b = 200*sparsify_matrix(np.random.rand(n_cells['inter'],n_cells['pyramidal']))
        # self.W_pp = 200*sparsify_matrix(np.random.rand(n_cells['inter'],n_cells['inter']))

        if W_pi is not None:
            self.W_pi = W_pi
        if W_ip_a is not None:
            self.W_ip_a = W_ip_a
        if W_ip_b is not None:
            self.W_ip_b = W_ip_b
        if W_pp is not None:
            self.W_pp = W_pp

        ## Normalize matrices to have the same sum of weights for each cell
        ## TODO: make this section look nicer
        ## TODO: When implementing plasticity, this should be done after each update

        # if self.W_pi.ndim == 1:
        #     self.W_pi = np.expand_dims(self.W_pi, axis=0)
        # if self.W_ip_a.ndim == 1:
        #     self.W_ip_a = np.expand_dims(self.W_ip_a, axis=0)
        # if self.W_ip_b.ndim == 1:
        #     self.W_ip_b = np.expand_dims(self.W_ip_b, axis=0)
        # 
        # if np.all(np.sum(self.W_pi, axis=1, keepdims=True) == 0):
        #     self.W_pi = np.zeros_like(self.W_pi)
        # else:
        #     self.W_pi = self.W_pi / np.sum(self.W_pi, axis=1, keepdims=True) * 1000
        # 
        # if np.all(np.sum(self.W_ip_a, axis=1, keepdims=True) == 0):
        #     self.W_ip_a = np.zeros_like(self.W_ip_a)
        # else:
        #     self.W_ip_a = self.W_ip_a / np.sum(self.W_ip_a, axis=1, keepdims=True) * 1000
        # 
        # if np.all(np.sum(self.W_ip_b, axis=1, keepdims=True) == 0):
        #     self.W_ip_b = np.zeros_like(self.W_ip_b)
        # else:
        #     self.W_ip_b = self.W_ip_b / np.sum(self.W_ip_b, axis=1, keepdims=True) * 1000

        #### 


        self.I_ip_a, self.I_pi_a, self.I_ip_b, self.I_pi_b = 0, 0, 0, 0

        self.events, self.bursts = [np.zeros(n_cells['pyramidal'])], [np.zeros(n_cells['pyramidal'])]
        self.inter_spikes_a, self.inter_spikes_b = np.zeros(n_cells['inter']), np.zeros(n_cells['inter'])

        
    def dynamics_basal(self, t, v):
        R_b, E_L_b, tau_b = self.pb['R'], self.pb['E_L'], self.pb['tau']
        v_dot = 1/tau_b * (E_L_b - v + R_b * (self.I_b(t) - self.W_pi@self.inter_spikes_b))

        return v_dot

  
    def dynamics_apical(self, t, v):
        R_a, E_L_a, tau_a = self.pa['R'], self.pa['E_L'], self.pa['tau']
        v_dot = 1/tau_a * (E_L_a - v + R_a * (self.I_a(t) - self.W_pi@self.inter_spikes_a))

        return v_dot
    

    def dynamics_interneuron_a(self, t, v):
        R_i, E_L_i, tau_i = self.pi['R'], self.pi['E_L'], self.pi['tau']



        v_dot = 1/self.tau_i * (E_L_i - v + R_i * self.W_ip_a @ self.spiking)

        return v_dot
    

    def dynamics_interneuron_b(self, t, v):
        R_i, E_L_i, tau_i = self.pi['R'], self.pi['E_L'], self.pi['tau']
        v_dot = 1/self.tau_i * (E_L_i - v + R_i * self.W_ip_b @ self.spiking)

        return v_dot
    
    ## TODO : Two things that could be improved: 
    ## 1. Make it possible to have differing amounts of the two types of interneurons (i.e., need W_pi_a and W_pi_b i guess)
    ## 2. Have them both in the same function (then i guess i would only need one W_pi and W_ip)


    def run_simulation(self, v0, t0, tn, dt):

        self.bursting = np.zeros(len(v0['basal']))
        self.spiking = np.zeros(len(v0['basal']))
        spike_count = [np.zeros(len(v0['basal']))]
        burst_count = [np.zeros(len(v0['basal']))]

        events = [[] for _ in range(len(v0['basal']))] 
        bursts = [[] for _ in range(len(v0['basal']))]

        t_values = [t0]

        dynamics_values = {
            'v_b'   : (self.dynamics_basal,         [v0['basal']]                ),
            'v_a'   : (self.dynamics_apical,        [v0['apical']]               ),
            'v_i_a' : (self.dynamics_interneuron_a, [v0['inter']]                ),     # TODO : Possibly use different initial values for apical and basal interneurons
            'v_i_b' : (self.dynamics_interneuron_b, [v0['inter']]                ),
            }
        
        values = {k: v[1] for k, v in dynamics_values.items()}
        
        while t_values[-1] < tn:

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

            burst_count.append(self.bursting)
            spike_count.append(self.spiking)
            self.events.append(spike_count[-1])
            [events[l[0]].append(t_new) for l in np.argwhere(self.spiking)]
            [bursts[l[0]].append(t_new) for l in np.argwhere(self.bursting)]

            t_values.append(t_new)
        
        v_values = values['v_b'], values['v_a'], values['v_i_a'], values['v_i_b']

        # TODO : get rid of third return value

        return t_values, v_values, v_values, np.array(spike_count), np.array(burst_count), events, bursts