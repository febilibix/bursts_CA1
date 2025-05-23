import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import matplotlib
print(matplotlib.matplotlib_fname())
plt.style.use('./plot_style.mplstyle')
import pandas as pd


def runge_kutta(t, y_values, f, dt):

    ### TODO: Use scipy biult in runge kutta instead

    y = y_values[-1]

    k1 = dt * f(t, y)
    k2 = dt * f(t + dt, y + k1)

    y_new = y + 0.5 * (k1 + k2)
    t_new = t + dt

    return np.array(y_new), t_new


def gauss(x, mu, sig):
    diff = mu[:, :, None] - x[:, None, :]  # Shape: (2, 350000, 225)
    squared_dist = np.sum(diff**2, axis=0)  # Sum over the spatial dimensions -> (350000, 225)
    return np.exp(-0.5 * squared_dist / sig**2)


class PyramidalCells():

    def __init__(
            self, 
            n_cells,
            len_edge,
            learning_rate = 0.05,
            dt = 0.01,
            seed = None
            ): 
        
        self.dt = dt
        
        # TODO: I changed them all by an order of magnitude
    
        self.pb = {"E_L": -65, "R": 10, "v_th": -45, "tau": 0.1}
        self.pa = {"E_L": -65, "R": 10, "v_th": -35, "tau": 0.5} 
        self.pib = {"E_L": -65, "R": 10, "v_th": -45, "tau": .25} 
        self.pia = {"E_L": -65, "R": 10, "v_th": -45, "tau": .05}
        
        self.eta, self.eta_use = learning_rate, 0

        n_int_a, n_int_b, n_pyr = n_cells['inter_a'], n_cells['inter_b'], n_cells['pyramidal']

        self.W_ip_a = 8000*np.ones((n_int_a, n_pyr))/(n_pyr) # 7000
        self.W_ip_b = 4000*np.ones((n_int_b, n_pyr))/(n_pyr)
        self.W_pi_a = 500*np.ones((n_pyr, n_int_a))/n_int_a # 200
        self.W_pi_b = 30*np.ones((n_pyr, n_int_b))/n_int_b # 30
        self.W_CA3 = np.random.rand(n_cells['pyramidal'], n_cells['CA3'])  

        W_CA3_norm = np.sum(self.W_CA3, axis=1)
        self.W_CA3 = self.W_CA3 / W_CA3_norm[:, np.newaxis] 
        self.W_CA3 = np.where(self.W_CA3 < 0, 0, self.W_CA3)

        self.inter_spikes_a, self.inter_spikes_b = np.zeros(n_cells['inter_a']), np.zeros(n_cells['inter_b'])
        self.n_cells = n_cells
        self.plast_count = 0

        self.dynamics_values = {
            'v_b'    : (self.dynamics_basal,         [np.ones(n_cells['pyramidal'])*self.pb['E_L']]),
            'v_a'    : (self.dynamics_apical,        [np.ones(n_cells['pyramidal'])*self.pa['E_L']]),
            'v_i_a'  : (self.dynamics_interneuron_a, [np.ones(n_cells['inter_a'])*self.pia['E_L']]),  
            'v_i_b'  : (self.dynamics_interneuron_b, [np.ones(n_cells['inter_b'])*self.pib['E_L']]),
            }
                
        self.alpha, self.alpha_use = 0.05, 0 
        self.ma_pc, self.mb_pc = 5*40, 30*32

        left = -len_edge/10
        right = len_edge + len_edge/10
        step_CA3 = int(np.sqrt(n_cells['CA3']))*1j
        step_EC = int(np.sqrt(n_cells['pyramidal']))*1j

        rng = np.random.default_rng(seed=seed)

        m_CA3_x, m_CA3_y = np.mgrid[left:right:step_CA3, left:right:step_CA3]
        self.m_CA3 = np.vstack((m_CA3_x.flatten(), m_CA3_y.flatten()))
        self.m_CA3_new = np.vstack((m_CA3_x.flatten(), m_CA3_y.flatten()))

        m_EC_x, m_EC_y = np.mgrid[0:len_edge:step_EC, 0:len_edge:step_EC]
        self.m_EC = np.vstack((m_EC_x.flatten(), m_EC_y.flatten()))
        self.m_EC_new = np.vstack((m_EC_x.flatten(), m_EC_y.flatten()))
        
        self.m_CA3 = rng.permutation(self.m_CA3, axis=1) 
        self.m_CA3_new = rng.permutation(self.m_CA3_new, axis=1)
        self.m_EC = rng.permutation(self.m_EC, axis=1)
        self.m_EC_new = rng.permutation(self.m_EC_new, axis=1)

        
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
        v_dot = 1/tau_i * (E_L_i - v + R_i * (self.W_ip_a @ self.spiking ))
        return v_dot
    

    def dynamics_interneuron_b(self, t, v):
        R_i, E_L_i, tau_i = self.pib['R'], self.pib['E_L'], self.pib['tau']
        v_dot = 1/tau_i * (E_L_i - v + R_i * (self.W_ip_b @ self.spiking)) 
        return v_dot
    

    def learn_place_cells(self, t_run, x_run, t_per_epoch, top_down = True, len_track = None, plasiticty = True):
        return self.retrieve_place_cells(t_run, x_run, t_per_epoch=t_per_epoch, top_down=top_down, plasticity=plasiticty)


    def create_activity_pc(self, x, region, n_cells, m, new_env, m_cells, m_cells_new, a): 
        sigma_pf = 2 # TODO: TUNE
        n_active = int(n_cells) 
        a = a if new_env else 1
        
        activity = np.zeros((n_active, x.shape[1] + 1))
        activity[:, :-1] = m * ((1-a) * gauss(x, m_cells_new, sigma_pf) + a * gauss(x, m_cells, sigma_pf))
    
        return activity 
    

    def run_one_epoch(self, t_epoch, plasticity = True): 
        dt = self.dt

        self.bursting = np.zeros(self.n_cells['pyramidal'])
        self.spiking = np.zeros(self.n_cells['pyramidal'])

        self.spike_count = np.zeros((int(round(t_epoch / dt)), self.n_cells['pyramidal']))
        self.burst_count = np.zeros((int(round(t_epoch / dt)), self.n_cells['pyramidal']))
        self.spike_count_int_a = np.zeros((int(round(t_epoch / dt)), self.n_cells['inter_a']))
        self.spike_count_int_b = np.zeros((int(round(t_epoch / dt)), self.n_cells['inter_b']))

        values = {k: v[1] for k, v in self.dynamics_values.items()}

        values_new = {}
        self.t_epoch = t_epoch

        t, t_old = 0, 0

        for t in range(1, int(round(t_epoch / dt)) + 1):
             
            for value_name, (dynamics, value) in self.dynamics_values.items():
                value_new, t_new = runge_kutta(
                    t_old, value, dynamics, dt)
                values_new[value_name] = np.array(value_new)

            self.spiking = (values_new['v_b'] > self.pb['v_th']).astype(int)
            # self.pb['v_th'] = self.pb['v_th'] + 50000*((self.spiking.sum()/self.n_cells['pyramidal'] - 0.03*4/25)**3)  # *4/25
            # print(self.pb['v_th'], self.spiking.sum()/self.n_cells['pyramidal'])

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
            self.spike_count_int_b[t-1, :] = self.inter_spikes_b

            t_old = t_new

        # self.alpha_use = 
        if plasticity:
            self.plasticity_step()
        self.values = values_new
        self.all_values = values


    def retrieve_place_cells(self, t_run, x_run, new_env = False, a = 0, t_per_epoch = None, top_down = False, plasticity = True, len_track = None):
        dt = self.dt
        m_a, m_b = self.ma_pc, self.mb_pc

        tn = t_run[-1]
        len_track = np.max(x_run)
        print(len_track)
        
        n_epochs = int(round(tn/t_per_epoch))
        ## TODO: I will probably need to stop keeping track of all these things; this is what uses so much memory
        ## And i guess it makes my code a lot slower as well 
        full_spike_count = np.zeros((int(round(tn/dt)+1), self.n_cells['pyramidal']))
        full_burst_count = np.zeros((int(round(tn/dt)+1), self.n_cells['pyramidal']))
        # self.full_CA3_activities = np.zeros((int(round(tn/dt)+1), self.n_cells['CA3']))
        # self.full_EC_activities = np.zeros((int(round(tn/dt)+1), self.n_cells['pyramidal']))

        for j in range(n_epochs):
            t0_epoch = j*t_per_epoch
            # self.alpha_use = self.alpha if t0_epoch > 100 else self.alpha/100*t0_epoch
            # self.eta_use = self.eta if t0_epoch > 100 else self.eta/100*t0_epoch

            x = x_run[:, int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt)]

            self.I_b = self.create_activity_pc(x, 'CA3', self.n_cells['CA3'], m_b, new_env, self.m_CA3, self.m_CA3_new, a)
            
            if top_down:
                self.I_a = self.create_activity_pc(x, 'EC', self.n_cells['pyramidal'], m_a, new_env, self.m_EC, self.m_EC_new, a = 0)
            else: 
                self.I_a = np.zeros((self.n_cells['pyramidal'], int(t_per_epoch/dt+1))) 

            self.Ib_trace = self.I_b[:, :x.shape[1]].T
            self.run_one_epoch(t_per_epoch, plasticity)

            full_spike_count[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt), :] = self.spike_count
            full_burst_count[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt), :] = self.burst_count
            # self.full_CA3_activities[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt), :] = self.I_b.T[:int(t_per_epoch/dt), :]
            # self.full_EC_activities[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt), :] = self.I_a.T[:int(t_per_epoch/dt), :]


        return full_spike_count, full_burst_count


    def plasticity_step(self): 

        mean_events = np.mean(self.spike_count, axis=0)
        mean_bursts = np.mean(self.burst_count, axis=0)

        mean_Ib = np.mean(self.Ib_trace, axis=0)/(self.n_cells['pyramidal'])

        delta_W = self.eta * (np.outer(mean_bursts + self.alpha_use*(mean_events), mean_Ib) )
        self.W_CA3 += delta_W   
        
        if np.sum(self.W_CA3) != 0:    
            W_CA3_norm = np.sum(self.W_CA3, axis=1)
            self.W_CA3 = self.W_CA3 / W_CA3_norm[:, np.newaxis] 
            self.W_CA3 = np.where(self.W_CA3 < 0, 0, self.W_CA3)
        else:
            print('Weights are zero')
            print(delta_W, self.W_CA3)
            quit()

        self.plast_count += 1

