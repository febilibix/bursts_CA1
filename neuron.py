import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import matplotlib
print(matplotlib.matplotlib_fname())
plt.style.use('./plot_style.mplstyle')
import csv


def runge_kutta(t, y_values, f, dt):

    ### TODO: Use scipy biult in runge kutta instead

    y = y_values[-1]

    k1 = dt * f(t, y)
    k2 = dt * f(t + dt, y + k1)

    y_new = y + 0.5 * (k1 + k2)
    t_new = t + dt

    return np.array(y_new), t_new


def gauss(x, mu, sig):
    return np.exp(-0.5 * ((mu[:, None] - x[None, :])**2) / sig**2)


class PyramidalCells():

    def __init__(
            self, 
            n_cells,
            len_track,
            learning_rate = 0.05,
            dt = 0.01,
            n_env = 2,
            ): 
        
        self.dt = dt
        
        self.pb  = {"E_L": -65, "R": 10, "v_th": -45, "tau": 0.25}
        self.pa = {"E_L": -65, "R": 10, "v_th": -45, "tau": 1} 
        self.pi  = {"E_L": -65, "R": 10, "v_th": -45, "tau": .1} 
        
        self.eta = learning_rate 
        self.tau_i = self.pi['tau']

        n_int_a, n_int_b, n_pyr = n_cells['inter_a'], n_cells['inter_b'], n_cells['pyramidal']

        self.W_ip_a = 7000*np.ones((n_int_a, n_pyr))/(n_pyr)
        self.W_ip_b = 4000*np.ones((n_int_b, n_pyr))/(n_pyr)
        self.W_pi_a = 200*np.ones((n_pyr, n_int_a))/n_int_a
        self.W_pi_b = 30*np.ones((n_pyr, n_int_b))/n_int_b
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
            'v_i_a'  : (self.dynamics_interneuron_a, [np.ones(n_cells['inter_a'])*self.pi['E_L']]),  
            'v_i_b'  : (self.dynamics_interneuron_b, [np.ones(n_cells['inter_b'])*self.pi['E_L']]),
            }
                
        self.alpha = 0.05
        self.ma_pc, self.mb_pc = 40, 32
        self.all_CA3_activities = np.zeros((n_cells['CA3'], (int(1000/dt))))

        # TODO: Will I need to use rng here like in 2D case? Apparently it's best practice in numpy for some reason

        self.all_m_CA3 = []
        for i in range(n_env):
            self.all_m_CA3.append(np.linspace(-len_track/10, len_track + len_track/10, n_cells['CA3']))
            np.random.shuffle(self.all_m_CA3[-1])


        self.all_m_EC = []
        for i in range(n_env):
            self.all_m_EC.append(np.linspace(0, len_track, n_cells['pyramidal']))
            np.random.shuffle(self.all_m_EC[-1])
        
        # TODO: This should catch old code for now 
        self.m_CA3, self.m_CA3_new = self.all_m_CA3[0], self.all_m_CA3[1]
        self.m_EC, self.m_EC_new = self.all_m_EC[0], self.all_m_EC[1]

        
    def dynamics_basal(self, t, v):
        R_b, E_L_b, tau_b = self.pb['R'], self.pb['E_L'], self.pb['tau']
        v_dot = 1/tau_b * (E_L_b - v + R_b * (self.W_CA3@self.I_b[:, int(t/self.dt)] - self.W_pi_b@self.inter_spikes_b))
        return v_dot

  
    def dynamics_apical(self, t, v):
        R_a, E_L_a, tau_a = self.pa['R'], self.pa['E_L'], self.pa['tau']
        v_dot = 1/tau_a * (E_L_a - v + R_a * (self.I_a[:, int(t/self.dt)] - self.W_pi_a@self.inter_spikes_a))
        return v_dot
    

    def dynamics_interneuron_a(self, t, v):
        R_i, E_L_i, tau_i = self.pi['R'], self.pi['E_L'], self.pi['tau']
        v_dot = 1/tau_i * (E_L_i - v + R_i * (self.W_ip_a @ self.spiking ))
        return v_dot
    

    def dynamics_interneuron_b(self, t, v):
        R_i, E_L_i, tau_i = self.pi['R'], self.pi['E_L'], self.pi['tau']
        v_dot = 1/tau_i * (E_L_i - v + R_i * (self.W_ip_b @ self.spiking)) 
        return v_dot
    

    def learn_place_cells(self, t_run, x_run, t_per_epoch, top_down = True, len_track = None, plasiticty = True):
        return self.retrieve_place_cells(t_run, x_run, t_per_epoch=t_per_epoch, top_down=top_down, plasticity=plasiticty)


    def create_activity_pc(self, x, len_track, n_cells, m, m_cells, m_cells_new, a): 
        sigma_pf = len_track/16 # len_track/8
        n_active = int(n_cells) 
        # a = a if new_env else 1
        
        activity = np.zeros((n_active, x.shape[0] + 5))
        activity[:, :-5] = m * ((1-a) * gauss(x, m_cells_new, sigma_pf) + a * gauss(x, m_cells, sigma_pf))
    
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
            self.bursting = ((values_new['v_a'] > self.pa['v_th']) & self.spiking).astype(int)
            
            self.inter_spikes_a = (values_new['v_i_a'] > self.pi['v_th']).astype(int)
            self.inter_spikes_b = (values_new['v_i_b'] > self.pi['v_th']).astype(int)
            
            values_new['v_a'] = np.where(self.spiking, self.pa['E_L'], values_new['v_a'])
            values_new['v_b'] = np.where(self.spiking, self.pb['E_L'], values_new['v_b'])

            values_new['v_i_b'] = np.where(self.inter_spikes_b, self.pi['E_L'], values_new['v_i_b'])
            values_new['v_i_a'] = np.where(self.inter_spikes_a, self.pi['E_L'], values_new['v_i_a'])
            
            [values[i].append(values_new[i]) for i in list(self.dynamics_values.keys())]

            self.burst_count[t-1, :] = self.bursting
            self.spike_count[t-1, :] = self.spiking
            self.spike_count_int_b[t-1, :] = self.inter_spikes_b

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
        full_spike_count = np.zeros((int(round(tn/dt)+1), self.n_cells['pyramidal']))
        full_burst_count = np.zeros((int(round(tn/dt)+1), self.n_cells['pyramidal']))
        self.full_CA3_activities = np.zeros((int(round(tn/dt)+1), self.n_cells['CA3']))
        self.full_EC_activities = np.zeros((int(round(tn/dt)+1), self.n_cells['pyramidal']))

        for j in range(n_epochs):
            t0_epoch = j*t_per_epoch

            x = x_run[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt)]

            self.I_b = self.create_activity_pc(x, len_track, self.n_cells['CA3'], m_b, self.m_CA3, self.all_m_CA3[int(new_env)], a)
            
            if top_down:
                self.I_a = self.create_activity_pc(x, len_track, self.n_cells['pyramidal'], m_a, self.m_EC, self.all_m_EC[int(new_env)], a=0)
            else: 
                self.I_a = np.zeros((self.n_cells['pyramidal'], int(t_per_epoch/dt+5))) 

            self.Ib_trace = self.I_b[:, :x.shape[0]].T
            self.run_one_epoch(t_per_epoch, plasticity)

            full_spike_count[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt), :] = self.spike_count
            full_burst_count[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt), :] = self.burst_count
            self.full_CA3_activities[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt), :] = self.I_b.T[:int(t_per_epoch/dt), :]
            self.full_EC_activities[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt), :] = self.I_a.T[:int(t_per_epoch/dt), :]

        return full_spike_count, full_burst_count


    def plasticity_step(self): 

        mean_events = np.mean(self.spike_count, axis=0)
        mean_bursts = np.mean(self.burst_count, axis=0)

        mean_Ib = np.mean(self.Ib_trace, axis=0)/(self.n_cells['pyramidal'])

        delta_W = self.eta * (np.outer(mean_bursts + self.alpha*(mean_events), mean_Ib) )
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

