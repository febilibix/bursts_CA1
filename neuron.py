import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import matplotlib
print(matplotlib.matplotlib_fname())
plt.style.use('./plot_style.mplstyle')
import csv


def runge_kutta(t_values, y_values, f, dt):
    t = t_values[-1]
    y = y_values[-1]

    k1 = dt * f(t, y)
    k2 = dt * f(t + dt, y + k1)

    y_new = y + 0.5 * (k1 + k2)
    t_new = t + dt

    return np.array(y_new), t_new


def sigmoid(x, x0, k):
    y = 1 / (1 + np.exp(-k*(x-x0)))
    return y

### TODO: I think at some point I want to combine some of the functionality of retreiving and 
###       learning into one function, for readability and to avoid code duplication


class PyramidalCells():

    def __init__(
            self, 
            n_cells,
            weights = dict(),
            learning_rate = 0.05,
            dt = 0.01, 
            p_active = (1,1)
            ): 
        
        
        self.dt = dt
        
        self.t_values = [0.]
        
        self.pb  = {"E_L": -65, "R": 10, "v_th": -45, "tau": 0.5}
        self.pa = {"E_L": -65, "R": 10, "v_th": -45, "tau": 1} 
        #  {"E_L": -65, "R": 10, "v_th": -50, "tau": 5}
        self.pi  = {"E_L": -65, "R": 10, "v_th": -50, "tau": .5} 
        
        self.eta = learning_rate 
        self.tau_i = self.pi['tau']
        self.p_active_CA3, self.p_active_EC = p_active

        n_int_a, n_int_b, n_pyr = n_cells['inter_a'], n_cells['inter_b'], n_cells['pyramidal']

        inh = 1
        if 'inh' in weights and weights['inh'] is not None:
            inh = weights['inh']
        self.W_ip_a = 140*np.ones((n_int_a, n_pyr))/(n_pyr*self.p_active_EC)
        self.W_ip_b = 7000*np.ones((n_int_b, n_pyr))/(n_pyr*self.p_active_EC)
        self.W_pi_a = .1*inh*70*np.ones((n_pyr, n_int_a))/n_int_a
        self.W_pi_b = .1*inh*70*np.ones((n_pyr, n_int_b))/n_int_b
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
            'v_b'    : (self.dynamics_basal,         [np.ones(n_cells['pyramidal'])*self.pb['E_L']]),
            'v_a'    : (self.dynamics_apical,        [np.ones(n_cells['pyramidal'])*self.pa['E_L']]),
            'v_i_a'  : (self.dynamics_interneuron_a, [np.ones(n_cells['inter_a'])*self.pi['E_L']]),  
            'v_i_b'  : (self.dynamics_interneuron_b, [np.ones(n_cells['inter_b'])*self.pi['E_L']]),
            'x_b'    : (self.synaptic_trace_pyr,     [np.zeros(n_cells['pyramidal'])]),
            'x_int_b': (self.synaptic_trace_ib,      [np.zeros(n_cells['inter_b'])]),
            }
        
        self.v0 = np.array([self.pb['E_L'], self.pa['E_L'], self.pi['E_L'], self.pi['E_L']])
        self.burst_rate = []
        self.spike_rate = []

        
        self.trace_events, self.trace_bursts = [], []
        self.alpha = 0.05

        self.ma_pc, self.mb_pc = 5*8,  2*16 # * self.n_cells['pyramidal'] # TODO: Ask federico about this scaling
        self.burst_collector = []

        self.all_CA3_activities = np.zeros((n_cells['CA3'], (int(1000/dt))))
        
        self.rho_0_spikes = 15 # 15 Hz target spike rate
        self.tau_stdp = 2
        self.lr_i_b = 0.5
        
        
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
        v_dot = 1/tau_i * (E_L_i - v + R_i * self.W_ip_a @ self.spiking + np.random.normal(0, 2, self.n_cells['inter_a']))
        return v_dot
    

    def dynamics_interneuron_b(self, t, v):
        R_i, E_L_i, tau_i = self.pi['R'], self.pi['E_L'], self.pi['tau']
        v_dot = 1/tau_i * (E_L_i - v + R_i * self.W_ip_b @ self.spiking + np.random.normal(0, 2, self.n_cells['inter_b']))
        return v_dot
    

    def synaptic_trace_pyr(self, t, x):
        # print(self.spiking)
        x_dot = -x/self.tau_stdp + 1*self.spiking
        return x_dot
    

    def synaptic_trace_ib(self, t, x):
        # print(self.inter_spikes_b)
        x_dot = -x/self.tau_stdp + 1*self.inter_spikes_b
        return x_dot
            

    def learn_place_cells(self, t_run, x_run, t_per_epoch, top_down = True):
        dt = self.dt

        m_a, m_b = self.ma_pc, self.mb_pc 

        tn = t_run[-1]
        len_track = np.max(x_run)
        n_epochs = int(round(tn/t_per_epoch))

        full_spike_count = np.zeros((int(round(tn/dt)+1), self.n_cells['pyramidal']))
        full_burst_count = np.zeros((int(round(tn/dt)+1), self.n_cells['pyramidal']))

        x = x_run[:int((t_per_epoch)/dt)]
        self.I_b, self.m_CA3, self.CA3_act = self.create_activity_pc(x, len_track, t_per_epoch, self.n_cells['CA3'], m_b, self.p_active_CA3)
        if top_down:
            self.I_a, self.m_EC, _ = self.create_activity_pc(x, len_track, t_per_epoch, self.n_cells['pyramidal'], m_a, self.p_active_EC)
        else:
            self.I_a = lambda t: np.zeros(self.n_cells['pyramidal'])
            self.m_EC = np.arange(0, len_track, len_track/self.n_cells['pyramidal']) ## TODO: I guess i will need to adapt this, not sure yet how

        for j in range(n_epochs):
            t0_epoch = j*t_per_epoch

            self.spike_count = np.zeros((int(round(t_per_epoch / dt)), self.n_cells['pyramidal']))
            self.burst_count = np.zeros((int(round(t_per_epoch / dt)), self.n_cells['pyramidal']))
            self.spike_count_int_a = np.zeros((int(round(t_per_epoch / dt)), self.n_cells['inter_a']))
            self.spike_count_int_b = np.zeros((int(round(t_per_epoch / dt)), self.n_cells['inter_b']))

            x = x_run[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt)]

            self.I_b, self.m_CA3, self.CA3_act = self.create_activity_pc(x, len_track, t_per_epoch, self.n_cells['CA3'], m_b, self.p_active_CA3, self.m_CA3)
            if top_down:
                self.I_a, self.m_EC, _ = self.create_activity_pc(x, len_track, t_per_epoch, self.n_cells['pyramidal'], m_a, self.p_active_EC, self.m_EC)
            else:
                self.I_a = lambda t: np.zeros(self.n_cells['pyramidal'])
            
            self.run_one_epoch(t_per_epoch)

            full_spike_count[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt), :] = self.spike_count
            full_burst_count[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt), :] = self.burst_count

            self.all_CA3_activities[:, int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt)] = self.CA3_act.T[:int(t_per_epoch/dt), :].T

        return full_spike_count, full_burst_count


    def create_activity_pc(self, x, len_track, t_per_epoch, n_cells, m, p_active, m_cells = None): 
        dt = self.dt
        sigma_pf = len_track/8
        # active_cells = np.random.choice([0, 1], size=(n_cells,), p=[0.3, 0.7])
        n_active = int(n_cells*p_active) 
        # TODO: I will probably need to make the m_cells a class attribute
        
        m_cells_new = np.linspace(-.5*sigma_pf, len_track + len_track/n_active + .5*sigma_pf, n_active)
        
        activity = np.zeros((n_active, x.shape[0] + 5))
        
        for i in range(x.shape[0]):
            activity[:, i] = m * np.exp(-0.5 * ((m_cells_new - x[i])**2) / sigma_pf**2)

        if m_cells is None:
            activity = np.vstack([activity, np.zeros((n_cells-n_active, x.shape[0] + 5))])
            m_cells = np.hstack([m_cells_new, np.zeros(n_cells - n_active)])

            activity_indices = np.arange(n_cells)
            np.random.shuffle(activity_indices)

            activity = activity[activity_indices, :]
            m_cells = m_cells[activity_indices] 

        else:
            new_activity = np.zeros((n_cells, x.shape[0] + 5))

            m_cells_new_indices = {value: idx for idx, value in enumerate(m_cells_new)}
            valid_indices = np.isin(m_cells, m_cells_new) & (m_cells != 0)
            mapped_indices = np.vectorize(m_cells_new_indices.get)(m_cells[valid_indices])
            new_activity[valid_indices] = activity[mapped_indices]
            activity = new_activity            

        return lambda t: activity[:, int(t/dt)], m_cells, activity
    

    def run_one_epoch(self, t_epoch, plasticity = True): 
        dt = self.dt

        self.t_values = [0.]
            
        self.bursting = np.zeros(self.n_cells['pyramidal'])
        self.spiking = np.zeros(self.n_cells['pyramidal'])

        values = {k: v[1] for k, v in self.dynamics_values.items()}

        self.Ib_trace = []
        values_new = {}

        t = 0

        while round(self.t_values[-1], 3) < t_epoch:
            t += 1
            
            for value_name, (dynamics, value) in self.dynamics_values.items():
                value_new, t_new = runge_kutta(
                    self.t_values, value, dynamics, dt)
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
            self.spike_count_int_a[t-1, :] = self.inter_spikes_a
            self.spike_count_int_b[t-1, :] = self.inter_spikes_b

            self.Ib_trace.append(self.I_b(t_new))
            self.t_values.append(t_new)

        t_epoch = self.t_values[-1]
        if plasticity:
            self.plasticity_step()
        self.values = values_new
        self.all_values = values

        self.burst_rate.append((self.burst_count.sum(axis = 0)/t_epoch).mean())
        self.spike_rate.append((self.spike_count.sum(axis = 0)/t_epoch).mean())


    def retrieve_place_cells(self, t_run, x_run, new_env = False, a = 0, t_per_epoch = None, top_down = False):
        dt = self.dt
        m_a, m_b = self.ma_pc, self.mb_pc

        tn = t_run[-1]
        len_track = np.max(x_run)
        if not t_per_epoch:
            t_per_epoch = tn
        
        n_epochs = int(round(tn/t_per_epoch))
        full_spike_count = np.zeros((int(round(tn/dt)+1), self.n_cells['pyramidal']))
        full_burst_count = np.zeros((int(round(tn/dt)+1), self.n_cells['pyramidal']))

        for j in range(n_epochs):
            t0_epoch = j*t_per_epoch

            x = x_run[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt)]
            self.spike_count = np.zeros((int(round(t_per_epoch / dt)), self.n_cells['pyramidal']))
            self.burst_count = np.zeros((int(round(t_per_epoch / dt)), self.n_cells['pyramidal']))
            self.spike_count_int_a = np.zeros((int(round(t_per_epoch / dt)), self.n_cells['inter_a']))
            self.spike_count_int_b = np.zeros((int(round(t_per_epoch / dt)), self.n_cells['inter_b']))
            if not hasattr(self, 'I_b_n'):
                self.I_b_n, self.m_CA3_n, self.CA3_act_n = self.create_activity_pc(x, len_track, t_per_epoch, self.n_cells['CA3'], m_b, self.p_active_CA3)
            self.I_b_o, self.m_CA3_o, self.CA3_act_o = self.create_activity_pc(x, len_track, t_per_epoch, self.n_cells['CA3'], m_b, self.p_active_CA3, self.m_CA3)
            
            if not new_env:
                self.I_b, self.m_CA3, self.CA3_act = self.I_b_o, self.m_CA3_o, self.CA3_act_o
            else:
                self.m_CA3 = self.m_CA3_o
                self.CA3_act = (1-a)*self.CA3_act_o + a*self.CA3_act_n
                self.I_b = lambda t: self.CA3_act[:, int(t/dt)]  

            zero_top_down = np.zeros((int(round(tn/dt)+10), self.n_cells['pyramidal']))
            
            if top_down and new_env:
                self.I_a, self.m_EC, _ = self.create_activity_pc(x, len_track, t_per_epoch, self.n_cells['pyramidal'], m_a, self.p_active_EC)
            elif top_down:
                self.I_a, self.m_EC, _ = self.create_activity_pc(x, len_track, t_per_epoch, self.n_cells['pyramidal'], m_a, self.p_active_EC, self.m_EC)
            else: self.I_a = lambda t: zero_top_down[int(t/dt), :]   

            self.run_one_epoch(t_per_epoch, plasticity=True)

            full_spike_count[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt), :] = self.spike_count
            full_burst_count[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt), :] = self.burst_count

        return full_spike_count, full_burst_count


    def plasticity_step(self): 
        
        last_events = self.spike_count
        mean_events = np.mean(last_events, axis=0)
        last_bursts = self.burst_count
        mean_bursts = np.mean(last_bursts, axis=0)

        # self.spike_count_int_a
        # self.spike_count_int_b


        x_b = np.array(self.dynamics_values['x_b'][1])[-last_events.shape[0]:]
        x_int_b = np.array(self.dynamics_values['x_int_b'][1])[-last_events.shape[0]:]
        alpha_spikes = 2 * self.rho_0_spikes * self.tau_stdp

        if self.plast_count % 100 == 0:
            print(x_b, x_int_b)
            print(x_b.shape, x_int_b.shape)


        self.W_pi_b = self.W_pi_b + self.lr_i_b * (np.einsum('ij,ij->j', x_b, last_events)[:, np.newaxis])/x_b.shape[0]
        self.W_pi_b = self.W_pi_b + self.lr_i_b * (np.einsum('ij,ij->j', x_int_b - alpha_spikes, self.spike_count_int_b)[np.newaxis, :])/x_b.shape[0]

        # print(np.abs(np.abs(self.W_pi_b)))

        Ib_arr = np.array(self.Ib_trace)
        mean_Ib = np.mean(Ib_arr, axis=0)/(self.n_cells['pyramidal'] * self.p_active_EC) #*50

        # TODO: Turn this into numpy array to do properly 
        self.trace_events.append(mean_events)
        self.trace_bursts.append(mean_bursts)

        non_0 = mean_bursts[np.where(mean_bursts > 0)]

        self.burst_collector.extend(list(non_0))

        delta_W = self.eta * (np.outer(mean_bursts + self.alpha*(mean_events), mean_Ib)) 
        self.W_CA3 += delta_W    

        if np.sum(self.W_CA3) != 0:    
            self.W_CA3 = self.W_CA3 / (np.sum(self.W_CA3, axis=1, keepdims=True)) #  * self.n_cells['pyramidal'])
            self.W_CA3 = np.where(self.W_CA3 < 0, 0, self.W_CA3)
   
        else:
            print('Weights are zero')
            print(delta_W, self.W_CA3)
            quit()

        self.Ib_trace = []
        self.plast_count += 1

        print('Plasticity step:', self.plast_count)


def sigmoid(x, a=1, k=1, x0=0):
    return a / (1 + np.exp(-k*(x-x0)))

plt.figure()
a = 0.002
x = np.arange(0, 4*a, 0.0001)
plt.plot(x, sigmoid(x, a = 0.01, k=5000, x0=0.005))
plt.savefig('plots/sigmoid.png')
plt.close()