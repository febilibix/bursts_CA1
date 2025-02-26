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
        
        self.pb  = {"E_L": -65, "R": 10, "v_th": -45, "tau": 0.25}
        self.pa = {"E_L": -65, "R": 10, "v_th": -45, "tau": 1} 
        self.pi  = {"E_L": -65, "R": 10, "v_th": -45, "tau": .1} 
        
        self.eta = learning_rate 
        self.tau_i = self.pi['tau']
        self.p_active_CA3, self.p_active_EC = p_active

        n_int_a, n_int_b, n_pyr = n_cells['inter_a'], n_cells['inter_b'], n_cells['pyramidal']

        self.w_pi_b_total, self.w_ip_b_total = 1000, 1.25
        # self.w_ffi_b_total = 0.1

        self.W_ip_a = 7000*np.random.rand(n_int_a, n_pyr)/(n_pyr*self.p_active_EC)
        self.W_ip_b = np.random.rand(n_int_b, n_pyr) # 7000*np.random.rand(n_int_b, n_pyr)/(n_pyr*self.p_active_EC)
        self.W_pi_a = 70*np.random.rand(n_pyr, n_int_a)/n_int_a
        self.W_pi_b = np.random.rand(n_pyr, n_int_b) # 70*np.random.rand(n_pyr, n_int_b)/n_int_b
        self.W_ffi_b = np.random.rand(n_int_b, n_cells['CA3'])
        self.W_CA3 = np.random.rand(n_cells['pyramidal'], n_cells['CA3'])  

        W_ip_b_norm = np.sum(self.W_ip_b, axis=1)
        W_ffi_b_norm = np.sum(self.W_ffi_b, axis=1)
        W_CA3_norm = np.sum(self.W_CA3, axis=1)
        W_pi_b_norm = np.sum(self.W_pi_b, axis=1)

        self.W_pi_b = self.w_pi_b_total * self.W_pi_b / W_pi_b_norm[:, np.newaxis]
        self.W_pi_b = np.where(self.W_pi_b < 0, 0, self.W_pi_b)

        self.W_ip_b = self.w_ip_b_total * self.W_ip_b / (W_ip_b_norm + W_ffi_b_norm)[:, np.newaxis]
        self.W_ip_b = np.where(self.W_ip_b < 0, 0, self.W_ip_b)
                  
        self.W_CA3 = self.W_CA3 / W_CA3_norm[:, np.newaxis] 
        self.W_CA3 = np.where(self.W_CA3 < 0, 0, self.W_CA3)

        self.W_ffi_b = self.w_ip_b_total * self.W_ffi_b / (W_ip_b_norm + W_ffi_b_norm)[:, np.newaxis]
        self.W_ffi_b = np.where(self.W_ffi_b < 0, 0, self.W_ffi_b)

        self.inter_spikes_a, self.inter_spikes_b = np.zeros(n_cells['inter_a']), np.zeros(n_cells['inter_b'])
        self.n_cells = n_cells
        self.plast_count = 0
        self.act_tr_count = 0 

        self.dynamics_values = {
            'v_b'    : (self.dynamics_basal,         [np.ones(n_cells['pyramidal'])*self.pb['E_L']]),
            'v_a'    : (self.dynamics_apical,        [np.ones(n_cells['pyramidal'])*self.pa['E_L']]),
            'v_i_a'  : (self.dynamics_interneuron_a, [np.ones(n_cells['inter_a'])*self.pi['E_L']]),  
            'v_i_b'  : (self.dynamics_interneuron_b, [np.ones(n_cells['inter_b'])*self.pi['E_L']]),
            }
                
        self.alpha = 0.05
        self.ma_pc, self.mb_pc = 40, 32
        self.all_CA3_activities = np.zeros((n_cells['CA3'], (int(1000/dt))))
        
        ### initialize parameters of inhibitory plasticity
        self.trace_events_post = np.ones(n_pyr)  # initialize trace of spiking activity
        self.trace_events_pre = np.ones(n_int_b)  # initialize trace of spiking activity
        self.rho_0_spikes = 15 # 15 Hz target spike rate 
        self.rho_0_spikes_pre = 15 # 15 Hz target spike rate
        self.tau_stdp_b = .02
        self.lr_i_b = 0.1
        self.events_decay = np.exp(-self.dt/self.tau_stdp_b)
        
        ## initialize parameters for inhibitory plasticity of dendritic compartment
        self.trace_bursts_post = np.ones(n_pyr)  # initialize trace of spiking activity
        self.trace_bursts_pre = np.ones(n_int_a)  # initialize trace of spiking activity
        self.rho_0_bursts = 2 # 2 Hz target burst rate 
        self.rho_0_bursts_pre = 2
        self.tau_stdp_a = .02 
        self.lr_i_a = .1
        self.bursts_decay = np.exp(-self.dt/self.tau_stdp_a)

       

        self.inh_plasticity = True

        self.target_active = n_cells['pyramidal'] * self.p_active_EC
        self.activity_trace = np.zeros((n_pyr, 10)) # 10 steps of activity trace, get rid of hardcoding

        
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
        v_dot = 1/tau_i * (E_L_i - v + R_i * self.W_ip_a @ self.spiking) #  + np.random.normal(0, 2, self.n_cells['inter_a']))
        return v_dot
    

    def dynamics_interneuron_b(self, t, v):
        R_i, E_L_i, tau_i = self.pi['R'], self.pi['E_L'], self.pi['tau']
        v_dot = 1/tau_i * (E_L_i - v + R_i * (self.W_ffi_b@self.I_b(t) + self.W_ip_b @ self.spiking)) #  + np.random.normal(0, 2, self.n_cells['inter_b']))
        return v_dot
    

    def learn_place_cells(self, t_run, x_run, t_per_epoch, top_down = True, len_track = None, plasiticty = True):
        dt = self.dt

        m_a, m_b = self.ma_pc, self.mb_pc 

        tn = t_run[-1]
        if len_track is None:
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

            x = x_run[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt)]

            self.I_b, self.m_CA3, self.CA3_act = self.create_activity_pc(x, len_track, t_per_epoch, self.n_cells['CA3'], m_b, self.p_active_CA3, self.m_CA3)
            if top_down:
                self.I_a, self.m_EC, _ = self.create_activity_pc(x, len_track, t_per_epoch, self.n_cells['pyramidal'], m_a, self.p_active_EC, self.m_EC)
            else:
                self.I_a = lambda t: np.zeros(self.n_cells['pyramidal'])
            
            self.run_one_epoch(t_per_epoch, plasticity = plasiticty)

            full_spike_count[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt), :] = self.spike_count
            full_burst_count[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt), :] = self.burst_count

            self.all_CA3_activities[:, int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt)] = self.CA3_act.T[:int(t_per_epoch/dt), :].T

        return full_spike_count, full_burst_count


    def create_activity_pc(self, x, len_track, t_per_epoch, n_cells, m, p_active, m_cells = None): 
        dt = self.dt
        sigma_pf = len_track/16 # len_track/8
        n_active = int(n_cells*p_active) 
        # TODO: I will probably need to make the m_cells a class attribute

        if self.n_cells['CA3'] == n_cells:
            m_cells_new = np.linspace(-2*sigma_pf, len_track + len_track/n_active + 2*sigma_pf, n_active)
        else:
            m_cells_new = np.linspace(0, len_track, n_active)
        
        activity = np.zeros((n_active, x.shape[0] + 5))
        activity[:, :-5] = m * np.exp(-0.5 * ((m_cells_new[:, None] - x[None, :])**2) / sigma_pf**2)

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
    

    def run_one_epoch(self, t_epoch, plasticity = True, inh_plasticity = True): 
        dt = self.dt

        self.t_values = np.zeros(int(round(t_epoch / dt)) + 1)  #### TODO: WHY DO I NEED THIS ??

        self.bursting = np.zeros(self.n_cells['pyramidal'])
        self.spiking = np.zeros(self.n_cells['pyramidal'])

        self.spike_count = np.zeros((int(round(t_epoch / dt)), self.n_cells['pyramidal']))
        self.burst_count = np.zeros((int(round(t_epoch / dt)), self.n_cells['pyramidal']))
        self.spike_count_int_a = np.zeros((int(round(t_epoch / dt)), self.n_cells['inter_a']))
        self.spike_count_int_b = np.zeros((int(round(t_epoch / dt)), self.n_cells['inter_b']))

        values = {k: v[1] for k, v in self.dynamics_values.items()}

        self.Ib_trace = np.zeros((int(round(t_epoch / dt)), self.n_cells['CA3']))
        values_new = {}

        self.tr_ev_pre = np.zeros((int(round(t_epoch / dt)), self.n_cells['inter_b']))
        self.tr_ev_post = np.zeros((int(round(t_epoch / dt)), self.n_cells['pyramidal']))
        self.tr_bu_pre = np.zeros((int(round(t_epoch / dt)), self.n_cells['inter_a']))
        self.tr_bu_post = np.zeros((int(round(t_epoch / dt)), self.n_cells['pyramidal']))

        self.t_epoch = t_epoch

        t, t_old = 0, 0

        # TODO: Fix this loop
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

            ### implement inhibitory plasticity:
            # self.update_traces()
            
            # self.tr_ev_pre[t-1, :] = self.trace_events_pre
            # self.tr_ev_post[t-1, :] = self.trace_events_post
            # self.tr_bu_pre[t-1, :] = self.trace_bursts_pre
            # self.tr_bu_post[t-1, :] = self.trace_bursts_post
            
            self.burst_count[t-1, :] = self.bursting
            self.spike_count[t-1, :] = self.spiking
            # self.spike_count_int_a[t-1, :] = self.inter_spikes_a
            # self.spike_count_int_b[t-1, :] = self.inter_spikes_b

            self.Ib_trace[t-1, :] = self.I_b(t_new)
            self.t_values[t] = t_new
            t_old = t_new

        t_epoch = self.t_values[t]
        if plasticity:
            self.plasticity_step()
        self.values = values_new
        self.all_values = values


    def update_traces(self):
        self.trace_events_post = self.trace_events_post*self.events_decay
        self.trace_events_pre = self.trace_events_pre*self.events_decay
        self.trace_events_post = np.where(self.spiking, self.trace_events_post + 1, self.trace_events_post)
        self.trace_events_pre = np.where(self.inter_spikes_b, self.trace_events_pre + 1, self.trace_events_pre)

        self.trace_bursts_post = self.trace_bursts_post*self.bursts_decay
        self.trace_bursts_pre = self.trace_bursts_pre*self.bursts_decay
        self.trace_bursts_post = np.where(self.bursting, self.trace_bursts_post + 1, self.trace_bursts_post)
        self.trace_bursts_pre = np.where(self.inter_spikes_a, self.trace_bursts_pre + 1, self.trace_bursts_pre)


    def retrieve_place_cells(self, t_run, x_run, new_env = False, a = 0, t_per_epoch = None, top_down = False, plasticity = True, len_track = None):
        # self.inh_plasticity = top_down
        dt = self.dt
        m_a, m_b = self.ma_pc, self.mb_pc

        tn = t_run[-1]
        if len_track is None:
            len_track = np.max(x_run)
        if not t_per_epoch:
            t_per_epoch = tn
        
        n_epochs = int(round(tn/t_per_epoch))
        full_spike_count = np.zeros((int(round(tn/dt)+1), self.n_cells['pyramidal']))
        full_burst_count = np.zeros((int(round(tn/dt)+1), self.n_cells['pyramidal']))

        for j in range(n_epochs):
            t0_epoch = j*t_per_epoch

            x = x_run[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt)]

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

            self.run_one_epoch(t_per_epoch, plasticity)

            full_spike_count[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt), :] = self.spike_count
            full_burst_count[int(t0_epoch/dt):int((t0_epoch + t_per_epoch)/dt), :] = self.burst_count

        return full_spike_count, full_burst_count


    def plasticity_step(self): 

        mean_events = np.mean(self.spike_count, axis=0)
        mean_bursts = np.mean(self.burst_count, axis=0)

        mean_spikes_int_b = np.mean(self.spike_count_int_b, axis=0)
        mean_Ib = np.mean(self.Ib_trace, axis=0)/(self.n_cells['pyramidal'] * self.p_active_EC)
        ## Inhibitory plasticity
        if self.inh_plasticity:
            # self.alpha_spikes = 2 * self.rho_0_spikes * self.tau_stdp_b
            # self.alpha_bursts = 2 * self.rho_0_bursts * self.tau_stdp_a
# 
            # self.alpha_pre_spikes = 2 * self.rho_0_spikes_pre * self.tau_stdp_b
            # self.alpha_pre_bursts = 2 * self.rho_0_bursts_pre * self.tau_stdp_a 
# 
            # self.W_pi_b = self.W_pi_b + self.lr_i_b * np.einsum('ij,ik->kj', self.spike_count_int_b, self.tr_ev_post) #  - self.alpha_spikes)
            # self.W_pi_b = self.W_pi_b + self.lr_i_b * np.einsum('ij,ik->jk', self.spike_count, self.tr_ev_pre)
            # self.W_pi_b = np.where(self.W_pi_b < 0, 0, self.W_pi_b)
            # 
            # self.W_pi_a = self.W_pi_a + self.lr_i_a * np.einsum('ij,ik->kj', self.spike_count_int_a, self.tr_bu_post) # - self.alpha_bursts)
            # self.W_pi_a = self.W_pi_a + self.lr_i_a * np.einsum('ij,ik->jk', self.burst_count, self.tr_bu_pre)
            # self.W_pi_a = np.where(self.W_pi_a < 0, 0, self.W_pi_a)

            delta_W_pi_b = .1 * (np.outer(mean_events, mean_spikes_int_b))
            delta_W_ip_b = .1 * (np.outer(mean_spikes_int_b, mean_events))
            delta_W_ff_b = .1 * (np.outer(mean_spikes_int_b, mean_Ib))

            self.W_pi_b += delta_W_pi_b
            self.W_ip_b += delta_W_ip_b
            self.W_ffi_b += delta_W_ff_b

            W_pi_b_norm = np.sum(self.W_pi_b, axis=1)
            W_ffi_b_norm = np.sum(self.W_ffi_b, axis=1)
            W_ip_b_norm = np.sum(self.W_ip_b, axis=1)

            self.W_pi_b = self.w_pi_b_total * self.W_pi_b / W_pi_b_norm[:, np.newaxis]
            self.W_pi_b = np.where(self.W_pi_b < 0, 0, self.W_pi_b)

            self.W_ip_b = self.w_ip_b_total * self.W_ip_b / (W_ip_b_norm + W_ffi_b_norm)[:, np.newaxis]
            self.W_ip_b = np.where(self.W_ip_b < 0, 0, self.W_ip_b)
                  
            self.W_ffi_b = self.w_ip_b_total * self.W_ffi_b / (W_ip_b_norm + W_ffi_b_norm)[:, np.newaxis]
            self.W_ffi_b = np.where(self.W_ffi_b < 0, 0, self.W_ffi_b)

            print(np.sum(self.W_ffi_b))


            # self.W_ip_b = self.W_ip_b + self.lr_i_b * np.einsum('ij,ik->jk', self.spike_count_int_b, self.tr_ev_post - self.alpha_pre_spikes)
            # self.W_ip_b = self.W_ip_b + self.lr_i_b * np.einsum('ij,ik->kj', self.spike_count, self.tr_ev_pre)
            # self.W_ip_b = np.where(self.W_ip_b < 0, 0, self.W_ip_b)
# 
            # self.W_ip_a = self.W_ip_a + self.lr_i_a * np.einsum('ij,ik->jk', self.spike_count_int_a, self.tr_bu_post - self.alpha_pre_bursts)
            # self.W_ip_a = self.W_ip_a + self.lr_i_a * np.einsum('ij,ik->kj', self.burst_count, self.tr_bu_pre)
            # self.W_ip_a = np.where(self.W_ip_a < 0, 0, self.W_ip_a)

            # print('apical:')
            # print(np.sum(self.W_ip_a))
            # print(np.sum(self.W_pi_a))

            # print('basal:')
            # print('n active:')
            # print(np.sum(self.W_ip_b))
            # self.activity_trace[:, self.act_tr_count] = mean_events/(self.dt) 
        # 
            # if self.act_tr_count == 9:
            # 
            #     print(np.mean(self.activity_trace, axis=1))
            #     self.n_active = (np.mean(self.activity_trace, axis=1) >= 2).sum()
# 
            #     # self.W_ip_b = self.W_ip_b + 1 * (self.n_active - self.target_active)
            #     self.W_ip_b = np.where(self.W_ip_b < 0, 0, self.W_ip_b)
            #     self.act_tr_count = 0
# 
            #     print(self.n_active)
            # print('firing rate:')
            # print(np.sum(self.W_pi_b))

        ## CA3 plasticity
    

         #*50
        # print(self.n_active)
        # print(mean_bursts/(self.dt), sigmoid(mean_bursts/(self.dt), 2.5, 10))

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

        self.Ib_trace = self.Ib_trace * 0
        self.plast_count += 1
        self.act_tr_count += 1

        print('Plasticity step:', self.plast_count)


def sigmoid(x, x0, k):
    y = 1 / (1 + np.exp(-k*(x-x0)))
    return y


plt.figure()
x = np.linspace(0, 10, 100)
y = sigmoid(x,7.5, 5)
plt.plot(x, y)
plt.savefig('plots/sigmoid.png')