import numpy as np
from collections import deque
import time
import tkinter as tk
from scipy.signal import butter, sosfilt_zi,sosfilt
import matplotlib
import pylsl
import math
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


plot_duration = 10  # how many seconds of data to show
update_interval = 30  # ms between screen updates
pull_interval = 100  # ms between each pull operation

class Inlet:
    def __init__(self, info: pylsl.StreamInfo):
        self.inlet = pylsl.StreamInlet(info, max_buflen=plot_duration,
                                       processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter)
        self.name = info.name()
        self.channel_count = info.channel_count()

    def pull_and_plot(self,):
        pass

class DataInlet(Inlet):
    dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]

    def __init__(self, info: pylsl.StreamInfo):#, plt: pg.PlotItem):
        super().__init__(info)
        bufsize = (2 * math.ceil(info.nominal_srate() * plot_duration), info.channel_count())
        self.buffer = np.empty(bufsize, dtype=self.dtypes[info.channel_format()])

    def pull_and_plot(self,):
        _, ts = self.inlet.pull_chunk(timeout=0.0,
                                      max_samples=self.buffer.shape[0],
                                      dest_obj=self.buffer)
        return self.buffer
    
class heat_gui(tk.Toplevel):
    def __init__(self, parent, task_trial, task_stim, target_profile_x,target_profile_y,stim_profile_x,stim_profile_y, trial_params,dev_select='FLX', vis_chan_mode='avg', vis_chan = 10,record = False, heat_dict = None, thermodes = None):
        super().__init__(parent)

        self.vis_buffer_len = 5
        self.vis_xlim_pad = 3
        self.EMG_avg_win = 100 #in samples
        self.vis_chan_mode  = vis_chan_mode
        self.vis_chan = int(vis_chan)
        self.task_trial = task_trial
        self.task_stim = task_stim
        self.force_holder = deque(list(np.empty(self.vis_buffer_len)))
        self.trig_holder = deque(list(np.empty(self.vis_buffer_len,dtype= bool)))
        self.stim_profile_x = stim_profile_x
        self.x_axis = np.linspace(0,1,self.vis_buffer_len)
        self.kill = False
        
        self.thermodes = thermodes
        self.heat_dict = heat_dict

        self.attributes('-fullscreen', True)
        self.title('Force Visualization')
        self.trial_params = trial_params
        self.rec_flag = record
        self.parent = parent
        # if self.rec_flag:
        self.parent.frame_exp.dump_trig = []
        self.parent.frame_exp.dump_heat = []
        self.parent.frame_exp.dump_force = []
        self.parent.frame_exp.dump_time = []

        if self.vis_chan_mode == 'single':
            self.vis_chan_slice = np.array([int(self.vis_chan)])
        elif self.vis_chan_mode == 'aux':
            self.vis_chan_slice = np.array([int(self.vis_chan) + self.parent.frame_exp.UNI_count-1])
        else:
            self.vis_chan_slice = np.arange(int(self.vis_chan))

        fig = Figure()
        self.disp_target = fig.add_subplot(111)
        
        self.disp_target.set_xlabel("Time (s)", fontsize=14)
        self.disp_target.set_ylabel("Torque (Nm)", fontsize=14)
        
        self.main_frame = tk.Frame(self, borderwidth=2, relief= 'solid')
        self.main_frame.pack(side="bottom", expand=True, fill="both")


        self.canvas_disp_target = FigureCanvasTkAgg(fig, master=self.main_frame,)  
        self.canvas_disp_target.draw()
        self.canvas_disp_target.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.disp_target.set_xlabel("Time (s)", fontsize=14)
        self.disp_target.set_ylabel("EMG (mV)/Torque (Nm)", fontsize=14)
        self.l_target = self.disp_target.plot(target_profile_x, target_profile_y, linewidth = 50, color = 'r')
        self.l_history = self.disp_target.plot(self.x_axis, self.force_holder, linewidth = 5, color = 'c',)
        self.l_current = self.disp_target.plot(self.x_axis, self.force_holder, linewidth = 13, color = 'b',)
        self.disp_target.set_xlim([0,self.trial_params['duration']])
        self.disp_target.set_ylim([0,self.trial_params['MVF']*0.4])

        self.canvas_disp_target.draw()

        self.stream_vis_button = tk.Button(self, text='START TRIAL', bg ='yellow')
        self.stream_vis_button['command'] = self.start_vis
        self.stream_vis_button.pack()
        self.stream_vis_button.place(x=100, y=100)
        self.stream_vis_button = tk.Button(self, text='STOP TRIAL', bg ='red')
        self.stream_vis_button['command'] = self.stop_vis
        self.stream_vis_button.pack()
        self.stream_vis_button.place(x=100, y=150)

        print("finding stream")
        stream = pylsl.resolve_stream('name', dev_select)
        for info in stream:
            print('name: ', info.name())
            print('channel count:', info.channel_count())
            print('sampling rate:', info.nominal_srate())
            print('type: ', info.type())
        self.inlet = DataInlet(stream[0])    
        
    def stop_vis(self):
        self.kill = True
        for key in self.thermodes.keys():
            self.thermodes[key].abort()
            self.thermodes[key].set_baseline(self.heat_dict[1][key]["BL"])
        self.inlet.inlet.close_stream()
        self.destroy()

    def start_vis(self):
        self.inlet.inlet.open_stream()

        if self.rec_flag:
            self.task_stim.write(False)
        self.task_trial.write(True)

        array_data = self.inlet.pull_and_plot()#
        if self.vis_chan_mode == 'aux':
            sos_raw = butter(3, [0.2, 20], 'bandpass', fs=2000, output='sos')
            sos_env= butter(3, 5, 'lowpass', fs=2000, output='sos')
            z_sos0 = sosfilt_zi(sos_raw)
            z_sos_raw=np.repeat(z_sos0[:, np.newaxis, :], len(self.vis_chan_slice), axis=1)
            z_sos0 = sosfilt_zi(sos_env)
            z_sos_env=np.repeat(z_sos0[:, np.newaxis, :], len(self.vis_chan_slice), axis=1)
        else:
            sos_raw = butter(3, [20, 500], 'bandpass', fs=2000, output='sos')
            sos_env= butter(3, 5, 'lowpass', fs=2000, output='sos')
            z_sos0 = sosfilt_zi(sos_raw)
            z_sos_raw=np.repeat(z_sos0[:, np.newaxis, :], len(self.vis_chan_slice), axis=1)
            z_sos0 = sosfilt_zi(sos_env)
            z_sos_env=np.repeat(z_sos0[:, np.newaxis, :], len(self.vis_chan_slice), axis=1)
        
        
        samples_raw, z_sos_raw= sosfilt(sos_raw, array_data[:self.EMG_avg_win,self.vis_chan_slice].T, zi=z_sos_raw)
        samples = abs(samples_raw) - np.min(abs(samples_raw),axis =0).reshape(1,-1)
        _, z_sos_env= sosfilt(sos_env, samples, zi=z_sos_env)
        # time.sleep(0.1)
        # if self.vis_chan_mode == 'aux':
        #     array_data = self.inlet.pull_and_plot()
        #     array_data_filt = np.abs(array_data[:self.EMG_avg_win,self.vis_chan_slice])
        #     array_data_scaled = np.abs(np.nan_to_num(array_data_filt,nan=0,posinf=0,neginf=0)).T
        #     baseline =  abs(np.mean(array_data_scaled))   
        
        for key in self.thermodes.keys():
            self.thermodes[key].set_quiet()
            self.thermodes[key].set_baseline(self.heat_dict[1][key]["BL"])


        baseline_list = []
        t0 = time.time()
        stim_ctr = 0
        curr_pulse_time = 1e16
        heat_dict = {1:12,2:13,3:24 }
        keys_temp = list(self.thermodes.keys())
        if stim_ctr<len(self.heat_dict):
            curr_pulse_time = self.heat_dict[1][keys_temp[0]]["INIT"]
        baseline = 0
        while time.time()-t0 < self.trial_params['duration'] and not self.kill:
            time.sleep(0.0001)
            self.trig_holder.popleft()
            
            stim = False
            if time.time()-t0 > curr_pulse_time and stim_ctr<len(self.heat_dict):
                stim = True
                self.task_stim.write(True)
                stim_ctr+=1
                
                if stim_ctr<len(self.heat_dict):
                    curr_pulse_time = self.heat_dict[stim_ctr+1][keys_temp[0]]["INIT"]
                else:
                    curr_pulse_time += self.trial_params['duration']
                self.trig_holder.append(1)

                for key in self.thermodes.keys():
                    self.thermodes[key].set_quiet()
                    self.thermodes[key].set_baseline(self.heat_dict[stim_ctr][key]["BL"])
                    self.thermodes[key].set_durations(list(self.heat_dict[stim_ctr][key]["HOLD"]))
                    self.thermodes[key].set_ramp_speed(list(self.heat_dict[stim_ctr][key]["URATE"]))
                    self.thermodes[key].set_return_speed(list(self.heat_dict[stim_ctr][key]["DRATE"]))
                    self.thermodes[key].set_temperatures(list(self.heat_dict[stim_ctr][key]["TGT"]))
                    self.thermodes[key].stimulate()    



            self.trig_holder.append(0)
            
            self.force_holder.popleft()
            array_data = self.inlet.pull_and_plot()

            if self.vis_chan_mode == 'aux':
                array_data_filt = np.abs(array_data[:self.EMG_avg_win,self.vis_chan_slice])
            else:
                samples_raw, z_sos_raw= sosfilt(sos_raw, array_data[:self.EMG_avg_win,self.vis_chan_slice].T, zi=z_sos_raw)
                samples = abs(samples_raw) - np.min(abs(samples_raw),axis =0).reshape(1,-1)
                array_data_filt, z_sos_env= sosfilt(sos_env, samples, zi=z_sos_env)
            array_data_scaled = np.abs(np.nan_to_num(array_data_filt,nan=0,posinf=0,neginf=0)).T
            force = abs(np.mean(array_data_scaled)) 


            if time.time()-t0 < 3:
                force = np.median(array_data_scaled)
                if self.vis_chan_mode == 'aux':
                    # force = force#*float(self.parent.frame_exp.conv_factor.get())
                    force = force*float(self.parent.frame_exp.conv_factor.get())

                baseline_list.append(force)
                baseline = np.median(baseline_list)

            else:
                if self.vis_chan_mode == 'aux':
                    force =(abs(np.median(array_data_scaled)-baseline))*float(self.parent.frame_exp.conv_factor.get())
                    # print("using", baseline)
                    # force = force*float(self.parent.frame_exp.conv_factor.get())
                else:
                    force = abs(np.median(array_data_scaled) - baseline)
                    
            self.force_holder.append(force)
            t_prev = time.time()-t0
            if stim==True:
                print(time.time()-t0,curr_pulse_time,stim,force)
            if self.rec_flag:
                self.task_stim.write(False)
                self.parent.frame_exp.dump_trig.append(self.trig_holder[-1])
                temperatures = []
                for t_key in self.thermodes.keys():
                    temperatures.append(self.thermodes[t_key].get_temperatures())
                self.parent.frame_exp.dump_heat.append(temperatures)
            """
            NOTE: Can be made faster by pre defining the length of dump_force and dump_time variables
            """
            self.parent.frame_exp.dump_time.append(t_prev)
            self.parent.frame_exp.dump_force.append(force)
            # disp_force = self.force_holder
            self.l_current[0].set_data(self.x_axis*(time.time()-t0-t_prev-0.1)+t_prev,np.mean(self.force_holder)*np.ones(self.vis_buffer_len))
            self.l_history[0].set_data(self.parent.frame_exp.dump_time,self.parent.frame_exp.dump_force)
            self.disp_target.set_xlim([time.time()-t0-self.vis_xlim_pad,time.time()-t0+self.vis_xlim_pad])
            self.canvas_disp_target.draw()
            self.update()
        self.inlet.inlet.close_stream()
        self.destroy()
