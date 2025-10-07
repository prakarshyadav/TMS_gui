import os
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt_zi,sosfilt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from collections import deque
from scipy.io import savemat
from tmsi_dual_interface.tmsi_libraries.TMSiFileFormats.file_writer import FileWriter, FileFormat
from tmsi_dual_interface.tmsi_libraries.TMSiSDK.device import ChannelType
from thermode.TcsControl_python3 import TcsDevice
from thermode.heat_stim_gui import heat_gui
import math
import cv2
import pylsl
import nidaqmx
import nidaqmx.system
from nidaqmx.constants import LineGrouping
from tkinter import simpledialog
import json
import TMS_lib.DuoMAG

"""
The code seems to be unstable if alternating between trail and rec mode
Also sometimes stream does not close properly (issue with sample_data_server.py) which causes code to crash after a while
"""

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
class DataInlet_reset(Inlet):
    dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]

    def __init__(self, info: pylsl.StreamInfo):#, plt: pg.PlotItem):
        super().__init__(info)
        self.bufsize = (2 * math.ceil(info.nominal_srate() * plot_duration), info.channel_count())
        self.info = info.channel_format()
        self.buffer = np.empty(self.bufsize, dtype=self.dtypes[self.info])

    def pull_and_plot(self,):
        _, ts = self.inlet.pull_chunk(timeout=0.0,
                                      max_samples=self.buffer.shape[0],
                                      dest_obj=self.buffer)
        out = self.buffer
        self.buffer = np.empty(self.bufsize, dtype=self.dtypes[self.info])
        return out

class check_MEPs_win(tk.Toplevel):  
    def __init__(self, parent, task_trial, task_stim, task_analog, DS8R_analog, DS8R_trig,target_profile_x,target_profile_y,stim_profile_x,stim_profile_y, DS8R_stim_x, DS8R_stim_y, TMS_dev, trial_params,dev_select='FLX', vis_chan_mode='avg', vis_chan = 10,vis_chan_mode_check='single', vis_chan_check = 35,record = False):
        super().__init__(parent)

        self.vis_buffer_len = 5
        self.vis_xlim_pad = 3
        self.EMG_avg_win = 100 #in samples
        self.vis_chan_mode  = vis_chan_mode
        self.vis_chan = int(vis_chan)        
        self.vis_chan_mode_check  = vis_chan_mode_check
        self.vis_chan_check = int(vis_chan_check)

        self.task_trial = task_trial
        self.task_stim = task_stim
        self.task_analog = task_analog
        self.DS8R_analog = DS8R_analog
        self.DS8R_trig = DS8R_trig
        self.force_holder = deque(list(np.empty(self.vis_buffer_len)))
        self.trig_holder = deque(list(np.empty(self.vis_buffer_len,dtype= bool)))
        self.stim_profile_x = deque(list(stim_profile_x))
        self.stim_profile_y = deque(list(stim_profile_y))
        self.DS8R_stim_profile_x = deque(list(DS8R_stim_x))
        self.DS8R_stim_profile_y = deque(list(DS8R_stim_y))
        self.tms_dev = TMS_dev
        self.x_axis = np.linspace(0,1,self.vis_buffer_len)
        self.kill = False

        self.attributes('-fullscreen', True)
        self.title('Force Visualization')
        self.trial_params = trial_params
        self.rec_flag = record
        self.parent = parent
        # if self.rec_flag:
        self.parent.frame_exp.dump_trig = []
        self.parent.frame_exp.dump_force = []
        self.parent.frame_exp.dump_time = []

        if self.vis_chan_mode == 'single':
            self.vis_chan_slice = np.array([int(self.vis_chan)])
        elif self.vis_chan_mode == 'aux':
            self.vis_chan_slice = np.array([int(self.vis_chan) + self.parent.frame_exp.UNI_count-1])
        else:
            self.vis_chan_slice = np.arange(int(self.vis_chan))

        if self.vis_chan_mode_check == 'single':
            self.vis_chan_slice_check = np.array([int(self.vis_chan_check)])
        elif self.vis_chan_mode_check == 'aux':
            self.vis_chan_slice_check = np.array([int(self.vis_chan_check) + self.parent.frame_exp.UNI_count-1])
        else:
            self.vis_chan_slice_check = np.arange(int(self.vis_chan_check))

        fig_data = Figure()
        self.disp_target = fig_data.add_subplot(111)
        
        fig_MEP = Figure()
        self.check_MEP_fig = fig_MEP.add_subplot(111)
        
        
        self.main_frame = tk.Frame(self, borderwidth=2, relief= 'solid')
        self.main_frame.pack(side="bottom", expand=True, fill="both")
        self.main_frame.grid_columnconfigure(0, weight=1,uniform=1)
        self.main_frame.grid_rowconfigure(0, weight=1,uniform=1)
        self.main_frame.grid_rowconfigure(1, weight=1,uniform=1)

        self.frame1=tk.Frame(self.main_frame,bg="red")
        self.frame1.grid(row=0, column=0, sticky='nsew')
        self.frame2=tk.Frame(self.main_frame,bg="black")
        self.frame2.grid(column=0,row=1,sticky='nsew')

        self.canvas_disp_target = FigureCanvasTkAgg(fig_data, master=self.frame1,)  
        self.canvas_disp_target.draw()
        self.canvas_disp_target.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.canvas_check_MEP_fig = FigureCanvasTkAgg(fig_MEP, master=self.frame2,)  
        self.canvas_check_MEP_fig.draw()
        self.canvas_check_MEP_fig.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.disp_target.set_xlabel("Time (s)", fontsize=14)
        self.disp_target.set_ylabel("EMG (mV)/Torque (Nm)", fontsize=14)
        self.check_MEP_fig.set_xlabel("Time (ms)", fontsize=14)
        self.check_MEP_fig.set_ylabel("EMG (mV)/AUX (V)", fontsize=14)

        self.l_target = self.disp_target.plot(target_profile_x, target_profile_y, linewidth = 25, color = 'r')
        self.l_history = self.disp_target.plot(self.x_axis, self.force_holder, linewidth = 5, color = 'c',)
        self.l_current = self.disp_target.plot(self.x_axis, self.force_holder, linewidth = 13, color = 'b',)
        
        self.x_axis_MEP = np.linspace(self.trial_params['MEP_winL'],self.trial_params['MEP_winU'],(self.trial_params['MEP_winU'] - self.trial_params['MEP_winL'])*2)#
        self.stim_line_y0 = self.check_MEP_fig.vlines(0,-1000,1000, linewidth = 3, color = 'k')
        self.stim_line_20 = self.check_MEP_fig.vlines(20,-1000,1000, linewidth = 1, color = 'c')
        self.stim_line_x0 = self.check_MEP_fig.hlines(0,self.trial_params['MEP_winL'],self.trial_params['MEP_winU'], linewidth = 0.5, color = 'c')
        self.stim_line_sd_U = self.check_MEP_fig.hlines(0.1,self.trial_params['MEP_winL'],self.trial_params['MEP_winU'], linewidth = 1, color = 'k', alpha =0.5)
        self.stim_line_sd_L = self.check_MEP_fig.hlines(-0.1,self.trial_params['MEP_winL'],self.trial_params['MEP_winU'], linewidth = 1, color = 'k',alpha = 0.5)
        self.MEP_amp = self.check_MEP_fig.text(0,0,'MEP: '+str(0))
        self.vis_MEP = self.check_MEP_fig.plot(self.x_axis_MEP, np.zeros_like(self.x_axis_MEP), linewidth = 2, color = 'r',)
        
        self.disp_target.set_xlim([0,self.trial_params['duration']])
        self.disp_target.set_ylim([0,self.trial_params['MVF']*0.4])

        self.check_MEP_fig.set_xlim([self.trial_params['MEP_winL'],self.trial_params['MEP_winU']])
        self.check_MEP_fig.set_ylim([-self.trial_params['MVF']*0.4,self.trial_params['MVF']*0.5])

        self.canvas_disp_target.draw()
        self.canvas_check_MEP_fig.draw()

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
        self.inlet_STA = DataInlet_reset(stream[0])    
    def stop_vis(self):
        self.kill = True
        self.inlet.inlet.close_stream()
        self.inlet_STA.inlet.close_stream()
        self.destroy()


    def start_vis(self):
        if self.rec_flag:
            self.task_stim.write(False)
        self.task_trial.write(True)
        self.inlet.inlet.open_stream()
        self.inlet_STA.inlet.open_stream()
        data_STA = self.inlet_STA.pull_and_plot()#
        array_data = self.inlet.pull_and_plot()#
        if self.vis_chan_mode == 'aux':
            sos_raw = butter(3, [20, 500], 'bandpass', fs=2000, output='sos')
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
        sos_raw_sta = butter(3, [20, 500], 'bandpass', fs=2000, output='sos')

        STA_raw = sosfilt(sos_raw,data_STA[:,self.vis_chan_slice_check].T)
        
        samples_raw, z_sos_raw= sosfilt(sos_raw, array_data[:self.EMG_avg_win,self.vis_chan_slice].T, zi=z_sos_raw)
        samples = abs(samples_raw) - np.min(abs(samples_raw),axis =0).reshape(1,-1)
        _, z_sos_env= sosfilt(sos_env, samples, zi=z_sos_env)

        # if self.vis_chan_mode == 'aux':
        #     array_data = self.inlet.pull_and_plot()
        #     array_data_filt = np.abs(array_data[:self.EMG_avg_win,self.vis_chan_slice])
        #     array_data_scaled = np.abs(np.nan_to_num(array_data_filt,nan=0,posinf=0,neginf=0)).T
        #     baseline =  abs(np.mean(array_data_scaled))


        t0 = time.time()
        stim_ctr = 0
        curr_pulse_time = 1e16
        MEP_update = False
        baseline = 0
        baseline_list = []
        data_STA = self.inlet_STA.pull_and_plot()#


        if len(self.stim_profile_y)>0:
            stim_amp = self.stim_profile_y[0]
            self.task_analog.write(stim_amp)
        if len(self.DS8R_stim_profile_y)>0:
            DS8R_amp = self.DS8R_stim_profile_y[0]
            self.DS8R_analog.write(DS8R_amp)
        
        while time.time()-t0 < self.trial_params['duration'] and not self.kill:
            time.sleep(0.0001)
            self.trig_holder.popleft()
            
            stim = False
            # tms_state = True
            if len(self.stim_profile_x)>0:
                if time.time()-t0 > self.stim_profile_x[0]:
                    MEP_update = False
                    stim_ctr+=1
                    stim = True
                    stim_time = self.stim_profile_x.popleft()
                    curr_pulse_time = stim_time
                    self.task_stim.write(True)
                    self.tms_dev[0].Pulse()
                    time.sleep(0.001)
                    self.task_stim.write(False)
                    self.trig_holder.append(1)
                    
                    self.stim_profile_y.popleft()
                    if len(self.stim_profile_y)>0:
                        stim_amp = self.stim_profile_y[0]
                    self.task_analog.write(stim_amp)

            DS8R_stim = False
            if len(self.DS8R_stim_profile_x)>0:
                if time.time()-t0 > self.DS8R_stim_profile_x[0]:
                    DS8R_stim = True
                    MEP_update = False
                    stim_ctr+=1
                    DS8R_time = self.DS8R_stim_profile_x.popleft()
                    curr_pulse_time = DS8R_time
                    self.DS8R_trig.write(True)
                    time.sleep(0.001)
                    self.DS8R_trig.write(False)
                    
                    self.DS8R_stim_profile_y.popleft()
                    if len(self.DS8R_stim_profile_y)>0:
                        DS8R_amp = self.DS8R_stim_profile_y[0]
                    self.DS8R_analog.write(DS8R_amp)

            self.trig_holder.append(0)

            if time.time()-t0 > (curr_pulse_time+0.5) and not MEP_update and stim_ctr > 0:
                MEP_update = True
                data_STA = self.inlet_STA.pull_and_plot()#
                trigs = data_STA[:,-3]
                plot_event_idx = np.where(np.abs(np.diff(trigs))>0)[0]#[-1]
                print("updating MEPs", plot_event_idx)
                if len(plot_event_idx)>1:
                    if self.vis_chan_mode_check == 'aux':
                        data_STA_filt = sosfilt(sos_raw_sta, data_STA[:,self.vis_chan_slice_check].T)
                        # data_STA_filt = np.abs(data_STA[:,self.vis_chan_slice_check])
                    else:
                        data_STA_filt = sosfilt(sos_raw_sta, data_STA[:,self.vis_chan_slice_check].T)
                    data_STA_scaled = np.nan_to_num(data_STA_filt,nan=0,posinf=0,neginf=0).reshape(-1)
                    plot_data = data_STA_scaled[plot_event_idx[-2]+self.trial_params['MEP_winL']*2:plot_event_idx[-2]+self.trial_params['MEP_winU']*2]
                    SD_bound = np.std(data_STA_scaled[plot_event_idx[-2]-250:plot_event_idx[-2]-50])
                    l_cut = 4; u_cut = 10
                    plot_data[abs(self.trial_params['MEP_winL']*2)-l_cut:abs(self.trial_params['MEP_winL']*2)+u_cut] = np.zeros(l_cut+u_cut)
                    y_MEP = max(0.05,np.max(np.abs(plot_data)))
                    self.check_MEP_fig.set_ylim([-y_MEP*1.1,y_MEP*1.1])
                    self.vis_MEP[0].set_data(self.x_axis_MEP,plot_data)
                    self.stim_line_sd_U.remove()
                    self.stim_line_sd_U = self.check_MEP_fig.hlines(SD_bound*5.5,self.trial_params['MEP_winL'],self.trial_params['MEP_winU'], linewidth = 1, color = 'k', alpha =0.5)
                    #set_segments([np.array([[SD_bound, self.trial_params['MEP_winL']], [SD_bound,self.trial_params['MEP_winU']]])])
                    self.stim_line_sd_L.remove()
                    self.stim_line_sd_L = self.check_MEP_fig.hlines(-SD_bound*5.5,self.trial_params['MEP_winL'],self.trial_params['MEP_winU'], linewidth = 1, color = 'k', alpha =0.5)

                    #set_segments([np.array([[-SD_bound, self.trial_params['MEP_winL']], [-SD_bound,self.trial_params['MEP_winU']]])])

                    p2p_amp = np.max(plot_data) - np.min(plot_data)
                    self.MEP_amp.set_text("P2P: "+str(abs(np.float16(p2p_amp))))
                    # plot_data = np.abs(np.diff(trigs))
                    # self.check_MEP_fig.set_xlim([0,1])
                    max_val =  max(SD_bound*6, np.max(np.abs(plot_data)*1.1))
                    self.check_MEP_fig.set_ylim([-max_val,max_val])
                    # self.vis_MEP[0].set_data(np.linspace(0,1,plot_data.shape[0]),plot_data)
                    self.canvas_check_MEP_fig.draw()
                else:
                    print("Warning Trigs not detected")

                
            self.force_holder.popleft()
            array_data = self.inlet.pull_and_plot()
            if self.vis_chan_mode == 'aux':
                array_data_filt = array_data[:self.EMG_avg_win,self.vis_chan_slice]+ self.parent.frame_exp.vis_scaling_offset
            else:
                samples_raw, z_sos_raw= sosfilt(sos_raw, array_data[:self.EMG_avg_win,self.vis_chan_slice].T, zi=z_sos_raw)
                samples = abs(samples_raw) - np.min(abs(samples_raw),axis =0).reshape(1,-1)
                array_data_filt, z_sos_env= sosfilt(sos_env, samples, zi=z_sos_env)
            array_data_scaled = np.nan_to_num(array_data_filt,nan=0,posinf=0,neginf=0).T
            force = np.median(array_data_scaled) 

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
            # if self.vis_chan_mode == 'aux' and time.time()-t0 < 3:
            #     force = abs(np.mean(array_data_scaled)) 
            #     baseline_list.append(force)
            #     baseline = np.mean(baseline_list)
            #     force = force*float(self.parent.frame_exp.conv_factor.get())
            #     # print("setting", baseline)
            # elif self.vis_chan_mode == 'aux' and time.time()-t0 > 3:
            #     force = abs(np.mean(array_data_scaled)) - baseline
            #     # print("using", baseline)
            #     force = force*float(self.parent.frame_exp.conv_factor.get())
            # else:
            #     force = abs(np.mean(array_data_scaled))
            # force = np.median(array_data_scaled)
            # print(force)
            self.force_holder.append(force)
            t_prev = time.time()-t0
            if stim==True:
                print(time.time()-t0,curr_pulse_time,stim,force)
            if self.rec_flag:
                self.task_stim.write(False)
                self.parent.frame_exp.dump_trig.append(self.trig_holder[-1])
            self.parent.frame_exp.dump_time.append(t_prev)
            self.parent.frame_exp.dump_force.append(force)
            disp_force = sorted(self.force_holder)
            self.l_current[0].set_data(self.x_axis*(time.time()-t0-t_prev-0.1)+t_prev,np.mean(disp_force)*np.ones(self.vis_buffer_len))
            self.l_history[0].set_data(self.parent.frame_exp.dump_time,self.parent.frame_exp.dump_force)
            self.disp_target.set_xlim([time.time()-t0-self.vis_xlim_pad,time.time()-t0+self.vis_xlim_pad])
            self.canvas_disp_target.draw()
            self.update()

        self.inlet_STA.inlet.close_stream()
        self.inlet.inlet.close_stream()
        self.destroy()

class display_force_data(tk.Toplevel):
    def __init__(self, parent, task_trial, task_stim, task_analog, DS8R_analog, DS8R_trig,target_profile_x,target_profile_y,stim_profile_x,stim_profile_y, DS8R_stim_x, DS8R_stim_y, TMS_dev,trial_params,dev_select='FLX', vis_chan_mode='avg', vis_chan = 10,record = False):
        super().__init__(parent)

        self.vis_buffer_len = 5
        self.vis_xlim_pad = 3
        self.EMG_avg_win = 100 #in samples
        self.vis_chan_mode  = vis_chan_mode
        self.vis_chan = int(vis_chan)
        self.task_trial = task_trial
        self.task_stim = task_stim
        self.task_analog = task_analog
        self.DS8R_analog = DS8R_analog
        self.DS8R_trig = DS8R_trig
        self.force_holder = deque(list(np.empty(self.vis_buffer_len)))
        self.trig_holder = deque(list(np.empty(self.vis_buffer_len,dtype= bool)))
        self.stim_profile_x = deque(list(stim_profile_x))
        self.stim_profile_y = deque(list(stim_profile_y))
        self.DS8R_stim_profile_x = deque(list(DS8R_stim_x))
        self.DS8R_stim_profile_y = deque(list(DS8R_stim_y))
        self.tms_dev = TMS_dev
        self.x_axis = np.linspace(0,1,self.vis_buffer_len)
        self.kill = False

        self.attributes('-fullscreen', True)
        self.title('Force Visualization')
        self.trial_params = trial_params
        self.rec_flag = record
        self.parent = parent
        # if self.rec_flag:
        self.parent.frame_exp.dump_trig = []
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
        self.l_target = self.disp_target.plot(target_profile_x, target_profile_y, linewidth = 25, color = 'r')
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
        baseline_list = []
        t0 = time.time()

        if len(self.stim_profile_y)>0:
            stim_amp = self.stim_profile_y[0]
            self.task_analog.write(stim_amp)
        if len(self.DS8R_stim_profile_y)>0:
            DS8R_amp = self.DS8R_stim_profile_y[0]
            self.DS8R_analog.write(DS8R_amp)

        baseline = 0
        tms_state = True
        while time.time()-t0 < self.trial_params['duration'] and not self.kill:
            time.sleep(0.0001)
            self.trig_holder.popleft()
            
            stim = False
            if len(self.stim_profile_x)>0:
                if time.time()-t0 > self.stim_profile_x[0]:
                    stim = True
                    stim_time = self.stim_profile_x.popleft()
                    self.task_stim.write(True)
                    if tms_state:
                        self.tms_dev[0].Pulse(stim_amp*10,True)
                        tms_state = False
                    else:
                        self.tms_dev[1].Pulse(stim_amp*10,True)
                        tms_state = True
                    time.sleep(0.001)
                    self.task_stim.write(False)
                    self.trig_holder.append(1)
                    self.stim_profile_y.popleft()
                    if len(self.stim_profile_y)>0:
                        stim_amp = self.stim_profile_y[0]
                    self.task_analog.write(stim_amp)

            DS8R_stim = False
            if len(self.DS8R_stim_profile_x)>0:
                if time.time()-t0 > self.DS8R_stim_profile_x[0]:
                    DS8R_stim = True
                    DS8R_time = self.DS8R_stim_profile_x.popleft()
                    self.DS8R_trig.write(True)
                    time.sleep(0.001)
                    self.DS8R_trig.write(False)
                    self.DS8R_stim_profile_y.popleft()
                    if len(self.DS8R_stim_profile_y)>0:
                        DS8R_amp = self.DS8R_stim_profile_y[0]
                    self.DS8R_analog.write(DS8R_amp)

            self.trig_holder.append(0)
            
            self.force_holder.popleft()
            array_data = self.inlet.pull_and_plot()

            if self.vis_chan_mode == 'aux':
                array_data_filt = array_data[:self.EMG_avg_win,self.vis_chan_slice] + self.parent.frame_exp.vis_scaling_offset
            else:
                samples_raw, z_sos_raw= sosfilt(sos_raw, array_data[:self.EMG_avg_win,self.vis_chan_slice].T, zi=z_sos_raw)
                samples = abs(samples_raw) - np.min(abs(samples_raw),axis =0).reshape(1,-1)
                array_data_filt, z_sos_env= sosfilt(sos_env, samples, zi=z_sos_env)
            array_data_scaled = np.nan_to_num(array_data_filt,nan=0,posinf=0,neginf=0).T
            force = np.median(array_data_scaled)


            if time.time()-t0 < 3:
                force = np.median(array_data_scaled)
                if self.vis_chan_mode == 'aux':
                    force = force *float(self.parent.frame_exp.conv_factor.get())

                baseline_list.append(force)
                baseline = np.median(baseline_list)
            else:

                if self.vis_chan_mode == 'aux':
                    force = (abs(np.median(array_data_scaled)-baseline))*float(self.parent.frame_exp.conv_factor.get())
                else:
                    force = abs(np.median(array_data_scaled) - baseline)

            self.force_holder.append(force)
            t_prev = time.time()-t0
            if stim==True:
                print('TMS', time.time()-t0,stim_time,stim,force)
                
            if DS8R_stim==True:
                print('DS8R', time.time()-t0,DS8R_time,DS8R_stim,force)
                
            if self.rec_flag:
                self.task_stim.write(False)
                self.parent.frame_exp.dump_trig.append(self.trig_holder[-1])
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
        self.DS8R_analog.write(0)
        self.task_analog.write(0)
        
        self.inlet.inlet.close_stream()
        self.destroy()

class APP:
    def __init__(self,parent,tmsi,dump_path):
        # super().__init__(parent)
        # self.title('Force Ramp Interface')
        # self.geometry('1400x1000')

        self.parent = parent

        """
        Buttons
        """
        self.parent.frame_exp.vis_scaling_offset = 50

        self.tmsi_dev = tmsi
        self.dump_path = dump_path
        self.start_rec_button = tk.Button(self.parent.frame_exp, text='START', bg ='green')
        self.start_rec_button['command'] = self.start_rec
        self.start_rec_button.pack()
        self.start_rec_button.place(x=10, y=10)

        self.stop_rec_button = tk.Button(self.parent.frame_exp, text='STOP', bg ='red')
        self.stop_rec_button['command'] = self.stop_rec
        self.stop_rec_button.pack()
        self.stop_rec_button.place(x=60, y=10)

        options = list(self.tmsi_dev.keys())
        self.vis_TMSi = tk.StringVar() 
        self.vis_TMSi.set(options[0])
        self.lbl_vis = ttk.Label(self.parent.frame_exp, text='Select TMSi to visualize:')
        self.lbl_vis.pack(fill='x', expand=True)
        self.lbl_vis.place(x=200, y=10)
        self.vis_tmsi_drop = tk.OptionMenu( self.parent.frame_exp , self.vis_TMSi , *options) #tk.Button(self, text='START', bg ='green')
        self.vis_tmsi_drop.pack()
        self.vis_tmsi_drop.place(x=350, y=10)
        
        self.lbl_vis_mode = ttk.Label(self.parent.frame_exp, text='Select vis mode:')
        self.lbl_vis_mode.pack(fill='x', expand=True)
        self.lbl_vis_mode.place(x=430, y=10)
        self.vis_chan_mode = tk.StringVar() 
        self.vis_mode_option1 = tk.Radiobutton(self.parent.frame_exp, text="Single Chan", variable=self.vis_chan_mode, value="single", command=self.set_vis_mode)
        self.vis_mode_option1.pack(fill='x', expand=True)
        self.vis_mode_option1.place(x=530, y=10)
        self.vis_mode_option2 = tk.Radiobutton(self.parent.frame_exp, text="Average", variable=self.vis_chan_mode, value="avg", command=self.set_vis_mode)
        self.vis_mode_option2.pack(fill='x', expand=True)
        self.vis_mode_option2.place(x=530, y=30)
        self.vis_mode_option3 = tk.Radiobutton(self.parent.frame_exp, text="Aux", variable=self.vis_chan_mode, value="aux", command=self.set_vis_mode)
        self.vis_mode_option3.pack(fill='x', expand=True)
        self.vis_mode_option3.place(x=530, y=50)

        options = [0,36]
        self.vis_chan = tk.StringVar() 
        self.vis_chan.set(options[1])
        self.vis_chan_drop = tk.OptionMenu( self.parent.frame_exp , self.vis_chan , *options) #tk.Button(self, text='START', bg ='green')
        self.vis_chan_drop.pack()
        self.vis_chan_drop.place(x=650, y=10)
        
        self.trial_ID = tk.StringVar()
        self.lbl_trial_ID = ttk.Label(self.parent.frame_exp, text='Trial Num:')
        self.lbl_trial_ID.pack(fill='x', expand=True)
        self.lbl_trial_ID.place(x=10, y=40)
        self.t_trial_ID = tk.Entry(self.parent.frame_exp, textvariable=self.trial_ID)
        self.t_trial_ID.insert(0, "1")
        self.t_trial_ID.pack(fill='x', expand=True)
        self.t_trial_ID.focus()
        self.t_trial_ID.place(x=150, y=40, width = 50)

        self.read_cur_trial_button = tk.Button(self.parent.frame_exp, text='READ CURRENT TRIAL', bg ='yellow')
        self.read_cur_trial_button['command'] = self.read_cur_trial
        self.read_cur_trial_button.pack()
        self.read_cur_trial_button.place(x=250, y=40)

        self.daq_name = tk.StringVar()
        self.lbl_daq_name = ttk.Label(self.parent.frame_exp, text='DAQ ID:')
        self.lbl_daq_name.pack(fill='x', expand=True)
        self.lbl_daq_name.place(x=10, y=100)
        self.t_daq_name = tk.Entry(self.parent.frame_exp, textvariable=self.daq_name)
        self.t_daq_name.insert(0, "Dev1")
        self.t_daq_name.pack(fill='x', expand=True)
        self.t_daq_name.focus()
        self.t_daq_name.place(x=150, y=100, width = 100)

        self.stim_chan = tk.StringVar()
        self.lbl_Ach_name = ttk.Label(self.parent.frame_exp, text='Stim init Chans:')
        self.lbl_Ach_name.pack(fill='x', expand=True)
        self.lbl_Ach_name.place(x=10, y=130)
        self.t_Ach_name = tk.Entry(self.parent.frame_exp, textvariable=self.stim_chan)
        self.t_Ach_name.insert(0, "port0/line1")
        self.t_Ach_name.pack(fill='x', expand=True)
        self.t_Ach_name.focus()
        self.t_Ach_name.place(x=150, y=130, width = 100)

        self.trial_chan = tk.StringVar()
        self.lbl_Dch_name = ttk.Label(self.parent.frame_exp, text='Trial init Chans:')
        self.lbl_Dch_name.pack(fill='x', expand=True)
        self.lbl_Dch_name.place(x=10, y=160)
        self.t_Dch_name = tk.Entry(self.parent.frame_exp, textvariable=self.trial_chan)
        self.t_Dch_name.insert(0, "port0/line2")
        self.t_Dch_name.pack(fill='x', expand=True)
        self.t_Dch_name.focus()
        self.t_Dch_name.place(x=150, y=160, width = 100)

        self.start_daq_button = tk.Button(self.parent.frame_exp, text='START DAQ', bg ='yellow')
        self.start_daq_button['command'] = self.start_DAQ
        self.start_daq_button.pack()
        self.start_daq_button.place(x=10, y=190)

        # self.stream_daq_button = tk.Button(self, text='STREAM DAQ', bg ='yellow')
        # self.stream_daq_button['command'] = self.stream_DAQ
        # self.stream_daq_button.pack()
        # self.stream_daq_button.place(x=200, y=190)

        self.test_force_read_button = tk.Button(self.parent.frame_exp, text='TRAINING', bg ='yellow')
        self.test_force_read_button['command'] = self.test_force_read
        self.test_force_read_button.pack()
        self.test_force_read_button.place(x=110, y=10)

        self.parent.frame_exp.conv_factor = tk.StringVar()
        self.lbl_conv_factor = ttk.Label(self.parent.frame_exp, text='Torque Const.:')
        self.lbl_conv_factor.pack(fill='x', expand=True)
        self.lbl_conv_factor.place(x=10, y=220)
        self.t_conv_factor = tk.Entry(self.parent.frame_exp, textvariable=self.parent.frame_exp.conv_factor)
        self.t_conv_factor.insert(0, "1")
        self.t_conv_factor.pack(fill='x', expand=True)
        self.t_conv_factor.focus()
        self.t_conv_factor.place(x=150, y=220, width = 100)

        self.MVC_duration = tk.StringVar()
        self.lbl_MVC_len = ttk.Label(self.parent.frame_exp, text='Duration of MVC (s):')
        self.lbl_MVC_len.pack(fill='x', expand=True)
        self.lbl_MVC_len.place(x=10, y=250)
        self.t_MVC_len = tk.Entry(self.parent.frame_exp, textvariable=self.MVC_duration)
        self.t_MVC_len.insert(0, "5")
        self.t_MVC_len.pack(fill='x', expand=True)
        self.t_MVC_len.focus()
        self.t_MVC_len.place(x=150, y=250, width = 100)

        self.start_MVC_button = tk.Button(self.parent.frame_exp, text='START MVC', bg ='yellow')
        self.start_MVC_button['command'] = self.get_MVC
        self.start_MVC_button.pack()
        self.start_MVC_button.place(x=10, y=280)

        self.lbl_max_force = ttk.Label(self.parent.frame_exp, text="Max Force",font=('Helvetica 16 bold'))
        self.lbl_max_force.pack(fill='x', expand=True)
        self.lbl_max_force.place(x=400, y=150)
        self.max_force = tk.StringVar()
        self.max_force.set('10')

        self.X_profile = tk.StringVar()
        self.lbl_X_profile = ttk.Label(self.parent.frame_exp, text='X axis times (s):')
        self.lbl_X_profile.pack(fill='x', expand=True)
        self.lbl_X_profile.place(x=10, y=330)
        self.t_X_profile = tk.Entry(self.parent.frame_exp, textvariable=self.X_profile)
        self.t_X_profile.insert(0, "0, 5, 10, 25, 30, 35, 50, 55, 60")
        self.t_X_profile.pack(fill='x', expand=True)
        self.t_X_profile.focus()
        self.t_X_profile.place(x=150, y=330, width = 300)

        self.Y_profile = tk.StringVar()
        self.lbl_Y_profile = ttk.Label(self.parent.frame_exp, text='MVC targets (0.X):')
        self.lbl_Y_profile.pack(fill='x', expand=True)
        self.lbl_Y_profile.place(x=10, y=360)
        self.t_Y_profile = tk.Entry(self.parent.frame_exp, textvariable=self.Y_profile)
        self.t_Y_profile.insert(0, "0, 0, 0.1, 0.1, 0.2, 0.1, 0.1, 0, 0")
        self.t_Y_profile.pack(fill='x', expand=True)
        self.t_Y_profile.focus()
        self.t_Y_profile.place(x=150, y=360, width = 300)
        
        self.lbl_max_force_num = ttk.Label(self.parent.frame_exp, textvariable=self.max_force,font=('Helvetica 30 bold'))
        self.lbl_max_force_num.pack(fill='x', expand=True)
        self.lbl_max_force_num.place(x=400, y=200)
        self.t_max_force_num = tk.Entry(self.parent.frame_exp, textvariable=self.max_force)
        self.t_max_force_num.pack(fill='x', expand=True)
        self.t_max_force_num.focus()
        self.t_max_force_num.place(x=400, y=250, width = 200)

        self.manualMVC_button = tk.Button(self.parent.frame_exp, text='PUSH MVC', bg ='yellow')
        self.manualMVC_button['command'] = self.manualMVC
        self.manualMVC_button.pack()
        self.manualMVC_button.place(x=400, y=280)
        
        self.start_vanilla_button = tk.Button(self.parent.frame_exp, text='PUSH TRACE', bg ='yellow')
        self.start_vanilla_button['command'] = self.do_vanilla
        self.start_vanilla_button.pack()
        self.start_vanilla_button.place(x=10, y=500)
        
        self.target_profile_x = [0]
        self.target_profile_y = [0]
        self.stim_profile_x = np.empty(0)
        self.stim_profile_y = np.empty(0)
        self.DS8R_profile_x = np.empty(0)
        self.DS8R_profile_y = np.empty(0)


        fig = Figure(figsize=(7, 4), dpi=100)
        self.disp_target = fig.add_subplot(111)
        
        self.disp_target.set_title("Ramp profile", fontsize=14)
        self.disp_target.set_xlabel("Time (s)", fontsize=14)
        self.disp_target.set_ylabel("Torque (Nm)", fontsize=14)
        self.disp_target_twin_ax = self.disp_target.twinx()
        self.disp_target_twin_ax.set_ylabel("DS8R Stim amplitude (mA)", fontsize=14)
        
        self.canvas_disp_target = FigureCanvasTkAgg(fig, master=self.parent.frame_exp)  
        self.canvas_disp_target.draw()
        self.canvas_disp_target.get_tk_widget().pack(side=tk.BOTTOM, fill='x', expand=True)
        self.canvas_disp_target.get_tk_widget().place(y=550,)


        self.lbl_vis_mode_check = ttk.Label(self.parent.frame_exp, text='Select feedback mode (Check):')
        self.lbl_vis_mode_check.pack(fill='x', expand=True)
        self.lbl_vis_mode_check.place(x=650, y=250)
        self.vis_chan_mode_check = tk.StringVar() 
        self.vis_mode_option1_check = tk.Radiobutton(self.parent.frame_exp, text="Single Chan", variable=self.vis_chan_mode_check, value="single", command=self.set_vis_mode_check)
        self.vis_mode_option1_check.pack(fill='x', expand=True)
        self.vis_mode_option1_check.place(x=850, y=250)
        self.vis_mode_option3_check = tk.Radiobutton(self.parent.frame_exp, text="Aux", variable=self.vis_chan_mode_check, value="aux", command=self.set_vis_mode_check)
        self.vis_mode_option3_check.pack(fill='x', expand=True)
        self.vis_mode_option3_check.place(x=850, y=280)

        options = [x for x in range(1,65)]
        self.vis_chan_check = tk.StringVar() 
        self.vis_chan_check.set(options[35])
        self.vis_chan_drop_check = tk.OptionMenu( self.parent.frame_exp , self.vis_chan_check , *options) #tk.Button(self, text='START', bg ='green')
        self.vis_chan_drop_check.pack()
        self.vis_chan_drop_check.place(x=850, y=310)
        
        self.do_vanilla()
        self.trl_duration = self.target_profile_x[-1]


        self.lbl_Astim_note = ttk.Label(self.parent.frame_exp, text='TMS: 10V->100%')
        self.lbl_Astim_note.pack(fill='x', expand=True)
        self.lbl_Astim_note.place(x=710, y=340)
        self.analog_chan = tk.StringVar()
        self.lbl_Astim_name = ttk.Label(self.parent.frame_exp, text='TMS stim AnaChan:')
        self.lbl_Astim_name.pack(fill='x', expand=True)
        self.lbl_Astim_name.place(x=710, y=400)
        self.t_lbl_Astim_name = tk.Entry(self.parent.frame_exp, textvariable=self.analog_chan)
        self.t_lbl_Astim_name.insert(0, "ao0")
        self.t_lbl_Astim_name.pack(fill='x', expand=True)
        self.t_lbl_Astim_name.focus()
        self.t_lbl_Astim_name.place(x=850, y=400, width = 100)

        self.analog_chan_minV = tk.StringVar()
        self.lbl_AstimMinV_name = ttk.Label(self.parent.frame_exp, text='Analog min (V):')
        self.lbl_AstimMinV_name.pack(fill='x', expand=True)
        self.t_AstimMinV = tk.Entry(self.parent.frame_exp, textvariable=self.analog_chan_minV)
        self.t_AstimMinV.insert(0, "-10")
        self.t_AstimMinV.pack(fill='x', expand=True)
        self.t_AstimMinV.focus()
        self.lbl_AstimMinV_name.place(x=710, y=430)
        self.t_AstimMinV.place(x=850, y=430)

        self.analog_chan_maxV = tk.StringVar()
        self.lbl_AstimMaxV = ttk.Label(self.parent.frame_exp, text='Analog max (V):')
        self.lbl_AstimMaxV.pack(fill='x', expand=True)
        self.t_AstimMaxV = tk.Entry(self.parent.frame_exp, textvariable=self.analog_chan_maxV)
        self.t_AstimMaxV.insert(0, "10")
        self.t_AstimMaxV.pack(fill='x', expand=True)
        self.t_AstimMaxV.focus()
        self.lbl_AstimMaxV.place(x=710, y=460)
        self.t_AstimMaxV.place(x=850, y=460)
        
        self.stim_intensity = tk.StringVar()
        self.lbl_stim_intensity = ttk.Label(self.parent.frame_exp, text='Peak stim intensity (V)')
        self.lbl_stim_intensity.pack(fill='x', expand=True)
        self.lbl_stim_intensity.place(x=710, y=370)
        self.t_stim_intensity = tk.Entry(self.parent.frame_exp, textvariable=self.stim_intensity)
        self.t_stim_intensity.insert(0, "0.1")
        self.t_stim_intensity.pack(fill='x', expand=True)
        self.t_stim_intensity.focus()
        self.t_stim_intensity.place(x=850, y=370, width = 100)

        self.stim_x_profile = tk.StringVar()
        self.stim_lbl_X_profile = ttk.Label(self.parent.frame_exp, text='TMS stim times (s):')
        self.stim_lbl_X_profile.pack(fill='x', expand=True)
        self.stim_lbl_X_profile.place(x=710, y=530)
        self.stim_t_X_profile = tk.Entry(self.parent.frame_exp, textvariable=self.stim_x_profile)
        self.stim_t_X_profile.insert(0, "10, 11, 12, 5, 10, 25, 30, 35, 50, 55, 60")
        self.stim_t_X_profile.pack(fill='x', expand=True)
        self.stim_t_X_profile.focus()
        self.stim_t_X_profile.place(x=850, y=530, width = 300)

        self.stim_y_profile = tk.StringVar()
        self.stim_lbl_Y_profile = ttk.Label(self.parent.frame_exp, text='TMS stim amp (%max):')
        self.stim_lbl_Y_profile.pack(fill='x', expand=True)
        self.stim_lbl_Y_profile.place(x=710, y=560)
        self.stim_t_Y_profile = tk.Entry(self.parent.frame_exp, textvariable=self.stim_y_profile)
        self.stim_t_Y_profile.insert(0, "0, 0.2, 0.4, 0.6, 0.8, 1")
        self.stim_t_Y_profile.pack(fill='x', expand=True)
        self.stim_t_Y_profile.focus()
        self.stim_t_Y_profile.place(x=850, y=560, width = 300)

        self.lbl_DS8R_stim_note = ttk.Label(self.parent.frame_exp, text='DS8R: 10V->50mA')
        self.lbl_DS8R_stim_note.pack(fill='x', expand=True)
        self.lbl_DS8R_stim_note.place(x=1200, y=340)
        self.DS8R_analog_chan = tk.StringVar()
        self.lbl_DS8R_Astim_name = ttk.Label(self.parent.frame_exp, text='DS8R stim AnaChan:')
        self.lbl_DS8R_Astim_name.pack(fill='x', expand=True)
        self.lbl_DS8R_Astim_name.place(x=1200, y=400)
        self.t_lbl_DS8R_Astim_name = tk.Entry(self.parent.frame_exp, textvariable=self.DS8R_analog_chan)
        self.t_lbl_DS8R_Astim_name.insert(0, "ao1")
        self.t_lbl_DS8R_Astim_name.pack(fill='x', expand=True)
        self.t_lbl_DS8R_Astim_name.focus()
        self.t_lbl_DS8R_Astim_name.place(x=1350, y=400, width = 100)

        self.DS8R_intensity = tk.StringVar()
        self.lbl_DS8R_intensity = ttk.Label(self.parent.frame_exp, text='Peak DS8R intensity (V)')
        self.lbl_DS8R_intensity.pack(fill='x', expand=True)
        self.lbl_DS8R_intensity.place(x=1200, y=370)
        self.t_DS8R_intensity = tk.Entry(self.parent.frame_exp, textvariable=self.DS8R_intensity)
        self.t_DS8R_intensity.insert(0, "0.1")
        self.t_DS8R_intensity.pack(fill='x', expand=True)
        self.t_DS8R_intensity.focus()
        self.t_DS8R_intensity.place(x=1350, y=370, width = 100)

        self.DS8R_analog_chan_minV = tk.StringVar()
        self.lbl_DS8R_AstimMinV_name = ttk.Label(self.parent.frame_exp, text='Analog min (V):')
        self.lbl_DS8R_AstimMinV_name.pack(fill='x', expand=True)
        self.t_DS8R_AstimMinV = tk.Entry(self.parent.frame_exp, textvariable=self.DS8R_analog_chan_minV)
        self.t_DS8R_AstimMinV.insert(0, "-10")
        self.t_DS8R_AstimMinV.pack(fill='x', expand=True)
        self.t_DS8R_AstimMinV.focus()
        self.lbl_DS8R_AstimMinV_name.place(x=1200, y=430)
        self.t_DS8R_AstimMinV.place(x=1350, y=430)

        self.DS8R_analog_chan_maxV = tk.StringVar()
        self.lbl_DS8R_AstimMaxV = ttk.Label(self.parent.frame_exp, text='Analog max (V):')
        self.lbl_DS8R_AstimMaxV.pack(fill='x', expand=True)
        self.t_DS8R_AstimMaxV = tk.Entry(self.parent.frame_exp, textvariable=self.DS8R_analog_chan_maxV)
        self.t_DS8R_AstimMaxV.insert(0, "10")
        self.t_DS8R_AstimMaxV.pack(fill='x', expand=True)
        self.t_DS8R_AstimMaxV.focus()
        self.lbl_DS8R_AstimMaxV.place(x=1200, y=460)
        self.t_DS8R_AstimMaxV.place(x=1350, y=460)

        self.DS8R_trig_chan = tk.StringVar()
        self.lbl_DS8R_Ach_name = ttk.Label(self.parent.frame_exp, text='DS8R Trig Chans:')
        self.lbl_DS8R_Ach_name.pack(fill='x', expand=True)
        self.lbl_DS8R_Ach_name.place(x=1200, y=490)
        self.t_DS8R_Ach_name = tk.Entry(self.parent.frame_exp, textvariable=self.DS8R_trig_chan)
        self.t_DS8R_Ach_name.insert(0, "port0/line3")
        self.t_DS8R_Ach_name.pack(fill='x', expand=True)
        self.t_DS8R_Ach_name.focus()
        self.t_DS8R_Ach_name.place(x=1350, y=490, width = 100)

        self.DS8R_x_profile = tk.StringVar()
        self.DS8R_lbl_X_profile = ttk.Label(self.parent.frame_exp, text='DS8R stim times (s):')
        self.DS8R_lbl_X_profile.pack(fill='x', expand=True)
        self.DS8R_lbl_X_profile.place(x=1200, y=530)
        self.DS8R_t_X_profile = tk.Entry(self.parent.frame_exp, textvariable=self.DS8R_x_profile)
        self.DS8R_t_X_profile.insert(0, "10, 11, 12, 5, 10, 25, 30, 35, 50, 55, 60")
        self.DS8R_t_X_profile.pack(fill='x', expand=True)
        self.DS8R_t_X_profile.focus()
        self.DS8R_t_X_profile.place(x=1350, y=530, width = 300)

        self.DS8R_y_profile = tk.StringVar()
        self.DS8R_lbl_Y_profile = ttk.Label(self.parent.frame_exp, text='DS8R stim amp (%max):')
        self.DS8R_lbl_Y_profile.pack(fill='x', expand=True)
        self.DS8R_lbl_Y_profile.place(x=1200, y=560)
        self.DS8R_t_Y_profile = tk.Entry(self.parent.frame_exp, textvariable=self.DS8R_y_profile)
        self.DS8R_t_Y_profile.insert(0, "0, 0.2, 0.4, 0.6, 0.8, 1")
        self.DS8R_t_Y_profile.pack(fill='x', expand=True)
        self.DS8R_t_Y_profile.focus()
        self.DS8R_t_Y_profile.place(x=1350, y=560, width = 300)

        self.pushstim_button = tk.Button(self.parent.frame_exp, text='PUSH STIM', bg ='yellow')
        self.pushstim_button['command'] = self.stim_push
        self.pushstim_button.pack()
        self.pushstim_button.place(x=710, y=600)
        
        self.clearstim_button = tk.Button(self.parent.frame_exp, text='Clear STIM', bg ='yellow')
        self.clearstim_button['command'] = self.stim_clear
        self.clearstim_button.pack()
        self.clearstim_button.place(x=1000, y=600)

        self.check_MEPs_button = tk.Button(self.parent.frame_exp, text='Check MEPs', bg ='yellow')
        self.check_MEPs_button['command'] = self.check_MEPs
        self.check_MEPs_button.pack()
        self.check_MEPs_button.place(x=900, y=190)

        self.DS8R_pushstim_button = tk.Button(self.parent.frame_exp, text='PUSH STIM', bg ='yellow')
        self.DS8R_pushstim_button['command'] = self.DS8R_stim_push
        self.DS8R_pushstim_button.pack()
        self.DS8R_pushstim_button.place(x=1200, y=600)

        self.param_file_path = tk.StringVar()
        self.lbl_param_file_path = ttk.Label(self.parent.frame_exp, text='Config file: ')
        self.lbl_param_file_path.pack(fill='x', expand=True)
        self.lbl_param_file_path.place(x=1000, y=20)
        self.t_param_file_path = tk.Entry(self.parent.frame_exp, textvariable=self.param_file_path)
        self.t_param_file_path.insert(0, os.path.join(self.dump_path,'params.json'))
        self.t_param_file_path.pack(fill='x', expand=True)
        self.t_param_file_path.focus()
        self.t_param_file_path.place(x=1100, y=20, width = 200)

        self.read_prev_trial_button = tk.Button(self.parent.frame_exp, text='PREV', bg ='yellow')
        self.read_prev_trial_button['command'] = self.read_prev_trial
        self.read_prev_trial_button.pack()
        self.read_prev_trial_button.place(x=1000, y=50)

        self.read_next_trial_button = tk.Button(self.parent.frame_exp, text='NEXT', bg ='yellow')
        self.read_next_trial_button['command'] = self.read_next_trial
        self.read_next_trial_button.pack()
        self.read_next_trial_button.place(x=1050, y=50)

        self.thresholding_file_path = tk.StringVar()
        self.lbl_thresholding_file_path = ttk.Label(self.parent.frame_exp, text='Thresholding file: ')
        self.lbl_thresholding_file_path.pack(fill='x', expand=True)
        self.lbl_thresholding_file_path.place(x=710, y=700)
        self.t_thresholding_file_path = tk.Entry(self.parent.frame_exp, textvariable=self.thresholding_file_path)
        self.t_thresholding_file_path.insert(0, os.path.join(self.dump_path,'thresholding.json'))
        self.t_thresholding_file_path.pack(fill='x', expand=True)
        self.t_thresholding_file_path.focus()
        self.t_thresholding_file_path.place(x=810, y=700, width = 200)
        
        self.read_thresholding_profile_button = tk.Button(self.parent.frame_exp, text='READ PROFILE', bg ='yellow')
        self.read_thresholding_profile_button['command'] = self.read_thresh_profile
        self.read_thresholding_profile_button.pack()
        self.read_thresholding_profile_button.place(x=1050, y=700)

        profile_options = ['Recrutiment Threshold']
        self.thresholding_profile = tk.StringVar() 
        self.thresholding_profile.set(options[0])
        self.thresholding_profile_drop = tk.OptionMenu( self.parent.frame_exp , self.thresholding_profile , *profile_options) #tk.Button(self, text='START', bg ='green')
        self.thresholding_profile_drop.pack()
        self.thresholding_profile_drop.place(x=810, y=750)
        
        self.push_thresholding_profile_button = tk.Button(self.parent.frame_exp, text='PUSH PROFILE', bg ='yellow')
        self.push_thresholding_profile_button['command'] = self.push_thresh_profile
        self.push_thresholding_profile_button.pack()
        self.push_thresholding_profile_button.place(x=1050, y=750)

        self.run_thresholding_button = tk.Button(self.parent.frame_exp, text='RUN THRESHOLDING EXP', bg ='yellow')
        self.run_thresholding_button['command'] = lambda: self.start_rec(flag = 'thresholding')
        self.run_thresholding_button.pack()
        self.run_thresholding_button.place(x=810, y=900)

        self.trial_finish_time = time.time()
        self.lbl_trial_timer = ttk.Label(self.parent.frame_exp, text='Time since last trial',font=('Helvetica 16 bold'))
        self.lbl_trial_timer.pack(fill='x', expand=True)
        self.lbl_trial_timer.place(x=1250, y=150)

        self.trial_time = tk.StringVar()
        self.trial_time.set('0')
        self.trial_timer = ttk.Label(self.parent.frame_exp, textvariable=self.trial_time,font=('Helvetica 32 bold'))
        self.trial_timer.pack(fill='x', expand=True)
        self.trial_timer.place(x=1250, y=200)
        self.time_update_started = False

        self.TMS1_COM = tk.StringVar()
        self.lbl_TMS1_COM = ttk.Label(self.parent.frame_exp, text='TMS 1 COM PORT:')
        self.lbl_TMS1_COM.pack(fill='x', expand=True)
        self.lbl_TMS1_COM.place(x=450, y=400)
        self.t_TMS1_COM = tk.Entry(self.parent.frame_exp, textvariable=self.TMS1_COM)
        self.t_TMS1_COM.insert(0, "COM9")
        self.t_TMS1_COM.pack(fill='x', expand=True)
        self.t_TMS1_COM.focus()
        self.t_TMS1_COM.place(x=600, y=400, width = 100)

        self.TMS2_COM = tk.StringVar()
        self.lbl_TMS2_COM = ttk.Label(self.parent.frame_exp, text='TMS 2 COM PORT:')
        self.lbl_TMS2_COM.pack(fill='x', expand=True)
        self.lbl_TMS2_COM.place(x=450, y=430)
        self.t_TMS2_COM = tk.Entry(self.parent.frame_exp, textvariable=self.TMS2_COM)
        self.t_TMS2_COM.insert(0, "COM10")
        self.t_TMS2_COM.pack(fill='x', expand=True)
        self.t_TMS2_COM.focus()
        self.t_TMS2_COM.place(x=600, y=430, width = 100)

        self.init_TMS_button = tk.Button(self.parent.frame_exp, text='INIT TMS DEV', bg ='yellow')
        self.init_TMS_button['command'] = lambda: self.init_TMS()
        self.init_TMS_button.pack()
        self.init_TMS_button.place(x=450, y=480)

        self.term_TMS_button = tk.Button(self.parent.frame_exp, text='TERM TMS DEV', bg ='red')
        self.term_TMS_button['command'] = lambda: self.term_TMS()
        self.term_TMS_button.pack()
        self.term_TMS_button.place(x=550, y=480)
        
    def init_TMS(self):
        self.TMS1 = TMS_lib.DuoMAG.DuoMAG(self.TMS1_COM.get())
        self.TMS2 = TMS_lib.DuoMAG.DuoMAG(self.TMS2_COM.get())
        self.init_TMS_button.config(bg = 'green')

    def term_TMS(self):
        self.init_TMS_button.config(bg = 'yellow')
        del self.TMS1, self.TMS2

    def time_update(self):
        self.trial_time.set(int(time.time() - self.trial_finish_time))
        self.parent.update()
        self.parent.after(1000, self.time_update)
        if not self.time_update_started:
            self.time_update_started = True

    def read_json(self, path, trial_ID):
        with open(path, 'r') as f:
            params = json.load(f)
        return params[str(trial_ID)]

    def update_json(self, path, trial_ID):
        with open(path, 'r') as f:
            params = json.load(f)
        params[str(trial_ID)]['completion_flag'] = True
        params[str(trial_ID)]['notes'] =  str(self.painscore)
        with open(path, "w") as outfile: 
            json.dump(params, outfile, indent = 4) 

    def read_csv(self, path, trial_ID):
        
        params = np.loadtxt(path,dtype='str',delimiter=',')
        titles = params[0]
        param_vals = params[1:][trial_ID-1]
        param_dict = {}
        for i,title in enumerate(titles):
            param_dict[title] = param_vals[i]
        return param_dict

    def update_csv(self, path, trial_ID):
        params = np.loadtxt(path,dtype='str',delimiter=',')
        params[1:][trial_ID-1][-1] = '1'
        params[1:][trial_ID-1][-2] = str(self.painscore)
        np.savetxt(path,params,fmt= "%s", delimiter=',')

    def update_params(self, param_dict):
        self.stim_clear()
        self.X_profile.set(str(param_dict["X_axis"])[1:-1])# = tk.StringVar()
        self.Y_profile.set(str(param_dict["Y_axis"])[1:-1])# = tk.StringVar()
        self.do_vanilla()

        if int(param_dict['TMSflag']):
            self.stim_x_profile.set(str(param_dict["X_stim"])[1:-1])
            self.stim_y_profile.set(str(param_dict["Y_stim"])[1:-1])
            # self.parent.frame_exp.stim_rate.set(float(param_dict['stim_interval']))# = tk.StringVar()
            # self.stim_start.set(float(param_dict['start_time']))# = tk.StringVar()
            # self.stim_stop.set(float(param_dict['stop_time']))# = tk.StringVar()
            self.stim_push()

        if int(param_dict['DS8Rflag']):
            self.DS8R_x_profile.set(str(param_dict["X_DS8R"])[1:-1])
            self.DS8R_y_profile.set(str(param_dict["Y_DS8R"])[1:-1])
            self.DS8R_stim_push()

        if int(param_dict['completion_flag']) == True:
            showinfo("Trial marked as completed", "This trial has been marked as completed make sure to not duplicate files")
        self.parent.update()

    def read_cur_trial(self):
        current_trial = int(self.trial_ID.get())
        trial_param_dict = self.read_json(self.param_file_path.get(),current_trial)
        self.update_params(trial_param_dict)
        self.t_trial_ID.delete(0, 'end')
        self.t_trial_ID.insert(0, str(current_trial))

        self.parent.update()

    def read_next_trial(self):
        self.trial_ID.set(str(int(self.trial_ID.get())+1))
        current_trial = int(self.trial_ID.get())

        trial_param_dict = self.read_json(self.param_file_path.get(),current_trial)
        self.update_params(trial_param_dict)
        
        self.t_trial_ID.delete(0, 'end')
        self.t_trial_ID.insert(0, str(current_trial))
        self.parent.update()

    def read_thresh_profile(self):
        with open(self.thresholding_file_path.get(), 'r') as f:
            params = json.load(f)
        profile_options = []
        self.thresholding_profile_drop['menu'].delete(0, 'end')
        for k in params.keys():
            profile_options.append(params[k]['stim_type'])
        for choice in profile_options:
            self.thresholding_profile_drop['menu'].add_command(label=choice,command=tk._setit(self.thresholding_profile, choice))
        self.parent.update()

    def push_thresh_profile(self):
        with open(self.thresholding_file_path.get(), 'r') as f:
            params = json.load(f)
        for k in params.keys():
            if params[k]['stim_type'] == self.thresholding_profile.get():
                self.update_params(params[k])
        self.parent.update()

    def read_prev_trial(self):
        self.trial_ID.set(str(int(self.trial_ID.get())-1))
        current_trial = int(self.trial_ID.get())

        trial_param_dict = self.read_json(self.param_file_path.get(),current_trial)
        self.update_params(trial_param_dict)

        self.t_trial_ID.delete(0, 'end')
        self.t_trial_ID.insert(0, str(current_trial))
        self.parent.update()

    def clear_therm_config(self):
        self.heat_dict = {}
        self.heat_profile_x = np.empty(0)
        self.heat_profile_y = np.empty(0)
        assert len(self.target_profile_x) == len(self.target_profile_y)
        self.disp_target.clear()
        self.canvas_disp_target.draw()
        self.do_vanilla()
        # self.push_dict = self.therm_param_list_gen(self.therm_params, self.therm_select)
        # for key in self.thermodes.keys():
        #     self.thermodes[key].set_baseline(self.push_dict[key]["BL"])
        #     self.thermodes[key].set_durations(self.push_dict[key]["HOLD"])
        #     self.thermodes[key].set_ramp_speed(self.push_dict[key]["URATE"])
        #     self.thermodes[key].set_return_speed(self.push_dict[key]["DRATE"])
        #     self.thermodes[key].set_temperatures(self.push_dict[key]["TGT"])
        # showinfo(title='Thermode param sent', message="Set Thermodes to Baseline")

    def select_contacts(self):
        self.therm_select = {}
        keys = list(self.thermodes.keys())
        if len(self.thermodes)==1:
            self.therm_select[keys[0]] = np.array([self.t1_c1_check.get(),
                                          self.t1_c2_check.get(),
                                          self.t1_c3_check.get(),
                                          self.t1_c4_check.get(),
                                          self.t1_c5_check.get(),
                                          ])
        else:
            self.therm_select[keys[0]] = np.array([self.t1_c1_check.get(),
                                          self.t1_c2_check.get(),
                                          self.t1_c3_check.get(),
                                          self.t1_c4_check.get(),
                                          self.t1_c5_check.get(),
                                          ])
            self.therm_select[keys[1]] = np.array([self.t2_c1_check.get(),
                                          self.t2_c2_check.get(),
                                          self.t2_c3_check.get(),
                                          self.t2_c4_check.get(),
                                          self.t2_c5_check.get(),
                                          ])

    def therm_param_list_gen(self,therm_params, contact_select, key):
        push_dict = {}
        for key2 in therm_params[key].keys():
            if key2 == 'TGT':
                push_dict[key2] = np.ones(5)*therm_params[key]["BL"]
                push_dict[key2][np.where(contact_select[key]==1)[0]] = np.ones(np.sum(contact_select[key]))*therm_params[key][key2]
            else:
                push_dict[key2] = np.ones(5)*therm_params[key][key2]
        return push_dict

    def push_therm_config(self):
        self.therm_params = {}
        keys = list(self.thermodes.keys())
        stim_ctr = len(self.heat_dict)
        if len(self.thermodes)==1:
            self.therm_params[keys[0]] = {
                "BL" : float(self.therm1_baseline.get()),
                "TGT" : float(self.therm1_tgt_temp.get()),
                "HOLD" : float(self.therm1_hold_duration.get()),
                "DRATE" : float(self.therm1_ramp_down_rate.get()),
                "URATE" : float(self.therm1_ramp_up_rate.get()),
                }
            self.heat_dict[stim_ctr+1] = {}
            self.heat_dict[stim_ctr+1][keys[0]] = self.therm_param_list_gen(self.therm_params,self.therm_select,keys[0])
            self.heat_dict[stim_ctr+1][keys[0]]["INIT"] = float(self.therm1_start_time.get())

            self.heat_profile_y = np.concatenate((self.heat_profile_y,[1]))
            self.heat_profile_x = np.unique(np.concatenate((self.heat_profile_x,[float(self.therm1_start_time.get())])))
            assert float(self.therm1_tgt_temp.get()) <60
        else:
            self.therm_params[keys[0]] = {
                "BL" : float(self.therm1_baseline.get()),
                "TGT" : float(self.therm1_tgt_temp.get()),
                "HOLD" : float(self.therm1_hold_duration.get()),
                "DRATE" : float(self.therm1_ramp_down_rate.get()),
                "URATE" : float(self.therm1_ramp_up_rate.get()),
                }
            self.therm_params[keys[1]] = {
                "BL" : float(self.therm2_baseline.get()),
                "TGT" : float(self.therm2_tgt_temp.get()),
                "HOLD" : float(self.therm2_hold_duration.get()),
                "DRATE" : float(self.therm2_ramp_down_rate.get()),
                "URATE" : float(self.therm2_ramp_up_rate.get()),
                }
            self.heat_dict[stim_ctr+1] = {}
            self.heat_dict[stim_ctr+1][keys[0]] = self.therm_param_list_gen(self.therm_params,self.therm_select,keys[0])
            self.heat_dict[stim_ctr+1][keys[0]]["INIT"] = float(self.therm1_start_time.get())
            self.heat_dict[stim_ctr+1][keys[1]] = self.therm_param_list_gen(self.therm_params,self.therm_select,keys[1])
            self.heat_dict[stim_ctr+1][keys[1]]["INIT"] = float(self.therm2_start_time.get())


            self.heat_profile_y = np.concatenate((self.heat_profile_y,[1],[1]))
            self.heat_profile_x = np.unique(np.concatenate((self.heat_profile_x,[float(self.therm1_start_time.get())],[float(self.therm2_start_time.get())])))
            assert float(self.therm2_tgt_temp.get()) < 50
            assert float(self.therm1_tgt_temp.get()) < 50



        self.disp_target.vlines(self.heat_profile_x,0,np.max(self.target_profile_y), linewidth = 3, color = 'g')
        self.canvas_disp_target.draw()

        # for key in self.heat_dict.keys():
        #     self.disp_target.vlines(self.heat_profile_x,0,np.max(self.target_profile_y), linewidth = 3, color = 'k')
        # self.canvas_disp_target.draw()

        # for key in self.thermodes.keys():
        #     self.thermodes[key].set_baseline(self.push_dict[key]["BL"])
        #     self.thermodes[key].set_durations(self.push_dict[key]["HOLD"])
        #     self.thermodes[key].set_ramp_speed(self.push_dict[key]["URATE"])
        #     self.thermodes[key].set_return_speed(self.push_dict[key]["DRATE"])
        #     self.thermodes[key].set_temperatures(self.push_dict[key]["TGT"])
        # showinfo(title='Thermode param sent', message="Pushed params to Thermode")

    def init_therm(self):

        label_1 = str(self.therm_1_name.get())
        label_2 = str(self.therm_2_name.get())
        self.thermodes = {}
        self.heat_dict ={}
        if label_2 != 'None':
            self.thermodes[label_1] = TcsDevice(port=label_1)
            self.thermodes[label_1].set_quiet()
            self.thermodes[label_1].set_baseline(float(self.therm1_baseline.get()))
            # self.heat_dict[label_1]={}
            self.thermodes[label_2] = TcsDevice(port=label_2)
            self.thermodes[label_2].set_quiet()
            self.thermodes[label_2].set_baseline(float(self.therm2_baseline.get()))

            # self.heat_dict[label_2]={}
        else:
            self.thermodes[label_1] = TcsDevice(port=label_1)
            self.thermodes[label_1].set_quiet()
            self.thermodes[label_1].set_baseline(float(self.therm1_baseline.get()))
            # self.heat_dict[label_1]={}
        self.select_contacts()
        self.stop_therm_button.config(bg = 'red')
        self.init_therm_button.config(bg = 'green')
        showinfo(title='Thermode started', message="Started "+str(len(self.thermodes))+" Thermode")
        # """
        # for debug
        # """
        # label_1 = str(self.therm_1_name.get())
        # label_2 = str(self.therm_2_name.get())
        # self.thermodes = {label_1:0}
        # self.heat_dict ={}
        # # if label_2 != 'None':
        # #     self.thermodes[label_1] = TcsDevice(port=label_1)
        # #     self.thermodes[label_1].set_quiet()
        # #     # self.heat_dict[label_1]={}
        # #     self.thermodes[label_2] = TcsDevice(port=label_2)
        # #     self.thermodes[label_2].set_quiet()
        # #     # self.heat_dict[label_2]={}
        # # else:
        # #     self.thermodes[label_1] = TcsDevice(port=label_1)
        # #     self.thermodes[label_1].set_quiet()
        # #     # self.heat_dict[label_1]={}
        # self.select_contacts()
        # self.stop_therm_button.config(bg = 'red')
        # self.init_therm_button.config(bg = 'green')
        
    def stop_therm(self):
        for key in self.thermodes.keys():
            self.thermodes[key].close()
        self.init_therm_button.config(bg = 'yellow')
        self.stop_therm_button.config(bg = 'red')

    def start_rec(self,flag = 'rec'):

        self.task_trial.write(False)
        print('starting')
        self.start_tmsi(flag)
        start_time = time.time()
        self.task_analog.write(float(self.stim_intensity.get()))
        trial_params = {
            "duration": self.trl_duration,
            "MVF": float(self.max_force.get()),
            }
        window = display_force_data(self.parent, 
                                    self.task_trial, 
                                    self.task_stim, 
                                    self.task_analog,
                                    self.DS8R_analog,
                                    self.DS8R_trig,
                                    self.target_profile_x,
                                    self.target_profile_y,
                                    self.stim_profile_x,
                                    self.stim_profile_y,
                                    self.DS8R_profile_x,
                                    self.DS8R_profile_y,
                                    TMS_dev=[self.TMS1,self.TMS2],
                                    trial_params=trial_params,
                                    dev_select=self.vis_TMSi.get(),
                                    vis_chan_mode = self.vis_chan_mode.get(),
                                    vis_chan = self.vis_chan.get(),
                                    record=True
                                    )
        window.grab_set()
        self.parent.wait_window(window)

        out_mat = {
            "time": np.array(self.parent.frame_exp.dump_time),
            "force": np.array(self.parent.frame_exp.dump_force),
            "trigs": np.array(self.parent.frame_exp.dump_trig),
            "target_profile": np.array((self.target_profile_x,self.target_profile_y)).T,
            "MVC": float(self.max_force.get())
                }
        
        
        self.task_trial.write(False)
        self.stop_tmsi()
        
        self.task_analog.write(0)
        self.trial_finish_time = time.time()
        if not self.time_update_started:
            self.time_update()
        self.stim_clear()
        # self.parent.withdraw()
        if flag =='rec':
            savemat(os.path.join(self.dump_path,'trial_'+ self.trial_ID.get()+'_'+str(start_time)+'_profiles'+".mat"), out_mat)
            nWin = tk.Tk()
            nWin.withdraw()
            self.painscore = simpledialog.askstring(title="Notes for trial", prompt = "Notes for trial "+self.trial_ID.get())
            nWin.destroy()
            self.update_json(self.param_file_path.get(),int(self.trial_ID.get()))
            self.read_next_trial()
        else:
            savemat(os.path.join(self.dump_path,'thresholding','thresholding_'+str(start_time)+'_profiles'+".mat"), out_mat)
            self.push_thresh_profile()
        self.parent.update()

    def set_vis_mode(self):
        self.vis_chan_drop['menu'].delete(0, 'end')
        
        ch_list = self.tmsi_dev[self.vis_TMSi.get()].dev.config.channels
        self.parent.frame_exp.UNI_count = 0
        self.parent.frame_exp.AUX_count = 0
        self.parent.frame_exp.BIP_count = 0
        self.parent.frame_exp.DUD_count = 0
        for idx, ch in enumerate(ch_list):
            if (ch.type.value == ChannelType.UNI.value):
                if ch.enabled == True:
                    self.parent.frame_exp.UNI_count+=1
            elif (ch.type.value == ChannelType.AUX.value):
                if ch.enabled == True:
                    self.parent.frame_exp.AUX_count += 1
            elif (ch.type.value == ChannelType.BIP.value):
                if ch.enabled == True:
                    self.parent.frame_exp.BIP_count += 1
            else :
                self.parent.frame_exp.DUD_count += 1

        if self.vis_chan_mode.get() == 'single':
            options = [x for x in range(1,65)]
            self.vis_chan.set(options[1])
            for choice in options:
                self.vis_chan_drop['menu'].add_command(label=choice,command=tk._setit(self.vis_chan, choice))
        elif self.vis_chan_mode.get() == 'aux':
            options = [x for x in range(1,self.parent.frame_exp.AUX_count+self.parent.frame_exp.BIP_count+1)]
            self.vis_chan.set(options[0])
            for choice in options:
                self.vis_chan_drop['menu'].add_command(label=choice,command=tk._setit(self.vis_chan, choice))
        else:
            ch_list = self.tmsi_dev[self.vis_TMSi.get()].dev.config.channels
            options = [x for x in reversed(range(1,self.parent.frame_exp.UNI_count))]
            self.vis_chan.set(options[1])
            for choice in options:
                self.vis_chan_drop['menu'].add_command(label=choice,command=tk._setit(self.vis_chan, choice))
 
    def set_vis_mode_check(self):
        self.vis_chan_drop_check['menu'].delete(0, 'end')
        
        ch_list = self.tmsi_dev[self.vis_TMSi.get()].dev.config.channels
        self.parent.frame_exp.UNI_count = 0
        self.parent.frame_exp.AUX_count = 0
        self.parent.frame_exp.BIP_count = 0
        self.parent.frame_exp.DUD_count = 0
        for idx, ch in enumerate(ch_list):
            if (ch.type.value == ChannelType.UNI.value):
                if ch.enabled == True:
                    self.parent.frame_exp.UNI_count+=1
            elif (ch.type.value == ChannelType.AUX.value):
                if ch.enabled == True:
                    self.parent.frame_exp.AUX_count += 1
            elif (ch.type.value == ChannelType.BIP.value):
                if ch.enabled == True:
                    self.parent.frame_exp.BIP_count += 1
            else :
                self.parent.frame_exp.DUD_count += 1

        if self.vis_chan_mode_check.get() == 'single':
            options = [x for x in range(1,65)]
            self.vis_chan_check.set(options[1])
            for choice in options:
                self.vis_chan_drop_check['menu'].add_command(label=choice,command=tk._setit(self.vis_chan_check, choice))
        elif self.vis_chan_mode_check.get() == 'aux':
            options = [x for x in range(1,self.parent.frame_exp.AUX_count+self.parent.frame_exp.BIP_count+1)]
            self.vis_chan_check.set(options[0])
            for choice in options:
                self.vis_chan_drop_check['menu'].add_command(label=choice,command=tk._setit(self.vis_chan_check, choice))
        else:
            ch_list = self.tmsi_dev[self.vis_TMSi.get()].dev.config.channels
            options = [x for x in reversed(range(1,self.parent.frame_exp.UNI_count))]
            self.vis_chan_check.set(options[1])
            for choice in options:
                self.vis_chan_drop_check['menu'].add_command(label=choice,command=tk._setit(self.vis_chan_check, choice))
 
    def manualMVC(self):
        self.manualMVC_button.config(bg = 'green')
        self.max_force.set(self.max_force.get())
        self.parent.update()

    def stop_rec(self,):
        print('stopping')

        self.trial_ID.set(str(int(self.trial_ID.get())+1))
        current_trial = int(self.trial_ID.get())
        self.t_trial_ID.delete(0, 'end')
        self.t_trial_ID.insert(0, str(current_trial))
        self.parent.update()

    def test_force_read(self):
        print('starting')
        self.test_force_read_button.config(bg = 'green')
        self.start_tmsi(flag = "no_rec")

        trial_params = {
            "duration": self.trl_duration,
            "MVF": float(self.max_force.get()),
            }
        # self.task_stim= [] 
        # self.task_trial = []
        window = display_force_data(self.parent, self.task_trial, 
                                    self.task_stim, 
                                    self.task_analog,
                                    self.DS8R_analog,
                                    self.DS8R_trig,
                                    self.target_profile_x,
                                    self.target_profile_y,
                                    stim_profile_x =  np.empty(0),
                                    stim_profile_y =  np.empty(0),
                                    DS8R_stim_x =  np.empty(0),
                                    DS8R_stim_y =  np.empty(0),
                                    TMS_dev=[self.TMS1,self.TMS2],
                                    trial_params=trial_params,
                                    dev_select=self.vis_TMSi.get(),
                                    vis_chan_mode = self.vis_chan_mode.get(),
                                    vis_chan = self.vis_chan.get(),
                                    record=False
                                    )
        window.grab_set()
        self.parent.wait_window(window)
        self.stop_tmsi(flag = "no_rec")
        self.test_force_read_button.config(bg = 'yellow')

    def check_MEPs(self):
        self.task_trial.write(False)
        print('starting')
        self.start_tmsi(flag='check')
        start_time = time.time()
        trial_params = {
            "duration": self.trl_duration,
            "MVF": float(self.max_force.get()),
            "MEP_winU": 100,
            "MEP_winL": -50,
            }
        window = check_MEPs_win(self.parent, self.task_trial, 
                                    self.task_stim, 
                                    self.task_analog,
                                    self.DS8R_analog,
                                    self.DS8R_trig,
                                    self.target_profile_x,
                                    self.target_profile_y,
                                    self.stim_profile_x,
                                    self.stim_profile_y,
                                    self.DS8R_profile_x,
                                    self.DS8R_profile_y,
                                    TMS_dev=[self.TMS1,self.TMS2],
                                    trial_params=trial_params,
                                    dev_select=self.vis_TMSi.get(),
                                    vis_chan_mode = self.vis_chan_mode.get(),
                                    vis_chan = self.vis_chan.get(),
                                    vis_chan_mode_check=self.vis_chan_mode_check.get(), 
                                    vis_chan_check = self.vis_chan_check.get(),
                                    record=True,
                                    )
        self.task_trial.write(False)
        window.grab_set()
        self.parent.wait_window(window)
        self.stop_tmsi(flag='rec')
        self.parent.update()

    def start_DAQ(self):
        self.task_stim = None
        self.task_trial = None
        self.task_analog = None

        self.DS8R_analog = None
        self.DS8R_trig = None

        daq_name = self.daq_name.get()

        self.task_trial = nidaqmx.Task("trial_trig")
        self.task_trial.do_channels.add_do_chan( daq_name+"/" + self.trial_chan.get(),line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
        self.task_trial.start()
        self.task_trial.write(False)
        self.task_stim = nidaqmx.Task("stim_trig")
        self.task_stim.do_channels.add_do_chan( daq_name+"/" + self.stim_chan.get(),line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
        self.task_stim.start()
        self.task_stim.write(False)
        
        self.task_analog = nidaqmx.Task("analog_val")
        self.task_analog.ao_channels.add_ao_voltage_chan( daq_name+"/" + self.analog_chan.get(), min_val = float(self.analog_chan_minV.get()), max_val= float(self.analog_chan_maxV.get()))
        self.task_analog.start()
        self.task_analog.write(float(self.stim_intensity.get()))

        self.DS8R_trig = nidaqmx.Task("DS8R_trig")
        self.DS8R_trig.do_channels.add_do_chan( daq_name+"/" + self.DS8R_trig_chan.get(),line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
        self.DS8R_trig.start()
        self.DS8R_trig.write(False)

        self.DS8R_analog = nidaqmx.Task("DS8R_stim")
        self.DS8R_analog.ao_channels.add_ao_voltage_chan( daq_name+"/" + self.DS8R_analog_chan.get(), min_val = float(self.DS8R_analog_chan_minV.get()), max_val= float(self.DS8R_analog_chan_maxV.get()))
        self.DS8R_analog.start()
        self.DS8R_analog.write(float(self.DS8R_intensity.get()))

        self.start_daq_button.config(bg = 'green')

    def stream_DAQ(self):
        self.stream_daq_button.config(bg = 'red')
        t0 = time.time()
        while time.time()-t0 < 5:
            print("trigs", self.task_trig.read(number_of_samples_per_channel=10))
            print("force", abs(np.mean(self.in_stream_force.read(number_of_samples_per_channel=10)))*float(self.parent.frame_exp.conv_factor.get()))
        self.stream_daq_button.config(bg = 'yellow')

    def start_tmsi(self,flag = "rec"):
        start_time = time.time()
        trial_num = self.trial_ID.get()
        dump_path = self.dump_path
        self.streams = {}
        self.file_writers = {}
        for key in self.tmsi_dev.keys():
            self.streams[key] = FileWriter(FileFormat.lsl, self.tmsi_dev[key].dev_name)
            self.streams[key].open(self.tmsi_dev[key].dev)
        
            if flag != "no_rec" and flag == "MVC":
                save_path = os.path.join(dump_path,'MVC','MVC_'+key+'.poly5')
                self.file_writers[key] = FileWriter(FileFormat.poly5, save_path)
                self.file_writers[key].open(self.tmsi_dev[key].dev)
            
            elif flag != "no_rec" and flag == "check":
                save_path = os.path.join(dump_path,'MEPs',key+'.poly5')
                self.file_writers[key] = FileWriter(FileFormat.poly5, save_path)
                self.file_writers[key].open(self.tmsi_dev[key].dev)
            
            elif flag != "no_rec" and flag == "thresholding":
                save_path = os.path.join(dump_path,'thresholding','thresholding_'+key+'.poly5')
                self.file_writers[key] = FileWriter(FileFormat.poly5, save_path)
                self.file_writers[key].open(self.tmsi_dev[key].dev)

            elif flag != "no_rec" and flag == "rec":
                save_path = os.path.join(dump_path,'trial_'+str(trial_num)+'_'+key+'.poly5')
                self.file_writers[key] = FileWriter(FileFormat.poly5, save_path)
                self.file_writers[key].open(self.tmsi_dev[key].dev)
                
            self.tmsi_dev[key].dev.start_measurement()
            # self.tmsi_dev[keysList[1]].dev.start_measurement()
        time.sleep(0.5)

    def stop_tmsi(self,flag='rec'):
        keysList = list(self.tmsi_dev.keys())
        time.sleep(0.2)
        for key in self.tmsi_dev.keys():
            if flag == "rec":
                self.file_writers[key].close()
                # self.file_writer2.close()
            self.streams[key].close()
            # self.stream_2.close()
            time.sleep(0.5)
            self.tmsi_dev[key].dev.stop_measurement()
            # self.tmsi_dev[keysList[1]].dev.stop_measurement()

    def get_MVC(self):
        self.task_trial.write(False)
        self.EMG_avg_win = 100
        trial_len = int(self.MVC_duration.get())
        max_force = 0
        self.start_MVC_button.config(bg = 'red')

        ch_list = self.tmsi_dev[self.vis_TMSi.get()].dev.config.channels
        self.parent.frame_exp.UNI_count = 0
        self.parent.frame_exp.AUX_count = 0
        self.parent.frame_exp.BIP_count = 0
        self.parent.frame_exp.DUD_count = 0
        for idx, ch in enumerate(ch_list):
            if (ch.type.value == ChannelType.UNI.value):
                if ch.enabled == True:
                    self.parent.frame_exp.UNI_count+=1
            elif (ch.type.value == ChannelType.AUX.value):
                if ch.enabled == True:
                    self.parent.frame_exp.AUX_count += 1
            elif (ch.type.value == ChannelType.BIP.value):
                if ch.enabled == True:
                    self.parent.frame_exp.BIP_count += 1
            else :
                self.parent.frame_exp.DUD_count += 1
        if self.vis_chan_mode.get() == 'single':
            self.vis_chan_slice = np.array([int(self.vis_chan.get())])
        elif self.vis_chan_mode.get() == 'aux':
            self.vis_chan_slice = np.array([int(self.vis_chan.get()) + self.parent.frame_exp.UNI_count-1])
        else:
            self.vis_chan_slice = np.arange(int(self.vis_chan.get()))

        if self.vis_chan_mode.get() == 'aux':
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
        
        self.start_tmsi(flag = "MVC")
        print("finding stream")
        stream = pylsl.resolve_stream('name', self.vis_TMSi.get())
        for info in stream:
            print('name: ', info.name())
            print('channel count:', info.channel_count())
            print('sampling rate:', info.nominal_srate())
            print('type: ', info.type())
        self.inlet = DataInlet(stream[0])    
        

        self.inlet.inlet.open_stream()
        self.task_trial.write(True)
        array_data = self.inlet.pull_and_plot()#

        samples_raw, z_sos_raw= sosfilt(sos_raw, array_data[:self.EMG_avg_win,self.vis_chan_slice].T, zi=z_sos_raw)
        samples = np.abs(samples_raw) - np.min(abs(samples_raw),axis =0).reshape(1,-1)
        _, z_sos_env= sosfilt(sos_env, samples, zi=z_sos_env)
        t0 = time.time()
        ctr = 0 # this counter prevents initial values (with filter artifact) from being saved into MVC
        baseline_list = []
        baseline = 0
        while time.time()-t0 < 3:
            if ctr<5:
                ctr+=1
                time.sleep(0.1)
                array_data = self.inlet.pull_and_plot()
                if self.vis_chan_mode.get() == 'aux':
                    array_data_filt = array_data[:self.EMG_avg_win,self.vis_chan_slice] + self.parent.frame_exp.vis_scaling_offset
                else:
                    samples_raw, z_sos_raw= sosfilt(sos_raw, array_data[:self.EMG_avg_win,self.vis_chan_slice].T, zi=z_sos_raw)
                    samples = np.abs(samples_raw) - np.min(abs(samples_raw),axis =0).reshape(1,-1)
                    array_data_filt, z_sos_env= sosfilt(sos_env, samples, zi=z_sos_env)
                array_data_scaled = np.nan_to_num(array_data_filt,nan=0,posinf=0,neginf=0).T
                curr_force = np.median(array_data_scaled)
                if self.vis_chan_mode.get() == 'aux':
                    curr_force = curr_force*float(self.parent.frame_exp.conv_factor.get())
                print("not saved",curr_force)
            else:
                time.sleep(0.1)
                array_data = self.inlet.pull_and_plot()
                if self.vis_chan_mode.get() == 'aux':
                    array_data_filt = array_data[:self.EMG_avg_win,self.vis_chan_slice]+ self.parent.frame_exp.vis_scaling_offset*float(self.parent.frame_exp.conv_factor.get())
                else:
                    samples_raw, z_sos_raw= sosfilt(sos_raw, array_data[:self.EMG_avg_win,self.vis_chan_slice].T, zi=z_sos_raw)
                    samples = np.abs(samples_raw) - np.min(abs(samples_raw),axis =0).reshape(1,-1)
                    array_data_filt, z_sos_env= sosfilt(sos_env, samples, zi=z_sos_env)
                array_data_scaled = np.nan_to_num(array_data_filt,nan=0,posinf=0,neginf=0).T
                baseline = np.median(array_data_scaled)
                # if self.vis_chan_mode.get() == 'aux':
                #     baseline = baseline*float(self.conv_factor.get())
                baseline_list.append(baseline)
                baseline = np.median(baseline_list)
                print("Baseline",baseline)

        
        showinfo(title='START MVC', message="START MVC")
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 2.5
        color = (0,0,0)
        thickness = 3
        go_image = np.zeros((300,300,3))
        go_image[:,:,1] = np.ones((300,300))*255
        textsize = cv2.getTextSize("GO!", font, fontScale, thickness,)[0]
        textX = (go_image.shape[1] - textsize[0]) // 2
        textY = (go_image.shape[0] + textsize[1]) // 2
        go_image = cv2.putText(go_image, "GO!",  (textX, textY), font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.namedWindow("Gesture", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Gesture", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Gesture', go_image)
        key = cv2.waitKey(1)
        t0 = time.time()
        while time.time()-t0 < trial_len:
            time.sleep(0.1)
            array_data = self.inlet.pull_and_plot()
            if self.vis_chan_mode.get() == 'aux':
                array_data_filt = array_data[:self.EMG_avg_win,self.vis_chan_slice] + self.parent.frame_exp.vis_scaling_offset#sosfilt(sos_raw, array_data[:self.EMG_avg_win,self.vis_chan_slice].T, zi=z_sos_raw)
                # samples = abs(samples_raw) - np.min(abs(samples_raw),axis =0).reshape(1,-1)
                # array_data_filt, z_sos_env= sosfilt(samples_raw.T, samples, zi=z_sos_env)
            else:
                samples_raw, z_sos_raw= sosfilt(sos_raw, array_data[:self.EMG_avg_win,self.vis_chan_slice].T, zi=z_sos_raw)
                samples = np.abs(samples_raw) - np.min(abs(samples_raw),axis =0).reshape(1,-1)
                array_data_filt, z_sos_env= sosfilt(sos_env, samples, zi=z_sos_env)
            array_data_scaled = np.nan_to_num(array_data_filt,nan=0,posinf=0,neginf=0).T
            curr_force = np.median(array_data_scaled)
            
            if self.vis_chan_mode == 'aux':
                curr_force =(abs(np.median(array_data_scaled) -baseline))*float(self.parent.frame_exp.conv_factor.get())
                # print("using", baseline)
                # curr_force = curr_force*float(self.conv_factor.get())
            else:
                curr_force = abs(np.median(array_data_scaled) - baseline)
            print(curr_force)
            if curr_force > max_force:
                max_force = curr_force
                self.max_force.set(str(np.float16(max_force)))
                self.parent.update()

        
        self.task_trial.write(False)
        self.inlet.inlet.close_stream()

        cv2.destroyAllWindows()

        self.stop_tmsi()
        # showinfo(title='STOP MVC', message="STOP MVC")
        self.start_MVC_button.config(bg = 'green')

    def do_vanilla(self):
        max_force = float(self.max_force.get())
        # peak_ramp_force = float(self.peak_ramp_force.get())
        # trl_duration = float(self.trl_duration.get())
        # init_wait = float(self.init_wait.get())
        x_profile = self.X_profile.get()
        y_profile = self.Y_profile.get()
        self.target_profile_x = np.array(x_profile.split(','),dtype = float)
        self.target_profile_y = np.array(y_profile.split(','),dtype = float) * max_force
        assert len(self.target_profile_x) == len(self.target_profile_y)
        self.stim_profile_x = np.empty(0)
        self.stim_profile_y = np.empty(0)
        self.DS8R_profile_x = np.empty(0)
        self.DS8R_profile_y = np.empty(0)

        self.disp_target.clear()
        self.disp_target.set_xlabel("Time (s)", fontsize=14)
        self.disp_target.set_ylabel("Torque (Nm)", fontsize=14)
        self.disp_target.plot(self.target_profile_x, self.target_profile_y, linewidth = 5, color = 'r')
        self.canvas_disp_target.draw()
        self.heat_dict = {}
        self.trl_duration = self.target_profile_x[-1]
        # self.start_sombrero_button.config(bg = 'yellow')
        # self.start_vanilla_button.config(bg = 'green')

    def DS8R_stim_push(self):
        
        DS8R_x_profile = self.DS8R_x_profile.get()
        DS8R_y_profile = self.DS8R_y_profile.get()
        
        DS8R_profile_x = np.array(DS8R_x_profile.split(','),dtype = float)
        DS8R_profile_y = np.array(DS8R_y_profile.split(','),dtype = float)*float(self.DS8R_intensity.get())
        
        self.DS8R_profile_x = np.unique(np.concatenate((self.DS8R_profile_x,DS8R_profile_x)))
        self.DS8R_profile_y = np.concatenate((self.DS8R_profile_y,DS8R_profile_y))
        for y_val, x_val in zip(self.DS8R_profile_y,self.DS8R_profile_x):
            self.disp_target_twin_ax.plot([x_val,x_val],[0,y_val], linewidth = 2, color = 'b')
        self.canvas_disp_target.draw()

        self.trl_duration = self.target_profile_x[-1]
        self.DS8R_pushstim_button.config(bg = 'green')

    def stim_push(self):
        
        TMS_xprofile = self.stim_x_profile.get()
        TMS_yprofile = self.stim_y_profile.get()
        
        trial_stim_profile_x = np.array(TMS_xprofile.split(','),dtype = float)
        trial_stim_profile_y = np.array(TMS_yprofile.split(','),dtype = float)*float(self.stim_intensity.get())
        
        self.stim_profile_x = np.unique(np.concatenate((self.stim_profile_x,trial_stim_profile_x)))
        self.stim_profile_y = np.concatenate((self.stim_profile_y,trial_stim_profile_y))
        self.disp_target.vlines(self.stim_profile_x,0,np.max(self.target_profile_y), linewidth = 2, color = 'k')
        self.canvas_disp_target.draw()

        self.trl_duration = self.target_profile_x[-1]
        self.pushstim_button.config(bg = 'green')

    def stim_clear(self):
        self.stim_profile_x = np.empty(0)
        self.stim_profile_y = np.empty(0)
        self.DS8R_profile_x = np.empty(0)
        self.DS8R_profile_y = np.empty(0)
        assert len(self.target_profile_x) == len(self.target_profile_y)
        self.disp_target.clear()
        self.disp_target_twin_ax.clear()
        self.canvas_disp_target.draw()
        self.do_vanilla()
        self.pushstim_button.config(bg = 'yellow')
        self.DS8R_pushstim_button.config(bg = 'yellow')

def main():
    tk_trial = APP([],{"FLX":[],"EXT":[]})
    tk_trial.mainloop()
    return None

if __name__ == "__main__":
    main()