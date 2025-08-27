import os
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, sosfilt_zi,sosfilt
# import nidaqmx
# import nidaqmx.system
# from nidaqmx.constants import LineGrouping
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from collections import deque
from scipy.io import savemat

from pylsl import StreamInlet, resolve_stream, StreamInfo, StreamOutlet
from tmsi_dual_interface.tmsi_libraries.TMSiFileFormats.file_writer import FileWriter, FileFormat
#luigi
import sys
import getopt
#contactile libraries
# from contactile_3dfb_lib import ListenerClass as lc
# from contactile_3dfb_lib import SensorClass as sc
import math
#luigi



import pylsl
import pyqtgraph as pg
# Basic parameters for the plotting window
plot_duration = 5  # how many seconds of data to show
update_interval = 30  # ms between screen updates
pull_interval = 100  # ms between each pull operation


class Inlet:
    """Base class to represent a plottable inlet"""
    def __init__(self, info: pylsl.StreamInfo):
        # create an inlet and connect it to the outlet we found earlier.
        # max_buflen is set so data older the plot_duration is discarded
        # automatically and we only pull data new enough to show it

        # Also, perform online clock synchronization so all streams are in the
        # same time domain as the local lsl_clock()
        # (see https://labstreaminglayer.readthedocs.io/projects/liblsl/ref/enums.html#_CPPv414proc_clocksync)
        # and dejitter timestamps
        self.inlet = pylsl.StreamInlet(info, max_buflen=plot_duration,
                                       processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter)
        # store the name and channel count
        self.name = info.name()
        self.channel_count = info.channel_count()

    def pull_and_plot(self, plot_time: float, plt: pg.PlotItem):
        """Pull data from the inlet and add it to the plot.
        :param plot_time: lowest timestamp that's still visible in the plot
        :param plt: the plot the data should be shown on
        """
        # We don't know what to do with a generic inlet, so we skip it.
        pass

class DataInlet(Inlet):
    """A DataInlet represents an inlet with continuous, multi-channel data that
    should be plotted as multiple lines."""
    dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]

    def __init__(self, info: pylsl.StreamInfo):#, plt: pg.PlotItem):
        super().__init__(info)
        # calculate the size for our buffer, i.e. two times the displayed data
        bufsize = (2 * math.ceil(info.nominal_srate() * plot_duration), info.channel_count())
        self.buffer = np.empty(bufsize, dtype=self.dtypes[info.channel_format()])
        empty = np.array([])
        # # create one curve object for each channel/line that will handle displaying the data
        # self.curves = [pg.PlotCurveItem(x=empty, y=empty, autoDownsample=True) for _ in range(self.channel_count)]
        # for curve in self.curves:
        #     plt.addItem(curve)

    def pull_and_plot(self,):
        # pull the data
        _, ts = self.inlet.pull_chunk(timeout=0.0,
                                      max_samples=self.buffer.shape[0],
                                      dest_obj=self.buffer)
        # ts will be empty if no samples were pulled, a list of timestamps otherwise
        # if ts:
        return self.buffer



class display_force_data(tk.Toplevel):
    def __init__(self, parent, stream_trig, stream_force, target_profile_x,target_profile_y, trial_params,dev_select='FLX', vis_chan_mode='avg', vis_chan = 10,record = False):
        super().__init__(parent)

        vis_buffer_len = 10
        self.vis_xlim_pad = 3
        self.EMG_avg_win = 100 #in samples
        self.vis_chan_mode  = vis_chan_mode
        self.vis_chan = int(vis_chan)

        self.force_holder = deque(list(np.empty(vis_buffer_len)))
        self.trig_holder = deque(list(np.empty(vis_buffer_len,dtype= bool)))
        self.x_axis = np.linspace(0,1,vis_buffer_len)

        self.attributes('-fullscreen', True)
        # self.geometry('1000x1000')
        self.title('Force Visualization')
        self.trial_params = trial_params
        self.stream_trig = stream_trig
        self.stream_force = stream_force
        self.rec_flag = record
        self.parent = parent
        if self.rec_flag:
            self.parent.dump_trig = []
            self.parent.dump_force = []
            self.parent.dump_time = []

        fig = Figure(figsize=(7, 4), dpi=100)
        self.disp_target = fig.add_subplot(111)
        
        self.disp_target.set_xlabel("Time (s)", fontsize=14)
        self.disp_target.set_ylabel("Force (N)", fontsize=14)
        
        self.canvas_disp_target = FigureCanvasTkAgg(fig, master=self)  
        self.canvas_disp_target.draw()
        self.canvas_disp_target.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.disp_target.set_xlabel("Time (s)", fontsize=14)
        self.disp_target.set_ylabel("Force (N)", fontsize=14)
        self.l_target = self.disp_target.plot(target_profile_x, target_profile_y, linewidth = 7, color = 'r')
        self.l_current = self.disp_target.plot(self.x_axis, self.force_holder, linewidth = 3, color = 'b',)
        self.disp_target.set_xlim([0,self.trial_params['duration']])
        self.disp_target.set_ylim([0,self.trial_params['MVF']*0.8])

        self.canvas_disp_target.draw()

        self.stream_vis_button = tk.Button(self, text='START TRIAL', bg ='yellow')
        self.stream_vis_button['command'] = self.start_vis
        self.stream_vis_button.pack()
        self.stream_vis_button.place(x=100, y=100)

        print("finding stream")
        stream = resolve_stream('name', dev_select)
        for info in stream:
            print('name: ', info.name())
            print('channel count:', info.channel_count())
            print('sampling rate:', info.nominal_srate())
            print('type: ', info.type())
            
        self.inlet = Inlet(stream[0])    
        sample_1, timestamp_1 = self.inlet.pull_and_plot()#(timeout=0.0,max_samples=self.EMG_avg_win)
        # sample_1, timestamp_1 = self.inlet.pull_chunk(timeout=0.0,max_samples=self.EMG_avg_win)
        time.sleep(0.1)

    def start_vis(self):
        t0 = time.time()
        time.sleep(0.055)
 
        sample_1, timestamp_1 = self.inlet.pull_chunk(timeout=0.0,max_samples=self.EMG_avg_win)
        n_chan = np.array(sample_1).shape[1]
        time.sleep(0.055)

                #     if Fc_hp and Fc_lp:
        sos_raw = butter(3, [20, 500], 'bandpass', fs=2000, output='sos')
        #         #     elif Fc_hp:
        #         #         sos=signal.butter(order, Fc_hp, 'highpass', fs=self.sample_rate, output='sos')
        #         #     elif Fc_lp:
        sos_env= butter(3, 5, 'lowpass', fs=2000, output='sos')
                    
        z_sos0 = sosfilt_zi(sos_raw)
        z_sos_raw=np.repeat(z_sos0[:, np.newaxis, :], 64, axis=1)
        
        z_sos0 = sosfilt_zi(sos_env)
        z_sos_env=np.repeat(z_sos0[:, np.newaxis, :], 64, axis=1)

        # nyq = 0.5 * 2000
        # b, a = butter(3, [20/ nyq, 500/ nyq], btype='band', analog=False)
        while time.time()-t0 < self.trial_params['duration']:
            time.sleep(0.055)
            t_prev = time.time()-t0
            
            self.trig_holder.popleft()
            # trig = self.stream_trig.read(number_of_samples_per_channel=10)
            trig = [0]
            self.trig_holder.append(trig[0])

            self.force_holder.popleft()


            sample_1, timestamp_1 = self.inlet.pull_chunk(timeout=0.0,max_samples=self.EMG_avg_win)



            array_data = np.array(sample_1).reshape(self.EMG_avg_win,n_chan)


            # samples_raw = np.zeros_like(array_data)
            # for i in range(array_data.shape[1]):
            #     samples_raw[:,i] = filtfilt(b, a, array_data[:,i])
            samples_raw, z_sos_raw= sosfilt(sos_raw, array_data[:,:64].T, zi=z_sos_raw)
            samples = abs(samples_raw) - np.min(abs(samples_raw),axis =0).reshape(1,-1)
            array_data_filt, z_sos_env= sosfilt(sos_env, samples, zi=z_sos_env)

            array_data_scaled = np.abs(np.nan_to_num(array_data_filt,nan=0,posinf=0,neginf=0)).T
            force_arr = np.mean(array_data_scaled,axis = 1)
            
            if self.vis_chan_mode == "single":
                force = abs(force_arr[self.vis_chan])
            elif self.vis_chan_mode == "aux":
                force = abs(force_arr[self.vis_chan+63])
            else:
                sorted_force = np.sort(force_arr)
                force = abs(np.mean(sorted_force[self.vis_chan:-self.vis_chan]))
            print(force)









            # force = abs(np.mean(self.stream_force.read(number_of_samples_per_channel=10)))*float(self.trial_params['conv_const'])
            
            #get force sensor. Force is 0 is no value is read from the buffer
            # res = self.parent.l.readAllInBuffer()
            # sensForce = np.zeros((1,3))
            # if res < 0:
            #     print('ERR: performanceTest::main(): readAllInBuffer returned error: ' + str(res))

            # elif res > 0: # Got one or more new samples (not guaranteed to be reading one sample at a time, just whatever is in the serial input buffer).
            #     # timestamp = self.sensors[0].getTimestamp_us()
            #     globalForces = self.parent.sensors[self.parent.SelectedSensor].getGlobalForce()
            #     sensForce = globalForces.reshape((1,3))
            
            # #split force in x,y,z coordinates
            # x_force = sensForce[:,0]
            # y_force = sensForce[:,1]
            # z_force = sensForce[:,2]
            # force = math.sqrt(math.pow(x_force,2) +  math.pow(y_force,2) +  math.pow(z_force,2))
            # force = 
            self.force_holder.append(force)

            if self.rec_flag:
                self.parent.dump_time.append(t_prev)
                self.parent.dump_trig.append(trig[0])
                self.parent.dump_force.append(force)

            self.l_current[0].set_data(self.x_axis*(time.time()-t0-t_prev)+t_prev,self.force_holder)
            self.disp_target.set_xlim([time.time()-t0-self.vis_xlim_pad,time.time()-t0+self.vis_xlim_pad])

            # print(t_prev, time.time()-t0, (self.x_axis + t_prev)*(time.time()-t0-t_prev))
            self.canvas_disp_target.draw()
            self.update()
        
        # self.inlet.close_stream()
        self.destroy()


class APP(tk.Toplevel):
    def __init__(self,parent,tmsi):
        super().__init__(parent)
        self.title('Force Ramp Interface')
        self.geometry('1200x1000')
        """
        Buttons
        """
        self.tmsi_dev = tmsi
        #luigi
        self.init_force_button = tk.Button(self, text='Init Force Button', bg ='gray')
        self.init_force_button['command'] = self.intForceButton
        self.init_force_button.pack()
        self.init_force_button.place(x=900, y=40)
        
        self.disc_force_button = tk.Button(self, text='Disconnect Force Button', bg ='gray')
        self.disc_force_button['command'] = self.disconnectForceButton
        self.disc_force_button.pack()
        self.disc_force_button.place(x=1000, y=40)
                
        self.SensComPort = tk.StringVar()
        self.lbl_SensComPort = ttk.Label(self, text='COM port:')
        self.lbl_SensComPort.pack(fill='x', expand=True)
        self.lbl_SensComPort.place(x=700, y=40)
        self.t_SensComPort = tk.Entry(self, textvariable=self.SensComPort)
        self.t_SensComPort.insert(0, "COM3")
        self.t_SensComPort.pack(fill='x', expand=True)
        self.t_SensComPort.focus()
        self.t_SensComPort.place(x=800, y=40, width = 50)
        
        self.SensorNumber = tk.StringVar()
        self.lbl_SensorNumber = ttk.Label(self, text='Sensor Num:')
        self.lbl_SensorNumber.pack(fill='x', expand=True)
        self.lbl_SensorNumber.place(x=700, y=70)
        self.t_SensorNumber = tk.Entry(self, textvariable=self.SensorNumber)
        self.t_SensorNumber.insert(0, "1")
        self.t_SensorNumber.pack(fill='x', expand=True)
        self.t_SensorNumber.focus()
        self.t_SensorNumber.place(x=800, y=70, width = 50)
        
        self.SensorFs = tk.StringVar()
        self.lbl_SensorFs = ttk.Label(self, text='Fs:')
        self.lbl_SensorFs.pack(fill='x', expand=True)
        self.lbl_SensorFs.place(x=700, y=100)
        self.t_SensorFs = tk.Entry(self, textvariable=self.SensorFs)
        self.t_SensorFs.insert(0, "2000")
        self.t_SensorFs.pack(fill='x', expand=True)
        self.t_SensorFs.focus()
        self.t_SensorFs.place(x=800, y=100, width = 50)
        
        self.MVC_value = tk.StringVar()
        self.lbl_MVC_value = ttk.Label(self, text='MVC value (N):')
        self.lbl_MVC_value.pack(fill='x', expand=True)
        self.lbl_MVC_value.place(x=10, y=280)
        self.t_MVC_value = tk.Entry(self, textvariable=self.MVC_value)
        self.t_MVC_value.insert(0, "0")
        self.t_MVC_value.pack(fill='x', expand=True)
        self.t_MVC_value.focus()
        self.t_MVC_value.place(x=150, y=280, width = 100)
        
        self.set_Manual_MVC_button = tk.Button(self, text='Set MVC', bg ='yellow')
        self.set_Manual_MVC_button['command'] = self.updateManualMVC
        self.set_Manual_MVC_button.pack()
        self.set_Manual_MVC_button.place(x=270, y=275)

        #luigi
        options = list(self.tmsi_dev.keys())
        self.vis_TMSi = tk.StringVar() 
        self.vis_TMSi.set(options[0])
        self.lbl_vis = ttk.Label(self, text='Select TMSi to visualize:')
        self.lbl_vis.pack(fill='x', expand=True)
        self.lbl_vis.place(x=200, y=10)
        self.vis_tmsi_drop = tk.OptionMenu( self , self.vis_TMSi , *options) #tk.Button(self, text='START', bg ='green')
        self.vis_tmsi_drop.pack()
        self.vis_tmsi_drop.place(x=350, y=10)
        
        self.lbl_vis_mode = ttk.Label(self, text='Select vis mode:')
        self.lbl_vis_mode.pack(fill='x', expand=True)
        self.lbl_vis_mode.place(x=450, y=10)
        self.vis_chan_mode = tk.StringVar() 
        self.vis_mode_option1 = tk.Radiobutton(self, text="Single Chan", variable=self.vis_chan_mode, value="single", command=self.set_vis_mode)
        self.vis_mode_option1.pack(fill='x', expand=True)
        self.vis_mode_option1.place(x=530, y=10)
        self.vis_mode_option2 = tk.Radiobutton(self, text="Average", variable=self.vis_chan_mode, value="avg", command=self.set_vis_mode)
        self.vis_mode_option2.pack(fill='x', expand=True)
        self.vis_mode_option2.place(x=530, y=30)

        options = [0,10,32,64]
        self.vis_chan = tk.StringVar() 
        self.vis_chan.set(options[1])
        self.vis_chan_drop = tk.OptionMenu( self , self.vis_chan , *options) #tk.Button(self, text='START', bg ='green')
        self.vis_chan_drop.pack()
        self.vis_chan_drop.place(x=650, y=10)
        
        self.start_rec_button = tk.Button(self, text='START', bg ='green')
        self.start_rec_button['command'] = self.start_rec
        self.start_rec_button.pack()
        self.start_rec_button.place(x=10, y=10)

        self.stop_rec_button = tk.Button(self, text='STOP', bg ='red')
        self.stop_rec_button['command'] = self.stop_rec
        self.stop_rec_button.pack()
        self.stop_rec_button.place(x=70, y=10)

        self.trial_ID = tk.StringVar()
        self.lbl_trial_ID = ttk.Label(self, text='Trial Num:')
        self.lbl_trial_ID.pack(fill='x', expand=True)
        self.lbl_trial_ID.place(x=10, y=40)
        self.t_trial_ID = tk.Entry(self, textvariable=self.trial_ID)
        self.t_trial_ID.insert(0, "1")
        self.t_trial_ID.pack(fill='x', expand=True)
        self.t_trial_ID.focus()
        self.t_trial_ID.place(x=150, y=40, width = 50)

        self.check_dir_button = tk.Button(self, text='CHECK DIR', bg ='yellow')
        self.check_dir_button['command'] = self.check_dir
        self.check_dir_button.pack()
        self.check_dir_button.place(x=250, y=40)

        self.dump_path = tk.StringVar()
        self.lbl_dump_path = ttk.Label(self, text='Dump Path:')
        self.lbl_dump_path.pack(fill='x', expand=True)
        self.lbl_dump_path.place(x=10, y=70)
        self.t_dump_path = tk.Entry(self, textvariable=self.dump_path)
        today = time.strftime("%Y%m%d")
        self.t_dump_path.insert(0, "data/PX/"+today+'/')
        self.t_dump_path.pack(fill='x', expand=True)
        self.t_dump_path.focus()
        self.t_dump_path.place(x=150, y=70, width = 500)

        self.daq_name = tk.StringVar()
        self.lbl_daq_name = ttk.Label(self, text='DAQ ID:')
        self.lbl_daq_name.pack(fill='x', expand=True)
        self.lbl_daq_name.place(x=10, y=100)
        self.t_daq_name = tk.Entry(self, textvariable=self.daq_name)
        self.t_daq_name.insert(0, "Dev3")
        self.t_daq_name.pack(fill='x', expand=True)
        self.t_daq_name.focus()
        self.t_daq_name.place(x=150, y=100, width = 100)

        self.analog_chan = tk.StringVar()
        self.lbl_Ach_name = ttk.Label(self, text='Analog Inp Chans:')
        self.lbl_Ach_name.pack(fill='x', expand=True)
        self.lbl_Ach_name.place(x=10, y=130)
        self.t_Ach_name = tk.Entry(self, textvariable=self.analog_chan)
        self.t_Ach_name.insert(0, "ai1")
        self.t_Ach_name.pack(fill='x', expand=True)
        self.t_Ach_name.focus()
        self.t_Ach_name.place(x=150, y=130, width = 100)

        self.digi_chan = tk.StringVar()
        self.lbl_Dch_name = ttk.Label(self, text='Digital Inp Chans:')
        self.lbl_Dch_name.pack(fill='x', expand=True)
        self.lbl_Dch_name.place(x=10, y=160)
        self.t_Dch_name = tk.Entry(self, textvariable=self.digi_chan)
        self.t_Dch_name.insert(0, "port0/line0")
        self.t_Dch_name.pack(fill='x', expand=True)
        self.t_Dch_name.focus()
        self.t_Dch_name.place(x=150, y=160, width = 100)

        self.start_daq_button = tk.Button(self, text='START DAQ', bg ='yellow')
        self.start_daq_button['command'] = self.start_DAQ
        self.start_daq_button.pack()
        self.start_daq_button.place(x=10, y=190)

        self.stream_daq_button = tk.Button(self, text='STREAM DAQ', bg ='yellow')
        self.stream_daq_button['command'] = self.stream_DAQ
        self.stream_daq_button.pack()
        self.stream_daq_button.place(x=200, y=190)

        self.test_force_read_button = tk.Button(self, text='TEST RIG', bg ='yellow')
        self.test_force_read_button['command'] = self.test_force_read
        self.test_force_read_button.pack()
        self.test_force_read_button.place(x=300, y=190)

        self.conv_factor = tk.StringVar()
        self.lbl_conv_factor = ttk.Label(self, text='Torque Const.:')
        self.lbl_conv_factor.pack(fill='x', expand=True)
        self.lbl_conv_factor.place(x=10, y=220)
        self.t_conv_factor = tk.Entry(self, textvariable=self.conv_factor)
        self.t_conv_factor.insert(0, "0.26959694")
        self.t_conv_factor.pack(fill='x', expand=True)
        self.t_conv_factor.focus()
        self.t_conv_factor.place(x=150, y=220, width = 100)

        self.MVC_duration = tk.StringVar()
        self.lbl_MVC_len = ttk.Label(self, text='Duration of MVC (s):')
        self.lbl_MVC_len.pack(fill='x', expand=True)
        self.lbl_MVC_len.place(x=10, y=250)
        self.t_MVC_len = tk.Entry(self, textvariable=self.MVC_duration)
        self.t_MVC_len.insert(0, "5")
        self.t_MVC_len.pack(fill='x', expand=True)
        self.t_MVC_len.focus()
        self.t_MVC_len.place(x=150, y=250, width = 100)
        
        self.start_MVC_button = tk.Button(self, text='Measure MVC', bg ='yellow')
        self.start_MVC_button['command'] = self.get_MVC
        self.start_MVC_button.pack()
        self.start_MVC_button.place(x=340, y=275)

        self.trl_duration = tk.StringVar()
        self.lbl_trl_duration = ttk.Label(self, text='Trial Duration (s):')
        self.lbl_trl_duration.pack(fill='x', expand=True)
        self.lbl_trl_duration.place(x=10, y=330)
        self.t_trl_duration = tk.Entry(self, textvariable=self.trl_duration)
        self.t_trl_duration.insert(0, "60")
        self.t_trl_duration.pack(fill='x', expand=True)
        self.t_trl_duration.focus()
        self.t_trl_duration.place(x=150, y=330, width = 100)

        self.init_wait = tk.StringVar()
        self.lbl_init_wait = ttk.Label(self, text='Ramp Delay (s):')
        self.lbl_init_wait.pack(fill='x', expand=True)
        self.lbl_init_wait.place(x=10, y=360)
        self.t_init_wait = tk.Entry(self, textvariable=self.init_wait)
        self.t_init_wait.insert(0, "5")
        self.t_init_wait.pack(fill='x', expand=True)
        self.t_init_wait.focus()
        self.t_init_wait.place(x=150, y=360, width = 100)

        self.peak_ramp_force = tk.StringVar()
        self.lbl_peak_ramp_force = ttk.Label(self, text='Max Ramp Force (x MVC):')
        self.lbl_peak_ramp_force.pack(fill='x', expand=True)
        self.lbl_peak_ramp_force.place(x=310, y=360)
        self.t_peak_ramp_force = tk.Entry(self, textvariable=self.peak_ramp_force)
        self.t_peak_ramp_force.insert(0, "0.3")
        self.t_peak_ramp_force.pack(fill='x', expand=True)
        self.t_peak_ramp_force.focus()
        self.t_peak_ramp_force.place(x=450, y=360, width = 100)

        self.lbl_max_force = ttk.Label(self, text="Max Force",font=('Helvetica 16 bold'))
        self.lbl_max_force.pack(fill='x', expand=True)
        self.lbl_max_force.place(x=400, y=150)
        self.max_force = tk.StringVar()
        self.max_force.set(self.MVC_value.get())
        self.lbl_max_force_num = ttk.Label(self, textvariable=self.max_force,font=('Helvetica 30 bold'))
        self.lbl_max_force_num.pack(fill='x', expand=True)
        self.lbl_max_force_num.place(x=400, y=200)

        self.start_vanilla_button = tk.Button(self, text='PUSH VANILLA', bg ='yellow')
        self.start_vanilla_button['command'] = self.do_vanilla
        self.start_vanilla_button.pack()
        self.start_vanilla_button.place(x=10, y=400)
        
        self.start_sombrero_button = tk.Button(self, text='PUSH SOMBRERO', bg ='yellow')
        self.start_sombrero_button['command'] = self.do_sombrero
        self.start_sombrero_button.pack()
        self.start_sombrero_button.place(x=310, y=400)

        self.sombrero_width = tk.StringVar()
        self.lbl_sombrero_width = ttk.Label(self, text='Sombrero hold (s):')
        self.lbl_sombrero_width.pack(fill='x', expand=True)
        self.lbl_sombrero_width.place(x=310, y=430)
        self.t_sombrero_width = tk.Entry(self, textvariable=self.sombrero_width)
        self.t_sombrero_width.insert(0, "10")
        self.t_sombrero_width.pack(fill='x', expand=True)
        self.t_sombrero_width.focus()
        self.t_sombrero_width.place(x=500, y=430, width = 100)

        self.sombrero_ramp = tk.StringVar()
        self.lbl_sombrero_ramp = ttk.Label(self, text='Sombrero ramp (s):')
        self.lbl_sombrero_ramp.pack(fill='x', expand=True)
        self.lbl_sombrero_ramp.place(x=310, y=460)
        self.t_sombrero_ramp = tk.Entry(self, textvariable=self.sombrero_ramp)
        self.t_sombrero_ramp.insert(0, "5")
        self.t_sombrero_ramp.pack(fill='x', expand=True)
        self.t_sombrero_ramp.focus()
        self.t_sombrero_ramp.place(x=500, y=460, width = 100)

        self.sombrero_force = tk.StringVar()
        self.lbl_sombrero_force = ttk.Label(self, text='Sombrero Ramp Force (x MVC):')
        self.lbl_sombrero_force.pack(fill='x', expand=True)
        self.lbl_sombrero_force.place(x=310, y=490)
        self.t_sombrero_force = tk.Entry(self, textvariable=self.sombrero_force)
        self.t_sombrero_force.insert(0, "0.1")
        self.t_sombrero_force.pack(fill='x', expand=True)
        self.t_sombrero_force.focus()
        self.t_sombrero_force.place(x=500, y=490, width = 100)

        self.target_profile_x = [0]
        self.target_profile_y = [0]

        fig = Figure(figsize=(7, 4), dpi=100)
        self.disp_target = fig.add_subplot(111)
        
        self.disp_target.set_xlabel("Time (s)", fontsize=14)
        self.disp_target.set_ylabel("Torque (Nm)", fontsize=14)
        
        self.canvas_disp_target = FigureCanvasTkAgg(fig, master=self)  
        self.canvas_disp_target.draw()
        self.canvas_disp_target.get_tk_widget().pack(side=tk.BOTTOM, fill='x', expand=True)
        self.canvas_disp_target.get_tk_widget().place(y=550,)
        
    #luigi
                 
    def intForceButton(self):
        print('Starting force button initialization...')
        self.init_force_button.config(bg = 'red')
        #get com port number
        port = self.SensComPort.get()
        sensorFs = self.SensorFs.get()
        
        # Initialise the sensors
        nSensors = 10
        self.SelectedSensor = int(self.SensorNumber.get())-1
        self.sensors = [sc.SensorClass() for _ in range(0,nSensors)]
        # Initialise the listener
        self.l = lc.ListenerClass(self.sensors,True)
        # Connect to the COM Port
        isOpen = self.l.connectToComPort(port)
        if not isOpen:
            print('Failed to open COM PORT')
            exit()
        else:
            self.init_force_button.config(bg = 'green')
            self.disc_force_button.config(bg = 'gray') 
                
        time.sleep(0.1)
        # Set sampling rate
        self.l.setSamplingPeriod(sensorFs) # 2000 us period
        time.sleep(0.1)
        # Bias the sensors (i.e., remove the offset. Make sure force sensor is not pressed)
        self.l.sendBiasRequest()
        time.sleep(0.1)
        
    def disconnectForceButton(self):
        self.l.disconnectFromComPort()
        self.disc_force_button.config(bg = 'green')
        self.init_force_button.config(bg = 'gray')
        
    def updateManualMVC(self):
        self.max_force.set(self.MVC_value.get())
        self.update()
        #luigi

    def set_vis_mode(self):
        self.vis_chan_drop['menu'].delete(0, 'end')
        if self.vis_chan_mode.get() == 'single':
            options = [x for x in range(1,65)]
            self.vis_chan.set(options[1])
            for choice in options:
                self.vis_chan_drop['menu'].add_command(label=choice,command=tk._setit(self.vis_chan, choice))
        elif self.vis_chan_mode.get() == 'aux':
            ch_list = self.tmsi_dev[self.vis_TMSi].dev.config.channels
            options = [x for x in range(1,len(ch_list)-64)]
            self.vis_chan.set(options[0])
            for choice in options:
                self.vis_chan_drop['menu'].add_command(label=choice,command=tk._setit(self.vis_chan, choice))
        else:
            options = [x for x in range(1,30)]
            self.vis_chan.set(options[1])
            for choice in options:
                self.vis_chan_drop['menu'].add_command(label=choice,command=tk._setit(self.vis_chan, choice))
            # self.vis_chan = tk.StringVar() 
            # self.vis_chan.set(options[1])
            # self.vis_chan_drop = tk.OptionMenu( self , self.vis_chan , *options) #tk.Button(self, text='START', bg ='green')
            # self.vis_chan_drop.pack()
            # self.vis_chan_drop.place(x=500, y=30)

    def start_rec(self,):
        print('starting')
        self.start_tmsi()
        start_time = time.time()
        trial_params = {
            "duration": float(self.trl_duration.get()),
            "conv_const": float(self.conv_factor.get()),
            "MVF": float(self.max_force.get()),
            }
        window = display_force_data(self, self.task_trig, 
                                    self.in_stream_force, 
                                    self.target_profile_x,
                                    self.target_profile_y,
                                    trial_params,
                                    dev_select=self.vis_TMSi.get(),
                                    vis_chan_mode = self.vis_chan_mode.get(),
                                    vis_chan = self.vis_chan.get(),
                                    record=True
                                    )
        window.grab_set()
        self.wait_window(window)

        out_mat = {
            "time": np.array(self.dump_time),
            "force": np.array(self.dump_force),
            "trigs": np.array(self.dump_trig),
            "target_profile": np.array((self.target_profile_x,self.target_profile_y)).T,
            "MVC": float(self.max_force.get())
                   }
        savemat(os.path.join(self.dump_path.get(),'trial_'+ self.trial_ID.get()+'_'+str(start_time)+'_dev1_'+".mat"), out_mat)
        self.stop_tmsi()
        self.trial_ID.set(str(int(self.trial_ID.get())+1))
        current_trial = int(self.trial_ID.get())
        self.t_trial_ID.delete(0, 'end')
        self.t_trial_ID.insert(0, str(current_trial))
        self.dump_path.get(), 
        self.trial_ID.get(),
        
        self.update()

    def stop_rec(self,):
        print('stopping')

        self.trial_ID.set(str(int(self.trial_ID.get())+1))
        current_trial = int(self.trial_ID.get())
        self.t_trial_ID.delete(0, 'end')
        self.t_trial_ID.insert(0, str(current_trial))
        self.update()

    def test_force_read(self):
        self.test_force_read_button.config(bg = 'red')
        print("force trace for acclamatization")
        trial_params = {
            "duration": float(self.trl_duration.get()),
            "conv_const": float(self.conv_factor.get()),
            "MVF": float(self.max_force.get()),
            }
        window = display_force_data(self, self.task_trig, 
                                    self.in_stream_force, 
                                    self.target_profile_x,
                                    self.target_profile_y,
                                    trial_params,
                                    )
        window.grab_set()
        self.wait_window(window)
        self.test_force_read_button.config(bg = 'yellow')

    def check_dir(self):
        dump_name = self.dump_path.get()
        if not os.path.isdir(dump_name):
            print("Dir not found, making it")
            os.makedirs(dump_name)
        self.check_dir_button.config(bg = 'green')

    def start_DAQ(self):
        daq_name = self.daq_name.get()
        di_chan_name = self.digi_chan.get()
        ai_chan_name = self.analog_chan.get()

        # self.task_trig = nidaqmx.Task("rec_trig")
        self.task_trig = []
        # self.task_trig.di_channels.add_di_chan(daq_name+"/" + di_chan_name, line_grouping=LineGrouping.CHAN_PER_LINE)

        # self.task_force = nidaqmx.Task("rec_force")
        self.task_force = []
        # self.task_force.ai_channels.add_ai_voltage_chan(daq_name+"/"+ai_chan_name)
        # self.in_stream_force = self.task_force.in_stream
        self.in_stream_force = []


        self.start_daq_button.config(bg = 'green')

    def stream_DAQ(self):
        self.stream_daq_button.config(bg = 'red')
        t0 = time.time()
        while time.time()-t0 < 5:
            print("trigs", self.task_trig.read(number_of_samples_per_channel=10))
            print("force", abs(np.mean(self.in_stream_force.read(number_of_samples_per_channel=10)))*float(self.conv_factor.get()))
        self.stream_daq_button.config(bg = 'yellow')

    def start_tmsi(self):
        start_time = time.time()
        trial_num = self.trial_ID.get()
        dump_path = self.dump_path.get()
        save_path1 = os.path.join(dump_path,'trial_'+str(trial_num)+'_'+str(start_time)+'_dev1_'+'.poly5')
        save_path2 = os.path.join(dump_path,'trial_'+str(trial_num)+'_'+str(start_time)+'_dev2_'+'.poly5')

        keysList = list(self.tmsi_dev.keys())

        self.file_writer1 = FileWriter(FileFormat.poly5, save_path1)
        self.file_writer1.open(self.tmsi_dev[keysList[0]].dev)
        self.stream_1 = FileWriter(FileFormat.lsl, self.tmsi_dev[keysList[0]].dev_name)
        self.stream_1.open(self.tmsi_dev[keysList[0]].dev)
        
        self.file_writer2 = FileWriter(FileFormat.poly5, save_path2)
        self.file_writer2.open(self.tmsi_dev[keysList[1]].dev)
        self.stream_2 = FileWriter(FileFormat.lsl, self.tmsi_dev[keysList[1]].dev_name)
        self.stream_2.open(self.tmsi_dev[keysList[1]].dev)
        
        self.tmsi_dev[keysList[0]].dev.start_measurement()
        self.tmsi_dev[keysList[1]].dev.start_measurement()
        time.sleep(0.5)

    def stop_tmsi(self):
        keysList = list(self.tmsi_dev.keys())
        self.file_writer1.close()
        self.stream_1.close()
        self.tmsi_dev[keysList[0]].dev.stop_measurement()
        self.file_writer2.close()
        self.stream_2.close()
        self.tmsi_dev[keysList[1]].dev.stop_measurement()

    def get_MVC(self):
        self.EMG_avg_win = 100
        trial_len = int(self.MVC_duration.get())
        t0 = time.time()
        max_force = 0
        self.start_MVC_button.config(bg = 'red')
        
        self.start_tmsi()

        print("finding stream")
        stream = resolve_stream('name', self.vis_TMSi.get())
        for info in stream:
            print('name: ', info.name())
            print('channel count:', info.channel_count())
            print('sampling rate:', info.nominal_srate())
            print('type: ', info.type())
        # self.inlet = StreamInlet(stream[0],int(self.EMG_avg_win*2))    
        self.inlet = DataInlet(stream[0])    
        showinfo(title='START MVC', message="START MVC")
        # time.sleep(0.055)

        # sample_1, timestamp_1 = self.inlet.pull_chunk(timeout=0.0,max_samples=self.EMG_avg_win)
        # time.sleep(0.055)

        # sample_1, timestamp_1 = self.inlet.pull_chunk(timeout=0.0,max_samples=self.EMG_avg_win)
        sample_1 = self.inlet.pull_and_plot()#
        print(sample_1)
        n_chan = np.array(sample_1).shape[1]

        array_data = sample_1[-100:,:64]


        # samples_raw = np.zeros_like(array_data)
        # for i in range(array_data.shape[1]):
        #     samples_raw[:,i] = filtfilt(b, a, array_data[:,i])


        samples_raw, z_sos_raw= sosfilt(sos_raw, array_data.T, zi=z_sos_raw)
        samples = abs(samples_raw) - np.min(abs(samples_raw),axis =0).reshape(1,-1)
        array_data_filt, z_sos_env= sosfilt(sos_env, samples, zi=z_sos_env)

        array_data_scaled = np.abs(np.nan_to_num(array_data_filt,nan=0,posinf=0,neginf=0)).T
        curr_force = np.median(array_data_scaled)
        print(curr_force)



        sample_1, timestamp_1 = self.inlet.pull_chunk(timeout=0.0,max_samples=self.EMG_avg_win)
        sample_1 = self.inlet.pull_and_plot()#
        print(sample_1)
        n_chan = np.array(sample_1).shape[1]

        array_data = sample_1[-100:,:64]
                #     if Fc_hp and Fc_lp:
        sos_raw = butter(3, [20, 500], 'bandpass', fs=2000, output='sos')
                #     elif Fc_hp:
                #         sos=signal.butter(order, Fc_hp, 'highpass', fs=self.sample_rate, output='sos')
                #     elif Fc_lp:
        sos_env= butter(3, 5, 'lowpass', fs=2000, output='sos')
                    
        z_sos0 = sosfilt_zi(sos_raw)
        z_sos_raw=np.repeat(z_sos0[:, np.newaxis, :], 64, axis=1)
        
        z_sos0 = sosfilt_zi(sos_env)
        z_sos_env=np.repeat(z_sos0[:, np.newaxis, :], 64, axis=1)

        # nyq = 0.5 * 2000
        # b, a = butter(3, [20/ nyq, 500/ nyq], btype='band', analog=False)
        while time.time()-t0 < trial_len:
            #read force from sensor
            # res = self.l.readAllInBuffer()
            # sensForce = np.zeros((1,3))
            # if res < 0:
            #     print('ERR: performanceTest::main(): readAllInBuffer returned error: ' + str(res))

            # elif res > 0: # Got one or more new samples (not guaranteed to be reading one sample at a time, just whatever is in the serial input buffer).
            #     # timestamp = self.sensors[0].getTimestamp_us()
            #     globalForces = self.sensors[self.SelectedSensor].getGlobalForce()
            #     sensForce = globalForces.reshape((1,3))
            
            # #split force in x,y,z coordinates
            # x_force = sensForce[:,0]
            # y_force = sensForce[:,1]
            # z_force = sensForce[:,2]
            
            # time.sleep(0.055)
            sample_1 = self.inlet.pull_and_plot()#
            print(sample_1)
            n_chan = np.array(sample_1).shape[1]

            array_data = sample_1[-100:,:64]


            # sample_1, timestamp_1 = self.inlet.pull_chunk(timeout=0.0,max_samples=self.EMG_avg_win)



            # array_data = np.array(sample_1).reshape(self.EMG_avg_win,n_chan)


            # samples_raw = np.zeros_like(array_data)
            # for i in range(array_data.shape[1]):
            #     samples_raw[:,i] = filtfilt(b, a, array_data[:,i])


            samples_raw, z_sos_raw= sosfilt(sos_raw, array_data[:,:64].T, zi=z_sos_raw)
            samples = abs(samples_raw) - np.min(abs(samples_raw),axis =0).reshape(1,-1)
            array_data_filt, z_sos_env= sosfilt(sos_env, samples, zi=z_sos_env)

            array_data_scaled = np.abs(np.nan_to_num(array_data_filt,nan=0,posinf=0,neginf=0)).T
            curr_force = np.median(array_data_scaled)
            print(curr_force)
            if curr_force > max_force:
                max_force = curr_force
                self.max_force.set(str(max_force))
                self.update()
        self.stop_tmsi()
        self.start_MVC_button.config(bg = 'green')
        # self.inlet.close_stream()
        showinfo(title='STOP MVC', message="STOP MVC")

    def do_sombrero(self):
        max_force = float(self.max_force.get())
        peak_ramp_force = float(self.peak_ramp_force.get())
        trl_duration = float(self.trl_duration.get())
        init_wait = float(self.init_wait.get())
        sombrero_width = float(self.sombrero_width.get())
        sombrero_ramp = float(self.sombrero_ramp.get())
        sombrero_force = float(self.sombrero_force.get())

        self.target_profile_x = [0, init_wait, init_wait+sombrero_ramp, init_wait+sombrero_ramp+sombrero_width, trl_duration//2, 
                                 trl_duration-init_wait-sombrero_ramp-sombrero_width, trl_duration-init_wait-sombrero_ramp, trl_duration-init_wait, trl_duration]
        self.target_profile_y = [0, 0, max_force*sombrero_force, max_force*sombrero_force, max_force*peak_ramp_force, max_force*sombrero_force, max_force*sombrero_force, 0, 0]
        assert len(self.target_profile_x) == len(self.target_profile_y)

        self.disp_target.clear()
        
        self.disp_target.set_xlabel("Time (s)", fontsize=14)
        self.disp_target.set_ylabel("Torque (Nm)", fontsize=14)
        self.disp_target.plot(self.target_profile_x, self.target_profile_y, linewidth = 5, color = 'r')
        self.canvas_disp_target.draw()
        self.start_vanilla_button.config(bg = 'yellow')
        self.start_sombrero_button.config(bg = 'green')

    def do_vanilla(self):
        max_force = float(self.max_force.get())
        peak_ramp_force = float(self.peak_ramp_force.get())
        trl_duration = float(self.trl_duration.get())
        init_wait = float(self.init_wait.get())

        self.target_profile_x = [0, init_wait, trl_duration//2, trl_duration-init_wait, trl_duration]
        self.target_profile_y = [0, 0, peak_ramp_force*max_force, 0, 0]
        assert len(self.target_profile_x) == len(self.target_profile_y)

        self.disp_target.clear()
        self.disp_target.set_xlabel("Time (s)", fontsize=14)
        self.disp_target.set_ylabel("Torque (Nm)", fontsize=14)
        self.disp_target.plot(self.target_profile_x, self.target_profile_y, linewidth = 5, color = 'r')
        self.canvas_disp_target.draw()

        self.start_sombrero_button.config(bg = 'yellow')
        self.start_vanilla_button.config(bg = 'green')



def main():
    tk_trial = APP()
    tk_trial.mainloop()
    return None

if __name__ == "__main__":
    main()