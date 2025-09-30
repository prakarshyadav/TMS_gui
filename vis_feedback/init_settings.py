import os
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
import numpy as np
import time
import json 
import yaml
from vis_feedback import render_emg,init_settings


def compose_trials(cont_dict, stim_dict, trl_dict, shuffle_seed):
    composed_trials = []
    for trl_id, trl in enumerate(trl_dict.keys()):
        contraction_params = cont_dict[trl_dict[trl]['contraction']]
        stimulation_params = stim_dict[trl_dict[trl]['stim']]
        for reps in range(trl_dict[trl]['repetition']):
            rep_dict = {
                'contraction': trl_dict[trl]['contraction'],
                'stim_type': trl_dict[trl]['stim'],
                'sham': trl_dict[trl]['sham'],
                'completion_flag': False,
                'notes': '',
                }
            if stimulation_params['shuffle_stim']:
                np.random.shuffle(stimulation_params['Y_DS8R'])
            rep_dict.update(contraction_params)
            rep_dict.update(stimulation_params)
            composed_trials.append(rep_dict)
    np.random.seed(shuffle_seed)
    np.random.shuffle(composed_trials)
    composed_trials_dict = {}
    for trl_id in range(len(composed_trials)):
        composed_trials[trl_id]['TrlNum'] = trl_id+1
        composed_trials_dict[trl_id+1] = composed_trials[trl_id]
    return composed_trials_dict

class APP:
    def __init__(self,parent,tmsi):
        # super().__init__(parent)
        self.parent = parent
        # self.notebook = ttk.Notebook(parent)
        # self.notebook.grid(row=1, column=0)

        # frame = ttk.Frame(self.notebook, width=2000, height=1000)
        # frame.pack(fill='both', expand=True)


        # self.notebook.add(frame, text='Muscle Map')

        """
        Buttons
        """
        self.tmsi = tmsi
        self.today = time.strftime("%Y%m%d")

        self.update_button = tk.Button(self.parent.frame_sett, text='Update', bg ='yellow')
        self.update_button['command'] = self.update_all
        self.update_button.pack()
        self.update_button.place(x=10, y=10)

        self.part_id = tk.StringVar()
        self.lbl_part_id = ttk.Label(self.parent.frame_sett, text='Participant ID:')
        self.lbl_part_id.pack(fill='x', expand=True)
        self.lbl_part_id.place(x=10, y=40)
        self.t_part_id = tk.Entry(self.parent.frame_sett, textvariable=self.part_id)
        self.t_part_id.insert(0, "PX")
        self.t_part_id.pack(fill='x', expand=True)
        self.t_part_id.focus()
        self.t_part_id.place(x=200, y=40, width = 200)

        self.expdate = tk.StringVar()
        self.lbl_expdate = ttk.Label(self.parent.frame_sett, text='Exp date/Exp ID:')
        self.lbl_expdate.pack(fill='x', expand=True)
        self.lbl_expdate.place(x=10, y=70)
        self.t_expdate = tk.Entry(self.parent.frame_sett, textvariable=self.expdate)
        self.t_expdate.insert(0, self.today)
        self.t_expdate.pack(fill='x', expand=True)
        self.t_expdate.focus()
        self.t_expdate.place(x=200, y=70, width = 200)

        self.trl_type_path = tk.StringVar()
        self.lbl_trl_type = ttk.Label(self.parent.frame_sett, text='Trial config')
        self.lbl_trl_type.pack(fill='x', expand=True)
        self.lbl_trl_type.place(x=10, y=480)
        self.t_trl_type = tk.Entry(self.parent.frame_sett, textvariable=self.trl_type_path)
        self.t_trl_type.insert(0, "./data/trial_types.yml")
        self.t_trl_type.pack(fill='x', expand=True)
        self.t_trl_type.focus()
        self.t_trl_type.place(x=200, y=480, width = 200)

        self.stim_type_path = tk.StringVar()
        self.lbl_stim_type = ttk.Label(self.parent.frame_sett, text='Stimulation config')
        self.lbl_stim_type.pack(fill='x', expand=True)
        self.lbl_stim_type.place(x=10, y=510)
        self.t_stim_type = tk.Entry(self.parent.frame_sett, textvariable=self.stim_type_path)
        self.t_stim_type.insert(0, "./data/stimulation_types.yml")
        self.t_stim_type.pack(fill='x', expand=True)
        self.t_stim_type.focus()
        self.t_stim_type.place(x=200, y=510, width = 200)

        self.cont_type_path = tk.StringVar()
        self.lbl_cont_type = ttk.Label(self.parent.frame_sett, text='Contraction config')
        self.lbl_cont_type.pack(fill='x', expand=True)
        self.lbl_cont_type.place(x=10, y=540)
        self.t_cont_type = tk.Entry(self.parent.frame_sett, textvariable=self.cont_type_path)
        self.t_cont_type.insert(0, "./data/contraction_types.yml")
        self.t_cont_type.pack(fill='x', expand=True)
        self.t_cont_type.focus()
        self.t_cont_type.place(x=200, y=540, width = 200)

        self.thScreen_type_path = tk.StringVar()
        self.lbl_thScreen_type = ttk.Label(self.parent.frame_sett, text='TMS/DS8R thresholding config')
        self.lbl_thScreen_type.pack(fill='x', expand=True)
        self.lbl_thScreen_type.place(x=10, y=570)
        self.t_thScreen_type = tk.Entry(self.parent.frame_sett, textvariable=self.thScreen_type_path)
        self.t_thScreen_type.insert(0, "./data/thresholding_types.yml")
        self.t_thScreen_type.pack(fill='x', expand=True)
        self.t_thScreen_type.focus()
        self.t_thScreen_type.place(x=200, y=570, width = 200)

        keys = list(self.tmsi.keys())
        if len(self.tmsi)>1:

            self.lbl_tmsi1 = ttk.Label(self.parent.frame_sett, text='TMSi 1 map:')
            self.lbl_tmsi1.pack(fill='x', expand=True)
            self.lbl_tmsi1.place(x=10, y=150)

            self.tmsi1_grid = tk.StringVar()
            self.lbl_tmsi1_grid = ttk.Label(self.parent.frame_sett, text='GRID:')
            self.lbl_tmsi1_grid.pack(fill='x', expand=True)
            self.lbl_tmsi1_grid.place(x=10, y=180)
            self.t_tmsi1_grid = tk.Entry(self.parent.frame_sett, textvariable=self.tmsi1_grid)
            self.t_tmsi1_grid.insert(0, keys[0])
            self.t_tmsi1_grid.pack(fill='x', expand=True)
            self.t_tmsi1_grid.focus()
            self.t_tmsi1_grid.place(x=60, y=180, width = 150)

            self.tmsi1_BIP1 = tk.StringVar()
            self.lbl_tmsi1_BIP1 = ttk.Label(self.parent.frame_sett, text='BIP 1-1:')
            self.lbl_tmsi1_BIP1.pack(fill='x', expand=True)
            self.lbl_tmsi1_BIP1.place(x=10, y=210)
            self.t_tmsi1_BIP1 = tk.Entry(self.parent.frame_sett, textvariable=self.tmsi1_BIP1)
            self.t_tmsi1_BIP1.insert(0, "Ipsilateral Biceps")
            self.t_tmsi1_BIP1.pack(fill='x', expand=True)
            self.t_tmsi1_BIP1.focus()
            self.t_tmsi1_BIP1.place(x=60, y=210, width = 150)

            self.tmsi1_BIP2 = tk.StringVar()
            self.lbl_tmsi1_BIP2 = ttk.Label(self.parent.frame_sett, text='BIP 1-2:')
            self.lbl_tmsi1_BIP2.pack(fill='x', expand=True)
            self.lbl_tmsi1_BIP2.place(x=10, y=240)
            self.t_tmsi1_BIP2 = tk.Entry(self.parent.frame_sett, textvariable=self.tmsi1_BIP2)
            self.t_tmsi1_BIP2.insert(0, 'Ipsilateral Triceps')
            self.t_tmsi1_BIP2.pack(fill='x', expand=True)
            self.t_tmsi1_BIP2.focus()
            self.t_tmsi1_BIP2.place(x=60, y=240, width = 150)

            self.tmsi1_BIP3 = tk.StringVar()
            self.lbl_tmsi1_BIP3 = ttk.Label(self.parent.frame_sett, text='BIP 2-1:')
            self.lbl_tmsi1_BIP3.pack(fill='x', expand=True)
            self.lbl_tmsi1_BIP3.place(x=10, y=270)
            self.t_tmsi1_BIP3 = tk.Entry(self.parent.frame_sett, textvariable=self.tmsi1_BIP3)
            self.t_tmsi1_BIP3.insert(0, 'Ipsilateral Thenar')
            self.t_tmsi1_BIP3.pack(fill='x', expand=True)
            self.t_tmsi1_BIP3.focus()
            self.t_tmsi1_BIP3.place(x=60, y=270, width = 150)

            self.tmsi1_BIP4 = tk.StringVar()
            self.lbl_tmsi1_BIP4 = ttk.Label(self.parent.frame_sett, text='BIP 2-2:')
            self.lbl_tmsi1_BIP4.pack(fill='x', expand=True)
            self.lbl_tmsi1_BIP4.place(x=10, y=300)
            self.t_tmsi1_BIP4 = tk.Entry(self.parent.frame_sett, textvariable=self.tmsi1_BIP4)
            self.t_tmsi1_BIP4.insert(0, 'Contralateral Thenar')
            self.t_tmsi1_BIP4.pack(fill='x', expand=True)
            self.t_tmsi1_BIP4.focus()
            self.t_tmsi1_BIP4.place(x=60, y=300, width = 150)

            self.tmsi1_AUX1 = tk.StringVar()
            self.lbl_tmsi1_AUX1 = ttk.Label(self.parent.frame_sett, text='AUX 1:')
            self.lbl_tmsi1_AUX1.pack(fill='x', expand=True)
            self.lbl_tmsi1_AUX1.place(x=10, y=330)
            self.t_tmsi1_AUX1 = tk.Entry(self.parent.frame_sett, textvariable=self.tmsi1_AUX1)
            self.t_tmsi1_AUX1.insert(0, 'Force sensor')
            self.t_tmsi1_AUX1.pack(fill='x', expand=True)
            self.t_tmsi1_AUX1.focus()
            self.t_tmsi1_AUX1.place(x=60, y=330, width = 150)

            self.tmsi1_AUX2 = tk.StringVar()
            self.lbl_tmsi1_AUX2 = ttk.Label(self.parent.frame_sett, text='AUX 2:')
            self.lbl_tmsi1_AUX2.pack(fill='x', expand=True)
            self.lbl_tmsi1_AUX2.place(x=10, y=360)
            self.t_tmsi1_AUX2 = tk.Entry(self.parent.frame_sett, textvariable=self.tmsi1_AUX2)
            self.t_tmsi1_AUX2.insert(0, 'N/A')
            self.t_tmsi1_AUX2.pack(fill='x', expand=True)
            self.t_tmsi1_AUX2.focus()
            self.t_tmsi1_AUX2.place(x=60, y=360, width = 150)

            self.tmsi1_AUX3 = tk.StringVar()
            self.lbl_tmsi1_AUX3 = ttk.Label(self.parent.frame_sett, text='AUX 3:')
            self.lbl_tmsi1_AUX3.pack(fill='x', expand=True)
            self.lbl_tmsi1_AUX3.place(x=10, y=390)
            self.t_tmsi1_AUX3 = tk.Entry(self.parent.frame_sett, textvariable=self.tmsi1_AUX3)
            self.t_tmsi1_AUX3.insert(0, 'N/A')
            self.t_tmsi1_AUX3.pack(fill='x', expand=True)
            self.t_tmsi1_AUX3.focus()
            self.t_tmsi1_AUX3.place(x=60, y=390, width = 150)


            self.lbl_tmsi2 = ttk.Label(self.parent.frame_sett, text='TMSi 2 map:')
            self.lbl_tmsi2.pack(fill='x', expand=True)
            self.lbl_tmsi2.place(x=310, y=150)

            self.tmsi2_grid = tk.StringVar()
            self.lbl_tmsi2_grid = ttk.Label(self.parent.frame_sett, text='GRID:')
            self.lbl_tmsi2_grid.pack(fill='x', expand=True)
            self.lbl_tmsi2_grid.place(x=310, y=180)
            self.t_tmsi2_grid = tk.Entry(self.parent.frame_sett, textvariable=self.tmsi2_grid)
            self.t_tmsi2_grid.insert(0, keys[1])
            self.t_tmsi2_grid.pack(fill='x', expand=True)
            self.t_tmsi2_grid.focus()
            self.t_tmsi2_grid.place(x=360, y=180, width = 150)

            self.tmsi2_BIP1 = tk.StringVar()
            self.lbl_tmsi2_BIP1 = ttk.Label(self.parent.frame_sett, text='BIP 1-1:')
            self.lbl_tmsi2_BIP1.pack(fill='x', expand=True)
            self.lbl_tmsi2_BIP1.place(x=310, y=210)
            self.t_tmsi2_BIP1 = tk.Entry(self.parent.frame_sett, textvariable=self.tmsi2_BIP1)
            self.t_tmsi2_BIP1.insert(0, "Contralateral Biceps")
            self.t_tmsi2_BIP1.pack(fill='x', expand=True)
            self.t_tmsi2_BIP1.focus()
            self.t_tmsi2_BIP1.place(x=360, y=210, width = 150)

            self.tmsi2_BIP2 = tk.StringVar()
            self.lbl_tmsi2_BIP2 = ttk.Label(self.parent.frame_sett, text='BIP 1-2:')
            self.lbl_tmsi2_BIP2.pack(fill='x', expand=True)
            self.lbl_tmsi2_BIP2.place(x=310, y=240)
            self.t_tmsi2_BIP2 = tk.Entry(self.parent.frame_sett, textvariable=self.tmsi2_BIP2)
            self.t_tmsi2_BIP2.insert(0, 'Contralateral Triceps')
            self.t_tmsi2_BIP2.pack(fill='x', expand=True)
            self.t_tmsi2_BIP2.focus()
            self.t_tmsi2_BIP2.place(x=360, y=240, width = 150)

            self.tmsi2_BIP3 = tk.StringVar()
            self.lbl_tmsi2_BIP3 = ttk.Label(self.parent.frame_sett, text='BIP 2-1:')
            self.lbl_tmsi2_BIP3.pack(fill='x', expand=True)
            self.lbl_tmsi2_BIP3.place(x=310, y=270)
            self.t_tmsi2_BIP3 = tk.Entry(self.parent.frame_sett, textvariable=self.tmsi2_BIP3)
            self.t_tmsi2_BIP3.insert(0, 'Contralateral Flexors')
            self.t_tmsi2_BIP3.pack(fill='x', expand=True)
            self.t_tmsi2_BIP3.focus()
            self.t_tmsi2_BIP3.place(x=360, y=270, width = 150)

            self.tmsi2_BIP4 = tk.StringVar()
            self.lbl_tmsi2_BIP4 = ttk.Label(self.parent.frame_sett, text='BIP 2-2:')
            self.lbl_tmsi2_BIP4.pack(fill='x', expand=True)
            self.lbl_tmsi2_BIP4.place(x=310, y=300)
            self.t_tmsi2_BIP4 = tk.Entry(self.parent.frame_sett, textvariable=self.tmsi2_BIP4)
            self.t_tmsi2_BIP4.insert(0, 'Contralateral Extensors')
            self.t_tmsi2_BIP4.pack(fill='x', expand=True)
            self.t_tmsi2_BIP4.focus()
            self.t_tmsi2_BIP4.place(x=360, y=300, width = 150)

            self.tmsi2_AUX1 = tk.StringVar()
            self.lbl_tmsi2_AUX1 = ttk.Label(self.parent.frame_sett, text='AUX 1:')
            self.lbl_tmsi2_AUX1.pack(fill='x', expand=True)
            self.lbl_tmsi2_AUX1.place(x=310, y=330)
            self.t_tmsi2_AUX1 = tk.Entry(self.parent.frame_sett, textvariable=self.tmsi2_AUX1)
            self.t_tmsi2_AUX1.insert(0, 'N/A')
            self.t_tmsi2_AUX1.pack(fill='x', expand=True)
            self.t_tmsi2_AUX1.focus()
            self.t_tmsi2_AUX1.place(x=360, y=330, width = 150)

            self.tmsi2_AUX2 = tk.StringVar()
            self.lbl_tmsi2_AUX2 = ttk.Label(self.parent.frame_sett, text='AUX 2:')
            self.lbl_tmsi2_AUX2.pack(fill='x', expand=True)
            self.lbl_tmsi2_AUX2.place(x=310, y=360)
            self.t_tmsi2_AUX2 = tk.Entry(self.parent.frame_sett, textvariable=self.tmsi2_AUX2)
            self.t_tmsi2_AUX2.insert(0, 'N/A')
            self.t_tmsi2_AUX2.pack(fill='x', expand=True)
            self.t_tmsi2_AUX2.focus()
            self.t_tmsi2_AUX2.place(x=360, y=360, width = 150)

            self.tmsi2_AUX3 = tk.StringVar()
            self.lbl_tmsi2_AUX3 = ttk.Label(self.parent.frame_sett, text='AUX 3:')
            self.lbl_tmsi2_AUX3.pack(fill='x', expand=True)
            self.lbl_tmsi2_AUX3.place(x=310, y=390)
            self.t_tmsi2_AUX3 = tk.Entry(self.parent.frame_sett, textvariable=self.tmsi2_AUX3)
            self.t_tmsi2_AUX3.insert(0, 'N/A')
            self.t_tmsi2_AUX3.pack(fill='x', expand=True)
            self.t_tmsi2_AUX3.focus()
            self.t_tmsi2_AUX3.place(x=360, y=390, width = 150)
        else:
            

            self.lbl_tmsi1 = ttk.Label(self.parent.frame_sett, text='TMSi 1 map:')
            self.lbl_tmsi1.pack(fill='x', expand=True)
            self.lbl_tmsi1.place(x=10, y=150)

            self.tmsi1_grid = tk.StringVar()
            self.lbl_tmsi1_grid = ttk.Label(self.parent.frame_sett, text='GRID:')
            self.lbl_tmsi1_grid.pack(fill='x', expand=True)
            self.lbl_tmsi1_grid.place(x=10, y=180)
            self.t_tmsi1_grid = tk.Entry(self.parent.frame_sett, textvariable=self.tmsi1_grid)
            self.t_tmsi1_grid.insert(0, keys[0])
            self.t_tmsi1_grid.pack(fill='x', expand=True)
            self.t_tmsi1_grid.focus()
            self.t_tmsi1_grid.place(x=60, y=180, width = 150)

            self.tmsi1_BIP1 = tk.StringVar()
            self.lbl_tmsi1_BIP1 = ttk.Label(self.parent.frame_sett, text='BIP 1-1:')
            self.lbl_tmsi1_BIP1.pack(fill='x', expand=True)
            self.lbl_tmsi1_BIP1.place(x=10, y=210)
            self.t_tmsi1_BIP1 = tk.Entry(self.parent.frame_sett, textvariable=self.tmsi1_BIP1)
            self.t_tmsi1_BIP1.insert(0, "Ipsilateral Biceps")
            self.t_tmsi1_BIP1.pack(fill='x', expand=True)
            self.t_tmsi1_BIP1.focus()
            self.t_tmsi1_BIP1.place(x=60, y=210, width = 150)

            self.tmsi1_BIP2 = tk.StringVar()
            self.lbl_tmsi1_BIP2 = ttk.Label(self.parent.frame_sett, text='BIP 1-2:')
            self.lbl_tmsi1_BIP2.pack(fill='x', expand=True)
            self.lbl_tmsi1_BIP2.place(x=10, y=240)
            self.t_tmsi1_BIP2 = tk.Entry(self.parent.frame_sett, textvariable=self.tmsi1_BIP2)
            self.t_tmsi1_BIP2.insert(0, 'Ipsilateral Triceps')
            self.t_tmsi1_BIP2.pack(fill='x', expand=True)
            self.t_tmsi1_BIP2.focus()
            self.t_tmsi1_BIP2.place(x=60, y=240, width = 150)

            self.tmsi1_BIP3 = tk.StringVar()
            self.lbl_tmsi1_BIP3 = ttk.Label(self.parent.frame_sett, text='BIP 2-1:')
            self.lbl_tmsi1_BIP3.pack(fill='x', expand=True)
            self.lbl_tmsi1_BIP3.place(x=10, y=270)
            self.t_tmsi1_BIP3 = tk.Entry(self.parent.frame_sett, textvariable=self.tmsi1_BIP3)
            self.t_tmsi1_BIP3.insert(0, 'Ipsilateral Thenar')
            self.t_tmsi1_BIP3.pack(fill='x', expand=True)
            self.t_tmsi1_BIP3.focus()
            self.t_tmsi1_BIP3.place(x=60, y=270, width = 150)

            self.tmsi1_BIP4 = tk.StringVar()
            self.lbl_tmsi1_BIP4 = ttk.Label(self.parent.frame_sett, text='BIP 2-2:')
            self.lbl_tmsi1_BIP4.pack(fill='x', expand=True)
            self.lbl_tmsi1_BIP4.place(x=10, y=300)
            self.t_tmsi1_BIP4 = tk.Entry(self, textvariable=self.tmsi1_BIP4)
            self.t_tmsi1_BIP4.insert(0, 'Contralateral Thenar')
            self.t_tmsi1_BIP4.pack(fill='x', expand=True)
            self.t_tmsi1_BIP4.focus()
            self.t_tmsi1_BIP4.place(x=60, y=300, width = 150)

            self.tmsi1_AUX1 = tk.StringVar()
            self.lbl_tmsi1_AUX1 = ttk.Label(self.parent.frame_sett, text='AUX 1:')
            self.lbl_tmsi1_AUX1.pack(fill='x', expand=True)
            self.lbl_tmsi1_AUX1.place(x=10, y=330)
            self.t_tmsi1_AUX1 = tk.Entry(self.parent.frame_sett, textvariable=self.tmsi1_AUX1)
            self.t_tmsi1_AUX1.insert(0, 'Force sensor')
            self.t_tmsi1_AUX1.pack(fill='x', expand=True)
            self.t_tmsi1_AUX1.focus()
            self.t_tmsi1_AUX1.place(x=60, y=330, width = 150)

            self.tmsi1_AUX2 = tk.StringVar()
            self.lbl_tmsi1_AUX2 = ttk.Label(self.parent.frame_sett, text='AUX 2:')
            self.lbl_tmsi1_AUX2.pack(fill='x', expand=True)
            self.lbl_tmsi1_AUX2.place(x=10, y=360)
            self.t_tmsi1_AUX2 = tk.Entry(self.parent.frame_sett, textvariable=self.tmsi1_AUX2)
            self.t_tmsi1_AUX2.insert(0, 'N/A')
            self.t_tmsi1_AUX2.pack(fill='x', expand=True)
            self.t_tmsi1_AUX2.focus()
            self.t_tmsi1_AUX2.place(x=60, y=360, width = 150)

            self.tmsi1_AUX3 = tk.StringVar()
            self.lbl_tmsi1_AUX3 = ttk.Label(self.parent.frame_sett, text='AUX 3:')
            self.lbl_tmsi1_AUX3.pack(fill='x', expand=True)
            self.lbl_tmsi1_AUX3.place(x=10, y=390)
            self.t_tmsi1_AUX3 = tk.Entry(self.parent.frame_sett, textvariable=self.tmsi1_AUX3)
            self.t_tmsi1_AUX3.insert(0, 'N/A')
            self.t_tmsi1_AUX3.pack(fill='x', expand=True)
            self.t_tmsi1_AUX3.focus()
            self.t_tmsi1_AUX3.place(x=60, y=390, width = 150)

        self.dump_path = tk.StringVar()
        self.lbl_dump_path = ttk.Label(self.parent.frame_sett, text='Dump Path:')
        self.lbl_dump_path.pack(fill='x', expand=True)
        self.lbl_dump_path.place(x=10, y=430)
        self.t_dump_path = tk.Entry(self.parent.frame_sett, textvariable=self.dump_path)
        self.t_dump_path.insert(0, os.path.join("data",self.part_id.get(),self.today))
        self.t_dump_path.pack(fill='x', expand=True)
        self.t_dump_path.focus()
        self.t_dump_path.place(x=200, y=430, width = 200)

        

        self.check_dir_button = tk.Button(self.parent.frame_sett, text='PUSH DIR', bg ='yellow')
        self.check_dir_button['command'] = self.check_dir
        self.check_dir_button.pack()
        self.check_dir_button.place(x=250, y=600)

    def update_all(self):
        self.dump_path.set(os.path.join("data",self.part_id.get(),self.expdate.get()))

    def check_dir(self):
        dump_name = self.dump_path.get()
        if not os.path.isdir(dump_name):
            print("Dir not found, making it")
            # os.makedirs(dump_name)
            os.makedirs(dump_name+'/MEPs')
            os.makedirs(dump_name+'/thresholding')
            os.makedirs(dump_name+'/MVC')
        
        keys = list(self.tmsi.keys())
        if len(self.tmsi)>1:
            muscle_map = {
                keys[0]:{
                "GRID": self.tmsi1_grid.get(),
                "BIP1-1": self.tmsi1_BIP1.get(),
                "BIP1-2": self.tmsi1_BIP2.get(),
                "BIP2-1": self.tmsi1_BIP3.get(),
                "BIP2-2": self.tmsi1_BIP4.get(),
                "AUX1": self.tmsi1_AUX1.get(),
                "AUX2": self.tmsi1_AUX2.get(),
                "AUX3": self.tmsi1_AUX3.get(),
                },

                keys[1]:{
                "GRID": self.tmsi2_grid.get(),
                "BIP1-1": self.tmsi2_BIP1.get(),
                "BIP1-2": self.tmsi2_BIP2.get(),
                "BIP2-1": self.tmsi2_BIP3.get(),
                "BIP2-2": self.tmsi2_BIP4.get(),
                "AUX1": self.tmsi2_AUX1.get(),
                "AUX2": self.tmsi2_AUX2.get(),
                "AUX3": self.tmsi2_AUX3.get(),
                },
                }
        else:
            
            muscle_map = {
                keys[0]:{
                "GRID": self.tmsi1_grid.get(),
                "BIP1-1": self.tmsi1_BIP1.get(),
                "BIP1-2": self.tmsi1_BIP2.get(),
                "BIP2-1": self.tmsi1_BIP3.get(),
                "BIP2-2": self.tmsi1_BIP4.get(),
                "AUX1": self.tmsi1_AUX1.get(),
                "AUX2": self.tmsi1_AUX2.get(),
                "AUX3": self.tmsi1_AUX3.get(),
                },}
        with open(dump_name+"/musclemap.json", "w") as outfile: 
            json.dump(muscle_map, outfile, indent = 4) 
        self.parent.dump_path = dump_name
        
        with open(self.cont_type_path.get(), 'r') as f:
            contraction_types = yaml.safe_load(f)
            
        with open(self.stim_type_path.get(), 'r') as f:
            stimulation_types = yaml.safe_load(f)
            
        with open(self.trl_type_path.get(), 'r') as f:
            trial_types = yaml.safe_load(f)

        with open(self.thScreen_type_path.get(), 'r') as f:
            thresh_types = yaml.safe_load(f)

        composed_th = compose_trials(contraction_types, stimulation_types, thresh_types, shuffle_seed = int(self.today))
        with open(dump_name+"/thresholding.json", "w") as outfile: 
            json.dump(composed_th, outfile, indent = 4) 

        composed_trl = compose_trials(contraction_types, stimulation_types, trial_types, shuffle_seed = int(self.today))
        with open(dump_name+"/params.json", "w") as outfile: 
            json.dump(composed_trl, outfile, indent = 4) 


        self.parent.notebook.forget(self.parent.frame_exp)
        self.parent.update()

        self.parent.frame_exp = ttk.Frame(self.parent.notebook, width=2000, height=1000)
        self.parent.frame_exp.pack(fill='both', expand=True)
        self.nb3 = render_emg.APP(self.parent, self.parent.tmsi_dev, self.parent.dump_path)
        self.parent.notebook.add(self.parent.frame_exp, text='Experiment controls')
        self.parent.update()

        showinfo(title='Information', message='Updated directory info')
        self.check_dir_button.config(bg = 'green')


def main():
    tk_trial = APP([],{"FLX":[],"EXT":[]})
    tk_trial.mainloop()
    return None

if __name__ == "__main__":
    main()