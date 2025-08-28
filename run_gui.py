
import tkinter as tk
from tkinter import ttk
import os, time
from tmsi_dual_interface import TMSi_gui
from vis_feedback import render_emg,init_settings

class APP_main(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title('Main interface')
        self.geometry('2000x1000')
        self.columnconfigure(0, weight=1)

        self.create_widgets()
    
    def create_widgets(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0)

        self.frame_tmsi = ttk.Frame(self.notebook, width=2000, height=1000)
        self.frame_tmsi.pack(fill='both', expand=True)
        self.nb1 = TMSi_gui.TMSi_GUI(self)
        self.notebook.add(self.frame_tmsi, text='TMSI init')

        self.tmsi_dev = {"FLX":[],"EXT":[]}

        self.frame_sett = ttk.Frame(self.notebook, width=2000, height=1000)
        self.frame_sett.pack(fill='both', expand=True)
        self.nb2 = init_settings.APP(self,self.tmsi_dev)
        self.notebook.add(self.frame_sett, text='Settings init')

        self.dump_path = 'data/PX/'+ time.strftime("%Y%m%d")

        self.frame_exp = ttk.Frame(self.notebook, width=2000, height=1000)
        self.frame_exp.pack(fill='both', expand=True)
        self.nb3 = render_emg.APP(self, self.tmsi_dev, self.dump_path)
        self.notebook.add(self.frame_exp, text='Experiment controls')

def main():
    tk_trial = APP_main()
    tk_trial.mainloop()
    return None

if __name__ == "__main__":
    main()