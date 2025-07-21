from trap_core import * # the backend w/ RNN & python logic 
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class MLBeatGenerator:
    def __init__(self, root):
        self.root = root
        self.setup_dark_theme()
        self.root.title("ML Trap Beat Generator")
        self.root.geometry("650x650") 
         
        self.last_generated_beat = None
        
        # GUI variables
        self.tempo = tk.IntVar(value=100)
        self.melody_speed = tk.DoubleVar(value=1.0)
        self.instruments = {name: tk.BooleanVar(value=True) for name in ["KICK", "SNARE", "CLAP", "HAT", "808"]}
        self.eq_settings = {band: tk.DoubleVar(value=1.0) for band in ["low", "mid", "high"]}

        self.setup_ui()

    def setup_dark_theme(self):
        style = ttk.Style()
        style.theme_use('clam')
        bg, fg, accent = "#121212", "#FFFFFF", "#BB86FC"
        style.configure('.', background=bg, foreground=fg)
        style.configure('TFrame', background=bg)
        style.configure('TLabel', background=bg, foreground=fg)
        style.configure('TButton', background="#1E1E1E", foreground=fg)
        style.configure('Accent.TButton', background=accent, foreground="#000000")
        style.configure('TCheckbutton', background=bg, foreground=fg)
        self.root.configure(bg=bg)

    def setup_ui(self):
        main = ttk.Frame(self.root, padding=15)
        main.pack(fill=tk.BOTH, expand=True)

        # Tempo controls
        ttk.Label(main, text="BPM Control").pack()
        self.bpm_label = ttk.Label(main, text=f"BPM: {self.tempo.get()}")
        self.bpm_label.pack()
        ttk.Scale(main, from_=20, to=200, variable=self.tempo,
                command=lambda v: self.update_display('bpm')).pack(fill=tk.X)

        # Melody controls
        ttk.Label(main, text="Melody Speed").pack()
        self.speed_label = ttk.Label(main, text=f"Melody Speed: {self.melody_speed.get():.1f}x")
        self.speed_label.pack()
        ttk.Scale(main, from_=0.2, to=2.0, variable=self.melody_speed,
                command=lambda v: self.update_display('speed')).pack(fill=tk.X)

        # Instruments
        inst_frame = ttk.LabelFrame(main, text="Instruments", padding=10)
        inst_frame.pack(fill=tk.X, pady=10)
        for name in self.instruments:
            ttk.Checkbutton(inst_frame, text=name, variable=self.instruments[name]).pack(side=tk.LEFT, padx=5)

        # EQ
        eq_frame = ttk.LabelFrame(main, text="EQ", padding=10)
        eq_frame.pack(fill=tk.X)
        for band in ["low", "mid", "high"]:
            ttk.Label(eq_frame, text=band.capitalize()).pack(side=tk.LEFT)
            ttk.Scale(eq_frame, from_=0.5, to=2.0, variable=self.eq_settings[band]).pack(side=tk.LEFT, expand=True)

        # Controls
        control_frame = ttk.Frame(main)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Buttons
        ttk.Button(control_frame, text="Generate", style='Accent.TButton', 
                 command=self.generate_beat).pack(side=tk.LEFT, expand=True, padx=2)
        
        ttk.Button(control_frame, text="Export", 
                 command=self.export_beat).pack(side=tk.LEFT, expand=True, padx=2)

        self.status = ttk.Label(main, text="Ready", relief=tk.SUNKEN)
        self.status.pack(fill=tk.X, pady=(5,0))

    def update_display(self, control):
        if control == 'bpm':
            self.bpm_label.config(text=f"BPM: {self.tempo.get()}")
        elif control == 'speed':
            self.speed_label.config(text=f"Melody Speed: {self.melody_speed.get():.1f}x")

    def generate_beat(self):
        self.status.config(text="Generating...")
        self.root.update()
        
        try:
            # Update global tempo variables
            global TEMPO_BPM, STEP_MS
            TEMPO_BPM = self.tempo.get()
            STEP_MS = int(60000 / TEMPO_BPM / STEPS_PER_BEAT)
            
            speed = self.melody_speed.get()
            selected = {k: v.get() for k, v in self.instruments.items()}

            patterns = create_rhythmic_patterns()
            for k in list(patterns.keys()):
                if not selected.get(k, True):
                    patterns[k] = [0] * TOTAL_STEPS

            beat = AudioSegment.silent(duration=TOTAL_STEPS * STEP_MS)
            if "MELODY" in loaded_samples:
                m = loaded_samples["MELODY"]
                m = m._spawn(m.raw_data, overrides={'frame_rate': int(m.frame_rate * speed)}).set_frame_rate(m.frame_rate)
                beat = beat.overlay(loop_melody_to_length(m, len(beat)))

            for step in range(TOTAL_STEPS):
                t = step * STEP_MS
                for name in patterns:
                    if patterns[name][step] and selected.get(name, True):
                        gain = -6 if name in ["808", "CLAP"] else (-3 if name == "808" else 0)
                        sample = loaded_samples[name]
                        if name == "808":
                            sample = sample.set_frame_rate(beat.frame_rate)
                        beat = beat.overlay(sample + gain, position=t)

            # Apply effects
            beat = beat.apply_gain(-3)
            beat = compress_dynamic_range(beat, threshold=-20, ratio=2.0)
            
            # Apply EQ settings with bounds checking
            low_eq = max(0.8, min(1.5, self.eq_settings['low'].get()))
            high_eq = max(0.8, min(1.5, self.eq_settings['high'].get()))
            
            beat = beat.high_pass_filter(int(30 * low_eq))
            beat = beat.low_pass_filter(int(15000 * high_eq))
            beat = beat.fade_in(100).fade_out(200)
            
            self.last_generated_beat = beat
            self.status.config(text="Ready - Beat Generated")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status.config(text="Error")

    def export_beat(self):
        if not self.last_generated_beat:
            messagebox.showwarning("Warning", "Generate a beat first!")
            return
            
        path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV", "*.wav"), ("MP3", "*.mp3")]
        )
        
        if path:
            try:
                fmt = path.split('.')[-1]
                self.last_generated_beat.export(path, format=fmt, bitrate="256k")
                messagebox.showinfo("Saved", f"Beat saved to {path}")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    MLBeatGenerator(root)
    root.mainloop()