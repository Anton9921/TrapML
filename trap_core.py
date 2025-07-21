import random
import torch
import torch.nn as nn
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range

# --- 6 Audio Layer Samples ---
samples = {
    "KICK": "samples/trap-basic-kick.wav",
    "SNARE": "samples/phonk-snare-lo-fi.wav",
    "CLAP": "samples/big-clap_D_minor.wav",
    "HAT": "samples/short-hi-hat_1bpm_C_major.wav",
    "808": "samples/rio - freedom 808.wav",
    "MELODY": "samples/playboi-carti-type-spacey-melody_150bpm_D_minor.wav",
}

# Initialize with default values
TEMPO_BPM = 100
BEATS_PER_BAR = 4
STEPS_PER_BEAT = 4
STEPS_PER_BAR = BEATS_PER_BAR * STEPS_PER_BEAT
BARS = 16

# Update tempo-related variables when BPM changes
def update_tempo_settings(new_bpm):
    global TEMPO_BPM, STEP_MS, TOTAL_STEPS
    TEMPO_BPM = new_bpm
    STEP_MS = int(60000 / TEMPO_BPM / STEPS_PER_BEAT)
    TOTAL_STEPS = STEPS_PER_BAR * BARS
update_tempo_settings(TEMPO_BPM)

# --- Load & EQ Samples ---
def apply_eq(sample: AudioSegment, eq_type: str):
    if eq_type == "KICK":
        return sample.low_pass_filter(120).high_pass_filter(40)
    elif eq_type == "SNARE":
        return sample.high_pass_filter(150).low_pass_filter(6000)
    elif eq_type == "CLAP":
        return sample.high_pass_filter(300)
    elif eq_type == "HAT":
        return sample.high_pass_filter(5000)
    elif eq_type == "808":
        return sample.low_pass_filter(150)
    return sample

loaded_samples = {}
for name, path in samples.items():
    try:
        sample = AudioSegment.from_wav(path).set_frame_rate(44100)
        sample = sample.apply_gain(-sample.max_dBFS)
        sample = apply_eq(sample, name)
        loaded_samples[name] = sample
    except FileNotFoundError:
        print(f"Warning: Missing sample file: {path}")
    except Exception as e:
        print(f"Error loading sample {name}: {str(e)}")

# --- RNN for Pattern Generation ---
class PatternRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1):
        super(PatternRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return torch.sigmoid(out)

def train_rnn_on_pattern(base_pattern, epochs=200):
    model = PatternRNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    input_seq = torch.tensor(base_pattern[:-1], dtype=torch.float32).view(1, -1, 1)
    target_seq = torch.tensor(base_pattern[1:], dtype=torch.float32).view(1, -1, 1)

    for _ in range(epochs):
        model.train()
        output = model(input_seq)
        loss = loss_fn(output, target_seq)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

# --- Pattern Logic ---
def create_rhythmic_patterns():
    base_patterns = {
        "KICK":  [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        "SNARE": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "CLAP":  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "HAT":   [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        "808":   [1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0],
    }
    
    patterns = {}
    epochs_config = {
        "KICK": 300,
        "SNARE": 250,
        "CLAP": 200,
        "HAT": 200,
        "808": 350  # More epochs for 808 for smoother basslines
    }

    # Train RNN for each instrument
    for name, base_pattern in base_patterns.items():
        model = train_rnn_on_pattern(base_pattern, epochs=epochs_config.get(name, 200))
        
        with torch.no_grad():
            # Create initial sequence from base pattern
            x = torch.tensor(base_pattern, dtype=torch.float32).view(1, -1, 1)
            
            # Generate multiple variations and blend them
            all_outputs = []
            for _ in range(3):  # Generate 3 variations
                rnn_out = model(x).squeeze().numpy()
                all_outputs.append(rnn_out)
            
            # Average the predictions for more stable results
            avg_output = sum(all_outputs) / len(all_outputs)
            generated_pattern = [1 if p > 0.5 else 0 for p in avg_output]
            
            # Repeat the pattern to full length
            patterns[name] = (generated_pattern * (TOTAL_STEPS // len(generated_pattern)))[:TOTAL_STEPS]

    return patterns

# --- Melody Looping ---
def loop_melody_to_length(melody, target_ms):
    looped = melody * (target_ms // len(melody) + 1)
    return looped[:target_ms].fade_in(100).fade_out(200)

# --- Slowing Down Melody ---
def slow_down_audio(audio: AudioSegment, factor: float) -> AudioSegment:
    if factor <= 0:
        raise ValueError("Speed factor must be positive")
    new_frame_rate = int(audio.frame_rate * factor)
    slowed = audio._spawn(audio.raw_data, overrides={'frame_rate': new_frame_rate})
    return slowed.set_frame_rate(audio.frame_rate)

# --- Beat Generation ---
def generate_beat_audio(selected_instruments=None, melody_speed=1.0):
    if selected_instruments is None:
        selected_instruments = {name: True for name in loaded_samples if name != "MELODY"}
    
    patterns = create_rhythmic_patterns()
    master = AudioSegment.silent(duration=TOTAL_STEPS * STEP_MS)
    
    # Add looped melody from sample folder
    if "MELODY" in loaded_samples and selected_instruments.get("MELODY", True):
        slowed_melody = slow_down_audio(loaded_samples["MELODY"], factor=melody_speed)
        melody = loop_melody_to_length(slowed_melody, TOTAL_STEPS * STEP_MS)
        master = master.overlay(melody, position=0)

    melody = melody.apply_gain(+10.0)  
    master = master.overlay(melody, position=0)

    # Generate track
    for step in range(TOTAL_STEPS):
        swing = random.randint(-8, 8) if step % 2 else 0  # human swing feel on off-beats
        t = step * STEP_MS + swing

        for name in patterns:
            if name not in selected_instruments or not selected_instruments[name]:
                continue
                
            if patterns[name][step]:
                sample = loaded_samples[name]
                gain = random.uniform(-10, -6) if name == "808" else random.uniform(-2.5, 0) if name == "HAT" else 0
                if name == "808":
                    sample = sample.set_frame_rate(master.frame_rate)
                master = master.overlay(sample + gain, position=t)
    
    return master

# --- Audio Processing ---
def process_audio(beat, eq_settings=None):
    if eq_settings is None:
        eq_settings = {'low': 1.0, 'mid': 1.0, 'high': 1.0}
    # Lower gain before compression
    beat = beat.apply_gain(-3)
    # Compress dynamics softly
    beat = compress_dynamic_range(beat, threshold=-20, ratio=2.0)

    # Apply EQ
    low_cutoff = int(30 * max(0.8, min(1.5, eq_settings['low'])))
    high_cutoff = int(15000 * max(0.8, min(1.5, eq_settings['high'])))
    beat = beat.high_pass_filter(low_cutoff)
    beat = beat.low_pass_filter(high_cutoff)
    # Soft limiter to avoid clipping
    peak = beat.max_dBFS
    if peak > -1.0:
        beat = beat.apply_gain(-(peak + 1.0))
    # Add fades
    beat = beat.fade_in(100).fade_out(200)
    
    return beat