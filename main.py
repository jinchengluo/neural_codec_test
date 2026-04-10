from os import listdir

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.signal import lfilter
from pathlib import Path
import torchaudio
import torchaudio.functional as F

def genererate_sound(duree=2.0, fs=24000, f0=110):

    t = np.linspace(0, duree, int(fs * duree), endpoint=False)
    
    source = 2 * ((t * f0) % 1.0) - 1
    
    formants = [730, 1090, 2440]
    largeurs_bande = [80, 90, 130]
    
    signal_final = np.zeros_like(source)
    
    for freq, lb in zip(formants, largeurs_bande):
        r = np.exp(-np.pi * lb / fs)
        theta = 2 * np.pi * freq / fs
        
        a = [1, -2 * r * np.cos(theta), r**2]
        b = [1 - r**2]
        
        signal_filtre = lfilter(b, a, source)
        signal_final += signal_filtre
    
    signal_final = signal_final / np.max(np.abs(signal_final))
    
    audio_16bit = (signal_final * 32767).astype(np.int16)
    return audio_16bit

# frequence_echantillonnage = 24000
# son_a = generate_sound(duree=2.0, fs=frequence_echantillonnage, f0=220)
# wavfile.write('a.wav', frequence_echantillonnage, son_a)

def rename_folder_files(repertoire_cible):
    dossier = Path(repertoire_cible)
    
    fichiers = sorted([f for f in dossier.iterdir() if f.is_file()])
    
    for index, fichier in enumerate(fichiers, start=1):
        extension = fichier.suffix
        
        nouveau_nom = f"{index:05d}{extension}"
        nouveau_chemin = fichier.with_name(nouveau_nom)
        
        fichier.rename(nouveau_chemin)
        print(f"Renamed : {fichier.name} -> {nouveau_nom}")

# rename_folder_files("./result/hhcodec_24")

def upsample(source_dir, target_dir, target_fs=16000):
    files = sorted(listdir(source_dir))
    
    for file in files:
        file_path = source_dir + "/" + file
        wav, sr = torchaudio.load(file_path)
        if sr != target_fs:
            wav = F.resample(wav, orig_freq=sr, new_freq=target_fs)
            torchaudio.save(target_dir + "/" + file, wav, target_fs)
        else:
            torchaudio.save(target_dir + "/" + file, wav, sr)

# upsample("./data/samples", "./data/samples_24k", target_fs=24000)