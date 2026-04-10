# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:39:44 2026

@author: Francois
"""
import torch
import torchaudio
import torchaudio.functional as F_audio
import numpy as np
import matplotlib.pyplot as plt

def synchro_signals(x, y, dmax_ms=80.0, tau_ms=15.0, sf=8000.0):
    # Assurer que les tenseurs sont en 1D
    x = x.squeeze()
    y = y.squeeze()
    device = x.device
    
    # search range
    P = int(sf * dmax_ms * 1.0e-03)
        
    # forgetting factor
    alpha = np.exp(-1.0 / (sf * tau_ms * 1.0e-03))

    # Envelop filter coefficients (PyTorch lfilter attend des tenseurs)
    # b = [1 - alpha], a = [1, -alpha]
    b = torch.tensor([1.0 - alpha, 0.0], device=device, dtype=torch.float32)
    a = torch.tensor([1.0, -alpha], device=device, dtype=torch.float32)
    
    # Envelop extraction
    # Note: F_audio.lfilter traite les signaux de gauche à droite
    def get_envelope(sig):
        # On inverse, on filtre, on ré-inverse, on re-filtre (Zero-phase filtering)
        rev_abs = torch.flip(torch.abs(sig), dims=[-1])
        et = F_audio.lfilter(rev_abs, a, b)
        ex = F_audio.lfilter(torch.flip(et, dims=[-1]), a, b)
        return ex

    ex = get_envelope(x)
    ey = get_envelope(y)
    
    # zero-mean centering
    ex = ex - torch.mean(ex)
    ey = ey - torch.mean(ey)
    
    # set common length
    L = min(len(ex), len(ey))
    
    delay = 0
    rn_max = -1.1
    
    # Fenêtres de calcul (pour éviter de recalculer rxx à chaque itération)
    ex_win = ex[P : L-P]
    rxx = torch.sum(ex_win ** 2)

    # Optimisation : On peut faire varier k
    for k in range(-P, P + 1):
        ey_win = ey[P + k : L - P + k]
        
        ryy = torch.sum(ey_win ** 2)
        rxy = torch.sum(ex_win * ey_win)
        
        # Ajout d'un petit epsilon pour éviter division par zéro
        rn = rxy / (torch.sqrt(rxx * ryy) + 1e-9)
        
        if rn > rn_max:
            rn_max = rn.item()
            delay = k
            
    # Output synchronised signals
    if delay == 0:
        xs, ys = x, y
    elif delay > 0:
        xs = x[:-delay]
        ys = y[delay:]
    else:
        xs = x[-delay:]
        ys = y[:delay]
    
    return delay, rn_max, xs, ys, ex, ey


# read signals
x, sampleRate = torchaudio.load("./data/x.wav")
y, sampleRate = torchaudio.load("./data/y.wav")

# synchronization process
delay, rn_max, xs, ys, ex, ey = synchro_signals(x, y, dmax_ms = 80.0, tau_ms=15.0, sf = sampleRate)

#♦ display
offset = 15500
N = 1000

# --- Display Enveloppes ---
plt.figure(figsize=(14,8))
plt.subplot(2,1,1)
# Utilisation de .numpy() pour matplotlib
plt.plot(ex[offset:offset+N].numpy() - torch.mean(ex).item(), 'b')
plt.plot(ey[offset:offset+N].numpy() - torch.mean(ey).item(), 'r')
plt.grid(True)
plt.title('Envelopes before synchronization')

plt.subplot(2,1,2)
# Ici on devrait normalement recalculer l'enveloppe de xs et ys 
# ou décaler ey avec le delay trouvé
plt.plot(ex[offset:offset+N].numpy() - torch.mean(ex).item(), 'b')
# On applique le décalage sur l'affichage
plt.plot(ey[offset+delay : offset+N+delay].numpy() - torch.mean(ey).item(), 'r')
plt.title(f'Envelopes after synchronization (delay: {delay})')
plt.grid(True)
plt.show()

# --- Display Signaux Bruts ---
plt.figure(figsize=(14,8))
plt.subplot(2,1,1)
plt.plot(x[offset:offset+N].numpy(), 'b')
plt.plot(y[offset:offset+N].numpy(), 'r')
plt.grid(True)
plt.title('Signals before synchronization')

plt.subplot(2,1,2)
# On utilise directement xs et ys renvoyés par la fonction
plt.plot(xs[offset:offset+N].numpy(), 'b')
plt.plot(ys[offset:offset+N].numpy(), 'r')
plt.grid(True)
plt.title('Signals after synchronization')
plt.show()