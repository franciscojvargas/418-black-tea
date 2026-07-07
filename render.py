"""
418 - black tea — hard groove OSCURO ATMOSFÉRICO (ambiente de club, cueva).
100 % sintetizado (sin samples), 145 BPM, 200 barras (~5:33) WAV estéreo 24-bit.

Identidad: GROOVE TRIBAL de toms sintéticos (el motor hardgroove) sobre kick
de cueva afinado a G1 + SUB monofónico de UNA nota (G1) en offbeats, rodando
con el swing; el b2 (Ab1) sólo como sombra deslizada al fin de frase. Encima:
stab dub Gm hueco y lavado (pre-ecos reversos, delay punteado), GONG metálico
cavernoso cada 16 compases, BREATH fantasma (exhalación formántica) que llama
desde el fondo, rim seco de madera, y una atmósfera continua de drone G0/G1
con wow de cinta + pad Gm que respira + cama de "lluvia en nave" band-pass.
Carácter por procesado y espacio, no por notas: phaser automatizado (pad y
ride), LP-wobble (toms, ride), HP respirando (rim), band-pass móvil (textura),
chorus decorrelado L/R (stab, breath). NATURAL, no máquina: round-robin de
tomas en toms y rim, swing que deriva ±0.012 en ciclos de 24 compases,
humanize ancho de timing/velocidad/pan y ghosts de tom sueltos. Evolución
por automatización CONTINUA y sustracción/adición de capas, sin drops ni
bajadas abruptas.

Estructura (compases): intro 0-23 / groove 24-55 / main 56-87 /
break profundo 88-111 (kick fuera 88-99, rebuild 100-111) /
peak 112-159 / late-groove 160-175 / outro 176-199.

Uso: venv/bin/python render.py
"""
import os
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

from engine.fx import (
    compress, sidechain_compress, reverb_plate, saturate, delay_pingpong,
    eq_low_cut, eq_lp, eq_low_shelf, eq_high_shelf_biquad, eq_mid_boost,
    stereo_width, mono_low_end, limiter_brickwall, soft_clip, mod_delay,
    fade_in, fade_out
)
from engine.atmos import reverse_swell, atmos_drone, shaker, crash, noise_sweep
from engine.automation import (
    rng, humanize_pos, humanize_vel, breakpoint_env, lfo,
    moving_lowpass, moving_highpass, phaser, reverb_space
)

RND = rng(418)  # semilla fija  ->  render reproducible

# --- Tempo y rejilla ---------------------------------------------------------
SR     = 44100
BPM    = 145.0
BARS   = 200
SPB    = 16             # steps por compás (semicorcheas, 4/4)
STEP_S = (60.0 / BPM) / 4.0

TOTAL_STEPS   = BARS * SPB
TOTAL_SAMPLES = int(TOTAL_STEPS * STEP_S * SR) + 2 * SR  # +2s de cola para reverbs

SWING = 0.56           # >0.5 retrasa los steps impares = swing tribal rodante
BAR_S = SPB * STEP_S

def spos(step):
    return int(step * STEP_S * SR)

def spos_sw(step):
    base = step * STEP_S * SR
    if step % 2 == 1:
        # Swing que RESPIRA: deriva ±0.012 en ciclos de 24 compases = el
        # groove se tensa y afloja como un percusionista, nunca cuantizado.
        sw = SWING + 0.012 * np.sin(2.0 * np.pi * (step / SPB) / 24.0)
        base += (sw - 0.5) * 2.0 * STEP_S * SR
    return int(base)

def place(buf, sample, pos, gain=1.0):
    n = len(sample)
    end = min(pos + n, len(buf))
    if end <= pos:
        return
    buf[pos:end] += sample[:end - pos] * gain

def _bp_filt(sig, lo, hi, order=4):
    b, a = butter(order, [lo / (SR / 2), hi / (SR / 2)], btype='band')
    return lfilter(b, a, sig)

# --- Frecuencias (Sol menor: oscuro, subterráneo) ----------------------------
F = {'G1': 49.00, 'Ab1': 51.91, 'G2': 98.00, 'Bb2': 116.54, 'D3': 146.83,
     'G3': 196.00, 'Bb3': 233.08, 'D4': 293.66, 'Ab4': 415.30}

# --- Síntesis (100 % numpy, sin samples) ------------------------------------
print("Sintetizando voces... ", end='', flush=True)

def synth_kick(dur=0.55, f_start=185.0, f_end=49.0, drop=0.022,
               knock_amt=0.85, drive=2.5):
    # Kick de CUEVA afinado a G1: redondo pero con la cola CONTENIDA — el
    # "roll" largo lo ponen rumble y sub, no el kick (un kick sub-pesado
    # apila graves y aplasta el limiter). Knock discreto ~200 Hz (piedra).
    n = int(dur * SR); t = np.arange(n) / SR
    f = f_end + (f_start - f_end) * np.exp(-t / drop)
    body  = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.14)
    knock = _bp_filt(RND.standard_normal(n), 130, 300, order=2)
    knock *= np.exp(-t / 0.013) * knock_amt
    out = np.tanh((body * 1.4 + knock) * drive) / np.tanh(drive)
    a = int(0.0008 * SR); out[:a] *= np.linspace(0, 1, a)
    nf = int(0.004 * SR); out[-nf:] *= np.linspace(1, 0, nf)
    return out * (0.97 / (np.max(np.abs(out)) + 1e-12))

def short_tail(sample, keep_ms, fade_ms=6.0):
    n = int(keep_ms * 0.001 * SR)
    if n >= len(sample):
        return sample.copy()
    out = sample[:n].copy()
    nf = min(int(fade_ms * 0.001 * SR), n)
    out[-nf:] *= np.linspace(1.0, 0.0, nf)
    return out

KICK_DN = synth_kick()
KICK_UP = short_tail(KICK_DN, keep_ms=140)

# TOMS = EL motor hardgroove: membranas sintéticas (caída de tono corta +
# slap de ruido) en tres alturas — tom grave, tom medio y "slap" de conga
# seca. Patrón tribal sincopado que rueda con el swing.
def synth_tom(f0=98.0, drop_ratio=1.7, dur=0.28, decay=0.085,
              slap=0.35, drive=1.8):
    n = int(dur * SR); t = np.arange(n) / SR
    f = f0 * (1.0 + (drop_ratio - 1.0) * np.exp(-t / 0.018))
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / decay)
    skin = _bp_filt(RND.standard_normal(n), 900, 3600, order=2)
    skin *= np.exp(-t / 0.006) * slap
    sig = np.tanh((body * 1.3 + skin) * drive)
    a = int(0.0008 * SR); sig[:a] *= np.linspace(0, 1, a)
    nf = int(0.004 * SR); sig[-nf:] *= np.linspace(1, 0, nf)
    return sig * (0.92 / (np.max(np.abs(sig)) + 1e-12))

# Round-robin: 3 tomas por tom (el ruido del slap avanza el RNG = cada toma
# es única, como golpes reales de mano) — el oído nunca oye el mismo sample.
TOM_LO  = [synth_tom(f0=F['G2'], drop_ratio=1.7, dur=0.30, decay=0.095, slap=0.30)
           for _ in range(3)]
TOM_MID = [synth_tom(f0=F['D3'], drop_ratio=1.6, dur=0.24, decay=0.070, slap=0.40)
           for _ in range(3)]
TOM_SLP = [synth_tom(f0=F['G3'], drop_ratio=1.9, dur=0.14, decay=0.030, slap=0.95,
                     drive=2.4) for _ in range(3)]   # conga-slap: casi todo ataque

# Patrones tribales (step, tom, gain, pan). El groove pregunta (A) y responde
# (B) cada 8 compases; C = variación de acentos que entra en el peak cada 4ª
# frase para que el motor no se memorice. Fill de subida al fin de frase de 16.
TOM_PAT_A = [
    (3, 'lo', 0.85, -0.30), (6, 'slp', 0.42, 0.30),
    (10, 'lo', 0.80, -0.25), (13, 'mid', 0.62, 0.35),
]
TOM_PAT_B = [
    (3, 'lo', 0.85, -0.30), (5, 'mid', 0.48, 0.30), (10, 'lo', 0.80, -0.25),
    (12, 'slp', 0.40, -0.40), (14, 'mid', 0.60, 0.35),
]
TOM_PAT_C = [
    (1, 'slp', 0.38, 0.35), (3, 'lo', 0.85, -0.30), (7, 'mid', 0.55, 0.30),
    (10, 'lo', 0.80, -0.25), (13, 'slp', 0.42, -0.35), (14, 'mid', 0.62, 0.35),
]
TOM_PAT_BREAK = [(3, 'lo', 0.60, -0.2), (10, 'mid', 0.45, 0.25)]
TOMS = {'lo': TOM_LO, 'mid': TOM_MID, 'slp': TOM_SLP}

def tom_take(nm):
    return TOMS[nm][int(RND.integers(0, 3))]

# RIM = rimshot de madera oscura: impulso + resonancia corta band-pass
# ~1.1 kHz + pop grave. Contrapunto seco y cercano del groove de toms.
def synth_rim(dur=0.08):
    n = int(dur * SR); t = np.arange(n) / SR
    click = RND.standard_normal(n) * np.exp(-t / 0.0011)
    ring  = _bp_filt(RND.standard_normal(n), 800, 1600, order=2) * np.exp(-t / 0.012)
    pop   = np.sin(2 * np.pi * 240 * t) * np.exp(-t / 0.011) * 0.6
    sig = np.tanh((click * 0.9 + ring * 1.6 + pop) * 2.2)
    nf = int(0.003 * SR); sig[-nf:] *= np.linspace(1, 0, nf)
    return sig * (0.9 / (np.max(np.abs(sig)) + 1e-12))

RIM = [synth_rim() for _ in range(2)]   # 2 tomas round-robin
RIM_PAT_A = [(2, 0.50), (9, 0.70)]
RIM_PAT_B = [(2, 0.50), (9, 0.70), (15, 0.38)]

def synth_hat(dur=0.16, hp=7200.0, ring=2.0):
    # Hat de carbón: ruido HP + ring-mod inarmónica (5.6/8.1 kHz) = siseo
    # oscuro pero PRESENTE. Oscuro ≠ apagado: la mordida vive en 2-6 kHz y
    # las hats tienen que cortar; lo oscuro lo ponen armonía y espacio.
    n = int(dur * SR); t = np.arange(n) / SR
    noise = eq_low_cut(RND.standard_normal(n), freq=hp)
    rm = noise * (0.45 + 0.32 * np.sin(2 * np.pi * 5647 * t)
                  + 0.23 * np.sin(2 * np.pi * 8117 * t))
    env = np.exp(-t / (0.010 * ring))
    sig = (noise * 0.5 + rm * 0.95) * env
    return sig * (0.85 / (np.max(np.abs(sig)) + 1e-12))

HAT_C = synth_hat(dur=0.065, ring=1.0)
HAT_O = synth_hat(dur=0.42, ring=6.5)
HAT_C_R = synth_hat(dur=0.065, ring=1.0)   # 2ª toma con ruido decorrelado
HAT_O_R = synth_hat(dur=0.42, ring=6.5)
# Canal R = blend de toma común + decorrelada: ancho sin cancelación en mono.
HAT_C_RB = HAT_C * 0.65 + HAT_C_R * 0.55
HAT_O_RB = HAT_O * 0.65 + HAT_O_R * 0.55

# RIDE = lavado de ceniza que abre el peak (no existe antes del compás 112):
# ruido oscuro + parciales inarmónicos graves = spray sucio, cero platillo hi-fi.
def synth_ride(dur=0.62):
    n = int(dur * SR); t = np.arange(n) / SR
    noise = eq_low_cut(RND.standard_normal(n), freq=3600) * np.exp(-t / 0.28)
    partials = [2683, 3907, 5741]
    metal = sum(np.sin(2 * np.pi * f * t) * 0.11 for f in partials) * np.exp(-t / 0.32)
    ping  = np.sin(2 * np.pi * 5099 * t) * np.exp(-t / 0.04) * 0.14
    sig = noise * 0.9 + metal + ping
    return sig * (0.8 / (np.max(np.abs(sig)) + 1e-12))

RIDE   = synth_ride()
RIDE_R = synth_ride()  # misma síntesis, ruido decorrelado (RND avanza)
RIDE_RB = RIDE * 0.65 + RIDE_R * 0.55

def synth_snap(dur=0.22):
    # Snap-backbeat oscuro: ráfagas band-pass 500-1900 (más grave que un
    # clap) + thump 190 Hz = palmada de cueva, con sala honda detrás.
    n = int(dur * SR); t = np.arange(n) / SR
    noise = _bp_filt(RND.standard_normal(n), 500, 1900, order=2)
    env = np.zeros(n)
    for b in (0.0, 0.010, 0.021):
        i = int(b * SR)
        if i < n: env[i:] += np.exp(-np.arange(n - i) / (0.011 * SR))
    env /= (np.max(env) + 1e-12)
    sig = noise * (env * 0.85 + np.exp(-t / 0.11) * 0.15)
    sig += np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.028) * 0.42
    return sig * (0.85 / (np.max(np.abs(sig)) + 1e-12))

SNAP = synth_snap()

SHAKER_S   = shaker(duration=0.050, tilt=1.4)
SHAKER_S_R = shaker(duration=0.050, tilt=2.2)      # semilla distinta = decorrelación L/R
SHAKER_RB  = SHAKER_S * 0.65 + SHAKER_S_R * 0.55   # blend mono-compatible para el canal R

print("OK")
print("Sintetizando armonía y atmósfera... ", end='', flush=True)

# STAB DUB HUECO Gm (G3-Bb3-D4 + Ab4 tenue = la sombra b2 arriba): squares
# redondeadas con detune estrecho -> band-pass 200-1900 = ventana dub oscura,
# sin brillo de saw. La sombra se intuye en la fricción del Ab, no se exhibe.
def synth_stab(dur=1.5):
    n = int(dur * SR); t = np.arange(n) / SR
    prng = np.random.default_rng(1913)
    sig = np.zeros(n)
    for f, g in ((F['G3'], 1.0), (F['Bb3'], 0.95), (F['D4'], 0.85),
                 (F['Ab4'], 0.32)):
        for det in (-7.0, 0.0, 6.0):
            fi = f * 2.0 ** (det / 1200.0)
            ph = 2 * np.pi * fi * t + prng.uniform(0, 2 * np.pi)
            sig += g * np.tanh(2.1 * np.sin(ph)) / 3.0
    sig = _bp_filt(sig, 200, 1900, order=2)
    env = np.exp(-t / 0.42)
    a = int(0.005 * SR); env[:a] *= np.linspace(0, 1, a)
    sig = np.tanh(sig * env * 1.4)
    nf = int(0.006 * SR); sig[-nf:] *= np.linspace(1, 0, nf)
    return sig * (0.85 / (np.max(np.abs(sig)) + 1e-12))

STAB = synth_stab()

# GONG = tam-tam cavernoso: parciales inarmónicos graves (ratios de gong real)
# con batido lento entre pares desafinados + estallido de ruido oscuro al
# ataque. Cada golpe es un EVENTO (1 por 16 compases), lejano y enorme.
def synth_gong(f0=73.4, dur=3.2):
    n = int(dur * SR); t = np.arange(n) / SR
    prng = np.random.default_rng(4181)
    sig = np.zeros(n)
    for r, g, dc in ((1.0, 1.0, 1.6), (1.48, 0.7, 1.2), (2.19, 0.55, 0.9),
                     (2.76, 0.45, 0.7), (3.63, 0.30, 0.5), (5.42, 0.18, 0.35)):
        for det in (-3.5, 3.5):
            fi = f0 * r * 2.0 ** (det / 1200.0)
            sig += g * 0.5 * np.sin(2 * np.pi * fi * t + prng.uniform(0, 2 * np.pi)) \
                   * np.exp(-t / dc)
    burst = _bp_filt(prng.standard_normal(n), 300, 2400, order=2)
    sig += burst * np.exp(-t / 0.05) * 0.5
    sig = np.tanh(sig * 1.2)
    a = int(0.002 * SR); sig[:a] *= np.linspace(0, 1, a)
    nf = int(0.05 * SR); sig[-nf:] *= np.linspace(1, 0, nf)
    return sig * (0.85 / (np.max(np.abs(sig)) + 1e-12))

GONG = synth_gong()

# BREATH = exhalación fantasma: ruido por formantes de susurro ("hhh"->"ahh")
# con swell de 2 compases y trémolo lento = presencia humana sin voz, la
# respiración de la sala. Tres colores (oscuro/medio/hueco).
def synth_breath(bars=2, color=0, seed=8018):
    dur = bars * BAR_S
    n = int(dur * SR); t = np.arange(n) / SR
    prng = np.random.default_rng(seed)
    src = prng.standard_normal(n)
    if color == 0:      # "hhh" oscuro
        sig = (_bp_filt(src, 380, 900) + _bp_filt(src, 1100, 1900) * 0.5)
    elif color == 1:    # "ahh" medio
        sig = (_bp_filt(src, 600, 1300) + _bp_filt(src, 1700, 2600) * 0.45)
    else:               # hueco (botella)
        sig = (_bp_filt(src, 260, 520) + _bp_filt(src, 2100, 2900) * 0.30)
    x = t / dur
    sig *= 0.75 + 0.25 * np.sin(2 * np.pi * 0.9 * t + prng.uniform(0, 2 * np.pi))
    sig *= np.sin(np.pi * np.clip(x, 0, 1)) ** 1.4      # swell entra/sale
    return (sig * (0.8 / (np.max(np.abs(sig)) + 1e-12))).astype(np.float64)

BREATH_D = synth_breath(bars=2, color=0, seed=8018)
BREATH_M = synth_breath(bars=2, color=1, seed=8019)
BREATH_H = synth_breath(bars=2, color=2, seed=8020)

# Sub-impacto: seno grave con caída de tono = "thud" que subraya aperturas.
def synth_boom(freq=36.0, dur=0.95):
    n = int(dur * SR); t = np.arange(n) / SR
    f = freq * (1.0 + 0.4 * np.exp(-t / 0.06))
    sig = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.30)
    a = int(0.003 * SR); sig[:a] *= np.linspace(0, 1, a)
    return np.tanh(sig * 1.1) * 0.7
BOOM = synth_boom()

# Atmósfera continua: drone de caverna G0/G1 (raíz+octava, hueco) + pad Gm.
DRONE_LOOP = atmos_drone(duration=8 * BAR_S, root=F['G1'], intervals=(0, 12),
                         cutoff=360.0)

def _make_pad_stereo(bars=8):
    # Pad Gm grave (G2-Bb2-D3): voces HUECAS (armónicos impares) con deriva
    # de afinado y spread estéreo + raíz senoidal = colchón de órgano de
    # cripta que respira. La 3ª menor (Bb) da la sombra, no el brillo.
    dur = bars * BAR_S
    n = int(dur * SR); t = np.arange(n) / SR
    prng = np.random.default_rng(6006)
    freqs = (F['G2'], F['Bb2'], F['D3'])
    L = np.zeros(n); R = np.zeros(n)
    spread = np.linspace(-12.0, 12.0, 7)   # cents por voz
    for f in freqs:
        L += 0.35 * np.sin(2 * np.pi * f * t + prng.uniform(0, 2 * np.pi))
        R += 0.35 * np.sin(2 * np.pi * f * t + prng.uniform(0, 2 * np.pi))
        for v in range(7):
            drift = prng.uniform(2.0, 5.5) * np.sin(
                2 * np.pi * prng.uniform(0.02, 0.06) * t + prng.uniform(0, 2 * np.pi))
            f_inst = f * 2.0 ** ((spread[v] + drift) / 1200.0)
            phase = 2 * np.pi * np.cumsum(f_inst) / SR + prng.uniform(0, 2 * np.pi)
            vce = sum((1.0 / k ** 1.7) * np.sin(k * phase + prng.uniform(0, 2 * np.pi))
                      for k in (1, 3, 5, 7))
            pan = v / 6.0 * 2.0 - 1.0
            L += vce * 0.6 * (1.0 - max(0.0, pan) * 0.7)
            R += vce * 0.6 * (1.0 - max(0.0, -pan) * 0.7)
    for buf in (L, R):
        buf *= 0.80 / (np.max(np.abs(buf)) + 1e-12)
    return L, R

PAD_LOOP_L, PAD_LOOP_R = _make_pad_stereo(8)

# Texturas de transición: sweeps de ruido + swells reversos (oscuros).
SWEEP_UP    = noise_sweep(duration=4 * BAR_S, f_start=260, f_end=7200, seed=181)
SWEEP_DOWN  = noise_sweep(duration=4 * BAR_S, f_start=300, f_end=6400, reverse=True, seed=183)
RSWELL      = reverse_swell(duration=2 * BAR_S, f_center=2800)
RSWELL_LONG = reverse_swell(duration=4 * BAR_S, f_center=2200)

print("OK")

# --- Buffers por capa (L/R) --------------------------------------------------
def _buf():
    # float32: mitad de RAM por buffer; el engine preserva dtype en todo el
    # pipeline y sobra precisión para 24-bit.
    return np.zeros(TOTAL_SAMPLES, dtype=np.float32)

kick_L, kick_R     = _buf(), _buf()
sub_L, sub_R       = _buf(), _buf()
tom_L, tom_R       = _buf(), _buf()
rim_L, rim_R       = _buf(), _buf()
hat_L, hat_R       = _buf(), _buf()
ride_L, ride_R     = _buf(), _buf()
snap_L, snap_R     = _buf(), _buf()
shaker_L, shaker_R = _buf(), _buf()
stab_L, stab_R     = _buf(), _buf()
gong_L, gong_R     = _buf(), _buf()
breath_L, breath_R = _buf(), _buf()
pad_L, pad_R       = _buf(), _buf()
drone_L, drone_R   = _buf(), _buf()
fx_L, fx_R         = _buf(), _buf()
impact_L, impact_R = _buf(), _buf()

# Cama de textura: "lluvia en nave" — ruido oscuro continuo (decorrelado L/R)
# que pega las capas y florece en el break; nivel y filtro se automatizan abajo.
tex_L = RND.standard_normal(TOTAL_SAMPLES, dtype=np.float32)
tex_R = RND.standard_normal(TOTAL_SAMPLES, dtype=np.float32)

def place_stereo(bufL, bufR, mono, pos, gain=1.0, pan=0.0):
    n = len(mono)
    e = min(pos + n, TOTAL_SAMPLES)
    if e <= pos:
        return
    gl = gain * (1.0 - max(0.0, pan) * 0.5)
    gr = gain * (1.0 - max(0.0, -pan) * 0.5)
    bufL[pos:e] += mono[:e - pos] * gl
    bufR[pos:e] += mono[:e - pos] * gr

def place_at_bar(bufL, bufR, mono, bar, gain=1.0, pan=0.0):
    place_stereo(bufL, bufR, mono, spos(bar * SPB), gain, pan)

def place_end_at_bar(bufL, bufR, mono, end_bar, gain=1.0):
    # Coloca un sample que TERMINA al inicio de end_bar (swells/risers).
    e = spos(end_bar * SPB); n = len(mono)
    s = max(0, e - n); seg = mono[-(e - s):]
    bufL[s:e] += seg * gain
    bufR[s:e] += seg * gain * 0.97

def tile_loop(bufL, bufR, loopL, loopR, start_bar, end_bar, gain=1.0):
    s = spos(start_bar * SPB); e = min(spos(end_bar * SPB), TOTAL_SAMPLES)
    nl = len(loopL); pos = s
    while pos < e:
        n = min(nl, e - pos)
        bufL[pos:pos + n] += loopL[:n] * gain
        bufR[pos:pos + n] += loopR[:n] * gain
        pos += n

# Atmósfera continua: pad y drone sostienen la cueva de principio a fin.
tile_loop(pad_L,   pad_R,   PAD_LOOP_L, PAD_LOOP_R, 0, 194, gain=1.0)
tile_loop(drone_L, drone_R, DRONE_LOOP, DRONE_LOOP, 0, 198, gain=1.0)
# Sub-impactos en aperturas de sección (no en el break = rompería el aire).
for _bb, _bg in ((24, 0.65), (56, 0.85), (112, 0.95)):
    place_stereo(impact_L, impact_R, BOOM, spos(_bb * SPB), gain=_bg)

# --- Mapa de secciones (hipnótico, sin drama EDM) ---------------------------
def in_groove(bar): return 24 <= bar < 56
def in_main(bar):   return 56 <= bar < 88
def in_break(bar):  return 88 <= bar < 112     # break profundo (rebuild 100-111)
def in_peak(bar):   return 112 <= bar < 160
def in_late(bar):   return 160 <= bar < 176
def in_outro(bar):  return bar >= 176

BREAK_NOKICK = frozenset(range(88, 100))       # 1ª mitad del break: sin kick
KICK_DROP_BARS = (55, 87, 159)                 # succión 1 compás al fin de sección

def beat_on(bar):
    return 8 <= bar < 192 and bar not in BREAK_NOKICK

# --- Texturas colocadas "cuando toca" (limpio = puntual, no constante) ------
place_end_at_bar(fx_L, fx_R, SWEEP_UP,    24, 0.28)
place_end_at_bar(fx_L, fx_R, RSWELL,      56, 0.26)
place_end_at_bar(fx_L, fx_R, SWEEP_DOWN,  88, 0.36)   # entra al break
place_end_at_bar(fx_L, fx_R, RSWELL_LONG, 112, 0.42)  # sale del break al peak
place_end_at_bar(fx_L, fx_R, RSWELL,      160, 0.22)
# Micro-eventos en el peak: swells suaves cada 8-16 barras = evolución continua.
place_end_at_bar(fx_L, fx_R, RSWELL,      128, 0.18)
place_end_at_bar(fx_L, fx_R, SWEEP_UP,    144, 0.16)
place_end_at_bar(fx_L, fx_R, RSWELL_LONG, 152, 0.20)
# Crashes oscuros en aperturas; en el break uno largo y sordo (deja aire).
for _cb, _cg in ((24, 0.36), (56, 0.44), (112, 0.55)):
    place_at_bar(fx_L, fx_R, crash(duration=2.2, bright=0.50), _cb, _cg)
place_at_bar(fx_L, fx_R, crash(duration=4.5, bright=0.32), 88, 0.30)

# BREATH: exhalaciones de 2 compases que terminan en los umbrales de sección
# + llamadas a distintos colores dentro del break y el peak. Siempre lejana.
for _vb, _vs, _vg in ((56, BREATH_D, 0.30), (88, BREATH_D, 0.44), (96, BREATH_M, 0.34),
                      (104, BREATH_H, 0.30), (112, BREATH_D, 0.46), (128, BREATH_M, 0.28),
                      (144, BREATH_H, 0.30), (152, BREATH_D, 0.26)):
    place_end_at_bar(breath_L, breath_R, _vs, _vb, _vg)

# GONG: un golpe cavernoso cada ~16 compases = campanada de la cueva.
for _gb, _gg, _gp in ((40, 0.55, 0.4), (72, 0.62, -0.4), (88, 0.80, 0.0),
                      (104, 0.55, 0.35), (120, 0.62, -0.35), (136, 0.58, 0.4),
                      (152, 0.60, -0.4), (176, 0.50, 0.0)):
    place_stereo(gong_L, gong_R, GONG, spos(_gb * SPB), gain=_gg, pan=_gp)

# Pre-ecos REVERSOS del stab: el acorde invertido crece hacia su propio golpe
# (succión dub). En el break antes de cada golpe; en el peak cada 8 compases.
STAB_REV = STAB[::-1].copy()
for _pb in range(88, 112, 2):
    place_end_at_bar(stab_L, stab_R, STAB_REV, _pb, 0.40)
for _pb in range(120, 160, 8):
    place_end_at_bar(stab_L, stab_R, STAB_REV, _pb, 0.30)

# Hat abierta INVERTIDA terminando en el downbeat cada 4 compases del peak =
# succión corta que marca la frase sin tocar la rejilla.
HAT_O_REV = HAT_O[::-1].copy()
for _hb in range(116, 160, 4):
    place_end_at_bar(hat_L, hat_R, HAT_O_REV, _hb, 0.30)

print(f"Secuenciando {BARS} barras... ", end='', flush=True)

# --- Secuenciación -----------------------------------------------------------
for bar in range(BARS):
    gs         = bar * SPB
    PHRASE_ALT = ((bar // 8) % 2 == 1)

    # KICK 4x4 (entra en 8, fuera en la 1ª mitad del break; succión fin de sección).
    kick_here = (beat_on(bar) and bar not in KICK_DROP_BARS)
    if kick_here:
        kgain = min(1.0, 0.55 + 0.12 * (bar - 8)) if bar < 13 else 1.0
        if in_outro(bar):
            kgain = max(0.6, 1.0 - (bar - 176) * 0.02)
        for s in (0, 4, 8, 12):
            smp = KICK_DN if s in (0, 8) else KICK_UP
            place(kick_L, smp, spos(gs + s), kgain)
            place(kick_R, smp, spos(gs + s), kgain)
        # Ghost kick: rebote antes del beat 3 (compases impares) y empujón
        # pre-downbeat al fin de frase = bounce hardgroove, alimenta el rumble.
        if bar % 2 == 1 and (in_groove(bar) or in_main(bar) or in_peak(bar)):
            place(kick_L, KICK_UP, spos_sw(gs + 7), 0.28 * kgain)
            place(kick_R, KICK_UP, spos_sw(gs + 7), 0.28 * kgain)
        if bar % 4 == 3 and (in_main(bar) or in_peak(bar)):
            place(kick_L, KICK_UP, spos_sw(gs + 15), 0.22 * kgain)
            place(kick_R, KICK_UP, spos_sw(gs + 15), 0.22 * kgain)

    # TOMS: el groove tribal (motor del tema). Round-robin de tomas, timing
    # y velocidad anchos, pan con jitter = manos, no máquina. Patrón C cada
    # 4ª frase del peak; ghosts sueltos en main/peak; fill de subida al fin
    # de cada frase de 16 en main/peak (16ths alternando lo/mid, crece).
    if (32 <= bar < 88) or (100 <= bar < 176):
        if in_break(bar) and bar < 100:
            pat = TOM_PAT_BREAK
        elif in_peak(bar) and (bar // 8) % 4 == 3:
            pat = TOM_PAT_C
        else:
            pat = TOM_PAT_B if PHRASE_ALT else TOM_PAT_A
        for st, nm, g, pn in pat:
            pt = humanize_pos(spos_sw(gs + st), 4.5, RND)
            place_stereo(tom_L, tom_R, tom_take(nm), pt,
                         gain=humanize_vel(g, 0.12, RND),
                         pan=pn + RND.uniform(-0.06, 0.06))
        if (in_main(bar) or in_peak(bar)) and RND.random() < 0.30:
            gst = 8 if RND.random() < 0.5 else 15
            place_stereo(tom_L, tom_R, tom_take('lo' if gst == 8 else 'mid'),
                         humanize_pos(spos_sw(gs + gst), 5.0, RND),
                         gain=humanize_vel(0.18, 0.05, RND),
                         pan=RND.uniform(-0.3, 0.3))
        if bar % 16 == 15 and (in_main(bar) or in_peak(bar)):
            for k in range(6):
                st = 10 + k
                nm = 'mid' if k % 2 else 'lo'
                place_stereo(tom_L, tom_R, tom_take(nm),
                             humanize_pos(spos_sw(gs + st), 3.5, RND),
                             gain=humanize_vel(0.30 + 0.07 * k, 0.06, RND),
                             pan=(0.3 if k % 2 else -0.3))

    # RIM: contrapunto seco de madera, escaso (2-3 golpes por compás).
    # Vuelve en el late-groove (160-172) = el descenso trae un elemento de
    # vuelta en vez de solo restar.
    if (40 <= bar < 88) or (112 <= bar < 172):
        pat = RIM_PAT_B if PHRASE_ALT else RIM_PAT_A
        for st, g in pat:
            pd  = humanize_pos(spos_sw(gs + st), 4.5, RND)
            pan = -0.35 if st % 4 == 1 else 0.25
            place_stereo(rim_L, rim_R, RIM[int(RND.integers(0, 2))], pd,
                         gain=humanize_vel(g, 0.11, RND),
                         pan=pan + RND.uniform(-0.05, 0.05))

    # HI-HATS: cerrada en offbeats (2,6,10,14) = motor; abierta 1 vez por
    # compás (step 14, pre-downbeat); ghosts de 16ths sólo en el peak.
    if bar >= 16 and (beat_on(bar) or in_break(bar)) and bar < 192:
        for st in (2, 6, 10, 14):
            pc = humanize_pos(spos_sw(gs + st), 2.0, RND)
            vc = humanize_vel(0.42, 0.06, RND)
            place(hat_L, HAT_C,    pc, vc * 0.82)
            place(hat_R, HAT_C_RB, pc, vc * 0.82)
        if in_peak(bar):
            for st in range(16):
                if st % 2 == 0 or RND.random() < 0.62:
                    continue
                pg = humanize_pos(spos_sw(gs + st), 2.5, RND)
                vg = humanize_vel(0.16, 0.05, RND)
                place(hat_L, HAT_C,    pg, vg * 0.82)
                place(hat_R, HAT_C_RB, pg, vg * 0.82)
        if bar >= 32:
            po = humanize_pos(spos_sw(gs + 14), 2.0, RND)
            vo = humanize_vel(0.5, 0.08, RND)
            place(hat_L, HAT_O,    po, vo * 0.78)
            place(hat_R, HAT_O_RB, po, vo * 0.78)
            if in_peak(bar):
                po = humanize_pos(spos_sw(gs + 6), 2.0, RND)
                vo = humanize_vel(0.4, 0.08, RND)
                place(hat_L, HAT_O,    po, vo * 0.78)
                place(hat_R, HAT_O_RB, po, vo * 0.78)

    # RIDE: 8ths lavados, SOLO en el peak (elemento nuevo que abre la sección).
    if 112 <= bar < 164:
        for st in (0, 2, 4, 6, 8, 10, 12, 14):
            pr = humanize_pos(spos_sw(gs + st), 2.0, RND)
            vr = humanize_vel(0.34 if st % 4 == 2 else 0.22, 0.06, RND)
            place(ride_L, RIDE,    pr, vr)
            place(ride_R, RIDE_RB, pr, vr)

    # SNAP: backbeat (2 y 4) con sala honda, groove tardío -> main y peak.
    if (40 <= bar < 88 or in_peak(bar)) and bar not in KICK_DROP_BARS:
        for st in (4, 12):
            pb = humanize_pos(spos(gs + st), 1.5, RND)
            place(snap_L, SNAP, pb, humanize_vel(0.46, 0.05, RND) * 0.92)
            place(snap_R, SNAP, pb, humanize_vel(0.46, 0.05, RND))

    # SHAKER: 16ths con acento en las CONTRAS (impares) = aire que empuja.
    if bar >= 16 and bar < 194:
        for st in range(16):
            if st % 2 == 0 and RND.random() < 0.30:
                continue
            ps  = humanize_pos(spos_sw(gs + st), 2.5, RND)
            vsh = humanize_vel(0.36 if st % 2 == 1 else 0.22, 0.05, RND)
            place(shaker_L, SHAKER_S,  ps, vsh)
            place(shaker_R, SHAKER_RB, ps, vsh)

    # STAB DUB: golpe Gm espaciado y TARDÍO (step 11 = offbeat dubby); cada 4
    # en groove/main, cada 2 en peak; en el break lava desde el downbeat.
    if (in_groove(bar) or in_main(bar)) and bar % 4 == 2 and bar >= 48:
        place_stereo(stab_L, stab_R, STAB, spos_sw(gs + 11), gain=0.85, pan=0.0)
    elif in_peak(bar) and bar % 2 == 0:
        place_stereo(stab_L, stab_R, STAB, spos_sw(gs + 11), gain=0.85, pan=0.0)
    elif in_break(bar):
        if bar % 2 == 0:
            place_stereo(stab_L, stab_R, STAB, spos(gs), gain=0.95, pan=0.0)
        if bar % 4 == 2:
            place_stereo(stab_L, stab_R, STAB, spos_sw(gs + 8), gain=0.60, pan=0.15)
    elif (in_late(bar) or in_outro(bar)) and bar % 4 == 0 and bar < 192:
        place_stereo(stab_L, stab_R, STAB, spos(gs), gain=0.70, pan=0.0)

print("OK")
print("Generando sub... ", end='', flush=True)

# --- SUB (monofónico, continuo: freq/amp/filtro por sample) ------------------
# UNA nota (G1) en offbeats de corchea, rodando con el swing = el pulso
# subterráneo. El b2 (Ab1) sólo como sombra deslizada al fin de frase.
SUB_PAT_A = [
    (2, 'G1', 0, 0), (6, 'G1', 1, 0), (10, 'G1', 0, 0), (14, 'G1', 0, 1),
]
SUB_PAT_B = [
    (2, 'G1', 0, 0), (6, 'G1', 1, 0), (10, 'Ab1', 0, 1), (12, 'G1', 0, 0),
    (14, 'G2', 0, 1),
]
SUB_PAT_BREAK = [(2, 'G1', 0, 0), (10, 'Ab1', 0, 1), (11, 'G1', 0, 0)]

def _sub_pattern(bar):
    # (patrón, ganancia) por compás; None = silencio. Mismo arco que el tema:
    # teaser en intro, línea completa groove/main/peak, mínimo en el break.
    if 16 <= bar < 24:
        return SUB_PAT_A, 0.55
    if in_groove(bar) or in_main(bar) or in_peak(bar):
        return (SUB_PAT_B if bar % 8 == 7 else SUB_PAT_A), 1.0
    if in_break(bar):
        if bar in BREAK_NOKICK:
            return SUB_PAT_BREAK, 0.55
        return SUB_PAT_A, 0.62 + 0.03 * (bar - 100)   # rebuild crece
    if in_late(bar):
        return SUB_PAT_A, 0.9
    if in_outro(bar) and bar < 188:
        return SUB_PAT_A, max(0.3, 1.0 - (bar - 176) * 0.06)
    return None, 0.0

_events = []
for bar in range(BARS):
    pat, pg = _sub_pattern(bar)
    if pat is None:
        continue
    for st, nm, acc, sl in pat:
        _events.append((spos_sw(bar * SPB + st), F[nm], acc, sl, pg))
_events.sort()

STEP_SMP = STEP_S * SR
sub_freq = np.full(TOTAL_SAMPLES, F['G1'])
sub_amp  = np.zeros(TOTAL_SAMPLES)
SUB_FENV = np.zeros(TOTAL_SAMPLES, dtype=np.float32)   # apertura de filtro por nota (Hz)
_atk = int(0.0015 * SR)

for i, (s0, f0, acc, sl, pg) in enumerate(_events):
    nxt = _events[i + 1][0] if i + 1 < len(_events) else TOTAL_SAMPLES
    legato = sl and (nxt - s0) < int(4 * STEP_SMP)
    end = min((nxt + int(0.010 * SR)) if legato else (s0 + int(0.85 * STEP_SMP)),
              TOTAL_SAMPLES)
    n = end - s0
    if n <= 0:
        continue
    sub_freq[s0:end] = f0
    if legato and nxt < TOTAL_SAMPLES:
        gl = min(int(0.060 * SR), nxt - s0)
        sub_freq[nxt - gl:nxt] = np.geomspace(f0, _events[i + 1][1], gl)
    t = np.arange(n) / SR
    env = pg * (1.10 if acc else 0.95) * np.exp(-t / 0.30)
    env[:min(_atk, n)] *= np.linspace(0, 1, min(_atk, n))
    nf = min(int(0.008 * SR), n)
    env[-nf:] *= np.linspace(1, 0, nf)
    sub_amp[s0:end] = np.maximum(sub_amp[s0:end], env)   # legato sin hueco
    fn = min(int(0.35 * SR), TOTAL_SAMPLES - s0)
    SUB_FENV[s0:s0 + fn] += ((600.0 if acc else 320.0) * pg *
                             np.exp(-np.arange(fn) / (0.11 * SR)))

_ph = 2 * np.pi * np.cumsum(sub_freq) / SR
# Square redondeada = throb hueco con armónicos que el LP móvil esculpe; el
# fundamental ya está en G1 (49 Hz), así que NO se apila otro sub 8vb debajo
# (lección de mezcla: el stack de graves aplasta el limiter).
sub_sig = (0.90 * np.tanh(2.2 * np.sin(_ph))
           + 0.22 * np.sin(2.0 * _ph)) * sub_amp   # 2º armónico = presencia G2
sub_L[:] = sub_sig
sub_R[:] = sub_sig
del _events, _ph, sub_sig, sub_freq, sub_amp
print("OK")
print("Procesando buses... ", end='', flush=True)

# --- Envolventes de energía (curvas continuas, sin saltos dentro de sección) -
SUB_ENERGY = breakpoint_env(TOTAL_SAMPLES, [
    (0, 0.0), (16, 0.6), (24, 0.9), (56, 1.0), (86, 0.95), (88, 0.55),
    (100, 0.75), (112, 1.0), (156, 1.0), (168, 0.75), (176, 0.5), (194, 0.0),
], bpm=BPM)
TOM_ENERGY = breakpoint_env(TOTAL_SAMPLES, [
    (0, 0.0), (32, 0.75), (56, 0.95), (86, 0.9), (88, 0.35), (100, 0.6),
    (112, 1.0), (156, 0.95), (168, 0.8), (176, 0.0),
], bpm=BPM)
RIM_ENERGY = breakpoint_env(TOTAL_SAMPLES, [
    (0, 0.0), (38, 0.0), (40, 0.75), (56, 0.9), (84, 0.85), (88, 0.0),
    (110, 0.0), (112, 1.0), (152, 0.95), (160, 0.75), (166, 0.85), (172, 0.0),
], bpm=BPM)
HAT_ENERGY = breakpoint_env(TOTAL_SAMPLES, [
    (0, 0.0), (16, 0.5), (24, 0.85), (56, 1.0), (86, 0.9), (90, 0.30),
    (104, 0.6), (112, 1.0), (156, 0.95), (176, 0.4), (194, 0.0),
], bpm=BPM)
RIDE_ENERGY = breakpoint_env(TOTAL_SAMPLES, [
    (0, 0.0), (110, 0.0), (116, 0.7), (132, 0.9), (156, 0.85), (164, 0.0),
], bpm=BPM)
SNAP_ENERGY = breakpoint_env(TOTAL_SAMPLES, [
    (0, 0.0), (38, 0.0), (40, 0.85), (84, 0.8), (88, 0.0), (110, 0.0),
    (112, 0.9), (156, 0.85), (160, 0.0),
], bpm=BPM)
SHAKER_ENERGY = breakpoint_env(TOTAL_SAMPLES, [
    (0, 0.0), (16, 0.4), (24, 0.65), (56, 0.8), (88, 0.25), (104, 0.55),
    (112, 0.9), (156, 0.85), (176, 0.25), (196, 0.0),
], bpm=BPM)
STAB_ENERGY = breakpoint_env(TOTAL_SAMPLES, [
    (0, 0.0), (46, 0.0), (48, 0.6), (56, 0.75), (88, 1.0), (112, 0.8),
    (144, 0.9), (160, 0.7), (192, 0.0),
], bpm=BPM)
GONG_ENERGY = breakpoint_env(TOTAL_SAMPLES, [
    (0, 0.0), (36, 0.5), (56, 0.7), (88, 1.0), (112, 0.7), (152, 0.7),
    (180, 0.5), (192, 0.0),
], bpm=BPM)
PAD_ENERGY = breakpoint_env(TOTAL_SAMPLES, [
    (0, 0.55), (24, 0.5), (56, 0.55), (88, 1.0), (112, 0.55), (144, 0.62),
    (160, 0.55), (194, 0.0),
], bpm=BPM)
DRONE_ENERGY = breakpoint_env(TOTAL_SAMPLES, [
    (0, 0.7), (24, 0.55), (56, 0.5), (88, 1.0), (112, 0.6), (176, 0.55),
    (198, 0.0),
], bpm=BPM)
TEX_ENERGY = breakpoint_env(TOTAL_SAMPLES, [
    (0, 0.55), (24, 0.4), (56, 0.5), (88, 1.0), (112, 0.45), (176, 0.6),
    (196, 0.0),
], bpm=BPM)
BREATH_ENERGY = breakpoint_env(TOTAL_SAMPLES, [
    (0, 1.0), (160, 1.0), (192, 0.0),
], bpm=BPM)
# Profundidad de reverb que respira: más cola en el break (aire) y el peak
# (tamaño); seca en groove = evolución de ESPACIO continua.
REV_ENERGY = breakpoint_env(TOTAL_SAMPLES, [
    (0, 0.55), (24, 0.7), (56, 0.85), (88, 1.3), (112, 1.0), (140, 1.15),
    (160, 0.8), (199, 0.4),
], bpm=BPM)

# Filtro del SUB: la palanca de evolución principal. Cutoff = base por sección
# (abre intro -> peak, se cierra en el break) + envolvente por nota (los
# acentos abren más) + LFO lento. Rango contenido = throb que se intuye.
SUB_CUT = np.clip(breakpoint_env(TOTAL_SAMPLES, [
    (0, 220), (16, 280), (24, 380), (56, 520), (88, 280), (100, 440),
    (112, 640), (152, 840), (160, 520), (176, 380), (192, 280),
], bpm=BPM) + lfo(TOTAL_SAMPLES, rate_bars=4.0, depth=80, center=0.0,
                  shape='tri', bpm=BPM) + SUB_FENV, 120, 1400)
# Filtro del STAB: LP que abre hacia el peak + wobble lento = dub vivo.
STAB_CUT = np.clip(breakpoint_env(TOTAL_SAMPLES, [
    (0, 650), (48, 850), (56, 1150), (88, 1600), (112, 1450), (152, 2300),
    (160, 850), (192, 650),
], bpm=BPM) + lfo(TOTAL_SAMPLES, rate_bars=6.0, depth=170, center=0.0,
                  shape='sine', bpm=BPM), 450, 2800)
# Delay throw dub del stab: el wet crece hacia el fin de cada frase de 8 y
# se recoge en el downbeat (rampa continua).
STAB_THROW = 0.30 * (1.0 + 0.8 * lfo(TOTAL_SAMPLES, rate_bars=8.0, depth=0.5,
                                     center=0.5, shape='saw', bpm=BPM))
# Filtro del drone: abre en el break = la caverna "florece".
DRONE_CUT = breakpoint_env(TOTAL_SAMPLES, [
    (0, 420), (56, 500), (88, 1000), (112, 480), (199, 360),
], bpm=BPM)
# Filtro del pad: respira por sección + LFO lento = nunca se congela.
PAD_CUT = np.clip(breakpoint_env(TOTAL_SAMPLES, [
    (0, 850), (24, 460), (56, 420), (88, 1400), (112, 500), (152, 600),
    (160, 640), (192, 560),
], bpm=BPM) + lfo(TOTAL_SAMPLES, rate_bars=6.0, depth=140, center=0.0,
                  shape='sine', bpm=BPM), 300, 1700)
SHAKER_CUT = np.clip(lfo(TOTAL_SAMPLES, rate_bars=8.0, depth=2400, center=8200,
                         shape='sine', bpm=BPM), 5000, 12000)
# Juego de filtros en la percusión: LP-wobble sobre los toms (ciclo 10
# compases) y HP respirando sobre el rim (contrafase) = el groove se
# adelgaza y engorda sin tocar la rejilla.
TOM_CUT = np.clip(lfo(TOTAL_SAMPLES, rate_bars=10.0, depth=900, center=3200,
                      shape='sine', bpm=BPM), 1800, 5000)
RIM_HP = np.clip(lfo(TOTAL_SAMPLES, rate_bars=10.0, depth=120, center=300,
                     shape='sine', phase=np.pi, bpm=BPM), 140, 500)
# LP wobble del ride (pseudo-phaser lento): nunca suena igual.
RIDE_CUT = np.clip(lfo(TOTAL_SAMPLES, rate_bars=6.0, depth=2800, center=7800,
                       shape='sine', bpm=BPM), 4000, 12000)
# Respiración de hats y rumble (ciclos largos): vida sin tocar la rejilla.
HAT_BREATH = lfo(TOTAL_SAMPLES, rate_bars=12.0, depth=0.07, center=1.0,
                 shape='sine', bpm=BPM)
RUMBLE_BREATH = lfo(TOTAL_SAMPLES, rate_bars=8.0, depth=0.10, center=1.0,
                    shape='sine', bpm=BPM)
# Filtro de la textura: LP lento 16 compases = la cama nunca es estática.
TEX_CUT = np.clip(lfo(TOTAL_SAMPLES, rate_bars=16.0, depth=380, center=820,
                      shape='sine', bpm=BPM), 360, 1400)
# HP en contrafase sobre la textura -> band-pass móvil (la cama "vocaliza").
TEX_HP = np.clip(lfo(TOTAL_SAMPLES, rate_bars=10.0, depth=160, center=310,
                     shape='sine', phase=np.pi, bpm=BPM), 140, 560)
# Centros de phaser (pad lento y profundo; ride más rápido = swoosh).
PAD_PHA  = lfo(TOTAL_SAMPLES, rate_bars=12.0, depth=420, center=750,
               shape='sine', bpm=BPM)
RIDE_PHA = lfo(TOTAL_SAMPLES, rate_bars=4.0, depth=850, center=1400,
               shape='sine', phase=np.pi / 3, bpm=BPM)

# --- Procesado por bus (low-end apretado, mids repartidos, mucho espacio) ----
# KICK: punch en 3.2k, cuerpo en 85, mud fuera.
for ch in [kick_L, kick_R]:
    ch[:] = eq_low_cut(ch, freq=40)
    ch[:] = eq_mid_boost(ch, freq=85,   gain_db=+1.5, q=1.2)
    ch[:] = eq_mid_boost(ch, freq=200,  gain_db=-2.0, q=1.4)
    ch[:] = eq_mid_boost(ch, freq=420,  gain_db=-3.0, q=1.0)
    ch[:] = eq_mid_boost(ch, freq=3200, gain_db=+6.0, q=1.6)
    ch[:] = eq_lp(ch, freq=7000)
    ch[:] = saturate(ch, drive=1.25, character='warm')
    ch[:] = compress(ch, threshold=0.38, ratio=3.5, attack_ms=1.5, release_ms=85)

kick_trigger = (kick_L + kick_R) * 0.5

# RUMBLE: cola del kick con reverb larga -> LP grave -> saturación -> duck al
# kick = retumbe rodante <220 Hz (firma del género), mono-compatible.
rumble_L = eq_lp(kick_L.copy(), freq=270)
rumble_R = eq_lp(kick_R.copy(), freq=270)
rumble_L = reverb_plate(rumble_L, decay=0.90, room=0.86, wet=1.0)
rumble_R = reverb_plate(rumble_R, decay=0.90, room=0.89, wet=1.0)
for ch in (rumble_L, rumble_R):
    ch[:] = eq_low_cut(ch, freq=36)
    ch[:] = eq_lp(ch, freq=220)
    ch[:] = saturate(ch, drive=1.2, character='warm')
rumble_L = sidechain_compress(rumble_L, kick_trigger, threshold=0.14, ratio=6.0,
                              attack_ms=2.0, release_ms=150.0)
rumble_R = sidechain_compress(rumble_R, kick_trigger, threshold=0.14, ratio=6.0,
                              attack_ms=2.0, release_ms=150.0)
rumble_L *= RUMBLE_BREATH
rumble_R *= RUMBLE_BREATH

# SUB: LP resonante moderado (q=1.8) -> saturación cálida = throb grave y
# disimulado, corriente subterránea; hueco en 250 para el kick, sidechain duro.
for ch in [sub_L, sub_R]:
    ch[:] = eq_low_cut(ch, freq=32)
    ch[:] = eq_mid_boost(ch, freq=250, gain_db=-1.5, q=1.2)
    ch[:] = compress(ch, threshold=0.34, ratio=4.0, attack_ms=6.0, release_ms=95, makeup_db=1.0)
    ch[:] = sidechain_compress(ch, kick_trigger, threshold=0.16, ratio=12.0,
                                attack_ms=0.5, release_ms=110.0)
sub_L[:] = moving_lowpass(sub_L, SUB_CUT, q=1.8)
sub_R[:] = moving_lowpass(sub_R, SUB_CUT, q=1.8)
sub_L[:] = saturate(sub_L, drive=1.45, character='warm')
sub_R[:] = saturate(sub_R, drive=1.45, character='warm')
sub_L[:] *= SUB_ENERGY; sub_R[:] *= SUB_ENERGY

# TOMS: cuerpo 180, piel 2.2k, LP-wobble lento, room corta oscura y sidechain
# suave = la tribu golpea cerca pero dentro de la cueva.
for ch in [tom_L, tom_R]:
    ch[:] = eq_low_cut(ch, freq=90)
    ch[:] = eq_mid_boost(ch, freq=180,  gain_db=+1.5, q=1.0)
    ch[:] = eq_mid_boost(ch, freq=420,  gain_db=-2.0, q=1.1)
    ch[:] = eq_mid_boost(ch, freq=2200, gain_db=+3.0, q=1.3)
    ch[:] = saturate(ch, drive=1.35, character='warm')
    ch[:] = compress(ch, threshold=0.40, ratio=3.5, attack_ms=4.0, release_ms=80)
    ch[:] = sidechain_compress(ch, kick_trigger, threshold=0.24, ratio=4.0,
                                attack_ms=1.0, release_ms=80.0)
tom_L[:] = moving_lowpass(tom_L, TOM_CUT, q=0.9)
tom_R[:] = moving_lowpass(tom_R, TOM_CUT, q=0.9)
tom_L[:] = reverb_space(tom_L, predelay_ms=8, decay=0.24, room=0.32, damping=0.72, wet=0.12)
tom_R[:] = reverb_space(tom_R, predelay_ms=11, decay=0.24, room=0.34, damping=0.72, wet=0.13)
tom_L[:] *= TOM_ENERGY; tom_R[:] *= TOM_ENERGY

# RIM: click presente (2.8k), seco y cerca; HP respirando + slap-delay dub
# corto duckeado = la madera rebota en la pared del fondo.
for ch in [rim_L, rim_R]:
    ch[:] = eq_mid_boost(ch, freq=500,  gain_db=+1.0, q=1.0)
    ch[:] = eq_mid_boost(ch, freq=2800, gain_db=+4.0, q=1.3)
    ch[:] = saturate(ch, drive=1.3, character='warm')
    ch[:] = compress(ch, threshold=0.45, ratio=3.0, attack_ms=8.0, release_ms=70)
    ch[:] = sidechain_compress(ch, kick_trigger, threshold=0.28, ratio=3.0,
                                attack_ms=1.5, release_ms=65.0)
rim_L[:] = moving_highpass(rim_L, RIM_HP, q=0.7)
rim_R[:] = moving_highpass(rim_R, RIM_HP, q=0.7)
RIM_DLY_S = (60.0 / BPM) * 0.75
rdl, rdr = delay_pingpong(rim_L, rim_R, RIM_DLY_S, feedback=0.36, n_echoes=5, wet=1.0)
rdl = sidechain_compress(rdl - rim_L, kick_trigger, threshold=0.22, ratio=5.0,
                         attack_ms=1.0, release_ms=60.0)
rdr = sidechain_compress(rdr - rim_R, kick_trigger, threshold=0.22, ratio=5.0,
                         attack_ms=1.0, release_ms=60.0)
rim_L[:] = rim_L + eq_lp(rdl, freq=3000) * 0.20
rim_R[:] = rim_R + eq_lp(rdr, freq=3000) * 0.20
del rdl, rdr
rim_L[:] = reverb_space(rim_L, predelay_ms=10, decay=0.24, room=0.32, damping=0.70, wet=0.10)
rim_R[:] = reverb_space(rim_R, predelay_ms=13, decay=0.24, room=0.34, damping=0.70, wet=0.11)
rim_L[:] *= RIM_ENERGY; rim_R[:] *= RIM_ENERGY

# HATS: dueñas del brillo medio-alto — presentes y con mordida (la cueva la
# ponen armonía y reverb, no unas hats apagadas).
for ch in [hat_L, hat_R]:
    ch[:] = eq_high_shelf_biquad(ch, freq=8500, gain_db=+4.0)
    ch[:] = eq_mid_boost(ch, freq=4500, gain_db=+2.5, q=1.0)
    ch[:] = eq_mid_boost(ch, freq=1200, gain_db=-1.0, q=1.2)
    ch[:] = compress(ch, threshold=0.55, ratio=2.5, attack_ms=2.5, release_ms=55)
hat_L[:] *= HAT_ENERGY * HAT_BREATH; hat_R[:] *= HAT_ENERGY * HAT_BREATH

# RIDE: lavado (HP 2.6k) + LP wobble + phaser = swoosh de ceniza vivo.
for ch in [ride_L, ride_R]:
    ch[:] = eq_low_cut(ch, freq=2600)
    ch[:] = eq_high_shelf_biquad(ch, freq=9500, gain_db=+2.5)
    ch[:] = compress(ch, threshold=0.50, ratio=2.5, attack_ms=3.0, release_ms=80)
    ch[:] = sidechain_compress(ch, kick_trigger, threshold=0.30, ratio=2.5,
                                attack_ms=1.5, release_ms=60.0)
ride_L[:] = moving_lowpass(ride_L, RIDE_CUT, q=0.8)
ride_R[:] = moving_lowpass(ride_R, RIDE_CUT, q=0.8)
ride_L[:] = phaser(ride_L, RIDE_PHA, stages=4, mix=0.40)
ride_R[:] = phaser(ride_R, RIDE_PHA * 1.12, stages=4, mix=0.40)
ride_L[:] *= RIDE_ENERGY; ride_R[:] *= RIDE_ENERGY

# SNAP: cuerpo 1k, sala honda oscura = backbeat de cueva.
for ch in [snap_L, snap_R]:
    ch[:] = eq_low_cut(ch, freq=150)
    ch[:] = eq_mid_boost(ch, freq=1000, gain_db=+2.0, q=1.0)
    ch[:] = eq_high_shelf_biquad(ch, freq=6000, gain_db=+1.0)
    ch[:] = compress(ch, threshold=0.40, ratio=4.0, attack_ms=1.0, release_ms=55)
snap_L[:] = reverb_space(snap_L, predelay_ms=16, decay=0.40, room=0.52, damping=0.60, wet=0.26)
snap_R[:] = reverb_space(snap_R, predelay_ms=21, decay=0.40, room=0.55, damping=0.60, wet=0.28)
snap_L[:] *= SNAP_ENERGY; snap_R[:] *= SNAP_ENERGY

# SHAKER: aire, filtro móvil que respira, sidechain.
for ch in [shaker_L, shaker_R]:
    ch[:] = eq_low_cut(ch, freq=3200)
    ch[:] = compress(ch, threshold=0.50, ratio=2.5, attack_ms=1.0, release_ms=40)
shaker_L[:] = moving_lowpass(shaker_L, SHAKER_CUT, q=0.70)
shaker_R[:] = moving_lowpass(shaker_R, SHAKER_CUT, q=0.70)
shaker_L[:] = sidechain_compress(shaker_L, kick_trigger, threshold=0.30, ratio=2.2,
                                 attack_ms=1.0, release_ms=50.0)
shaker_R[:] = sidechain_compress(shaker_R, kick_trigger, threshold=0.30, ratio=2.2,
                                 attack_ms=1.0, release_ms=50.0)
shaker_L[:] *= SHAKER_ENERGY; shaker_R[:] *= SHAKER_ENERGY

# STAB: HP + LP móvil + sidechain + DELAY DUB (corcheas punteadas) con throw
# que respira por frase + reverb honda = EL lavado armónico del tema.
for ch in [stab_L, stab_R]:
    ch[:] = eq_low_cut(ch, freq=220)
    ch[:] = eq_mid_boost(ch, freq=500, gain_db=-1.5, q=1.0)
    ch[:] = compress(ch, threshold=0.36, ratio=4.0, attack_ms=3.0, release_ms=90)
    ch[:] = sidechain_compress(ch, kick_trigger, threshold=0.20, ratio=6.0,
                                attack_ms=0.8, release_ms=90.0)
stab_L[:] = moving_lowpass(stab_L, STAB_CUT, q=1.1)
stab_R[:] = moving_lowpass(stab_R, STAB_CUT, q=1.1)
stab_L[:] = saturate(stab_L, drive=1.1, character='warm')
stab_R[:] = saturate(stab_R, drive=1.1, character='warm')
# Chorus lento decorrelado L/R = el stab se ensancha y ondula sin engordar.
stab_L[:] = mod_delay(stab_L, base_ms=15.0, depth_ms=3.5, rate_hz=0.29,
                      mix=0.45, phase=0.0)
stab_R[:] = mod_delay(stab_R, base_ms=17.0, depth_ms=4.0, rate_hz=0.25,
                      mix=0.45, phase=1.7)
STAB_DLY_S = (60.0 / BPM) * 0.75
sdl, sdr = delay_pingpong(stab_L, stab_R, STAB_DLY_S, feedback=0.52, n_echoes=7, wet=1.0)
sdl = sidechain_compress(sdl - stab_L, kick_trigger, threshold=0.20, ratio=6.0,
                         attack_ms=1.0, release_ms=70.0)
sdr = sidechain_compress(sdr - stab_R, kick_trigger, threshold=0.20, ratio=6.0,
                         attack_ms=1.0, release_ms=70.0)
stab_L[:] = stab_L + eq_lp(sdl, freq=2600) * STAB_THROW
stab_R[:] = stab_R + eq_lp(sdr, freq=2600) * STAB_THROW
del sdl, sdr
stab_L[:] = reverb_space(stab_L, predelay_ms=18, decay=0.52, room=0.62, damping=0.68, wet=0.32)
stab_R[:] = reverb_space(stab_R, predelay_ms=24, decay=0.52, room=0.65, damping=0.68, wet=0.35)
stab_L[:] *= STAB_ENERGY; stab_R[:] *= STAB_ENERGY

# GONG: banda acotada, delay dub largo + reverb ENORME = campanada al fondo
# de la caverna que se aleja rebotando.
for ch in [gong_L, gong_R]:
    ch[:] = eq_low_cut(ch, freq=90)
    ch[:] = eq_lp(ch, freq=4200)
    ch[:] = compress(ch, threshold=0.45, ratio=3.0, attack_ms=4.0, release_ms=120)
GONG_DLY_S = (60.0 / BPM) * 1.5
gdl, gdr = delay_pingpong(gong_L, gong_R, GONG_DLY_S, feedback=0.5, n_echoes=5, wet=1.0)
gong_L[:] = gong_L + eq_lp(gdl - gong_L, freq=2400) * 0.35
gong_R[:] = gong_R + eq_lp(gdr - gong_R, freq=2400) * 0.35
del gdl, gdr
gong_L[:] = reverb_space(gong_L, predelay_ms=28, decay=0.72, room=0.85, damping=0.55, wet=0.55)
gong_R[:] = reverb_space(gong_R, predelay_ms=36, decay=0.72, room=0.88, damping=0.55, wet=0.60)
gong_L[:] = sidechain_compress(gong_L, kick_trigger, threshold=0.24, ratio=4.0,
                               attack_ms=1.5, release_ms=100.0)
gong_R[:] = sidechain_compress(gong_R, kick_trigger, threshold=0.24, ratio=4.0,
                               attack_ms=1.5, release_ms=100.0)
gong_L[:] *= GONG_ENERGY; gong_R[:] *= GONG_ENERGY

# BREATH: banda de susurro acotada, chorus asimétrico L/R + throws de delay
# fantasma + reverb larga oscura = la sala respira al fondo.
for ch in [breath_L, breath_R]:
    ch[:] = eq_low_cut(ch, freq=250)
    ch[:] = eq_lp(ch, freq=3800)
    ch[:] = compress(ch, threshold=0.40, ratio=3.0, attack_ms=8.0, release_ms=120)
breath_L[:] = mod_delay(breath_L, base_ms=19.0, depth_ms=2.5, rate_hz=0.21,
                        mix=0.40, phase=0.0)
breath_R[:] = mod_delay(breath_R, base_ms=24.0, depth_ms=3.0, rate_hz=0.17,
                        mix=0.40, phase=2.1)
BREATH_DLY_S = (60.0 / BPM) * 1.5
bdl, bdr = delay_pingpong(breath_L, breath_R, BREATH_DLY_S, feedback=0.55, n_echoes=6, wet=1.0)
breath_L[:] = breath_L + eq_lp(bdl - breath_L, freq=2800) * 0.45
breath_R[:] = breath_R + eq_lp(bdr - breath_R, freq=2800) * 0.45
del bdl, bdr
breath_L[:] = reverb_space(breath_L, predelay_ms=32, decay=0.68, room=0.80, damping=0.55, wet=0.60)
breath_R[:] = reverb_space(breath_R, predelay_ms=40, decay=0.68, room=0.84, damping=0.55, wet=0.65)
breath_L[:] = sidechain_compress(breath_L, kick_trigger, threshold=0.24, ratio=3.5,
                                 attack_ms=2.0, release_ms=110.0)
breath_R[:] = sidechain_compress(breath_R, kick_trigger, threshold=0.24, ratio=3.5,
                                 attack_ms=2.0, release_ms=110.0)
breath_L[:] *= BREATH_ENERGY; breath_R[:] *= BREATH_ENERGY

# PAD: oscuro (LP móvil) + phaser lento + reverb larga = cripta ancha.
for ch in [pad_L, pad_R]:
    ch[:] = eq_low_cut(ch, freq=70)
    ch[:] = saturate(ch, drive=1.15, character='warm')
    ch[:] = compress(ch, threshold=0.55, ratio=2.0, attack_ms=30, release_ms=200)
    ch[:] = sidechain_compress(ch, kick_trigger, threshold=0.30, ratio=3.0,
                                attack_ms=2.0, release_ms=120.0)
pad_L[:] = moving_lowpass(pad_L, PAD_CUT, q=0.7)
pad_R[:] = moving_lowpass(pad_R, PAD_CUT, q=0.7)
pad_L[:] = phaser(pad_L, PAD_PHA, stages=4, mix=0.35)
pad_R[:] = phaser(pad_R, PAD_PHA * 0.9, stages=4, mix=0.35)
pad_L[:] = reverb_plate(pad_L, decay=0.60, room=0.72, wet=0.46)
pad_R[:] = reverb_plate(pad_R, decay=0.60, room=0.74, wet=0.50)
pad_L[:] *= PAD_ENERGY; pad_R[:] *= PAD_ENERGY

# DRONE: muy oscuro (LP automatizado) + wow de cinta + reverb = el suelo
# de la caverna, siempre moviéndose.
for ch in [drone_L, drone_R]:
    ch[:] = eq_low_cut(ch, freq=40)
    ch[:] = saturate(ch, drive=1.2, character='warm')
    ch[:] = compress(ch, threshold=0.50, ratio=2.0, attack_ms=30, release_ms=200)
    ch[:] = sidechain_compress(ch, kick_trigger, threshold=0.25, ratio=4.0,
                                attack_ms=2.0, release_ms=140.0)
drone_L[:] = moving_lowpass(drone_L, DRONE_CUT, q=0.8)
drone_R[:] = moving_lowpass(drone_R, DRONE_CUT, q=0.8)
drone_L[:] = mod_delay(drone_L, base_ms=11.0, depth_ms=3.0, rate_hz=0.05,
                       mix=0.5, phase=0.0)
drone_R[:] = mod_delay(drone_R, base_ms=11.0, depth_ms=3.0, rate_hz=0.05,
                       mix=0.5, phase=np.pi)
drone_L[:] = reverb_plate(drone_L, decay=0.64, room=0.64, wet=0.36)
drone_R[:] = reverb_plate(drone_R, decay=0.64, room=0.67, wet=0.40)
drone_L[:] *= DRONE_ENERGY; drone_R[:] *= DRONE_ENERGY

# FX (sweeps, swells, crashes): reverb amplia, sidechain.
for ch in [fx_L, fx_R]:
    ch[:] = eq_low_cut(ch, freq=160)
    ch[:] = eq_high_shelf_biquad(ch, freq=8500, gain_db=+0.5)
    ch[:] = compress(ch, threshold=0.50, ratio=2.5, attack_ms=3, release_ms=130)
fx_L[:] = reverb_plate(fx_L, decay=0.60, room=0.68, wet=0.42)
fx_R[:] = reverb_plate(fx_R, decay=0.60, room=0.72, wet=0.46)
fx_L[:] = sidechain_compress(fx_L, kick_trigger, threshold=0.25, ratio=3.0,
                             attack_ms=2.0, release_ms=110.0)
fx_R[:] = sidechain_compress(fx_R, kick_trigger, threshold=0.25, ratio=3.0,
                             attack_ms=2.0, release_ms=110.0)

# TEXTURA: band-pass MÓVIL (LP y HP en contrafase = la cama "vocaliza")
# + trémolo lento, sidechain suave.
TEX_TREM = lfo(TOTAL_SAMPLES, rate_bars=6.0, depth=0.25, center=0.75,
               shape='sine', bpm=BPM)
for nm_ch in ('L', 'R'):
    ch = tex_L if nm_ch == 'L' else tex_R
    ch[:] = eq_low_cut(ch, freq=140)
    ch[:] = eq_lp(ch, freq=2200)
    ch[:] = moving_lowpass(ch, TEX_CUT, q=0.8)
    ch[:] = moving_highpass(ch, TEX_HP, q=0.8)
    ch[:] = saturate(ch, drive=1.1, character='warm')
    ch[:] = sidechain_compress(ch, kick_trigger, threshold=0.28, ratio=3.0,
                               attack_ms=2.0, release_ms=120.0)
    ch[:] *= TEX_ENERGY * TEX_TREM

# IMPACT: sub-boom de transición. Grave apretado, SIN duck (marca el cambio).
for ch in [impact_L, impact_R]:
    ch[:] = eq_low_cut(ch, freq=30)
    ch[:] = eq_lp(ch, freq=130)
    ch[:] = saturate(ch, drive=1.3, character='warm')

# --- Reverb SEND compartida (la sala común = LA cueva, cohesión) -------------
send_L = (stab_L * 0.50 + tom_L * 0.28 + rim_L * 0.30 + snap_L * 0.45 +
          breath_L * 0.40 + gong_L * 0.55 + shaker_L * 0.20)
send_R = (stab_R * 0.50 + tom_R * 0.28 + rim_R * 0.30 + snap_R * 0.45 +
          breath_R * 0.40 + gong_R * 0.55 + shaker_R * 0.20)
send_L = eq_low_cut(send_L, freq=300); send_R = eq_low_cut(send_R, freq=300)
rev_L = reverb_plate(send_L, decay=0.64, room=0.76, wet=1.0)
rev_R = reverb_plate(send_R, decay=0.64, room=0.80, wet=1.0)
rev_L = eq_high_shelf_biquad(rev_L, freq=8500, gain_db=-2.5)
rev_R = eq_high_shelf_biquad(rev_R, freq=8500, gain_db=-2.5)
rev_L = sidechain_compress(rev_L, kick_trigger, threshold=0.20, ratio=4.0,
                           attack_ms=2.0, release_ms=120.0)
rev_R = sidechain_compress(rev_R, kick_trigger, threshold=0.20, ratio=4.0,
                           attack_ms=2.0, release_ms=120.0)
rev_L *= REV_ENERGY; rev_R *= REV_ENERGY

print("OK")

# --- Niveles de mezcla -------------------------------------------------------
# Balance club: kick manda pero sin aplastar (sub+low ~55 %), la percusión
# de 2-10 kHz PRESENTE (mordida = nivel pro), atmósfera audible.
GAIN_KICK   = 0.78
GAIN_RUMBLE = 0.45
GAIN_SUB    = 0.44
GAIN_TOM    = 0.85
GAIN_RIM    = 1.10
GAIN_HAT    = 3.60
GAIN_RIDE   = 2.20
GAIN_SNAP   = 1.75
GAIN_SHAKER = 5.20
GAIN_STAB   = 0.75
GAIN_GONG   = 1.25
GAIN_BREATH = 1.35
GAIN_PAD    = 0.42
GAIN_DRONE  = 0.44
GAIN_TEX    = 0.065
GAIN_FX     = 2.00
GAIN_REV    = 0.75
GAIN_IMPACT = 0.28

# Bus paralelo de batería (NY compression): la suma de percusión aplastada
# fuerte y mezclada por debajo = densidad y pegada de club; los transitorios
# limpios los pone la señal seca.
def _drum_par(k, t, h, sn, sh, rm, rd):
    d = (k * GAIN_KICK + t * GAIN_TOM + h * GAIN_HAT + sn * GAIN_SNAP +
         sh * GAIN_SHAKER + rm * GAIN_RIM + rd * GAIN_RIDE)
    return compress(d, threshold=0.16, ratio=9.0, attack_ms=9.0,
                    release_ms=140, makeup_db=7.0)
dpar_L = _drum_par(kick_L, tom_L, hat_L, snap_L, shaker_L, rim_L, ride_L)
dpar_R = _drum_par(kick_R, tom_R, hat_R, snap_R, shaker_R, rim_R, ride_R)
GAIN_DPAR = 0.22

mix_L = (kick_L * GAIN_KICK + rumble_L * GAIN_RUMBLE + sub_L * GAIN_SUB +
         tom_L * GAIN_TOM + rim_L * GAIN_RIM + hat_L * GAIN_HAT +
         ride_L * GAIN_RIDE + snap_L * GAIN_SNAP + shaker_L * GAIN_SHAKER +
         stab_L * GAIN_STAB + gong_L * GAIN_GONG + breath_L * GAIN_BREATH +
         pad_L * GAIN_PAD + drone_L * GAIN_DRONE + tex_L * GAIN_TEX +
         fx_L * GAIN_FX + rev_L * GAIN_REV + impact_L * GAIN_IMPACT +
         dpar_L * GAIN_DPAR)
mix_R = (kick_R * GAIN_KICK + rumble_R * GAIN_RUMBLE + sub_R * GAIN_SUB +
         tom_R * GAIN_TOM + rim_R * GAIN_RIM + hat_R * GAIN_HAT +
         ride_R * GAIN_RIDE + snap_R * GAIN_SNAP + shaker_R * GAIN_SHAKER +
         stab_R * GAIN_STAB + gong_R * GAIN_GONG + breath_R * GAIN_BREATH +
         pad_R * GAIN_PAD + drone_R * GAIN_DRONE + tex_R * GAIN_TEX +
         fx_R * GAIN_FX + rev_R * GAIN_REV + impact_R * GAIN_IMPACT +
         dpar_R * GAIN_DPAR)

# --- Export de stems ---------------------------------------------------------
print("Exportando stems... ", end='', flush=True)
STEMS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stems')
os.makedirs(STEMS_DIR, exist_ok=True)
STEMS = [
    ('kick',    kick_L,   kick_R,   GAIN_KICK),
    ('rumble',  rumble_L, rumble_R, GAIN_RUMBLE),
    ('sub',     sub_L,    sub_R,    GAIN_SUB),
    ('toms',    tom_L,    tom_R,    GAIN_TOM),
    ('rim',     rim_L,    rim_R,    GAIN_RIM),
    ('hats',    hat_L,    hat_R,    GAIN_HAT),
    ('ride',    ride_L,   ride_R,   GAIN_RIDE),
    ('snap',    snap_L,   snap_R,   GAIN_SNAP),
    ('shaker',  shaker_L, shaker_R, GAIN_SHAKER),
    ('stab',    stab_L,   stab_R,   GAIN_STAB),
    ('gong',    gong_L,   gong_R,   GAIN_GONG),
    ('breath',  breath_L, breath_R, GAIN_BREATH),
    ('pad',     pad_L,    pad_R,    GAIN_PAD),
    ('drone',   drone_L,  drone_R,  GAIN_DRONE),
    ('texture', tex_L,    tex_R,    GAIN_TEX),
    ('fx',      fx_L,     fx_R,     GAIN_FX),
    ('rev',     rev_L,    rev_R,    GAIN_REV),
    ('impact',  impact_L, impact_R, GAIN_IMPACT),
]
for name, L, R, g in STEMS:
    stem = np.column_stack([(L * g).astype(np.float32), (R * g).astype(np.float32)])
    sf.write(os.path.join(STEMS_DIR, f'{name}.wav'), stem, SR, subtype='PCM_24')
print(f"OK ({len(STEMS)} pistas  ->  stems/)")

# Liberar buffers ya volcados: el máster sólo necesita mix_L/mix_R.
del STEMS, stem
del kick_L, kick_R, rumble_L, rumble_R, sub_L, sub_R, tom_L, tom_R
del rim_L, rim_R, hat_L, hat_R, ride_L, ride_R, snap_L, snap_R
del shaker_L, shaker_R, stab_L, stab_R, gong_L, gong_R, breath_L, breath_R
del pad_L, pad_R, drone_L, drone_R, tex_L, tex_R, fx_L, fx_R, rev_L, rev_R
del impact_L, impact_R, send_L, send_R, kick_trigger, dpar_L, dpar_R
del SUB_ENERGY, TOM_ENERGY, RIM_ENERGY, HAT_ENERGY, RIDE_ENERGY, SNAP_ENERGY
del SHAKER_ENERGY, STAB_ENERGY, GONG_ENERGY, PAD_ENERGY, DRONE_ENERGY
del TEX_ENERGY, BREATH_ENERGY, REV_ENERGY
del SUB_CUT, SUB_FENV, STAB_CUT, STAB_THROW, DRONE_CUT, PAD_CUT, SHAKER_CUT
del TOM_CUT, RIM_HP, RIDE_CUT
del HAT_BREATH, RUMBLE_BREATH, TEX_CUT, TEX_HP, TEX_TREM, PAD_PHA, RIDE_PHA

# --- Master bus --------------------------------------------------------------
print("Master bus... ", end='', flush=True)

# 0) DJ filter: HP suave en intro (0-8) y outro (188-199).
DJ_HP_CUT = breakpoint_env(TOTAL_SAMPLES, [
    (0, 280), (8, 20), (188, 20), (194, 240), (199, 500),
], bpm=BPM)
mix_L = moving_highpass(mix_L, DJ_HP_CUT, q=0.7)
mix_R = moving_highpass(mix_R, DJ_HP_CUT, q=0.7)

# 1) EQ master: sub domado, mud (250) fuera, dureza (3.1k) suave, aire abierto.
for ch in [mix_L, mix_R]:
    ch[:] = eq_low_cut(ch, freq=26)
    ch[:] = eq_low_shelf(ch, freq=45, gain_db=-3.0)
    ch[:] = eq_mid_boost(ch, freq=250, gain_db=-1.8, q=1.1)
    ch[:] = eq_mid_boost(ch, freq=600, gain_db=+1.3, q=0.7)   # cuerpo del groove
    ch[:] = eq_mid_boost(ch, freq=2800, gain_db=+2.5, q=0.8)  # mordida/presencia
    ch[:] = eq_high_shelf_biquad(ch, freq=10500, gain_db=+1.0)

# 1b) Apertura progresiva: crossfade continuo hacia versión más brillante
#     (shelf +3 dB @8.5k) que crece intro -> peak y cae en el outro.
AIR_ENV = breakpoint_env(TOTAL_SAMPLES, [
    (0, 0.0), (24, 0.30), (56, 0.50), (88, 0.35), (112, 1.0), (152, 1.0),
    (160, 0.55), (176, 0.35), (199, 0.12),
], bpm=BPM)
mix_L = mix_L * (1 - AIR_ENV) + eq_high_shelf_biquad(mix_L, freq=8500, gain_db=+3.0) * AIR_ENV
mix_R = mix_R * (1 - AIR_ENV) + eq_high_shelf_biquad(mix_R, freq=8500, gain_db=+3.0) * AIR_ENV

# 2) Glue compressor suave.
mix_L[:] = compress(mix_L, threshold=0.60, ratio=1.6, attack_ms=20, release_ms=280)
mix_R[:] = compress(mix_R, threshold=0.60, ratio=1.6, attack_ms=20, release_ms=280)

# 2b) Saturación de cinta sutil.
mix_L[:] = saturate(mix_L, drive=1.04, character='warm')
mix_R[:] = saturate(mix_R, drive=1.04, character='warm')

# 2c) DJ LP "suck": LP resonante que cierra ~2.4 compases hacia el fin de
#     sección y abre de golpe en el downbeat = tensión larga -> release.
MASTER_LP_CUT = breakpoint_env(TOTAL_SAMPLES, [
    (0, 18000),
    (53.5, 18000), (55.9, 600), (55.98, 600), (56.0, 18000),      # -> main
    (85.5, 18000), (87.9, 600), (87.98, 600), (88.0, 18000),      # -> break
    (157.5, 18000), (159.9, 600), (159.98, 600), (160.0, 18000),  # -> late
], bpm=BPM)
mix_L = moving_lowpass(mix_L, MASTER_LP_CUT, q=1.2)
mix_R = moving_lowpass(mix_R, MASTER_LP_CUT, q=1.2)

# 3) Graves a mono (<150 Hz) + ensanchar medios/agudos.
mix_L, mix_R = mono_low_end(mix_L, mix_R, crossover_hz=150)
mix_L, mix_R = stereo_width(mix_L, mix_R, amount=0.55)

# 3b) M/S: high-shelf sólo en los lados = aire/anchura sin descentrar el mid.
mid  = (mix_L + mix_R) * 0.5
side = (mix_L - mix_R) * 0.5
side = eq_high_shelf_biquad(side, freq=8200, gain_db=+1.8)
mix_L = mid + side
mix_R = mid - side

# 3c) Arco macro sutil pre-limiter: groove < main < peak (~0.7 dB de viaje).
MACRO_ENV = breakpoint_env(TOTAL_SAMPLES, [
    (0, 0.93), (24, 0.96), (56, 0.98), (88, 0.92), (112, 1.0),
    (156, 1.0), (160, 0.96), (176, 0.93), (199, 0.90),
], bpm=BPM)
mix_L *= MACRO_ENV; mix_R *= MACRO_ENV

# 4) Empuje a +6.5 dB: nivel de club competitivo dejando ~8 dB de crest.
peak_pre = max(np.max(np.abs(mix_L)), np.max(np.abs(mix_R)), 1e-9)
target   = 10 ** (+6.5 / 20.0) / peak_pre
mix_L *= target; mix_R *= target

# 5) Soft clip suave: sólo lima picos.
mix_L = soft_clip(mix_L, ceiling=0.97, knee=0.85)
mix_R = soft_clip(mix_R, ceiling=0.97, knee=0.85)

# 6) Fades.
mix_L = fade_in(mix_L,  2.0); mix_R = fade_in(mix_R,  2.0)
mix_L = fade_out(mix_L, 4.0); mix_R = fade_out(mix_R, 4.0)

# 6b) DC fuera (la saturación asimétrica lo introduce; roba headroom): HP 20 Hz.
mix_L = eq_low_cut(mix_L, freq=20)
mix_R = eq_low_cut(mix_R, freq=20)

# 7) Brickwall final a -0.8 dBFS.
mix_L = limiter_brickwall(mix_L, threshold_db=-0.8, lookahead_ms=3.0, release_ms=120)
mix_R = limiter_brickwall(mix_R, threshold_db=-0.8, lookahead_ms=3.0, release_ms=120)

print("OK")

# --- Dither TPDF (ruido triangular ±1 LSB) antes de cuantizar a 24-bit --------
def tpdf_dither(x, bits=24):
    q = 2.0 ** -(bits - 1)
    d = RND.random(len(x)) - RND.random(len(x))
    return x + d * q
mix_L = tpdf_dither(mix_L); mix_R = tpdf_dither(mix_R)

# --- Export final + métricas -------------------------------------------------
stereo  = np.column_stack([mix_L.astype(np.float32), mix_R.astype(np.float32)])
peak_db = 20 * np.log10(np.max(np.abs(stereo)) + 1e-10)
rms_db  = 20 * np.log10(np.sqrt(np.mean(stereo**2)) + 1e-10)

output_file = '418-black-tea.wav'
sf.write(output_file, stereo, SR, subtype='PCM_24')

duration_s = TOTAL_SAMPLES / SR
mm, ss = divmod(duration_s, 60)
print(f"\n Exportado: {output_file}")
print(f"   {BARS} barras  |  {BPM} BPM  |  {int(mm)}:{ss:04.1f}  |  44.1kHz / 24-bit estéreo")
print(f"   Peak: {peak_db:+.1f} dBFS   RMS: {rms_db:+.1f} dBFS")
print(f"   Reproducir: ffplay {output_file}")
