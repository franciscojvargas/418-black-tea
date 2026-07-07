"""
Atmósfera y FX de transición: drone, swells, sweeps de ruido, crash y
percusión corta (shaker).

Todas las funciones devuelven float32 mono, declicado y normalizado, listo
para colocar/sumar sin clicks.
"""
import numpy as np
from scipy.signal import butter, lfilter

SR = 44100


def _edge_fade(sig, fade_in_ms=0.4, fade_out_ms=6.0, sr=SR):
    n = len(sig)
    fi = min(int(fade_in_ms * 0.001 * sr), n // 2)
    fo = min(int(fade_out_ms * 0.001 * sr), n // 2)
    if fi > 0:
        sig[:fi] *= 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fi))
    if fo > 0:
        sig[-fo:] *= 0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, fo))
    return sig


def _lp(cutoff, sr=SR, order=4):
    return butter(order, cutoff / (sr / 2), btype='low')


def _hp(cutoff, sr=SR, order=2):
    return butter(order, cutoff / (sr / 2), btype='high')


def _bp(lo, hi, sr=SR, order=4):
    return butter(order, [lo / (sr / 2), hi / (sr / 2)], btype='band')


# ─── Transiciones ─────────────────────────────────────────────────────────────

def reverse_swell(duration=2.0, f_center=3500.0, sr=SR):
    """
    Swell reverso (cymbal invertido): brillo que crece y corta en el downbeat.
    Se coloca terminando justo en la entrada de la sección  ->  succión.
    """
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(321)
    noise = rng.standard_normal(n)
    b, a = _bp(f_center * 0.5, f_center * 2.2, sr)
    shimmer = lfilter(b, a, noise)
    # Decaimiento exponencial  ->  al invertir, queda un swell ascendente
    env = np.exp(-t * 2.6)
    sig = shimmer * env
    sig = sig[::-1].copy()            # invertir  ->  crece hacia el final
    sig *= 1.0 + 0.15 * np.sin(2 * np.pi * 6.0 * t)  # leve movimiento
    sig = sig / np.max(np.abs(sig) + 1e-10) * 0.80
    return _edge_fade(sig, 30.0, 1.0).astype(np.float32)


# ─── Atmósfera ────────────────────────────────────────────────────────────────

def atmos_drone(duration=8.0, root=55.0, intervals=(0, 3, 7, 12),
                cutoff=1400.0, brightness=0.0, sr=SR):
    """
    Drone oscuro y vivo: sub-seno + cluster de saws con deriva de afinado lenta
    por voz + cama de ruido band-pass. `brightness` (0..1) abre el cutoff.
    """
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(1717)
    sig = np.zeros(n)

    # Sub-seno puro = fundamento cálido
    sig += 0.9 * np.sin(2 * np.pi * root * t + rng.uniform(0, 2 * np.pi))

    # Cluster de saws con deriva de afinación lenta por voz = beating vivo
    for semi in intervals:
        f0 = root * (2.0 ** (semi / 12.0))
        for cents in (-9, -3, 4, 11):
            drift_rate = rng.uniform(0.03, 0.09)
            drift_amt = rng.uniform(3.0, 7.0)
            inst = cents + drift_amt * np.sin(2 * np.pi * drift_rate * t +
                                              rng.uniform(0, 2 * np.pi))
            f = f0 * (2.0 ** (inst / 1200.0))
            # fase = integral de la frecuencia instantánea
            phase = 2 * np.pi * np.cumsum(f) / sr + rng.uniform(0, 2 * np.pi)
            saw = np.zeros(n)
            for k in range(1, 10):
                saw += (1.0 / k) * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
            sig += saw * (2 / np.pi) * 0.5

    # Cama de aire: ruido band-pass grave con trémolo lento
    air = lfilter(*_bp(180, 900, sr), rng.standard_normal(n))
    air *= 0.14 * (0.6 + 0.4 * np.sin(2 * np.pi * 0.05 * t + rng.uniform(0, 2 * np.pi)))
    sig += air

    fc = float(np.clip(cutoff * (1.0 + brightness), 200, sr / 2 * 0.9))
    b, a = _lp(fc, sr)
    sig = lfilter(b, a, sig)

    # Swell de entrada/salida (raised-cosine) + trémolo lento muy suave
    swell = np.ones(n)
    ramp = int(min(0.45, duration * 0.35) * sr)
    if ramp > 0:
        swell[:ramp] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ramp))
        swell[-ramp:] = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, ramp))
    sig *= swell * (1.0 + 0.08 * np.sin(2 * np.pi * 0.12 * t))

    b, a = _hp(40, sr)
    sig = lfilter(b, a, sig)
    sig = sig / np.max(np.abs(sig) + 1e-10) * 0.85
    return _edge_fade(sig, 12.0, 18.0).astype(np.float32)


# ─── Percusión corta ──────────────────────────────────────────────────────────

def shaker(duration=0.055, tilt=0.0, sr=SR):
    """Shaker: ruido agudo y corto para 16ths. `tilt` cambia semilla y brillo."""
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(int(404 + tilt * 100))
    noise = rng.standard_normal(n)
    b, a = _hp(6500 + tilt * 1500, sr)
    noise = lfilter(b, a, noise)
    shape = np.clip(0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n)), 0.0, 1.0)
    env = np.exp(-t * 130) * shape ** 0.3
    sig = noise * env
    sig = sig / np.max(np.abs(sig) + 1e-10) * 0.55
    return _edge_fade(sig, 0.3, 2.0).astype(np.float32)


# ─── Ear-candy puntual ────────────────────────────────────────────────────────

def crash(duration=1.9, bright=1.0, sr=SR):
    """Platillo crash: ruido brillante con cola larga. Marca aperturas de sección."""
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(606)
    noise = rng.standard_normal(n)
    b, a = _hp(2500 * bright, sr)
    noise = lfilter(b, a, noise)
    # parciales metálicos inarmónicos para timbre de platillo
    partials = [3400, 4730, 6210, 8400, 11200]
    tone = sum(0.08 * np.sin(2 * np.pi * f * t) for f in partials)
    sig = (noise * 0.85 + tone) * np.exp(-t * 2.4)
    sig = sig / np.max(np.abs(sig) + 1e-10) * 0.82
    return _edge_fade(sig, 0.3, 30.0).astype(np.float32)


def noise_sweep(duration=2.0, f_start=300.0, f_end=9000.0, reverse=False,
                seed=11, sr=SR):
    """Barrido de ruido (riser): band-pass cuyo centro se desplaza f_start -> f_end
    por bloques, con swell. `reverse` lo convierte en caída."""
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(int(seed))
    noise = rng.standard_normal(n)
    out = np.zeros(n)
    block = 1024
    n_blocks = int(np.ceil(n / block))
    for bi in range(n_blocks):
        s0 = bi * block
        s1 = min(s0 + block, n)
        frac = bi / max(n_blocks - 1, 1)
        fc = f_start * (f_end / f_start) ** frac          # barrido logarítmico
        lo = np.clip(fc * 0.7, 30, sr / 2 * 0.85)
        hi = np.clip(fc * 1.6, lo + 50, sr / 2 * 0.9)
        b, a = _bp(lo, hi, sr)
        out[s0:s1] = lfilter(b, a, noise[s0:s1])
    swell = (t / duration) ** 1.6                          # crece hacia el final
    out *= swell
    if reverse:
        out = out[::-1].copy()
    out = out / np.max(np.abs(out) + 1e-10) * 0.80
    return _edge_fade(out, 20.0, 20.0).astype(np.float32)
