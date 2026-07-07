"""
Movimiento, automatización y profundidad: curvas breakpoint + LFO
tempo-sincronizado, humanize de timing/velocidad, filtros móviles
(cutoff automatizado) y reverb con pre-delay + damping.

Convenciones: SR=44100; curvas largas en float32 (RAM), DSP interno float64.
RNG sembrado para que los renders sean reproducibles.
"""
import numpy as np
from scipy.signal import lfilter

SR = 44100


# ─── RNG reproducible ─────────────────────────────────────────────────────────

def rng(seed=42):
    """Generador sembrado: mismo seed  ->  mismo 'desorden' en cada render."""
    return np.random.default_rng(seed)


# ─── Automatización: curvas en el tiempo ──────────────────────────────────────

def bar_to_sample(bar, bpm=142.0, spb=16, sr=SR):
    step_s = (60.0 / bpm) / 4.0
    return int(bar * spb * step_s * sr)


def breakpoint_env(total_samples, points, bpm=142.0, spb=16, sr=SR, curve='lin'):
    """
    Curva de automatización a partir de puntos (bar, valor).

    points: lista [(bar, valor), ...] ordenada por bar.
    curve : 'lin' interpola lineal; 'exp' interpola en dominio log (suena
            más natural para cutoffs/volúmenes porque el oído es logarítmico).

    Devuelve un array de longitud total_samples listo para multiplicar como
    ganancia o usar como cutoff sample a sample.
    """
    # float32: las curvas duran todo el tema (millones de samples); a esta
    # resolución la precisión extra de float64 no aporta y duplica la RAM.
    out = np.empty(total_samples, dtype=np.float32)
    pts = [(bar_to_sample(b, bpm, spb, sr), v) for b, v in points]

    # Extender extremos para cubrir todo el buffer
    if pts[0][0] > 0:
        pts.insert(0, (0, pts[0][1]))
    if pts[-1][0] < total_samples:
        pts.append((total_samples, pts[-1][1]))

    for (s0, v0), (s1, v1) in zip(pts[:-1], pts[1:]):
        s0 = max(0, min(s0, total_samples))
        s1 = max(0, min(s1, total_samples))
        if s1 <= s0:
            continue
        if curve == 'exp':
            lv = np.linspace(np.log(max(v0, 1e-6)), np.log(max(v1, 1e-6)), s1 - s0)
            out[s0:s1] = np.exp(lv)
        else:
            out[s0:s1] = np.linspace(v0, v1, s1 - s0)
    return out


def lfo(total_samples, rate_bars=4.0, depth=1.0, center=0.0,
        shape='sine', phase=0.0, bpm=142.0, sr=SR):
    """
    LFO tempo-sincronizado. rate_bars = nº de compases por ciclo
    (4.0 = un ciclo cada 4 compases; 0.25 = 4 ciclos por compás).
    Devuelve center + depth * forma(t), en [center-depth, center+depth].
    """
    bar_s   = (60.0 / bpm) * 4.0
    period  = rate_bars * bar_s
    t       = np.arange(total_samples) / sr
    ph      = 2 * np.pi * (t / period) + phase
    if shape == 'sine':
        w = np.sin(ph)
    elif shape == 'tri':
        w = 2.0 / np.pi * np.arcsin(np.sin(ph))
    elif shape == 'saw':           # rampa ascendente (0 -> 1) por ciclo
        w = 2.0 * (ph / (2 * np.pi) % 1.0) - 1.0
    elif shape == 'ramp_down':
        w = 1.0 - 2.0 * (ph / (2 * np.pi) % 1.0)
    else:
        w = np.sin(ph)
    return (center + depth * w).astype(np.float32)


def humanize_pos(pos, time_jitter_ms, rng_, sr=SR):
    """Mueve un onset ±jitter ms aleatoriamente. Mata el feel 'machine-gun'."""
    j = (rng_.random() * 2 - 1) * time_jitter_ms * 0.001 * sr
    return int(pos + j)


def humanize_vel(vel, vel_jitter, rng_):
    """Varía la velocidad ±vel_jitter (fracción). 0.12 = ±12 %."""
    return float(vel * (1.0 + (rng_.random() * 2 - 1) * vel_jitter))


def _lp_coeffs(cutoff_hz, q, sr=SR):
    w0     = 2 * np.pi * np.clip(cutoff_hz, 20.0, sr / 2 * 0.98) / sr
    alpha  = np.sin(w0) / (2 * np.clip(q, 0.3, 18.0))
    cos_w0 = np.cos(w0)
    b0 = (1 - cos_w0) / 2
    b1 = 1 - cos_w0
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    return (np.array([b0 / a0, b1 / a0, b2 / a0]),
            np.array([1.0, a1 / a0, a2 / a0]))


def moving_lowpass(signal, cutoff_curve, q=1.2, block=256, sr=SR):
    """
    Low-pass resonante con cutoff automatizado (cutoff_curve = array por sample).
    Procesa por bloques recalculando el biquad  ->  barridos suaves de filtro.
    q > 2 añade resonancia audible.
    """
    sig = signal.astype(np.float64)
    n   = len(sig)
    out = np.empty(n, dtype=np.float64)
    zi  = np.zeros(2)
    for s0 in range(0, n, block):
        s1  = min(s0 + block, n)
        c   = float(np.mean(cutoff_curve[s0:s1]))
        b, a = _lp_coeffs(c, q, sr)
        out[s0:s1], zi = lfilter(b, a, sig[s0:s1], zi=zi)
    return out.astype(signal.dtype)


def _hp_coeffs(cutoff_hz, q, sr=SR):
    w0     = 2 * np.pi * np.clip(cutoff_hz, 20.0, sr / 2 * 0.98) / sr
    alpha  = np.sin(w0) / (2 * np.clip(q, 0.3, 18.0))
    cos_w0 = np.cos(w0)
    b0 = (1 + cos_w0) / 2
    b1 = -(1 + cos_w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    return (np.array([b0 / a0, b1 / a0, b2 / a0]),
            np.array([1.0, a1 / a0, a2 / a0]))


def moving_highpass(signal, cutoff_curve, q=0.7, block=256, sr=SR):
    """
    High-pass con cutoff automatizado (gemelo de moving_lowpass).
    Con cutoff alto recorta graves (filtro DJ); a ~20 Hz pasa todo.
    """
    sig = signal.astype(np.float64)
    n   = len(sig)
    out = np.empty(n, dtype=np.float64)
    zi  = np.zeros(2)
    for s0 in range(0, n, block):
        s1  = min(s0 + block, n)
        c   = float(np.mean(cutoff_curve[s0:s1]))
        b, a = _hp_coeffs(c, q, sr)
        out[s0:s1], zi = lfilter(b, a, sig[s0:s1], zi=zi)
    return out.astype(signal.dtype)


def _ap_coeffs(center_hz, q, sr=SR):
    w0     = 2 * np.pi * np.clip(center_hz, 40.0, sr / 2 * 0.95) / sr
    alpha  = np.sin(w0) / (2 * np.clip(q, 0.3, 18.0))
    cos_w0 = np.cos(w0)
    a0 = 1 + alpha
    return (np.array([(1 - alpha) / a0, -2 * cos_w0 / a0, (1 + alpha) / a0]),
            np.array([1.0, -2 * cos_w0 / a0, (1 - alpha) / a0]))


def phaser(signal, center_curve, stages=4, mix=0.5, q=0.7, block=256, sr=SR):
    """
    Phaser: cadena de allpass de 2º orden con centro automatizado (array por
    sample); cada etapa se separa en frecuencia. mix ~0.3-0.5 = notches suaves.
    """
    sig = signal.astype(np.float64)
    n   = len(sig)
    out = sig.copy()
    for s in range(stages):
        zi  = np.zeros(2)
        res = np.empty(n)
        for s0 in range(0, n, block):
            s1 = min(s0 + block, n)
            c  = float(np.mean(center_curve[s0:s1])) * (1.0 + 0.35 * s)
            b, a = _ap_coeffs(c, q, sr)
            res[s0:s1], zi = lfilter(b, a, out[s0:s1], zi=zi)
        out = res
    return ((1.0 - mix) * sig + mix * out).astype(signal.dtype)


# ─── Profundidad: reverb con pre-delay + damping ──────────────────────────────

def reverb_space(signal, sr=SR, predelay_ms=18.0, decay=0.42,
                 room=0.55, damping=0.35, wet=0.20):
    """
    Reverb con más profundidad que reverb_plate: pre-delay (separa la fuente
    de la cola), damping (low-pass en la cola = reverb oscura) y combs en
    proporciones primas + allpass difusor.
    wet puede ser escalar o curva (array) para automatizar el envío.
    """
    s   = signal.astype(np.float64)
    n   = len(s)
    pre = int(predelay_ms * 0.001 * sr)

    src = np.zeros(n)
    if pre < n:
        src[pre:] = s[:n - pre]
    else:
        src = s.copy()

    comb_ms = [29.7, 37.1, 41.1, 43.7, 47.3, 53.0, 59.1]
    tail = np.zeros(n)
    for d_ms in comb_ms:
        d = int(d_ms * 0.001 * sr * room)
        if d < 1 or d >= n:
            continue
        fb = np.zeros(n)
        fb[d:] = src[:n - d] * decay
        tail += fb / len(comb_ms)

    # Allpass difusor
    apd = int(10.3 * 0.001 * sr)
    if apd < n:
        ap = np.zeros(n)
        ap[apd:] = tail[:n - apd] * 0.7
        tail = tail * 0.3 + ap * 0.5

    # Damping: low-pass de 1er orden en la cola
    if damping > 0:
        alpha = np.exp(-2 * np.pi * (1.0 - damping) * 4000.0 / sr)
        tail = lfilter([1 - alpha], [1.0, -alpha], tail)

    wet_arr = wet if np.isscalar(wet) else wet[:n]
    return (s + tail * wet_arr).astype(signal.dtype)
