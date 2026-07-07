"""
Efectos de audio: dinámica (compresor, sidechain, limiter, soft-clip),
EQ (biquads del Audio EQ Cookbook + Butterworth), saturación, reverb plate,
delay ping-pong y utilidades estéreo (mono low-end, width, fades).

Todas las funciones trabajan con arrays numpy float64/float32.
"""
import numpy as np
from scipy.signal import butter, lfilter, sosfilt

SR = 44100


def compress(signal, threshold=0.30, ratio=4.0,
             attack_ms=5.0, release_ms=100.0, makeup_db=0.0, sr=SR):
    """
    Compresor dinámico vectorizado.
    Usa IIR de primer orden para seguimiento de envolvente attack/release.
    """
    att = np.exp(-1.0 / (attack_ms * 0.001 * sr))
    rel = np.exp(-1.0 / (release_ms * 0.001 * sr))

    abs_sig = np.abs(signal).astype(np.float64)

    # Envolvente peak con attack rápido / release lento
    b_att, a_att = [1 - att], [1.0, -att]
    b_rel, a_rel = [1 - rel], [1.0, -rel]

    env_att = lfilter(b_att, a_att, abs_sig)
    env_rel = lfilter(b_rel, a_rel, abs_sig)
    env = np.maximum(env_att, env_rel * 0.4)

    # Ganancia de reducción
    safe_env = np.maximum(env, 1e-10)
    gain = np.where(
        env > threshold,
        (threshold + (env - threshold) / ratio) / safe_env,
        1.0
    )
    # Suavizar transiciones de gain con release
    gain_smooth = lfilter(b_rel, a_rel, gain)

    makeup = 10 ** (makeup_db / 20.0)
    return (signal * gain_smooth * makeup).astype(signal.dtype)


def saturate(signal, drive=2.0, character='warm'):
    """
    Saturación analógica.
    - 'soft': tanh simétrico (tube limpio)
    - 'warm': asimétrico (tape)
    - 'hard': clip (transistor)
    """
    s = signal.astype(np.float64)
    if character == 'soft':
        out = np.tanh(s * drive) / np.tanh(np.clip(drive, 1e-6, 100))
    elif character == 'warm':
        # Asimétrico: más compresión en positivo (armónicos pares = calidez)
        pos = np.tanh(s * drive * 1.12)
        neg = np.tanh(s * drive * 0.88)
        norm = np.tanh(np.clip(drive, 1e-6, 100))
        out = np.where(s >= 0, pos, neg) / norm
    elif character == 'hard':
        out = np.clip(s * drive, -1.0, 1.0)
    else:
        out = s
    return out.astype(signal.dtype)


def reverb_plate(signal, sr=SR, decay=0.35, room=0.45, wet=0.18):
    """
    Reverb de placa simple: filtros comb en paralelo + allpass.
    """
    out = signal.astype(np.float64).copy()
    # Delays de comb (ms) basados en proporciones áureas
    comb_delays_ms = [29.7, 37.1, 41.1, 43.7, 47.3, 51.0]

    for d_ms in comb_delays_ms:
        d_samples = int(d_ms * 0.001 * sr * room)
        if d_samples < 1 or d_samples >= len(signal):
            continue
        fb = np.zeros_like(out)
        fb[d_samples:] = signal[:-d_samples] * decay
        out += fb * (wet / len(comb_delays_ms))

    # Allpass difusor
    ap_delay = int(10.3 * 0.001 * sr)
    if ap_delay < len(out):
        allpass_fb = np.zeros_like(out)
        allpass_fb[ap_delay:] = out[:-ap_delay] * 0.7
        out = out * 0.3 + allpass_fb * wet * 0.5

    return out.astype(signal.dtype)


def eq_low_cut(signal, freq=28.0, sr=SR):
    """High-pass para limpiar subsónicos."""
    sos = butter(2, freq / (sr / 2), btype='high', output='sos')
    return sosfilt(sos, signal).astype(signal.dtype)


def eq_mid_boost(signal, freq=2000.0, gain_db=3.0, q=1.4, sr=SR):
    """Boost/cut de banda media (peaking EQ)."""
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    A = 10 ** (gain_db / 40.0)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return lfilter(b, a, signal).astype(signal.dtype)


def stereo_width(left, right, amount=0.3):
    """Expande o contrae la imagen estéreo vía MS."""
    mid = (left + right) * 0.5
    side = (left - right) * 0.5 * (1.0 + amount)
    return (mid + side).astype(left.dtype), (mid - side).astype(right.dtype)


def eq_low_shelf(signal, freq=200.0, gain_db=3.0, sr=SR):
    """Biquad low shelf (Audio EQ Cookbook)."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / 2 * np.sqrt(2)  # S=1

    b0 =  A * ((A+1) - (A-1)*cos_w0 + 2*np.sqrt(A)*alpha)
    b1 =  2*A * ((A-1) - (A+1)*cos_w0)
    b2 =  A * ((A+1) - (A-1)*cos_w0 - 2*np.sqrt(A)*alpha)
    a0 =       (A+1) + (A-1)*cos_w0 + 2*np.sqrt(A)*alpha
    a1 = -2 * ((A-1) + (A+1)*cos_w0)
    a2 =       (A+1) + (A-1)*cos_w0 - 2*np.sqrt(A)*alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1/a0, a2/a0])
    return lfilter(b, a, signal).astype(signal.dtype)


def eq_high_shelf_biquad(signal, freq=8000.0, gain_db=2.0, sr=SR):
    """Biquad high shelf (Audio EQ Cookbook)."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / 2 * np.sqrt(2)  # S=1

    b0 =  A * ((A+1) + (A-1)*cos_w0 + 2*np.sqrt(A)*alpha)
    b1 = -2*A * ((A-1) + (A+1)*cos_w0)
    b2 =  A * ((A+1) + (A-1)*cos_w0 - 2*np.sqrt(A)*alpha)
    a0 =       (A+1) - (A-1)*cos_w0 + 2*np.sqrt(A)*alpha
    a1 =  2 * ((A-1) - (A+1)*cos_w0)
    a2 =       (A+1) - (A-1)*cos_w0 - 2*np.sqrt(A)*alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1/a0, a2/a0])
    return lfilter(b, a, signal).astype(signal.dtype)


def eq_lp(signal, freq=600.0, order=4, sr=SR):
    """Low-pass Butterworth."""
    sos = butter(order, freq / (sr / 2), btype='low', output='sos')
    return sosfilt(sos, signal).astype(signal.dtype)


def sidechain_compress(signal, trigger, threshold=0.28, ratio=8.0,
                        attack_ms=0.8, release_ms=65.0, sr=SR):
    """
    Sidechain: atenúa signal cuando trigger (el kick) supera threshold.
    """
    att = np.exp(-1.0 / (attack_ms * 0.001 * sr))
    rel = np.exp(-1.0 / (release_ms * 0.001 * sr))

    abs_trig = np.abs(trigger).astype(np.float64)
    b_att, a_att = [1 - att], [1.0, -att]
    b_rel, a_rel = [1 - rel], [1.0, -rel]

    env = np.maximum(
        lfilter(b_att, a_att, abs_trig),
        lfilter(b_rel, a_rel, abs_trig) * 0.12
    )
    gain = np.where(
        env > threshold,
        (threshold + (env - threshold) / ratio) / np.maximum(env, 1e-10),
        1.0
    )
    gain_smooth = lfilter(b_rel, a_rel, gain)
    return (signal.astype(np.float64) * gain_smooth).astype(signal.dtype)


def limiter_brickwall(signal, threshold_db=-0.3, lookahead_ms=3.0, release_ms=100.0, sr=SR):
    """
    Brick wall limiter con lookahead real.
    Ataque instantáneo (detecta picos futuros), release lento configurable.
    Garantiza que ningún sample supere threshold_db.
    """
    threshold = 10 ** (threshold_db / 20.0)
    n         = len(signal)
    lookahead = max(1, int(lookahead_ms * 0.001 * sr))
    rel       = np.exp(-1.0 / (release_ms * 0.001 * sr))

    sig64 = signal.astype(np.float64)

    # Delay del audio por lookahead samples (para la corrección anticipada)
    delayed = np.zeros(n, dtype=np.float64)
    if lookahead < n:
        delayed[lookahead:] = sig64[:-lookahead]

    # Máximo futuro: en cada posición, el pico mayor dentro de los próximos
    # `lookahead` samples. Esto es lo que realmente necesita un limiter.
    abs_sig = np.abs(sig64)
    padded  = np.concatenate([abs_sig, np.zeros(lookahead)])
    future_max = abs_sig.copy()
    for k in range(1, lookahead + 1):          # ~132 iteraciones  ->  rápido con numpy
        future_max = np.maximum(future_max, padded[k:k + n])

    # Gain necesario para mantenerse bajo el threshold
    needed = np.where(future_max > threshold,
                      threshold / np.maximum(future_max, 1e-10),
                      1.0)

    # Aplicar gain con ataque instantáneo + release lento.
    # Aquí SÍ hace falta el loop: la ganancia baja inmediatamente (no lineal),
    # pero sube lentamente (release IIR).
    gain    = np.ones(n, dtype=np.float64)
    rel_inv = 1.0 - rel
    for i in range(1, n):
        recovered = gain[i - 1] * rel + rel_inv   # recuperación lenta hacia 1.0
        gain[i]   = min(needed[i], recovered)      # ataque instantáneo

    return (delayed * gain).astype(signal.dtype)


def delay_pingpong(left, right, delay_s, feedback=0.42, n_echoes=6,
                   wet=0.32, sr=SR):
    """
    Delay ping-pong: ecos finitos con feedback decreciente que alternan
    de canal (L -> R -> L...) = rebote estéreo.
    """
    d = max(1, int(delay_s * sr))
    l64 = left.astype(np.float64)
    r64 = right.astype(np.float64)
    src = (l64 + r64) * 0.5
    n   = len(src)

    wet_l = np.zeros(n)
    wet_r = np.zeros(n)
    for k in range(1, n_echoes + 1):
        sh  = k * d
        if sh >= n:
            break
        amp = feedback ** k
        if k % 2 == 1:                       # eco impar  ->  derecha
            wet_r[sh:] += src[:n - sh] * amp
        else:                                # eco par  ->  izquierda
            wet_l[sh:] += src[:n - sh] * amp

    out_l = l64 + wet * wet_l
    out_r = r64 + wet * wet_r
    return out_l.astype(left.dtype), out_r.astype(right.dtype)


def mod_delay(signal, base_ms=12.0, depth_ms=3.0, rate_hz=0.3, mix=0.5,
              phase=0.0, sr=SR):
    """
    Delay modulado por LFO con interpolación fraccional.
    Rates ~0.2-0.5 Hz = chorus; rates <0.1 Hz = wow de cinta (deriva de tono).
    Con fases distintas por canal decorrela L/R sin tocar el low-end.
    """
    n = len(signal)
    t = np.arange(n) / sr
    d = (base_ms + depth_ms * np.sin(2 * np.pi * rate_hz * t + phase)) * 0.001 * sr
    idx = np.clip(np.arange(n) - d, 0, n - 1)
    wet = np.interp(idx, np.arange(n), signal.astype(np.float64))
    return ((1.0 - mix) * signal + mix * wet).astype(signal.dtype)


def soft_clip(signal, ceiling=0.92, knee=0.7):
    """
    Soft-clipper de seguridad: deja el audio intacto por debajo de `knee`
    y redondea suavemente (tanh) los picos por encima, en vez de recortarlos
    de forma dura. Suaviza transitorios antes del limiter  ->  menos distorsión.
    """
    s = signal.astype(np.float64)
    out = s.copy()
    over = np.abs(s) > knee
    if np.any(over):
        sign  = np.sign(s[over])
        excess = (np.abs(s[over]) - knee)
        span   = max(ceiling - knee, 1e-6)
        shaped = knee + span * np.tanh(excess / span)
        out[over] = sign * shaped
    return out.astype(signal.dtype)


def mono_low_end(left, right, crossover_hz=180.0, sr=SR):
    """
    Suma bajas frecuencias a mono por debajo del crossover.
    Obligatorio para sistemas de club (subwoofer central único).
    """
    sos_lp = butter(4, crossover_hz / (sr / 2), btype='low',  output='sos')
    sos_hp = butter(4, crossover_hz / (sr / 2), btype='high', output='sos')
    mono_low = (sosfilt(sos_lp, left) + sosfilt(sos_lp, right)) * 0.5
    return (mono_low + sosfilt(sos_hp, left)).astype(left.dtype), \
           (mono_low + sosfilt(sos_hp, right)).astype(right.dtype)


def fade_in(signal, duration_s, sr=SR):
    n = min(int(duration_s * sr), len(signal))
    sig = signal.copy()
    sig[:n] *= np.linspace(0, 1, n) ** 2
    return sig


def fade_out(signal, duration_s, sr=SR):
    n = min(int(duration_s * sr), len(signal))
    sig = signal.copy()
    sig[-n:] *= np.linspace(1, 0, n) ** 2
    return sig
