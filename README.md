```
               )    (
              (  )   )
               )  ( (
            _____________
            \###########/
             \#########/
              `-------'

       4 1 8 · b l a c k   t e a
```

# 418 - black tea

[![SoundCloud](https://img.shields.io/badge/▶_Escuchar_en-SoundCloud-ff5500?logo=soundcloud&logoColor=white)](https://soundcloud.com/418techno/418-black-tea)

**Hard groove oscuro y atmosférico** —ambiente de club— generado
100 % por código: sin samples, sin DAW, sin plugins. Todo el tema —síntesis,
secuenciación, mezcla y máster— es Python puro sobre numpy/scipy, y
`soundfile` solo para escribir el WAV.

Este repositorio contiene **solo el código**. El audio (`418-black-tea.wav`
y los stems) es la salida de `render.py`: no se versiona, se regenera en local.

## Renderizar

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python render.py
```

Produce:

- `418-black-tea.wav` — mezcla final, estéreo 24-bit / 44.1 kHz (~5:33)
- `stems/` — las 18 pistas por separado, ya procesadas y a nivel de mezcla

El render es **determinista**: todos los generadores de ruido usan semillas
fijas, así que el mismo código produce siempre el mismo WAV, bit a bit.

## El tema

| | |
|---|---|
| Tempo | 145 BPM |
| Duración | 200 barras (~5:33) |
| Tonalidad | Sol menor |
| Referencia | Óscar Mulero / Reeko / Planetary Assault Systems / Jeff Mills oscuro |
| Licencia | CC BY-NC 4.0 (ver `LICENSE`) |

El motor del tema es un **groove tribal de toms sintéticos** —tom grave, tom
medio y slap de conga seca— en patrón sincopado de pregunta/respuesta que
rueda con un swing del 0.56: la firma hardgroove. Debajo, un **kick de cueva
afinado a G1** alimenta el rumble reverberado del género, y un **sub
monofónico de UNA nota** (Sol) late en los offbeats de corchea; su única
disonancia es el **b2** (Ab), que aparece solo como sombra deslizada al fin
de frase. La armonía es un **stab dub Gm hueco** con la sombra b2 arriba,
golpeado tarde (step 11) en chorus lento y delay a corcheas punteadas, con
**pre-ecos reversos** que succionan hacia cada golpe. La atmósfera la ponen
un **gong cavernoso** (parciales inarmónicos con batido, uno cada ~16
compases, enorme y lejano), una **exhalación fantasma** (ruido formántico de
susurro que respira en los umbrales de sección), un **rim de madera** seco
con slap-delay, drone de caverna G0/G1 con wow de cinta, pad Gm de cripta y
una cama de "lluvia en nave" con band-pass móvil.

El carácter está en el **procesado en movimiento perpetuo**: phaser
automatizado en pad y ride, LP-wobble en toms y ride, HP respirando en el
rim, band-pass móvil que hace "vocalizar" la textura, chorus decorrelado L/R
en stab y breath, wow de cinta en el drone, y el cutoff del sub como palanca
macro de la evolución (abre de intro a peak).

Y en que **nada suena a máquina**: los toms y el rim usan **round-robin de
tomas** sintetizadas (cada golpe es un sample distinto, como manos reales),
el **swing deriva ±0.012 en ciclos de 24 compases** (el groove se tensa y
afloja como un percusionista), el humanize de timing/velocidad/pan es ancho,
y hay ghosts de tom sueltos que caen donde quieren.

Es un **DJ-tool lineal**: no hay drops ni bajadas abruptas. La energía
evoluciona por automatización continua, envolventes de energía, LFOs lentos
y sustracción/adición de capas.

### Estructura

| Sección | Barras | Qué pasa |
|---|---|---|
| Intro | 0–23 | la caverna desde el segundo 0 (drone+pad+textura); kick en el 8, teaser del sub filtrado en el 16 |
| Groove | 24–55 | sub completo; toms en el 32, rim y snap en el 40, stab en el 48 |
| Main | 56–87 | sección profunda: cutoff del sub abre, stab más presente, gong, fills de toms |
| Break | 88–111 | sin kick 88–99: pad/drone/textura florecen, stab lava cada compás, el breath llama, gong grande; rebuild 100–111 |
| Peak | 112–159 | máxima energía; entra el **ride**, ghosts de hats, sub abierto, stab cada 2 compases |
| Late groove | 160–175 | el brillo y el stab se recogen pero el **rim vuelve** con más delay — el descenso trae algo, no solo resta |
| Outro | 176–199 | decae por capas, con high-pass de salida DJ-friendly |

Las transiciones funcionan por **succión**: un low-pass resonante en el
máster se cierra durante ~2.4 compases al final de la sección y abre de golpe
en el downbeat, apoyado por un compás sin kick y un sub-boom de impacto.

### Las 18 capas (= stems)

| Stem | Rol |
|---|---|
| `kick` | 4x4 de cueva afinado a G1, redondo con la cola contenida (el roll lo ponen rumble y sub); ghost kicks (step 7 impares, 15 cada 4) = bounce |
| `rumble` | cola reverberada del kick filtrada <220 Hz — el retumbe firma del género |
| `sub` | EL pulso: una nota (G1) en offbeats con acentos y sombra b2 deslizada; LP resonante con envolvente por nota + cutoff macro |
| `toms` | el groove tribal: tom G2 + tom D3 + conga-slap con round-robin de 3 tomas, patrones A/B (+C en el peak), ghosts sueltos, LP-wobble, fills de subida al fin de frase |
| `rim` | rimshot de madera (2 tomas round-robin), seco y cerca, HP respirando + slap-delay dub duckeado; vuelve en el late-groove |
| `hats` | cerrada en offbeats + abierta pre-downbeat (y su reverso succionando cada 4 compases del peak); canal R decorrelado |
| `ride` | lavado de ceniza en corcheas con phaser automatizado, exclusivo del peak |
| `snap` | backbeat oscuro (band-pass 500–1900 + thump) con sala honda, main y peak |
| `shaker` | 16ths con acento en las contras, filtro móvil que respira |
| `stab` | dub-stab Gm con sombra b2 (squares redondeadas band-pass) en chorus lento L/R, delay punteado con throws por frase y pre-ecos reversos |
| `gong` | tam-tam cavernoso (parciales inarmónicos con batido lento) cada ~16 compases, delay dub largo + reverb enorme |
| `breath` | exhalación fantasma: ruido formántico de susurro a tres colores, chorus + throws de delay + reverb larga |
| `pad` | voces huecas Gm (G2-Bb2-D3) con deriva de afinado por voz + phaser lento — la cripta |
| `drone` | zumbido de caverna en G1 (raíz+octava) con wow de cinta y LP que florece en el break |
| `texture` | "lluvia en nave": ruido oscuro con band-pass móvil que vocaliza — pega las capas |
| `fx` | texturas puntuales: sweeps de ruido, swells reversos, crashes sordos |
| `rev` | reverb send compartida (la cueva común) — cohesión y profundidad |
| `impact` | sub-boom que subraya las aperturas de sección |

## Cómo funciona el código

### `render.py` — la partitura

Script lineal que se lee de arriba a abajo, en cinco fases:

1. **Rejilla** — constantes de tempo (145 BPM, 16 steps/compás, swing 0.56) y
   helpers para convertir steps/compases a samples y colocar audio en buffers.
2. **Síntesis** — cada voz se genera con numpy: osciladores, ruido, envolventes
   exponenciales y filtros. Nada de audio externo.
3. **Secuenciación** — un bucle por compás coloca cada golpe según la sección
   activa, con humanize de timing/velocidad/pan, round-robin de tomas, swing
   que respira, frases alternas cada 8 compases y fills de toms al final de
   frase. El sub se genera aparte como señal continua (curvas de
   freq/amp/filtro por sample, para slides y envolvente de filtro por nota).
4. **Buses** — cada capa recibe su cadena: EQ  ->  saturación  ->  compresión  ->
   sidechain al kick  ->  delay/reverb. Las envolventes de energía
   (`breakpoint_env`) y los filtros móviles hacen que cada capa entre, salga
   y evolucione por sección de forma continua.
5. **Máster** — DJ filter de entrada/salida  ->  EQ  ->  apertura progresiva de aire
    ->  glue compressor  ->  saturación de cinta  ->  LP "suck" de transiciones  ->
   graves a mono (<150 Hz)  ->  width M/S  ->  arco macro de nivel  ->  push +6.5 dB  ->
   soft clip  ->  fades  ->  limiter brickwall (−0.8 dBFS)  ->  dither TPDF.

### `engine/` — el motor DSP reutilizable

Independiente del tema; solo depende de numpy/scipy.

| Módulo | Contenido |
|---|---|
| `fx.py` | dinámica (compresor, sidechain, limiter con lookahead, soft-clip), EQ (biquads + Butterworth), saturación, reverb plate, delay ping-pong, mod-delay (chorus/wow), mono low-end, stereo width, fades |
| `atmos.py` | generadores de atmósfera: drone armónico, swell reverso, sweep de ruido, crash, shaker |
| `automation.py` | movimiento: `breakpoint_env` (curvas por compás), `lfo` tempo-sincronizado, filtros móviles (`moving_lowpass` / `moving_highpass`), `phaser` automatizado, `reverb_space` (pre-delay + damping), humanize y RNG sembrado |

## Licencia

Código y música bajo **CC BY-NC 4.0** — uso no comercial con atribución
(ver `LICENSE`).
