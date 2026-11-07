import struct
import numpy as np
from numba import njit

FRAME_SIZE = 1024
HOP = FRAME_SIZE // 2
N_COEFFS = FRAME_SIZE // 2

DEFAULT_DELTA = 0.18
DEFAULT_TARGET_SNR_DB = 24.0
DEFAULT_DELTA_CURVE = np.array([1.0, 1.0, 1.1, 1.3, 1.6, 2.0, 2.5, 3.0], dtype=np.float32)
MIN_BAND_SIZE = 3
SCALE_MIN = -32
SCALE_MAX = 31
SCALE_BITS = 6
ATH_OFFSET_DB = 96.0

AUDIO_HDR_FMT = '<IBHQf8ffBI'
AUDIO_HDR_SIZE = struct.calcsize(AUDIO_HDR_FMT)

_WINDOW = np.sin(np.pi * (np.arange(FRAME_SIZE) + 0.5) / FRAME_SIZE).astype(np.float64)
_n = np.arange(FRAME_SIZE, dtype=np.float64)[:, None]
_k = np.arange(N_COEFFS, dtype=np.float64)[None, :]
_MDCT_MATRIX = np.cos(np.pi / N_COEFFS * (_n + 0.5 + N_COEFFS / 2) * (_k + 0.5))
_IMDCT_MATRIX = (_MDCT_MATRIX.T * (2.0 / N_COEFFS)).astype(np.float64)

@njit(cache=True, inline='always')
def _get_bit(buf, pos):
    return (buf[pos >> 3] >> (7 - (pos & 7))) & 1

@njit(cache=True, inline='always')
def _set_bit(buf, pos, val):
    if val:
        buf[pos >> 3] |= (1 << (7 - (pos & 7)))

@njit(cache=True)
def _encode_frame_channel(quantized, band_starts, band_ends, scales, out_buf, bit_pos):
    n_bands = len(band_starts)
    n_coeffs = len(quantized)
    last_nz = 0
    for i in range(n_coeffs):
        if quantized[i] != 0:
            last_nz = i + 1
    for i in range(9, -1, -1):
        _set_bit(out_buf, bit_pos, (last_nz >> i) & 1)
        bit_pos += 1
    if last_nz == 0:
        return bit_pos

    band_present = np.zeros(n_bands, dtype=np.int32)
    band_rice_k = np.zeros(n_bands, dtype=np.int32)
    for b in range(n_bands):
        s = band_starts[b]
        e = band_ends[b]
        if s >= last_nz:
            break
        be = min(e, last_nz)
        any_nz = False
        total_zz = 0
        for i in range(s, be):
            v = quantized[i]
            if v != 0:
                any_nz = True
            total_zz += (v << 1) if v >= 0 else ((-v) << 1) - 1
        if any_nz:
            band_present[b] = 1
            mean = total_zz / (be - s)
            k = 0
            m = mean
            while m > 2.0 and k < 15:
                m /= 2.0
                k += 1
            band_rice_k[b] = k

    for b in range(n_bands):
        s = band_starts[b]
        if s >= last_nz:
            break
        _set_bit(out_buf, bit_pos, band_present[b])
        bit_pos += 1

    for b in range(n_bands):
        s = band_starts[b]
        e = band_ends[b]
        if s >= last_nz:
            break
        if band_present[b] == 0:
            continue
        scale_enc = scales[b] - SCALE_MIN
        for i in range(SCALE_BITS - 1, -1, -1):
            _set_bit(out_buf, bit_pos, (scale_enc >> i) & 1)
            bit_pos += 1
        k = band_rice_k[b]
        for i in range(3, -1, -1):
            _set_bit(out_buf, bit_pos, (k >> i) & 1)
            bit_pos += 1
        be = min(e, last_nz)
        for i in range(s, be):
            v = quantized[i]
            zz = (v << 1) if v >= 0 else ((-v) << 1) - 1
            q = zz >> k
            r = zz - (q << k)
            for _ in range(q):
                bit_pos += 1
            _set_bit(out_buf, bit_pos, 1)
            bit_pos += 1
            for i2 in range(k - 1, -1, -1):
                _set_bit(out_buf, bit_pos, (r >> i2) & 1)
                bit_pos += 1
    return bit_pos

@njit(cache=True)
def _decode_frame_channel(in_buf, bit_pos, quantized, scales, band_starts, band_ends):
    n_bands = len(band_starts)
    n_coeffs = len(quantized)
    for i in range(n_coeffs):
        quantized[i] = 0
    for i in range(n_bands):
        scales[i] = 0

    last_nz = 0
    for i in range(10):
        last_nz = (last_nz << 1) | _get_bit(in_buf, bit_pos)
        bit_pos += 1
    if last_nz == 0:
        return bit_pos

    present = np.zeros(n_bands, dtype=np.int32)
    for b in range(n_bands):
        s = band_starts[b]
        if s >= last_nz:
            break
        present[b] = _get_bit(in_buf, bit_pos)
        bit_pos += 1

    for b in range(n_bands):
        s = band_starts[b]
        e = band_ends[b]
        if s >= last_nz:
            break
        if present[b] == 0:
            continue
        scale_enc = 0
        for i in range(SCALE_BITS):
            scale_enc = (scale_enc << 1) | _get_bit(in_buf, bit_pos)
            bit_pos += 1
        scales[b] = scale_enc + SCALE_MIN
        k = 0
        for i in range(4):
            k = (k << 1) | _get_bit(in_buf, bit_pos)
            bit_pos += 1
        be = min(e, last_nz)
        for i in range(s, be):
            q = 0
            while _get_bit(in_buf, bit_pos) == 0:
                q += 1
                bit_pos += 1
            bit_pos += 1
            r = 0
            for i2 in range(k):
                r = (r << 1) | _get_bit(in_buf, bit_pos)
                bit_pos += 1
            zz = (q << k) | r
            quantized[i] = (zz >> 1) if (zz & 1) == 0 else -((zz + 1) >> 1)
    return bit_pos

@njit(cache=True)
def _encode_body_njit(q_flat, band_starts, band_ends, scales_flat, out_buf):
    bit_pos = 0
    for fc in range(q_flat.shape[0]):
        bit_pos = _encode_frame_channel(q_flat[fc], band_starts, band_ends, scales_flat[fc], out_buf, bit_pos)
    return bit_pos

@njit(cache=True)
def _decode_body_njit(body_arr, q_flat, scales_flat, band_starts, band_ends):
    bit_pos = 0
    for fc in range(q_flat.shape[0]):
        bit_pos = _decode_frame_channel(body_arr, bit_pos, q_flat[fc], scales_flat[fc], band_starts, band_ends)
    return bit_pos

def _make_bark_bands(sr, n_coeffs, min_size=MIN_BAND_SIZE):
    bark_hz = np.array([0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, sr / 2], dtype=np.float64)
    bin_centers = (np.arange(n_coeffs) + 0.5) * (sr / (2 * n_coeffs))
    raw = []
    for i in range(len(bark_hz) - 1):
        start = int(np.searchsorted(bin_centers, bark_hz[i], side='left'))
        end = int(np.searchsorted(bin_centers, bark_hz[i + 1], side='left'))
        if end > start:
            raw.append((start, end))
    if raw:
        if raw[0][0] > 0: raw[0] = (0, raw[0][1])
        if raw[-1][1] < n_coeffs: raw[-1] = (raw[-1][0], n_coeffs)
    merged = []
    acc = None
    for s, e in raw:
        if acc is None:
            acc = [s, e]
        elif acc[1] - acc[0] < min_size:
            acc[1] = e
        else:
            merged.append((acc[0], acc[1]))
            acc = [s, e]
    if acc is not None:
        merged.append((acc[0], acc[1]))
    return merged

def _ath_thresholds(sr, n_coeffs, offset_db=ATH_OFFSET_DB):
    freqs = np.maximum(np.arange(n_coeffs) * sr / FRAME_SIZE, 20.0)
    f_khz = freqs / 1000.0
    ath_db = np.minimum(3.64 * f_khz ** (-0.8) - 6.5 * np.exp(-0.6 * (f_khz - 3.3) ** 2) + 1e-3 * f_khz ** 4, 75.0)
    return (10.0 ** ((ath_db - offset_db) / 10.0)).astype(np.float64)

def _encode_body(coeffs, sr, delta, curve_bands, band_starts, band_ends, target_snr_db):
    n_frames, _, ch = coeffs.shape
    n_bands = len(band_starts)
    
    ath_band = np.array([np.mean(_ath_thresholds(sr, N_COEFFS)[s:e]) for s, e in zip(band_starts, band_ends)])
    
    energies = np.zeros((n_frames, n_bands, ch), dtype=np.float64)
    for b in range(n_bands):
        energies[:, b, :] = np.mean(coeffs[:, band_starts[b]:band_ends[b], :] ** 2, axis=1)

    mask_smooth = energies.copy()
    for shift in (1, 2, 3):
        mask_smooth = np.maximum(mask_smooth, np.roll(energies, shift, axis=0))
        mask_smooth = np.maximum(mask_smooth, np.roll(energies, -shift, axis=0))

    spread = np.zeros_like(mask_smooth)
    for b in range(n_bands):
        acc = np.zeros((n_frames, ch), dtype=np.float64)
        for db in range(-4, 5):
            b2 = b + db
            if 0 <= b2 < n_bands:
                acc += mask_smooth[:, b2, :] * (1.0 - abs(db) / 5.0)
        spread[:, b, :] = acc

    mask = spread + ath_band[None, :, None]
    base_deltas = delta * curve_bands.astype(np.float64)
    delta_max = np.sqrt(mask / 20.0)
    delta_floor = base_deltas[None, :, None] / np.geomspace(4.0, 0.5, n_bands)[None, :, None]
    deltas = np.maximum(delta_floor, np.minimum(base_deltas[None, :, None], delta_max))

    with np.errstate(divide='ignore', invalid='ignore'):
        scale = np.round(2 * np.log2(deltas / base_deltas[None, :, None])).astype(np.int32)
    np.clip(scale, SCALE_MIN, SCALE_MAX, out=scale)
    deltas = base_deltas[None, :, None] * (2.0 ** (scale / 2.0))

    quantized = np.zeros_like(coeffs, dtype=np.int32)
    for b in range(n_bands):
        d = deltas[:, b, :]
        quantized[:, band_starts[b]:band_ends[b], :] = np.round(coeffs[:, band_starts[b]:band_ends[b], :] / d[:, None, :])
    np.clip(quantized, -32767, 32767, out=quantized)

    q_flat = np.ascontiguousarray(quantized.transpose(0, 2, 1).reshape(n_frames * ch, N_COEFFS))
    scales_flat = np.ascontiguousarray(scale.transpose(0, 2, 1).reshape(n_frames * ch, n_bands))
    
    out_buf = np.zeros(n_frames * ch * 3072, dtype=np.uint8)
    bit_pos = _encode_body_njit(q_flat, band_starts, band_ends, scales_flat, out_buf)
    return out_buf[:(bit_pos + 7) // 8].tobytes(), bit_pos

def encode_audio(pcm, sr, ch, delta, target_snr_db):
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    n_samples = pcm.shape[0]
    n_frames = (n_samples + HOP - 1) // HOP + 1
    
    pcm_padded = np.zeros(((n_frames + 1) * HOP, ch), dtype=np.float64)
    pcm_padded[HOP:HOP + n_samples] = pcm

    idx = np.arange(n_frames)[:, None] * HOP + np.arange(FRAME_SIZE)[None, :]
    coeffs_L = (pcm_padded[idx, 0] * _WINDOW[None, :]) @ _MDCT_MATRIX

    if ch == 2:
        coeffs_R = (pcm_padded[idx, 1] * _WINDOW[None, :]) @ _MDCT_MATRIX
        coeffs_lr = np.stack([coeffs_L, coeffs_R], axis=2)
        coeffs_ms = np.stack([(coeffs_L + coeffs_R) * 0.5, (coeffs_L - coeffs_R) * 0.5], axis=2)
    else:
        coeffs_lr, coeffs_ms = coeffs_L[:, :, None], None

    bark_bands = _make_bark_bands(sr, N_COEFFS)
    band_starts = np.array([s for s, _ in bark_bands], dtype=np.int32)
    band_ends = np.array([e for _, e in bark_bands], dtype=np.int32)
    curve_bands = np.interp(np.linspace(0, 1, len(bark_bands)), np.linspace(0, 1, 8), DEFAULT_DELTA_CURVE).astype(np.float32)

    body_lr, bits_lr = _encode_body(coeffs_lr, sr, delta, curve_bands, band_starts, band_ends, target_snr_db)
    
    if ch == 2:
        body_ms, bits_ms = _encode_body(coeffs_ms, sr, delta, curve_bands, band_starts, band_ends, target_snr_db)
        use_ms, body = (1, body_ms) if bits_ms < bits_lr else (0, body_lr)
    else:
        use_ms, body = 0, body_lr

    hdr = struct.pack(AUDIO_HDR_FMT, sr, ch, FRAME_SIZE, n_samples, float(delta), *[float(x) for x in DEFAULT_DELTA_CURVE], float(target_snr_db), use_ms, len(body))
    return hdr + body

def decode_audio(payload):
    (sr, ch, fs, n_samples, delta, c0, c1, c2, c3, c4, c5, c6, c7, target_snr_db, use_ms, body_len) = struct.unpack_from(AUDIO_HDR_FMT, payload, 0)
    
    body = payload[AUDIO_HDR_SIZE:AUDIO_HDR_SIZE + body_len]
    n_frames = (n_samples + HOP - 1) // HOP + 1
    
    bark_bands = _make_bark_bands(sr, N_COEFFS)
    band_starts = np.array([s for s, _ in bark_bands], dtype=np.int32)
    band_ends = np.array([e for _, e in bark_bands], dtype=np.int32)
    
    curve_orig = np.array([c0, c1, c2, c3, c4, c5, c6, c7], dtype=np.float32)
    curve_bands = np.interp(np.linspace(0, 1, len(bark_bands)), np.linspace(0, 1, 8), curve_orig).astype(np.float32)

    q_flat = np.zeros((n_frames * ch, N_COEFFS), dtype=np.int32)
    scales_flat = np.zeros((n_frames * ch, len(bark_bands)), dtype=np.int32)
    _decode_body_njit(np.frombuffer(body, dtype=np.uint8), q_flat, scales_flat, band_starts, band_ends)

    coeffs = np.zeros((n_frames, N_COEFFS, ch), dtype=np.float64)
    for b in range(len(bark_bands)):
        d = delta * curve_bands[b] * (2.0 ** (scales_flat[:, b].reshape(n_frames, ch) / 2.0))
        q_band = q_flat[:, band_starts[b]:band_ends[b]].reshape(n_frames, ch, band_ends[b] - band_starts[b]).transpose(0, 2, 1)
        coeffs[:, band_starts[b]:band_ends[b], :] = q_band * d[:, None, :]

    frames = (coeffs.transpose(0, 2, 1).reshape(-1, N_COEFFS) @ _IMDCT_MATRIX).reshape(n_frames, ch, FRAME_SIZE).transpose(0, 2, 1) * _WINDOW[None, :, None]
    
    pcm_enc = np.zeros(((n_frames + 1) * HOP, ch), dtype=np.float64)
    for f in range(n_frames):
        pcm_enc[f * HOP : f * HOP + FRAME_SIZE] += frames[f]

    pcm_padded = np.stack([pcm_enc[:, 0] + pcm_enc[:, 1], pcm_enc[:, 0] - pcm_enc[:, 1]], axis=1) if use_ms and ch == 2 else pcm_enc
    return np.clip(pcm_padded[HOP:HOP + n_samples], -1.0, 1.0), sr, ch, n_samples

def get_audio_info(payload):
    (sr, ch, _, n_samples, delta, _, _, _, _, _, _, _, _, target_snr_db, use_ms, body_len) = struct.unpack_from(AUDIO_HDR_FMT, payload, 0)
    return dict(sr=sr, ch=ch, n_samples=n_samples, delta=delta, target_snr_db=target_snr_db, use_ms=use_ms, body_len=body_len)