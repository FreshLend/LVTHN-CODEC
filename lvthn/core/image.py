import os
import struct
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import zstandard as zstd
from numba import njit, prange

IMG_HDR_FMT = '<IIBBf'
IMG_HDR_SIZE = struct.calcsize(IMG_HDR_FMT)

_NUM_CPU = os.cpu_count() or 4

FLAG_ALPHA = 1 << 0
FLAG_INT16_RLE = 1 << 1

ZIGZAG_ORDER = np.array([
    0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
], dtype=np.int32)

_C_8X8 = np.zeros((8, 8), dtype=np.float32)
for _k in range(8):
    _a = np.sqrt(1.0 / 8.0) if _k == 0 else np.sqrt(2.0 / 8.0)
    for _n in range(8):
        _C_8X8[_k, _n] = _a * np.cos(np.pi * (2 * _n + 1) * _k / 16.0)


@njit(cache=True, fastmath=True, boundscheck=False, nogil=True, parallel=True)
def _rgb_to_oklab_njit(rgb, out):
    h, w = rgb.shape[0], rgb.shape[1]
    for i in prange(h):
        for j in range(w):
            r = rgb[i, j, 0] / 255.0
            g = rgb[i, j, 1] / 255.0
            b = rgb[i, j, 2] / 255.0
            if r > 0.04045:
                r = ((r + 0.055) / 1.055) ** 2.4
            else:
                r = r / 12.92
            if g > 0.04045:
                g = ((g + 0.055) / 1.055) ** 2.4
            else:
                g = g / 12.92
            if b > 0.04045:
                b = ((b + 0.055) / 1.055) ** 2.4
            else:
                b = b / 12.92
            l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
            m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
            s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
            l = (l ** (1.0 / 3.0)) if l > 0 else 0.0
            m = (m ** (1.0 / 3.0)) if m > 0 else 0.0
            s = (s ** (1.0 / 3.0)) if s > 0 else 0.0
            L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s
            a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
            bb = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
            out[i, j, 0] = L * 255.0
            out[i, j, 1] = a * 255.0 + 128.0
            out[i, j, 2] = bb * 255.0 + 128.0


@njit(cache=True, fastmath=True, boundscheck=False, nogil=True, parallel=True)
def _oklab_to_rgb_njit(oklab, out):
    h, w = oklab.shape[0], oklab.shape[1]
    for i in prange(h):
        for j in range(w):
            L = oklab[i, j, 0] / 255.0
            a = (oklab[i, j, 1] - 128.0) / 255.0
            b = (oklab[i, j, 2] - 128.0) / 255.0
            l = L + 0.3963377774 * a + 0.2158037573 * b
            m = L - 0.1055613458 * a - 0.0638541728 * b
            s = L - 0.0894841775 * a - 1.2914855480 * b
            l = l ** 3
            m = m ** 3
            s = s ** 3
            r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
            g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
            bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
            if r > 0.0031308:
                r = 1.055 * (r ** (1.0 / 2.4)) - 0.055
            else:
                r = 12.92 * r
            if g > 0.0031308:
                g = 1.055 * (g ** (1.0 / 2.4)) - 0.055
            else:
                g = 12.92 * g
            if bb > 0.0031308:
                bb = 1.055 * (bb ** (1.0 / 2.4)) - 0.055
            else:
                bb = 12.92 * bb
            r = max(0.0, min(1.0, r))
            g = max(0.0, min(1.0, g))
            bb = max(0.0, min(1.0, bb))
            out[i, j, 0] = int(r * 255.0 + 0.5)
            out[i, j, 1] = int(g * 255.0 + 0.5)
            out[i, j, 2] = int(bb * 255.0 + 0.5)


@njit(cache=True, boundscheck=False, nogil=True)
def _rle_encode_all_fast(zigzag_blocks, out_pairs_1d):
    n_blocks = zigzag_blocks.shape[0]
    out_pos = 0
    for b in range(n_blocks):
        run = 0
        for k in range(64):
            v = zigzag_blocks[b, k]
            if v == 0:
                run += 1
            else:
                while run > 15:
                    out_pairs_1d[out_pos] = 15
                    out_pairs_1d[out_pos + 1] = 0
                    out_pos += 2
                    run -= 16
                out_pairs_1d[out_pos] = run
                out_pairs_1d[out_pos + 1] = v
                out_pos += 2
                run = 0
        if run > 0:
            out_pairs_1d[out_pos] = 0
            out_pairs_1d[out_pos + 1] = 0
            out_pos += 2
    return out_pos // 2


@njit(cache=True, boundscheck=False, nogil=True)
def _rle_decode_all_fast(pairs_1d, n_pairs, n_blocks, out_zigzag):
    idx = 0
    for b in range(n_blocks):
        for k in range(64):
            out_zigzag[b, k] = 0
        pos = 0
        while pos < 64 and idx < n_pairs:
            run = pairs_1d[idx * 2]
            val = pairs_1d[idx * 2 + 1]
            idx += 1
            if val == 0 and run == 0:
                break
            pos += run
            if pos < 64:
                out_zigzag[b, pos] = val
                pos += 1
    return idx


@njit(cache=True, fastmath=True, boundscheck=False, nogil=True)
def _dct_quant_zigzag_njit(channel, C, q_matrix, zigzag_order, zigzag_out):
    h, w = channel.shape
    hb = (h + 7) // 8
    wb = (w + 7) // 8
    nb = hb * wb
    for bi in range(nb):
        bh = bi // wb
        bw = bi % wb
        blk = np.empty((8, 8), dtype=np.float32)
        for i in range(8):
            ii = bh * 8 + i
            if ii >= h:
                ii = h - 1
            for j in range(8):
                jj = bw * 8 + j
                if jj >= w:
                    jj = w - 1
                blk[i, j] = channel[ii, jj]
        tmp = np.empty((8, 8), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                s = np.float32(0.0)
                for k in range(8):
                    s += C[i, k] * blk[k, j]
                tmp[i, j] = s
        dct = np.empty((8, 8), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                s = np.float32(0.0)
                for k in range(8):
                    s += tmp[i, k] * C[j, k]
                dct[i, j] = s
        for k in range(64):
            pos = zigzag_order[k]
            i = pos // 8
            j = pos % 8
            zigzag_out[bi, k] = int(np.rint(dct[i, j] / q_matrix[i, j]))


@njit(cache=True, fastmath=True, boundscheck=False, nogil=True)
def _dezigzag_idct_njit(zigzag_blocks, q_matrix, C, zigzag_order, out):
    h, w = out.shape
    hb = (h + 7) // 8
    wb = (w + 7) // 8
    nb = hb * wb
    for bi in range(nb):
        bh = bi // wb
        bw = bi % wb
        blk = np.zeros((8, 8), dtype=np.float32)
        for k in range(64):
            pos = zigzag_order[k]
            i = pos // 8
            j = pos % 8
            blk[i, j] = zigzag_blocks[bi, k] * q_matrix[i, j]
        tmp = np.empty((8, 8), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                s = np.float32(0.0)
                for k in range(8):
                    s += C[k, i] * blk[k, j]
                tmp[i, j] = s
        for i in range(8):
            ii = bh * 8 + i
            if ii >= h:
                continue
            for j in range(8):
                jj = bw * 8 + j
                if jj >= w:
                    continue
                s = np.float32(0.0)
                for k in range(8):
                    s += tmp[i, k] * C[k, j]
                out[ii, jj] = s


@njit(cache=True, boundscheck=False, nogil=True)
def _pack_int16_njit(pairs_1d, n_pairs, out):
    for i in range(n_pairs):
        run = pairs_1d[i * 2]
        val = pairs_1d[i * 2 + 1]
        out[i * 3] = run & 0xFF
        out[i * 3 + 1] = val & 0xFF
        out[i * 3 + 2] = (val >> 8) & 0xFF


@njit(cache=True, boundscheck=False, nogil=True)
def _unpack_int16_njit(packed, n_pairs, out_pairs):
    for i in range(n_pairs):
        run = packed[i * 3]
        lo = packed[i * 3 + 1]
        hi = packed[i * 3 + 2]
        val = lo | (hi << 8)
        if val >= 32768:
            val -= 65536
        out_pairs[i * 2] = run
        out_pairs[i * 2 + 1] = val


def _get_q_matrix(quality, is_chroma):
    q = np.clip(quality, 1, 100)
    step = (100 - q) / 2.0 + 1.0
    if is_chroma:
        step *= 1.5
    return np.ones((8, 8), dtype=np.float32) * step


def _encode_channel_fast(channel, q_matrix):
    h, w = channel.shape
    hb = (h + 7) // 8
    wb = (w + 7) // 8
    nb = hb * wb
    zigzag = np.zeros((nb, 64), dtype=np.int32)
    _dct_quant_zigzag_njit(channel, _C_8X8, q_matrix, ZIGZAG_ORDER, zigzag)
    out_pairs = np.zeros(nb * 65 * 2, dtype=np.int32)
    n_pairs = _rle_encode_all_fast(zigzag, out_pairs)
    packed = np.empty(n_pairs * 3, dtype=np.uint8)
    _pack_int16_njit(out_pairs, n_pairs, packed)
    return packed.tobytes()


def _decode_channel_fast(packed_bytes, q_matrix, h, w, use_int16):
    if use_int16:
        packed = np.frombuffer(packed_bytes, dtype=np.uint8)
        n_pairs = len(packed) // 3
        pairs_1d = np.empty(n_pairs * 2, dtype=np.int32)
        _unpack_int16_njit(packed, n_pairs, pairs_1d)
    else:
        pairs_1d = np.frombuffer(packed_bytes, dtype='<i4').astype(np.int32)
        n_pairs = len(pairs_1d) // 2
    hb = (h + 7) // 8
    wb = (w + 7) // 8
    nb = hb * wb
    zigzag = np.zeros((nb, 64), dtype=np.int32)
    _rle_decode_all_fast(pairs_1d, n_pairs, nb, zigzag)
    out = np.zeros((h, w), dtype=np.float32)
    _dezigzag_idct_njit(zigzag, q_matrix, _C_8X8, ZIGZAG_ORDER, out)
    return out


def encode_image(pixels, width, height, channels=3, quality=90, level=19):
    has_alpha = (channels == 4)
    flags = FLAG_INT16_RLE | (FLAG_ALPHA if has_alpha else 0)

    if has_alpha:
        rgb_pixels = pixels[:, :, :3]
        alpha_channel = pixels[:, :, 3].astype(np.float32)
    else:
        rgb_pixels = pixels

    oklab = np.zeros((height, width, 3), dtype=np.float32)
    _rgb_to_oklab_njit(rgb_pixels, oklab)

    q_luma = _get_q_matrix(quality, is_chroma=False)
    q_chroma = _get_q_matrix(quality, is_chroma=True)
    q_alpha = _get_q_matrix(quality, is_chroma=False)

    channels_data = [
        (np.ascontiguousarray(oklab[:, :, 0]), q_luma),
        (np.ascontiguousarray(oklab[:, :, 1]), q_chroma),
        (np.ascontiguousarray(oklab[:, :, 2]), q_chroma),
    ]
    if has_alpha:
        channels_data.append((np.ascontiguousarray(alpha_channel), q_alpha))

    num_ch = len(channels_data)
    threads_per_ch = max(1, _NUM_CPU // num_ch)

    def encode_one(task):
        ch_data, qm = task
        packed_rle = _encode_channel_fast(ch_data, qm)
        cctx = zstd.ZstdCompressor(level=level, threads=threads_per_ch, write_content_size=True)
        return cctx.compress(packed_rle)

    with ThreadPoolExecutor(max_workers=num_ch) as ex:
        compressed_channels = list(ex.map(encode_one, channels_data))

    comp_lens = [len(c) for c in compressed_channels]
    header = struct.pack(IMG_HDR_FMT, width, height, channels, flags, float(quality))
    meta = struct.pack(f'<{channels}I', *comp_lens)
    body = b''.join(compressed_channels)
    return header + meta + body


def decode_image(payload):
    if len(payload) < IMG_HDR_SIZE:
        raise ValueError("Image payload too small")

    width, height, channels, flags, quality = struct.unpack_from(IMG_HDR_FMT, payload, 0)
    has_alpha = bool(flags & FLAG_ALPHA)
    use_int16 = bool(flags & FLAG_INT16_RLE)

    pos = IMG_HDR_SIZE
    comp_lens = struct.unpack_from(f'<{channels}I', payload, pos)
    pos += channels * 4

    compressed_channels = []
    for l in comp_lens:
        compressed_channels.append(payload[pos:pos + l])
        pos += l

    q_luma = _get_q_matrix(quality, is_chroma=False)
    q_chroma = _get_q_matrix(quality, is_chroma=True)
    q_alpha = _get_q_matrix(quality, is_chroma=False)

    q_matrices = [q_luma, q_chroma, q_chroma]
    if has_alpha:
        q_matrices.append(q_alpha)

    hb = (height + 7) // 8
    wb = (width + 7) // 8
    max_out = hb * wb * 520 + 1024

    tasks = list(zip(compressed_channels, q_matrices))

    def decode_one(task):
        comp, qm = task
        dctx = zstd.ZstdDecompressor()
        packed = dctx.decompress(comp, max_output_size=max_out)
        return _decode_channel_fast(packed, qm, height, width, use_int16)

    with ThreadPoolExecutor(max_workers=channels) as ex:
        channel_results = list(ex.map(decode_one, tasks))

    oklab = np.zeros((height, width, 3), dtype=np.float32)
    oklab[:, :, 0] = channel_results[0]
    oklab[:, :, 1] = channel_results[1]
    oklab[:, :, 2] = channel_results[2]

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    _oklab_to_rgb_njit(oklab, rgb)

    if has_alpha:
        alpha = np.clip(channel_results[3], 0, 255).astype(np.uint8)
        return np.dstack((rgb, alpha))

    return rgb


def get_image_info(payload):
    width, height, channels, flags, quality = struct.unpack_from(IMG_HDR_FMT, payload, 0)
    return dict(
        width=width, height=height, channels=channels,
        has_alpha=bool(flags & FLAG_ALPHA), quality=quality
    )