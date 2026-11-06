import struct
import numpy as np
from scipy.fft import dctn, idctn
import zstandard as zstd
from numba import njit

IMG_HDR_FMT = '<IIBBf'
IMG_HDR_SIZE = struct.calcsize(IMG_HDR_FMT)

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

@njit(cache=True)
def _rgb_to_oklab_njit(rgb, out):
    h, w = rgb.shape[0], rgb.shape[1]
    for i in range(h):
        for j in range(w):
            r = rgb[i, j, 0] / 255.0
            g = rgb[i, j, 1] / 255.0
            b = rgb[i, j, 2] / 255.0
            
            if r > 0.04045: r = ((r + 0.055) / 1.055) ** 2.4
            else: r = r / 12.92
            if g > 0.04045: g = ((g + 0.055) / 1.055) ** 2.4
            else: g = g / 12.92
            if b > 0.04045: b = ((b + 0.055) / 1.055) ** 2.4
            else: b = b / 12.92
            
            l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
            m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
            s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
            
            l = (l ** (1.0/3.0)) if l > 0 else 0.0
            m = (m ** (1.0/3.0)) if m > 0 else 0.0
            s = (s ** (1.0/3.0)) if s > 0 else 0.0
            
            L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s
            a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
            bb = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
            
            out[i, j, 0] = L * 255.0
            out[i, j, 1] = a * 255.0 + 128.0
            out[i, j, 2] = bb * 255.0 + 128.0

@njit(cache=True)
def _oklab_to_rgb_njit(oklab, out):
    h, w = oklab.shape[0], oklab.shape[1]
    for i in range(h):
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
            
            if r > 0.0031308: r = 1.055 * (r ** (1.0/2.4)) - 0.055
            else: r = 12.92 * r
            if g > 0.0031308: g = 1.055 * (g ** (1.0/2.4)) - 0.055
            else: g = 12.92 * g
            if bb > 0.0031308: bb = 1.055 * (bb ** (1.0/2.4)) - 0.055
            else: bb = 12.92 * bb
            
            r = max(0.0, min(1.0, r))
            g = max(0.0, min(1.0, g))
            bb = max(0.0, min(1.0, bb))
            
            out[i, j, 0] = int(r * 255.0 + 0.5)
            out[i, j, 1] = int(g * 255.0 + 0.5)
            out[i, j, 2] = int(bb * 255.0 + 0.5)

@njit(cache=True)
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
                    out_pairs_1d[out_pos+1] = 0
                    out_pos += 2
                    run -= 16
                out_pairs_1d[out_pos] = run
                out_pairs_1d[out_pos+1] = v
                out_pos += 2
                run = 0
        if run > 0:
            out_pairs_1d[out_pos] = 0
            out_pairs_1d[out_pos+1] = 0
            out_pos += 2
    return out_pos // 2

@njit(cache=True)
def _rle_decode_all_fast(pairs_1d, n_pairs, n_blocks, out_zigzag):
    idx = 0
    for b in range(n_blocks):
        for k in range(64):
            out_zigzag[b, k] = 0
        pos = 0
        while pos < 64 and idx < n_pairs:
            run = pairs_1d[idx*2]
            val = pairs_1d[idx*2+1]
            idx += 1
            if val == 0 and run == 0:
                break
            pos += run
            if pos < 64:
                out_zigzag[b, pos] = val
                pos += 1
    return idx

def _get_q_matrix(quality, is_chroma):
    q = np.clip(quality, 1, 100)
    step = (100 - q) / 2.0 + 1.0
    if is_chroma:
        step *= 1.5
    return np.ones((8, 8), dtype=np.float32) * step

def _encode_channel_fast(channel, q_matrix):
    h, w = channel.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='edge') if (pad_h or pad_w) else channel
        
    h_blocks = padded.shape[0] // 8
    w_blocks = padded.shape[1] // 8
    
    blocks = np.ascontiguousarray(padded.reshape(h_blocks, 8, w_blocks, 8).transpose(0, 2, 1, 3))
    dct_blocks = dctn(blocks, type=2, norm='ortho', axes=(-2, -1))
    quantized = np.round(dct_blocks / q_matrix).astype(np.int32)
    
    flat = quantized.reshape(-1, 64)
    zigzag = np.ascontiguousarray(flat[:, ZIGZAG_ORDER])
    
    n_blocks = zigzag.shape[0]
    max_pairs = n_blocks * 65
    out_pairs = np.zeros(max_pairs * 2, dtype=np.int32)
    n_pairs = _rle_encode_all_fast(zigzag, out_pairs)
    
    return out_pairs[:n_pairs * 2]

def _decode_channel_fast(pairs_1d, q_matrix, h, w):
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    h_blocks = (h + pad_h) // 8
    w_blocks = (w + pad_w) // 8
    n_blocks = h_blocks * w_blocks
    
    zigzag = np.zeros((n_blocks, 64), dtype=np.int32)
    n_pairs = len(pairs_1d) // 2
    _rle_decode_all_fast(pairs_1d, n_pairs, n_blocks, zigzag)
    
    flat = np.zeros((n_blocks, 64), dtype=np.int32)
    flat[:, ZIGZAG_ORDER] = zigzag
    
    blocks = flat.reshape(h_blocks, w_blocks, 8, 8).astype(np.float32) * q_matrix
    spatial_blocks = idctn(blocks, type=2, norm='ortho', axes=(-2, -1))
    
    padded = spatial_blocks.transpose(0, 2, 1, 3).reshape(h_blocks * 8, w_blocks * 8)
    return padded[:h, :w]

def encode_image(pixels, width, height, channels=3, quality=90, level=19):
    has_alpha = (channels == 4)
    flags = 1 if has_alpha else 0
    
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

    raw_pairs = []
    raw_pairs.append(_encode_channel_fast(oklab[:, :, 0], q_luma))
    raw_pairs.append(_encode_channel_fast(oklab[:, :, 1], q_chroma))
    raw_pairs.append(_encode_channel_fast(oklab[:, :, 2], q_chroma))
    if has_alpha:
        raw_pairs.append(_encode_channel_fast(alpha_channel, q_alpha))

    compressed_channels = []
    cctx = zstd.ZstdCompressor(level=level, threads=-1, write_content_size=True)
    for pairs in raw_pairs:
        compressed_channels.append(cctx.compress(pairs.tobytes()))

    comp_lens = [len(c) for c in compressed_channels]
    
    header = struct.pack(IMG_HDR_FMT, width, height, channels, flags, float(quality))
    meta = struct.pack(f'<{channels}I', *comp_lens)
    
    body = b''.join(compressed_channels)
    return header + meta + body

def decode_image(payload):
    if len(payload) < IMG_HDR_SIZE:
        raise ValueError("Image payload too small")

    width, height, channels, flags, quality = struct.unpack_from(IMG_HDR_FMT, payload, 0)
    has_alpha = bool(flags & 1)
    
    pos = IMG_HDR_SIZE
    comp_lens = struct.unpack_from(f'<{channels}I', payload, pos)
    pos += channels * 4
    
    compressed_channels = []
    for l in comp_lens:
        compressed_channels.append(payload[pos:pos+l])
        pos += l

    raw_pairs = []
    dctx = zstd.ZstdDecompressor()
    for comp in compressed_channels:
        packed = dctx.decompress(comp, max_output_size=width*height*64*4)
        raw_pairs.append(np.frombuffer(packed, dtype='<i4').astype(np.int32))

    q_luma = _get_q_matrix(quality, is_chroma=False)
    q_chroma = _get_q_matrix(quality, is_chroma=True)
    q_alpha = _get_q_matrix(quality, is_chroma=False)

    oklab = np.zeros((height, width, 3), dtype=np.float32)
    oklab[:, :, 0] = _decode_channel_fast(raw_pairs[0], q_luma, height, width)
    oklab[:, :, 1] = _decode_channel_fast(raw_pairs[1], q_chroma, height, width)
    oklab[:, :, 2] = _decode_channel_fast(raw_pairs[2], q_chroma, height, width)
    
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    _oklab_to_rgb_njit(oklab, rgb)

    if has_alpha:
        alpha_channel = _decode_channel_fast(raw_pairs[3], q_alpha, height, width)
        alpha = np.clip(alpha_channel, 0, 255).astype(np.uint8)
        return np.dstack((rgb, alpha))
    
    return rgb

def get_image_info(payload):
    width, height, channels, flags, quality = struct.unpack_from(IMG_HDR_FMT, payload, 0)
    return dict(
        width=width, height=height, channels=channels, 
        has_alpha=bool(flags & 1), quality=quality
    )