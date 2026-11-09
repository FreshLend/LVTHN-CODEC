import sys
import time
from pathlib import Path
import numpy as np
from scipy.io import wavfile

from .core import container, audio, image

PROFILES = {
    'default': {'delta': 0.18, 'snr': 24.0},
    'compact': {'delta': 0.25, 'snr': 20.0},
    'detail':  {'delta': 0.05, 'snr': 24.0},
    'speech':  {'delta': 0.10, 'snr': 20.0},
    'master':  {'delta': 0.03, 'snr': 30.0},
}

IMG_PROFILES = {
    'fast':    {'quality': 85, 'level': 5},
    'default': {'quality': 92, 'level': 12},
    'max':     {'quality': 98, 'level': 19},
}

def encode_pcm(pcm, sr, ch, delta, target_snr_db, mode='monolithic'):
    payload = audio.encode_audio(pcm, sr, ch, delta, target_snr_db, mode=mode)
    return container.wrap(container.MEDIA_AUDIO, payload)

def decode_lvn(data):
    media_type, payload = container.unwrap(data)
    if media_type == container.MEDIA_AUDIO:
        return audio.decode_audio(payload)
    elif media_type == container.MEDIA_IMAGE:
        return image.decode_image(payload)
    else:
        raise ValueError(f'Unsupported media_type: 0x{media_type:02x}')

def decode_lvn_info(data):
    media_type, payload = container.unwrap(data)
    if media_type == container.MEDIA_AUDIO:
        info = audio.get_audio_info(payload)
        info['media_type'] = media_type
        info['total'] = len(data)
        return info
    elif media_type == container.MEDIA_IMAGE:
        info = image.get_image_info(payload)
        info['media_type'] = media_type
        info['total'] = len(data)
        return info
    return dict(media_type=media_type, total=len(data))

def read_wav(path):
    sr, d = wavfile.read(path)
    if d.dtype == np.int16: d = d.astype(np.float64) / 32768.0
    elif d.dtype == np.int32: d = d.astype(np.float64) / 2147483648.0
    elif d.dtype == np.uint8: d = (d.astype(np.float64) - 128.0) / 128.0
    else: d = d.astype(np.float64)
    if d.ndim == 1: d = d[:, None]
    return d, sr

def write_wav(path, pcm, sr):
    wavfile.write(path, sr, np.clip(pcm * 32767, -32768, 32767).astype(np.int16))

def read_image(path):
    from PIL import Image
    img = Image.open(path)
    if img.mode == 'RGBA': return np.array(img), 4
    else: return np.array(img.convert('RGB')), 3

def write_image(path, pixels, channels):
    from PIL import Image
    img = Image.fromarray(pixels)
    img.save(path)

def main():
    if len(sys.argv) < 2:
        print('LVTHN-CODEC')
        print('Commands: encode | decode | info | encode-img | decode-img | profiles')
        sys.exit(1)
    cmd, args = sys.argv[1], sys.argv[2:]

    if cmd == 'encode':
        if len(args) < 2: sys.exit('usage: encode in.wav out.lvn [profile|delta] [target_snr_db] [mode]')
        pcm, sr = read_wav(args[0])
        arg2 = args[2] if len(args) > 2 else 'default'
        if arg2 in PROFILES:
            delta = PROFILES[arg2]['delta']
            tsnr = PROFILES[arg2]['snr']
            print(f'[audio profile: {arg2}] delta={delta} snr={tsnr}')
        else:
            delta = float(arg2)
            tsnr = float(args[3]) if len(args) > 3 else 24.0
        mode = args[4] if len(args) > 4 else 'monolithic'
        
        t0 = time.time()
        data = encode_pcm(pcm, sr, pcm.shape[1], delta, target_snr_db=tsnr, mode=mode)
        Path(args[1]).write_bytes(data)
        print(f'OK {args[1]} {len(data)} bytes ({pcm.size * 2 / len(data):.2f}x vs PCM) encode={time.time() - t0:.2f}s mode={mode}')

    elif cmd == 'decode':
        if len(args) < 2: sys.exit('usage: decode in.lvn out.wav')
        pcm, sr, ch, n = decode_lvn(Path(args[0]).read_bytes())
        write_wav(args[1], pcm, sr)
        print(f'OK {args[1]} {n} samples, {ch} ch, {sr} Hz')

    elif cmd == 'encode-img':
        if len(args) < 2: sys.exit('usage: encode-img in.png out.lvn [profile|quality] [level]')
        pixels, ch = read_image(args[0])
        arg2 = args[2] if len(args) > 2 else 'default'
        if arg2 in IMG_PROFILES:
            q = IMG_PROFILES[arg2]['quality']
            lvl = IMG_PROFILES[arg2]['level']
            print(f'[image profile: {arg2}] quality={q} level={lvl}')
        else:
            q = int(arg2)
            lvl = int(args[3]) if len(args) > 3 else 19
        t0 = time.time()
        payload = image.encode_image(pixels, pixels.shape[1], pixels.shape[0], ch, q, lvl)
        data = container.wrap(container.MEDIA_IMAGE, payload)
        Path(args[1]).write_bytes(data)
        print(f'OK {args[1]} {len(data)} bytes encode={time.time() - t0:.2f}s')

    elif cmd == 'decode-img':
        if len(args) < 2: sys.exit('usage: decode-img in.lvn out.png')
        mt, payload = container.unwrap(Path(args[0]).read_bytes())
        if mt != container.MEDIA_IMAGE: sys.exit('Not an image')
        pixels = image.decode_image(payload)
        write_image(args[1], pixels, pixels.shape[2] if len(pixels.shape) > 2 else 1)
        print(f'OK {args[1]}')

    elif cmd == 'info':
        if not args: sys.exit('usage: info file.lvn')
        info = decode_lvn_info(Path(args[0]).read_bytes())
        if info['media_type'] == container.MEDIA_AUDIO:
            mode_str = 'chunked' if info.get('mode') == 1 else 'monolithic'
            print(f"LVTHN v1 audio {info['sr']} Hz {info['ch']} ch {info['n_samples'] / info['sr']:.2f}s delta={info['delta']:.4f} mode={mode_str}")
            print(f"  target_snr_db: {info['target_snr_db']:.1f} total: {info['total']}")
        elif info['media_type'] == container.MEDIA_IMAGE:
            print(f"LVTHN v1 image {info['width']}x{info['height']} alpha={info['has_alpha']} quality={info['quality']:.0f} total: {info['total']}")

    elif cmd == 'profiles':
        print('Audio profiles:')
        for name, p in PROFILES.items():
            print(f'  {name:10s} delta={p["delta"]:<6} snr={p["snr"]}')
        print('Image profiles:')
        for name, p in IMG_PROFILES.items():
            print(f'  {name:10s} quality={p["quality"]:<4} level={p["level"]}')

if __name__ == '__main__':
    main()