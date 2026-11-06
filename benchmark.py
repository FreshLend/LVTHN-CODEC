import os
import sys
import time
import argparse
import numpy as np
from PIL import Image

from core import container, audio, image
import codec

def _snr(ref, got):
    err = ref - got
    return 10 * np.log10(np.sum(ref ** 2) / max(np.sum(err ** 2), 1e-20))

def _psnr(ref, got):
    mse = np.mean((ref.astype(np.float32) - got.astype(np.float32)) ** 2)
    if mse == 0: return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)

def _bench_audio(path, profile_name):
    try:
        pcm, sr = codec.read_wav(path)
    except Exception as e:
        print(f"  [!] Error reading {path}: {e}")
        return None

    dur = pcm.shape[0] / sr
    raw_bytes = pcm.size * 2
    
    delta = codec.PROFILES[profile_name]['delta']
    tsnr = codec.PROFILES[profile_name]['snr']

    t0 = time.time()
    payload = audio.encode_audio(pcm, sr, pcm.shape[1], delta, tsnr)
    data = container.wrap(container.MEDIA_AUDIO, payload)
    t_enc = time.time() - t0

    t0 = time.time()
    out_pcm, _, _, n2 = codec.decode_lvn(data)
    t_dec = time.time() - t0

    hop = audio.HOP
    snr = _snr(pcm[2*hop:n2-2*hop], out_pcm[2*hop:n2-2*hop])
    rtf_enc = t_enc / dur
    rtf_dec = t_dec / dur
    kbps = len(data) * 8 / dur / 1000

    return {
        'name': os.path.basename(path),
        'profile': profile_name,
        'dur': dur,
        'size': len(data),
        'ratio': raw_bytes / len(data),
        'kbps': kbps,
        'snr': snr,
        't_enc': t_enc,
        't_dec': t_dec,
        'rtf_enc': rtf_enc,
        'rtf_dec': rtf_dec
    }

def _bench_image(path, profile_name):
    try:
        img = Image.open(path)
        if img.mode == 'RGBA':
            pixels = np.array(img)
            ch = 4
        else:
            pixels = np.array(img.convert('RGB'))
            ch = 3
    except Exception as e:
        print(f"  [!] Error reading {path}: {e}")
        return None
    
    h, w = pixels.shape[:2]
    mp = (h * w) / 1_000_000
    
    q = codec.IMG_PROFILES[profile_name]['quality']
    lvl = codec.IMG_PROFILES[profile_name]['level']

    t0 = time.time()
    payload = image.encode_image(pixels, w, h, ch, q, lvl)
    data = container.wrap(container.MEDIA_IMAGE, payload)
    t_enc = time.time() - t0

    t0 = time.time()
    out_pixels = codec.decode_lvn(data)
    t_dec = time.time() - t0

    psnr = _psnr(pixels, out_pixels)
    orig_size = os.path.getsize(path)
    
    return {
        'name': os.path.basename(path),
        'profile': profile_name,
        'mp': mp,
        'orig_size': orig_size,
        'size': len(data),
        'ratio': orig_size / len(data),
        'psnr': psnr,
        't_enc': t_enc,
        't_dec': t_dec,
        'mp_enc': mp / t_enc,
        'mp_dec': mp / t_dec
    }

def _resolve_files(paths, extensions):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for f in os.listdir(p):
                if f.lower().endswith(extensions):
                    files.append(os.path.join(p, f))
        elif os.path.isfile(p):
            if p.lower().endswith(extensions):
                files.append(p)
            else:
                print(f"  [!] Skipped {p} (wrong extension)")
        else:
            print(f"  [!] Path not found: {p}")
    return files

def main():
    parser = argparse.ArgumentParser(description="LVTHN-CODEC Benchmark Tool")
    parser.add_argument('--audio', nargs='+', default=None, help='Audio files or folders (wav)')
    parser.add_argument('--image', nargs='+', default=None, help='Image files or folders (png/jpg)')
    
    parser.add_argument('--audio-profiles', nargs='+', default=['default', 'compact'], 
                        choices=list(codec.PROFILES.keys()), help='Audio profiles to test')
    parser.add_argument('--image-profiles', nargs='+', default=['fast', 'default', 'max'],
                        choices=list(codec.IMG_PROFILES.keys()), help='Image profiles to test')
    
    args = parser.parse_args()

    if not args.audio and not args.image:
        parser.print_help()
        sys.exit(1)

    print("=" * 80)
    print(" LVTHN-CODEC BENCHMARK")
    print("=" * 80)

    # --- AUDIO BENCHMARK ---
    if args.audio:
        audio_files = _resolve_files(args.audio, ('.wav',))
        if not audio_files:
            print("\n[ AUDIO ] No valid files found.")
        else:
            print("\n[ AUDIO ]")
            print(f"{'File':<20} | {'Profile':<10} | {'Dur':<6} | {'kbps':<6} | {'SNR':<6} | {'Enc RTF':<8} | {'Dec RTF':<8} | {'Ratio'}")
            print("-" * 95)
            
            for f in audio_files:
                for profile in args.audio_profiles:
                    res = _bench_audio(f, profile)
                    if res:
                        print(f"{res['name']:<20} | {res['profile']:<10} | {res['dur']:>5.0f}s | {res['kbps']:>5.0f}k | {res['snr']:>5.1f}dB | {res['rtf_enc']:>8.3f} | {res['rtf_dec']:>8.3f} | {res['ratio']:.1f}x")

    # --- IMAGE BENCHMARK ---
    if args.image:
        image_files = _resolve_files(args.image, ('.png', '.jpg', '.jpeg'))
        if not image_files:
            print("\n[ IMAGES ] No valid files found.")
        else:
            print("\n[ IMAGES ]")
            print(f"{'File':<20} | {'Profile':<10} | {'Res':<10} | {'PSNR':<7} | {'Enc':<6} | {'Dec':<6} | {'Enc MP/s':<9} | {'Ratio'}")
            print("-" * 95)

            for f in image_files:
                for profile in args.image_profiles:
                    res = _bench_image(f, profile)
                    if res:
                        print(f"{res['name']:<20} | {res['profile']:<10} | {res['mp']:.2f} MP   | {res['psnr']:>5.1f}dB | {res['t_enc']:>5.2f}s | {res['t_dec']:>5.2f}s | {res['mp_enc']:>7.1f} MP/s | {res['ratio']:.1f}x")

    print("\nDone.")

if __name__ == '__main__':
    main()