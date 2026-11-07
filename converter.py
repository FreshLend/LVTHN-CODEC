import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from lvthn import (
    PROFILES, IMG_PROFILES, decode_lvn, encode_pcm, read_wav, write_wav,
    read_image, write_image
)
from lvthn.core import container, image

AUDIO_EXTS = ('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.opus', '.wma')
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif')


def _ffmpeg_ok():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _to_wav(src, wav, sr=None, ch=None):
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', src]
    if sr:
        cmd += ['-ar', str(sr)]
    if ch:
        cmd += ['-ac', str(ch)]
    cmd += [wav]
    subprocess.run(cmd, check=True)


def _from_wav(wav, dst, sr=None, ch=None, bitrate=None):
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', wav]
    if sr:
        cmd += ['-ar', str(sr)]
    if ch:
        cmd += ['-ac', str(ch)]
    if bitrate:
        cmd += ['-b:a', bitrate]
    cmd += [dst]
    subprocess.run(cmd, check=True)


def to_lvn_audio(src, dst, profile='default', sr=None, ch=None):
    delta = PROFILES.get(profile, PROFILES['default'])['delta']
    tsnr = PROFILES.get(profile, PROFILES['default'])['snr']

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp = f.name
    try:
        _to_wav(src, tmp, sr, ch)
        pcm, sr2 = read_wav(tmp)
        data = encode_pcm(pcm, sr2, pcm.shape[1], delta, tsnr)
        Path(dst).write_bytes(data)
        print(f'OK [Audio] {src} -> {dst}  {len(data)} bytes  (profile={profile})')
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def to_lvn_image(src, dst, profile='default'):
    q = IMG_PROFILES.get(profile, IMG_PROFILES['default'])['quality']
    lvl = IMG_PROFILES.get(profile, IMG_PROFILES['default'])['level']

    pixels, ch = read_image(src)
    payload = image.encode_image(pixels, pixels.shape[1], pixels.shape[0], ch, q, lvl)
    data = container.wrap(container.MEDIA_IMAGE, payload)
    Path(dst).write_bytes(data)
    print(f'OK [Image] {src} -> {dst}  {len(data)} bytes  (profile={profile})')


def from_lvn(src, dst, sr=None, ch=None, bitrate=None):
    data = Path(src).read_bytes()
    media_type, payload = container.unwrap(data)

    if media_type == container.MEDIA_AUDIO:
        pcm, sr2, _, _ = decode_lvn(data)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            tmp = f.name
        try:
            write_wav(tmp, pcm, sr2)
            _from_wav(tmp, dst, sr, ch, bitrate)
            print(f'OK [Audio] {src} -> {dst}')
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    elif media_type == container.MEDIA_IMAGE:
        pixels = decode_lvn(data)
        write_image(dst, pixels, pixels.shape[2] if len(pixels.shape) > 2 else 1)
        print(f'OK [Image] {src} -> {dst}')


def main():
    p = argparse.ArgumentParser(description="LVTHN Converter (Audio & Image)")
    sub = p.add_subparsers(dest='cmd', required=True)

    st = sub.add_parser('to-lvn', help='Convert audio/image to .lvn')
    st.add_argument('input')
    st.add_argument('output')
    st.add_argument('--profile', default='default')
    st.add_argument('--rate', type=int, default=None, help='Audio sample rate')
    st.add_argument('--channels', type=int, default=None, help='Audio channels')

    sf = sub.add_parser('from-lvn', help='Convert .lvn to audio/image')
    sf.add_argument('input')
    sf.add_argument('output')
    sf.add_argument('--rate', type=int, default=None)
    sf.add_argument('--channels', type=int, default=None)
    sf.add_argument('--bitrate', default=None)

    a = p.parse_args()

    if a.cmd == 'to-lvn':
        ext = Path(a.input).suffix.lower()
        if ext in AUDIO_EXTS:
            if not _ffmpeg_ok():
                sys.exit('ffmpeg not found in PATH (required for audio)')
            to_lvn_audio(a.input, a.output, a.profile, a.rate, a.channels)
        elif ext in IMAGE_EXTS:
            to_lvn_image(a.input, a.output, a.profile)
        else:
            sys.exit(f'Unsupported extension: {ext}')

    elif a.cmd == 'from-lvn':
        data = Path(a.input).read_bytes()
        media_type, _ = container.unwrap(data)

        if media_type == container.MEDIA_AUDIO:
            if not _ffmpeg_ok():
                sys.exit('ffmpeg not found in PATH (required for audio)')
            from_lvn(a.input, a.output, a.rate, a.channels, a.bitrate)
        elif media_type == container.MEDIA_IMAGE:
            from_lvn(a.input, a.output)
        else:
            sys.exit('Unknown media type')


if __name__ == '__main__':
    main()