# LVTHN-CODEC

A modular, patent-clean, and highly efficient media codec written in Python. 
It uses a universal container (`.lvn`) that supports both **Audio** and **Images**, applying modern psychoacoustic and psychovisual models to outperform industry standards like MP3 and PNG.

## Installation

```bash
pip install lvthn-codec
```
*(Note: To use the FFmpeg converter features, ensure `ffmpeg` is installed in your system PATH).*

## Usage as a Python Library

You can import the codec directly into your Python projects.

```python
import numpy as np
import lvthn

# --- Audio Example ---
pcm, sr = lvthn.read_wav("track.wav")

# Encode (mode can be 'monolithic' or 'chunked')
lvn_data = lvthn.encode_pcm(pcm, sr, pcm.shape[1], 0.18, 24.0, mode='monolithic')

# Decode
out_pcm, out_sr, out_ch, n_samples = lvthn.decode_lvn(lvn_data)
lvthn.write_wav("restored.wav", out_pcm, out_sr)

# --- Image Example ---
from PIL import Image
from lvthn.core import container, image

pixels = np.array(Image.open("image.png"))
payload = image.encode_image(pixels, pixels.shape[1], pixels.shape[0], 3, 92, 12)
lvn_data = container.wrap(container.MEDIA_IMAGE, payload)
restored_pixels = image.decode_image(payload)
```

## Usage from the Command Line (CLI)

The package includes a CLI tool accessible via the `lvthn` command.

```bash
# Encode audio (Chunked mode for streaming)
lvthn encode track.wav track.lvn compact chunked

# Decode audio
lvthn decode track.lvn out.wav

# Encode image
lvthn encode-img image.png image.lvn fast

# Decode image
lvthn decode-img image.lvn out.png

# Get file info
lvthn info track.lvn
```

## Performance

### Audio (average across music tracks)
| Profile | Mode | Bitrate | SNR | Enc RTF | Dec RTF |
|---|---|---|---|---|---|
| `compact` | Monolithic | ~51 kbps | 23.2 dB | 0.010 | 0.010 |
| `compact` | Chunked | ~55 kbps | 23.2 dB | 0.080 | 0.050 |
| `default` | Monolithic | ~58 kbps | 24.9 dB | 0.009 | 0.009 |
| `default` | Chunked | ~62 kbps | 24.9 dB | 0.077 | 0.048 |
| `master` | Monolithic | ~114 kbps | 35.7 dB | 0.010 | 0.010 |

### Audio (average across speech recordings)
| Profile | Mode | Bitrate | SNR | Enc RTF | Dec RTF |
|---|---|---|---|---|---|
| `compact` | Monolithic | ~43 kbps | 30.6 dB | 0.004 | 0.003 |
| `default` | Monolithic | ~49 kbps | 32.7 dB | 0.004 | 0.003 |
| `master` | Monolithic | ~88 kbps | 45.9 dB | 0.004 | 0.004 |

*(RTF < 0.015 means encoding is over 60x faster than real time. Chunked mode adds ~4 kbps overhead for streaming support).*

### Images (average across real photographs)
| Profile | PSNR | Enc Time | Dec Time | Compression |
|---|---|---|---|---|
| `fast` | 39.9 dB | 0.01s | 0.01s | 7.8x |
| `default` | 42.9 dB | 0.02s | 0.01s | 5.6x |
| `max` | 47.5 dB | 0.06s | 0.01s | 3.0x |

*(PSNR > 40 dB is visually indistinguishable from the original. Larger compression ratios are achieved on simple images, lower on textured photographs).*

## Profiles

### Audio Profiles
| Profile | Description | SNR | Best for |
|---|---|---|---|
| `compact` | Maximum compression | 20.0 | Everyday listening, saving space |
| `default` | Balanced | 24.0 | High-quality music |
| `speech` | Voice optimized | 20.0 | Podcasts, audiobooks |
| `detail` | Preserves quiet details | 24.0 | Lo-Fi, acoustic, vinyl |
| `master` | Archival quality | 30.0 | Mastering, studio archives |

### Image Profiles
| Profile | Quality | Zstd Level | Best for |
|---|---|---|---|
| `fast` | 85 | 5 | Quick preview, game textures |
| `default` | 92 | 12 | Web images, everyday use |
| `max` | 98 | 19 | Archiving, professional photography |

## Links

*   [GitHub Repository](https://github.com/FreshLend/LVTHN-CODEC)

## License

This project is licensed under the MIT License. It is built exclusively on patent-free algorithms (MDCT, Oklab, DCT, Rice coding, Zstandard). See the `PATENTS` file on GitHub for a detailed breakdown.