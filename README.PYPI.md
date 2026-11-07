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
# 1. Read a WAV file
pcm, sr = lvthn.read_wav("track.wav")

# 2. Encode to LVN
# Arguments: pcm_array, sample_rate, channels, delta, target_snr_db
lvn_data = lvthn.encode_pcm(pcm, sr, pcm.shape[1], 0.18, 24.0)

# 3. Decode back to PCM
out_pcm, out_sr, out_ch, n_samples = lvthn.decode_lvn(lvn_data)

# 4. Write to WAV
lvthn.write_wav("restored.wav", out_pcm, out_sr)


# --- Image Example ---
from PIL import Image
from lvthn.core import container, image

# 1. Read an image
pixels = np.array(Image.open("image.png"))

# 2. Encode to LVN (quality 92, zstd level 12)
payload = image.encode_image(pixels, pixels.shape[1], pixels.shape[0], 3, 92, 12)
lvn_data = container.wrap(container.MEDIA_IMAGE, payload)

# 3. Decode back to pixels
restored_pixels = image.decode_image(payload)
```

## Usage from the Command Line (CLI)

The package includes a CLI tool accessible via the `lvthn` command.

```bash
# Encode audio
lvthn encode track.wav track.lvn compact

# Decode audio
lvthn decode track.lvn out.wav

# Encode image
lvthn encode-img image.png image.lvn fast

# Decode image
lvthn decode-img image.lvn out.png

# Get file info
lvthn info track.lvn
```

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
| `fast` | 85 | 1 | Quick preview, game textures |
| `default` | 92 | 12 | Web images, everyday use |
| `max` | 98 | 19 | Archiving, professional photography |

## Links

*   [GitHub Repository](https://github.com/FreshLend/LVTHN-CODEC)

## License

This project is licensed under the MIT License. It is built exclusively on patent-free algorithms (MDCT, Oklab, DCT, Rice coding, Zstandard). See the `PATENTS` file on GitHub for a detailed breakdown.