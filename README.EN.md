# LVTHN-CODEC [RU](https://github.com/FreshLend/LVTHN-CODEC/blob/main/README.md) | **EN**

A modular, patent-clean, and highly efficient media codec written in Python. 
It uses a universal container (`.lvn`) that supports both **Audio** and **Images**, applying modern psychoacoustic and psychovisual models to outperform industry standards like MP3 and PNG.

## Features

*   **Universal container (`.lvn`):** One format for everything. The decoder automatically detects what is inside.
*   **Audio codec:** MDCT + Mid/Side stereo + Bark scale bands + Absolute Threshold of Hearing (ATH) + Adaptive quantization.
*   **Image codec:** Oklab color space + Block DCT + RLE + Zstandard.
*   **Performance:** 
    *   Audio outperforms MP3 in size by **~15%** at the same perceived quality.
    *   Images outperform PNG in size by **8 to 9 times** with visually indistinguishable quality.
    *   High speed: Audio RTF < 0.015, 4K image encoding takes ~1.2 seconds, decoding takes ~0.8 seconds.

## Installation

Ensure you have Python 3.10+ installed.

```bash
pip install lvthn-codec sounddevice PySide6
```
*(Note: For `converter.py` to work with formats like MP3 or JPG, you need to install `ffmpeg` and add it to the system PATH environment variable).*

## Usage

### 1. Codec CLI
Direct encoding and decoding using the installed `lvthn` command.
```bash
# Audio
lvthn encode track.wav track.lvn compact
lvthn decode track.lvn out.wav

# Images
lvthn encode-img image.png image.lvn fast
lvthn decode-img image.lvn out.png

# File info
lvthn info track.lvn
```

### 2. Converter (`converter.py`)
Converts any supported format (via ffmpeg) to `.lvn` and back.
```bash
python converter.py to-lvn track.mp3 track.lvn --profile default
python converter.py from-lvn track.lvn track_out.mp3 --bitrate 192k
```

### 3. Player (`player.py`)
Modern GUI player and image viewer based on PySide6.
```bash
python player.py track.lvn
python player.py image.lvn
```

### 4. Benchmark (`benchmark.py`)
Testing speed and compression ratio on your files.
```bash
python benchmark.py --audio audio.wav --image image.png --image-profiles fast max
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

## Benchmarks

### Audio (acoustic.wav, 179 sec)
| Profile | Bitrate | SNR | Enc RTF | Dec RTF | Compression |
|---|---|---|---|---|---|
| `compact` | 113 kbps | 27.5 dB | 0.013 | 0.009 | 13.5x |
| `default` | 127 kbps | 29.4 dB | 0.012 | 0.009 | 12.1x |
| `speech` | 155 kbps | 32.9 dB | 0.013 | 0.010 | 9.9x |
| `detail` | 193 kbps | 37.3 dB | 0.013 | 0.010 | 8.0x |
| `master` | 225 kbps | 40.8 dB | 0.014 | 0.010 | 6.8x |

*(RTF < 0.015 means encoding is 60x faster than real time. Decoding is ~100x faster).*

### Images (4K 8.29 MP)
| Profile | PSNR | Enc Time | Dec Time | Compression |
|---|---|---|---|---|
| `fast` | 42.1 dB | 1.15s | 1.03s | 9.8x |
| `default` | 44.7 dB | 1.72s | 0.86s | 8.3x |
| `max` | 48.3 dB | 8.33s | 0.99s | 4.7x |

*(PSNR > 40 dB is visually indistinguishable from the original).*

## Architecture

*   `lvthn/codec.py` - Main CLI and API.
*   `lvthn/core/container.py` - Universal LVTHN wrapper.
*   `lvthn/core/audio.py` - Audio math (MDCT, Bark, Psy).
*   `lvthn/core/image.py` - Image math (Oklab, DCT).
*   `player.py` - GUI (PySide6).
*   `converter.py` - FFmpeg wrapper.
*   `benchmark.py` - Performance testing tool.

## License and Patents

The project is distributed under the **MIT** license. 
It is built exclusively on patent-free algorithms (MDCT, Oklab, DCT, Rice coding, Zstandard). See the `PATENTS` file for a detailed breakdown.