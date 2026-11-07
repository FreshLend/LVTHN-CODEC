# LVTHN-CODEC [RU](https://github.com/FreshLend/LVTHN-CODEC/blob/main/README.md) | **EN**

A modular, patent-clean, and highly efficient media codec written in Python. 
It uses a universal container (`.lvn`) that supports both **Audio** and **Images**, applying modern psychoacoustic and psychovisual models to outperform industry standards like MP3 and PNG.

## Features

*   **Universal container (`.lvn`):** One format for everything. The decoder automatically detects what is inside.
*   **Audio codec:** MDCT + Mid/Side stereo + Bark scale bands + Absolute Threshold of Hearing (ATH) + Adaptive quantization.
*   **Image codec:** Oklab color space + Block DCT + RLE + Zstandard.
*   **Performance:** 
    *   Audio outperforms MP3 in size by **~15%** at the same perceived quality.
    *   Images outperform PNG in size by **6 to 7 times** with visually indistinguishable quality.
    *   High speed: Audio RTF < 0.02, 4K image decoding takes about 3 seconds.

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

### Audio (179 sec track)
| File | Profile | Bitrate | SNR | Enc RTF | Dec RTF | Compression |
|---|---|---|---|---|---|---|
| acoustic.wav | compact | 113 kbps | 27.5 dB | 0.022 | 0.009 | 13.5x |
| acoustic.wav | master | 225 kbps | 40.8 dB | 0.020 | 0.010 | 6.8x |

### Images (4K 8.29 MP)
| File | Profile | PSNR | Enc Time | Dec Time | Compression |
|---|---|---|---|---|---|
| 4k_image.png | fast | 42.1 dB | 4.10s | 3.14s | 7.1x |
| 4k_image.png | max | 48.3 dB | 29.32s | 3.14s | 4.3x |

*(PSNR > 40 dB is considered visually indistinguishable from the original. RTF < 1.0 means faster than real time).*

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