# LVTHN-CODEC [RU](https://github.com/FreshLend/LVTHN-CODEC/blob/main/README.md) | **EN**

A modular, patent-clean, and highly efficient media codec written in Python. 
It uses a universal container (`.lvn`) that supports both **Audio** and **Images**, applying modern psychoacoustic and psychovisual models to outperform industry standards like MP3 and PNG.

## Features

*   **Universal container (`.lvn`):** One format for everything. The decoder automatically detects what is inside.
*   **Audio codec:** MDCT + Mid/Side stereo + Bark scale bands + Absolute Threshold of Hearing (ATH) + Adaptive quantization.
*   **Streaming Support:** Audio can be encoded in Monolithic mode (best compression) or Chunked mode (best for streaming and low memory usage).
*   **Image codec:** Oklab color space + Block DCT + RLE + Zstandard.
*   **Performance:** 
    *   Audio outperforms MP3 in size by ~15% at the same perceived quality (up to 3x on speech).
    *   Images outperform PNG in size by 5 to 10 times depending on content complexity.
    *   High speed: Audio RTF < 0.015, image encoding takes ~0.01 to 0.06 seconds depending on resolution.

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
# Audio (Monolithic mode)
lvthn encode track.wav track.lvn compact monolithic

# Audio (Chunked streaming mode)
lvthn encode track.wav track.lvn compact chunked

# Images
lvthn encode-img image.png image.lvn fast

# Decode
lvthn decode track.lvn out.wav
lvthn decode-img image.lvn out.png

# File info
lvthn info track.lvn
```

### 2. Converter (`converter.py`)
Converts any supported format (via ffmpeg) to `.lvn` and back.
```bash
python converter.py to-lvn track.mp3 track.lvn --profile default --mode chunked
python converter.py from-lvn track.lvn track_out.mp3 --bitrate 192k
```

### 3. Player (`player.py`)
Modern GUI player and image viewer based on PySide6. Supports seamless seeking for both monolithic and chunked audio.
```bash
python player.py track.lvn
python player.py image.lvn
```

### 4. Benchmark (`benchmark.py`)
Testing speed and compression ratio on your files.
```bash
python benchmark.py --audio audio.wav --image image.png --image-profiles fast max --audio-modes monolithic chunked
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
| `fast` | 85 | 5 | Quick preview, game textures |
| `default` | 92 | 12 | Web images, everyday use |
| `max` | 98 | 19 | Archiving, professional photography |

## Benchmarks

### Audio (average across music tracks)
| Profile | Mode | Bitrate | SNR | Enc RTF | Dec RTF |
|---|---|---|---|---|---|
| `compact` | Monolithic | ~51 kbps | 23.2 dB | 0.010 | 0.010 |
| `compact` | Chunked | ~55 kbps | 23.2 dB | 0.080 | 0.050 |
| `default` | Monolithic | ~58 kbps | 24.9 dB | 0.009 | 0.009 |
| `default` | Chunked | ~62 kbps | 24.9 dB | 0.077 | 0.048 |
| `speech` | Monolithic | ~74 kbps | 28.1 dB | 0.010 | 0.009 |
| `detail` | Monolithic | ~96 kbps | 32.4 dB | 0.010 | 0.009 |
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