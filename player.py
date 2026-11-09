import sys
import struct
import queue
from pathlib import Path
import numpy as np
import sounddevice as sd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PySide6.QtCore import Qt, QTimer, QRectF, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter

from lvthn import decode_lvn
from lvthn.core import container, audio

class ImageViewer(QGraphicsView):
    def __init__(self, pixels):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        pixels = np.ascontiguousarray(pixels)
        h, w, c = pixels.shape
        fmt = QImage.Format_RGBA8888 if c == 4 else QImage.Format_RGB888
        img = QImage(pixels.data, w, h, c * w, fmt).copy()
        pixmap = QPixmap.fromImage(img)
        self.item = QGraphicsPixmapItem(pixmap)
        self.item.setTransformationMode(Qt.SmoothTransformation)
        self.scene.addItem(self.item)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(Qt.black)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.zoom_factor = 1.15

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.scale(self.zoom_factor, self.zoom_factor)
        else:
            self.scale(1.0 / self.zoom_factor, 1.0 / self.zoom_factor)

    def fit_to_window(self):
        self.fitInView(QRectF(self.item.pixmap().rect()), Qt.KeepAspectRatio)

    def showEvent(self, event):
        super().showEvent(event)
        self.fit_to_window()


class MonolithicAudioPlayer(QWidget):
    def __init__(self, pcm, sr):
        super().__init__()
        self.pcm = np.ascontiguousarray(pcm, dtype=np.float32)
        self.sr = sr
        self.total_frames = len(self.pcm)
        self.current_frame = 0
        self.playing = False

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(40, 40, 40, 40)
        self.layout.setSpacing(20)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-size: 24px; font-family: Consolas, monospace; color: #ddd;")
        self.layout.addWidget(self.time_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.total_frames)
        self.slider.sliderMoved.connect(self.on_seek)
        self.slider.sliderPressed.connect(self.on_slider_press)
        self.slider.sliderReleased.connect(self.on_slider_release)
        self.layout.addWidget(self.slider)

        btn_layout = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.setFixedSize(150, 50)
        self.btn_play.setStyleSheet("font-size: 18px; background-color: #333; color: white; border-radius: 5px;")
        self.btn_play.clicked.connect(self.toggle_play)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_play)
        btn_layout.addStretch()
        self.layout.addLayout(btn_layout)

        self.is_seeking = False
        self.stream = sd.OutputStream(samplerate=self.sr, channels=self.pcm.shape[1], callback=self._audio_callback, blocksize=2048)
        self.stream.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(50)

    def toggle_play(self):
        if self.playing:
            self.playing = False
            self.btn_play.setText("Play")
        else:
            if self.current_frame >= self.total_frames:
                self.current_frame = 0
            self.playing = True
            self.btn_play.setText("Pause")

    def on_seek(self, value): self.current_frame = value
    def on_slider_press(self): self.is_seeking = True
    def on_slider_release(self): self.is_seeking = False

    def _audio_callback(self, outdata, frames, time_info, status):
        if not self.playing:
            outdata.fill(0)
            return
        chunk = self.pcm[self.current_frame:self.current_frame + frames]
        if len(chunk) == 0:
            self.playing = False
            outdata.fill(0)
            return
        if len(chunk) < frames:
            outdata[:len(chunk)] = chunk
            outdata[len(chunk):] = 0
            self.current_frame += len(chunk)
            self.playing = False
        else:
            outdata[:] = chunk
            self.current_frame += frames

    def update_ui(self):
        if not self.is_seeking: self.slider.setValue(self.current_frame)
        curr_sec = int(self.current_frame / self.sr)
        tot_sec = int(self.total_frames / self.sr)
        self.time_label.setText(f"{curr_sec // 60:02d}:{curr_sec % 60:02d} / {tot_sec // 60:02d}:{tot_sec % 60:02d}")
        if not self.playing and self.current_frame >= self.total_frames:
            self.btn_play.setText("Play")

    def shutdown(self):
        self.playing = False
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


class ChunkDecodeThread(QThread):
    finished = Signal()
    def __init__(self, payload, start_sample):
        super().__init__()
        self.payload = payload
        self.start_sample = start_sample
        self.queue = queue.Queue(maxsize=20)
        self._stop_event = False

    def run(self):
        pos = audio.AUDIO_HDR_SIZE
        target_idx = self.start_sample // audio.HOP
        skip_to = max(0, target_idx - 1)
        
        current_idx = 0
        while pos < len(self.payload) and current_idx < skip_to:
            clen = struct.unpack_from('<I', self.payload, pos)[0]
            pos += 4 + clen
            current_idx += 1
            
        decoder = audio.AudioDecoder(self.payload[:audio.AUDIO_HDR_SIZE])
        
        first_emit = True
        skip_samples = self.start_sample % audio.HOP
        
        while pos < len(self.payload) and not self._stop_event:
            clen = struct.unpack_from('<I', self.payload, pos)[0]
            pos += 4
            chunk_data = self.payload[pos:pos+clen]
            pos += clen
            
            out_pcm = decoder.decode_chunk(chunk_data)
            
            if current_idx >= target_idx and out_pcm is not None:
                if first_emit:
                    if len(out_pcm) <= skip_samples:
                        skip_samples -= len(out_pcm)
                        current_idx += 1
                        continue
                    out_pcm = out_pcm[skip_samples:]
                    first_emit = False
                
                if len(out_pcm) > 0:
                    while not self._stop_event:
                        try:
                            self.queue.put(out_pcm, timeout=0.1)
                            break
                        except queue.Full:
                            pass
            current_idx += 1

    def stop(self):
        self._stop_event = True


class ChunkedAudioPlayer(QWidget):
    def __init__(self, payload, sr, ch, n_samples):
        super().__init__()
        self.payload = payload
        self.sr = sr
        self.ch = ch
        self.total_frames = n_samples
        self.current_frame = 0
        self.playing = False
        
        self.queue = queue.Queue()
        self.thread = None
        self.audio_buffer = np.zeros((0, self.ch), dtype=np.float32)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(40, 40, 40, 40)
        self.layout.setSpacing(20)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-size: 24px; font-family: Consolas, monospace; color: #ddd;")
        self.layout.addWidget(self.time_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.total_frames)
        self.slider.sliderMoved.connect(self.on_slider_moved)
        self.slider.sliderPressed.connect(self.on_slider_press)
        self.slider.sliderReleased.connect(self.on_slider_release)
        self.layout.addWidget(self.slider)

        btn_layout = QHBoxLayout()
        self.btn_play = QPushButton("Play (Streaming)")
        self.btn_play.setFixedSize(150, 50)
        self.btn_play.setStyleSheet("font-size: 18px; background-color: #333; color: white; border-radius: 5px;")
        self.btn_play.clicked.connect(self.toggle_play)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_play)
        btn_layout.addStretch()
        self.layout.addLayout(btn_layout)

        self.is_seeking = False
        
        self.stream = sd.OutputStream(samplerate=self.sr, channels=self.ch, callback=self._audio_callback, blocksize=2048)
        self.stream.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(50)

    def start_decoding(self, start_sample=0):
        if self.thread:
            self.thread.stop()
            self.thread.wait()
        self.queue = queue.Queue(maxsize=20)
        self.audio_buffer = np.zeros((0, self.ch), dtype=np.float32)
        self.thread = ChunkDecodeThread(self.payload, start_sample)
        self.thread.setObjectName("ChunkDecodeThread")
        self.thread.queue = self.queue
        self.thread.start()

    def toggle_play(self):
        if self.playing:
            self.playing = False
            self.btn_play.setText("Play (Streaming)")
            if self.thread:
                self.thread.stop()
                self.thread.wait()
                self.thread = None
        else:
            if self.current_frame >= self.total_frames:
                self.current_frame = 0
            self.start_decoding(self.current_frame)
            self.playing = True
            self.btn_play.setText("Pause")

    def on_slider_moved(self, value):
        self.current_frame = value

    def on_slider_press(self): 
        self.is_seeking = True
        if self.playing:
            if self.thread:
                self.thread.stop()
                self.thread.wait()
                self.thread = None

    def on_slider_release(self): 
        self.is_seeking = False
        if self.playing:
            self.start_decoding(self.current_frame)

    def _audio_callback(self, outdata, frames, time_info, status):
        if not self.playing:
            outdata.fill(0)
            return
            
        while len(self.audio_buffer) < frames:
            try:
                chunk = self.queue.get_nowait()
                self.audio_buffer = np.vstack([self.audio_buffer, chunk])
            except queue.Empty:
                break
                
        if len(self.audio_buffer) == 0:
            outdata.fill(0)
            if self.thread and not self.thread.isRunning() and self.queue.empty():
                self.playing = False
                self.current_frame = self.total_frames
            return
            
        take = min(len(self.audio_buffer), frames)
        outdata[:take] = self.audio_buffer[:take]
        if take < frames:
            outdata[take:] = 0
            
        self.audio_buffer = self.audio_buffer[take:]
        self.current_frame += take

    def update_ui(self):
        if not self.is_seeking: 
            self.slider.setValue(self.current_frame)
        curr_sec = int(self.current_frame / self.sr)
        tot_sec = int(self.total_frames / self.sr)
        self.time_label.setText(f"{curr_sec // 60:02d}:{curr_sec % 60:02d} / {tot_sec // 60:02d}:{tot_sec % 60:02d}")
        
        if not self.playing and self.current_frame >= self.total_frames:
            if self.btn_play.text() != "Play (Streaming)":
                self.btn_play.setText("Play (Streaming)")
            if self.thread:
                self.thread.stop()
                self.thread.wait()
                self.thread = None

    def shutdown(self):
        self.playing = False
        if self.thread:
            self.thread.stop()
            self.thread.wait()
            self.thread = None
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


class MainWindow(QMainWindow):
    def __init__(self, path):
        super().__init__()
        self.resize(1280, 720)
        self.setStyleSheet("""
            QMainWindow { background-color: #222; }
            QSlider::groove:horizontal {
                border: 1px solid #333; height: 8px; background: #111; border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0078D7; border: 1px solid #0078D7; width: 18px; margin: -5px 0; border-radius: 9px;
            }
            QSlider::sub-page:horizontal { background: #0078D7; border-radius: 4px; }
        """)

        data = path.read_bytes()
        media_type, payload = container.unwrap(data)

        if media_type == container.MEDIA_AUDIO:
            info = audio.get_audio_info(payload)
            mode_str = "Chunked (Streaming)" if info.get('mode') == 1 else "Monolithic"
            self.setWindowTitle(f"LVTHN Player - {path.name} [{mode_str}]")
            
            if info.get('mode') == 1:
                sr, ch, n_samples = info['sr'], info['ch'], info['n_samples']
                self.setCentralWidget(ChunkedAudioPlayer(payload, sr, ch, n_samples))
                self.resize(600, 250)
            else:
                pcm, sr, ch, n = decode_lvn(data)
                self.setCentralWidget(MonolithicAudioPlayer(pcm, sr))
                self.resize(600, 250)
                
        elif media_type == container.MEDIA_IMAGE:
            self.setWindowTitle(f"LVTHN Player - {path.name}")
            pixels = decode_lvn(data)
            self.setCentralWidget(ImageViewer(pixels))
        else:
            sys.exit('Unsupported media type')

    def closeEvent(self, event):
        cw = self.centralWidget()
        if cw is not None and hasattr(cw, 'shutdown'):
            cw.shutdown()
        super().closeEvent(event)


def main():
    if len(sys.argv) < 2:
        sys.exit('usage: player.py <file.lvn>')
    path = Path(sys.argv[1])
    if not path.exists():
        sys.exit(f'File not found: {path}')
    print('Initializing...', flush=True)
    app = QApplication(sys.argv)
    window = MainWindow(path)
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()