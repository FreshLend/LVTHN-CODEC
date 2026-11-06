import sys
from pathlib import Path
import numpy as np
import sounddevice as sd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QImage, QPixmap, QPainter

from codec import decode_lvn
from core import container

class ImageViewer(QGraphicsView):
    def __init__(self, pixels):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        h, w, c = pixels.shape
        if c == 4:
            fmt = QImage.Format_RGBA8888
        else:
            fmt = QImage.Format_RGB888
            
        bytes_per_line = c * w
        img = QImage(pixels.data, w, h, bytes_per_line, fmt).copy()
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
        
        # Отключаем ползунки прокрутки
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.zoom_factor = 1.15

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.scale(self.zoom_factor, self.zoom_factor)
        else:
            self.scale(1.0 / self.zoom_factor, 1.0 / self.zoom_factor)

    def fit_to_window(self):
        rect = QRectF(self.item.pixmap().rect())
        self.fitInView(rect, Qt.KeepAspectRatio)

    def showEvent(self, event):
        super().showEvent(event)
        self.fit_to_window()


class AudioPlayer(QWidget):
    def __init__(self, pcm, sr):
        super().__init__()
        self.pcm = pcm
        self.sr = sr
        self.total_frames = len(pcm)
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
        
        self.stream = sd.OutputStream(
            samplerate=self.sr, 
            channels=self.pcm.shape[1],
            callback=self._audio_callback, 
            blocksize=2048
        )
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

    def on_seek(self, value):
        self.current_frame = value

    def on_slider_press(self):
        self.is_seeking = True

    def on_slider_release(self):
        self.is_seeking = False

    def _audio_callback(self, outdata, frames, time_info, status):
        if not self.playing:
            outdata.fill(0)
            return
            
        chunk = self.pcm[self.current_frame : self.current_frame + frames]
        
        if len(chunk) == 0:
            self.playing = False
            outdata.fill(0)
            return
            
        if len(chunk) < frames:
            outdata[:len(chunk)] = chunk.astype(np.float32)
            outdata[len(chunk):] = 0
            self.current_frame += len(chunk)
            self.playing = False
        else:
            outdata[:] = chunk.astype(np.float32)
            self.current_frame += frames

    def update_ui(self):
        if not self.is_seeking:
            self.slider.setValue(self.current_frame)
            
        curr_sec = int(self.current_frame / self.sr)
        tot_sec = int(self.total_frames / self.sr)
        self.time_label.setText(f"{curr_sec//60:02d}:{curr_sec%60:02d} / {tot_sec//60:02d}:{tot_sec%60:02d}")
        
        if not self.playing and self.current_frame >= self.total_frames:
            self.btn_play.setText("Play")

    def closeEvent(self, event):
        self.playing = False
        self.stream.stop()
        self.stream.close()
        event.accept()


class MainWindow(QMainWindow):
    def __init__(self, path):
        super().__init__()
        self.setWindowTitle(f"LVTHN Player - {path.name}")
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
        media_type, _ = container.unwrap(data)
        
        if media_type == container.MEDIA_AUDIO:
            pcm, sr, ch, n = decode_lvn(data)
            self.setCentralWidget(AudioPlayer(pcm, sr))
            self.resize(600, 250)
        elif media_type == container.MEDIA_IMAGE:
            pixels = decode_lvn(data)
            self.setCentralWidget(ImageViewer(pixels))
        else:
            sys.exit('Unsupported media type')


def main():
    if len(sys.argv) < 2:
        sys.exit('usage: player.py <file.lvn>')
        
    path = Path(sys.argv[1])
    if not path.exists():
        sys.exit(f'File not found: {path}')

    print('Decoding...', flush=True)
    
    app = QApplication(sys.argv)
    window = MainWindow(path)
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()