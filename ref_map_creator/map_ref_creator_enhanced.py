#!/usr/bin/env python3
"""
Map Reference Creator - Integrated System with Enhanced UI
System z zakładkami pokazującymi każdy etap przetwarzania
"""

import sys
import os
import json
import subprocess
import time
from datetime import datetime
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFormLayout, QMessageBox, QProgressBar, QTabWidget,
    QScrollArea, QSplitter
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtGui import QPixmap


class WorkerThread(QThread):
    """Worker thread do wykonywania operacji w tle"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, coordinates, zoom, output_folder):
        super().__init__()
        self.coordinates = coordinates
        self.zoom = zoom
        self.output_folder = output_folder
        self.should_stop = False
        
    def run(self):
        try:
            # Krok 1: Pobieranie danych
            self.progress.emit("\n" + "="*70)
            self.progress.emit("KROK 1/3: Pobieranie ortofotomapy i danych OSM")
            self.progress.emit("="*70)
            
            test_script = os.path.join(os.path.dirname(__file__), 'test_cli.py')
            if not os.path.exists(test_script):
                test_script = 'test_cli.py'
                if not os.path.exists(test_script):
                    test_script = 'test.py'
            
            self.progress.emit(f"Uruchamiam test_cli.py z parametrami:")
            self.progress.emit(f"  Środek: lat={self.coordinates['center_lat']}, lon={self.coordinates['center_lon']}")
            self.progress.emit(f"  Rozmiar: {self.coordinates['size_deg']}°")
            self.progress.emit(f"  Zoom: {self.zoom}")
            self.progress.emit("")
            
            process = subprocess.Popen(
                [sys.executable, '-u', test_script,
                 str(self.coordinates['center_lat']),
                 str(self.coordinates['center_lon']),
                 str(self.coordinates['size_deg']),
                 str(self.zoom)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in iter(process.stdout.readline, ''):
                if self.should_stop:
                    process.kill()
                    self.finished.emit(False, "Przerwano przez użytkownika")
                    return
                self.progress.emit(line.rstrip())
                
            process.wait()
            
            if process.returncode != 0:
                self.finished.emit(False, f"Błąd podczas pobierania danych (kod: {process.returncode})")
                return
            
            coord_folders = [d for d in os.listdir('.')
                           if os.path.isdir(d) and d.startswith('lat_')]
            if not coord_folders:
                self.finished.emit(False, "Nie znaleziono utworzonego folderu ze współrzędnymi")
                return
            
            coord_folder = sorted(coord_folders)[-1]
            self.progress.emit(f"\n Dane pobrane do folderu: {coord_folder}")
            
            if self.should_stop:
                self.finished.emit(False, "Przerwano przez użytkownika")
                return
            
            # Krok 2: Segmentacja
            self.progress.emit("\n" + "="*70)
            self.progress.emit("KROK 2/3: Segmentacja ortofotomapy sieci neuronową")
            self.progress.emit("="*70)
            
            segmentation_script = os.path.join(os.path.dirname(__file__), 'segmentation_cli.py')
            if not os.path.exists(segmentation_script):
                segmentation_script = 'segmentation_cli.py'
                if not os.path.exists(segmentation_script):
                    segmentation_script = 'segmentation.py'
            
            orto_image = os.path.join(coord_folder, 'image_orto.jpg')
            
            self.progress.emit(f"Uruchamiam segmentację dla: {orto_image}")
            self.progress.emit("To może potrwać kilka minut...")
            self.progress.emit("")
            
            process = subprocess.Popen(
                [sys.executable, '-u', segmentation_script, orto_image, coord_folder],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            dot_counter = 0
            for line in iter(process.stdout.readline, ''):
                if self.should_stop:
                    process.kill()
                    self.finished.emit(False, "Przerwano przez użytkownika")
                    return
                    
                self.progress.emit(line.rstrip())
                
                if "Processing patch" in line or "Progress:" in line:
                    dot_counter += 1
                    if dot_counter % 10 == 0:
                        self.progress.emit("." * 5)
                
            process.wait()
            
            if process.returncode != 0:
                self.finished.emit(False, f"Błąd podczas segmentacji (kod: {process.returncode})")
                return
            
            self.progress.emit(f"\n Segmentacja zakończona")
            
            if self.should_stop:
                self.finished.emit(False, "Przerwano przez użytkownika")
                return
            
            # Krok 3: Łączenie
            self.progress.emit("\n" + "="*70)
            self.progress.emit("KROK 3/3: Łączenie map topograficznych i segmentacji NN")
            self.progress.emit("="*70)
            
            merge_script = os.path.join(os.path.dirname(__file__), 'merge_topo_nn_cli.py')
            if not os.path.exists(merge_script):
                merge_script = 'merge_topo_nn_cli.py'
                if not os.path.exists(merge_script):
                    merge_script = 'merge_topo_nn.py'
            
            self.progress.emit(f"Uruchamiam merge_topo_nn_cli.py dla folderu: {coord_folder}")
            self.progress.emit("")
            
            process = subprocess.Popen(
                [sys.executable, '-u', merge_script, coord_folder, 'thesis', '50'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in iter(process.stdout.readline, ''):
                if self.should_stop:
                    process.kill()
                    self.finished.emit(False, "Przerwano przez użytkownika")
                    return
                self.progress.emit(line.rstrip())
                
            process.wait()
            
            if process.returncode != 0:
                self.finished.emit(False, f"Błąd podczas łączenia map (kod: {process.returncode})")
                return
            
            self.progress.emit(f"\n Mapy połączone pomyślnie")
            
            # Podsumowanie
            self.progress.emit("\n" + "="*70)
            self.progress.emit("PROCES ZAKOŃCZONY POMYŚLNIE!")
            self.progress.emit("="*70)
            self.progress.emit(f"\nWszystkie pliki zapisane w: {coord_folder}/")
            self.progress.emit("\nWygenerowane pliki:")
            self.progress.emit("  • image_orto.jpg - ortofotomapa")
            self.progress.emit("  • segmentation_mask.png - maska OSM")
            self.progress.emit("  • segmentation_nn_raw.png - segmentacja NN")
            self.progress.emit("  • reference_map.png - mapa referencyjna (wynik)")
            self.progress.emit("  • reference_map_colored.png - kolorowa wizualizacja")
            self.progress.emit("  • reference_map_comparison.png - porównanie wszystkich map")
            self.progress.emit("  • metadata.txt - metadane")
            
            self.finished.emit(True, coord_folder)
            
        except Exception as e:
            import traceback
            error_msg = f"Błąd: {str(e)}\n{traceback.format_exc()}"
            self.progress.emit(error_msg)
            self.finished.emit(False, error_msg)
    
    def stop(self):
        self.should_stop = True


class MapReferenceCreatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.current_folder = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Map Reference Creator - Enhanced UI")
        self.setGeometry(50, 50, 1600, 900)
        
        # Widget centralny
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Główny splitter (pionowy podział)
        main_splitter = QSplitter(Qt.Horizontal)
        central_widget_layout = QVBoxLayout(central_widget)
        central_widget_layout.addWidget(main_splitter)
        
        # ==== LEWA SEKCJA - Kontrola ====
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Grupa mapy
        map_group = QGroupBox("📍 Wybór Regionu")
        map_layout = QVBoxLayout()
        
        self.web = QWebEngineView()
        self.web.setHtml(self._get_leaflet_html())
        self.web.page().loadFinished.connect(self.on_map_loaded)
        self.web.setMaximumHeight(250)
        map_layout.addWidget(self.web)
        
        self.get_bounds_btn = QPushButton("📥 Pobierz współrzędne z mapy")
        self.get_bounds_btn.clicked.connect(self.get_map_bounds)
        self.get_bounds_btn.setEnabled(False)
        map_layout.addWidget(self.get_bounds_btn)
        
        map_group.setLayout(map_layout)
        left_layout.addWidget(map_group)
        
        # Grupa parametrów
        params_group = QGroupBox("⚙️ Parametry Przetwarzania")
        params_layout = QFormLayout()
        
        self.lat_input = QDoubleSpinBox()
        self.lat_input.setRange(-90, 90)
        self.lat_input.setDecimals(6)
        self.lat_input.setValue(54.352)
        params_layout.addRow("Szerokość geogr.:", self.lat_input)
        
        self.lon_input = QDoubleSpinBox()
        self.lon_input.setRange(-180, 180)
        self.lon_input.setDecimals(6)
        self.lon_input.setValue(18.646)
        params_layout.addRow("Długość geogr.:", self.lon_input)
        
        self.size_input = QDoubleSpinBox()
        self.size_input.setRange(0.001, 0.1)
        self.size_input.setDecimals(4)
        self.size_input.setValue(0.003)
        self.size_input.setSingleStep(0.001)
        params_layout.addRow("Rozmiar (°):", self.size_input)
        
        self.zoom_input = QSpinBox()
        self.zoom_input.setRange(14, 19)
        self.zoom_input.setValue(18)
        params_layout.addRow("Zoom:", self.zoom_input)
        
        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)
        
        # Przyciski akcji
        action_group = QGroupBox("🚀 Akcje")
        action_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("▶ ROZPOCZNIJ")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14pt;
                font-weight: bold;
                padding: 12px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.start_btn.clicked.connect(self.start_processing)
        action_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("■ ZATRZYMAJ")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 12pt;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #da190b; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        action_layout.addWidget(self.stop_btn)
        
        action_group.setLayout(action_layout)
        left_layout.addWidget(action_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        left_layout.addWidget(self.progress_bar)
        
        # Log
        log_label = QLabel("📋 Dziennik:")
        log_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        left_layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(250)
        self.log_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Courier New', monospace;
                font-size: 9pt;
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
        """)
        left_layout.addWidget(self.log_text)
        
        main_splitter.addWidget(left_widget)
        
        # ==== PRAWA SEKCJA - Wizualizacje ====
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        viz_header = QLabel("🖼️ Wizualizacje Etapów Przetwarzania")
        viz_header.setStyleSheet("font-size: 16pt; font-weight: bold; margin-bottom: 10px;")
        right_layout.addWidget(viz_header)
        
        # Przyciski wyboru wizualizacji
        viz_buttons = QHBoxLayout()
        
        self.btn_show_orto = QPushButton("📷 Ortofoto")
        self.btn_show_orto.clicked.connect(self.show_ortophoto)
        self.btn_show_orto.setEnabled(False)
        self.btn_show_orto.setStyleSheet("padding: 8px; font-weight: bold;")
        viz_buttons.addWidget(self.btn_show_orto)
        
        self.btn_show_osm = QPushButton("🗺️ OSM")
        self.btn_show_osm.clicked.connect(self.show_osm_map)
        self.btn_show_osm.setEnabled(False)
        self.btn_show_osm.setStyleSheet("padding: 8px; font-weight: bold;")
        viz_buttons.addWidget(self.btn_show_osm)
        
        self.btn_show_seg = QPushButton("🧠 Segmentacja")
        self.btn_show_seg.clicked.connect(self.show_segmentation)
        self.btn_show_seg.setEnabled(False)
        self.btn_show_seg.setStyleSheet("padding: 8px; font-weight: bold;")
        viz_buttons.addWidget(self.btn_show_seg)
        
        self.btn_show_ref = QPushButton("⭐ Referencyjna")
        self.btn_show_ref.clicked.connect(self.show_reference)
        self.btn_show_ref.setEnabled(False)
        self.btn_show_ref.setStyleSheet("padding: 8px; font-weight: bold;")
        viz_buttons.addWidget(self.btn_show_ref)
        
        right_layout.addLayout(viz_buttons)
        
        # Zakładki wizualizacji
        self.viz_tabs = QTabWidget()
        self.viz_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #cccccc;
                border-radius: 5px;
            }
            QTabBar::tab {
                background: #e0e0e0;
                padding: 10px 15px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #4CAF50;
                color: white;
            }
        """)
        
        # Dodaj placeholder
        welcome_label = QLabel(
            "👋 Witaj!\n\n"
            "1. Wybierz region na mapie po lewej\n"
            "2. Kliknij 'Pobierz współrzędne'\n"
            "3. Kliknij 'ROZPOCZNIJ'\n"
            "4. Poczekaj na zakończenie (~5 min)\n"
            "5. Wyświetl wyniki używając przycisków powyżej\n\n"
            " Tutaj zobaczysz wszystkie etapy przetwarzania"
        )
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("""
            font-size: 14pt;
            color: #666;
            padding: 50px;
        """)
        self.viz_tabs.addTab(welcome_label, "Instrukcja")
        
        right_layout.addWidget(self.viz_tabs)
        
        main_splitter.addWidget(right_widget)
        
        # Ustaw proporcje (1:2)
        main_splitter.setSizes([500, 1000])
        
        # Status bar
        self.statusBar().showMessage("Gotowy - wybierz region i kliknij ROZPOCZNIJ")
        
        # Początkowy log
        self.log("=" * 60)
        self.log("Map Reference Creator - Enhanced UI")
        self.log("=" * 60)
        self.log("System gotowy do pracy!")
        self.log("Wybierz region na mapie lub wprowadź współrzędne ręcznie")
        self.log("")
    
    def on_map_loaded(self, ok):
        if ok:
            self.get_bounds_btn.setEnabled(True)
            self.log(" Mapa załadowana")
            js = f"map.setView([{self.lat_input.value()}, {self.lon_input.value()}], 13);"
            self.web.page().runJavaScript(js)
    
    def get_map_bounds(self):
        self.web.page().runJavaScript("getMapBounds();", self.handle_map_bounds)
    
    def handle_map_bounds(self, result):
        if result:
            center_lat = (result['north'] + result['south']) / 2
            center_lon = (result['east'] + result['west']) / 2
            size_deg = max(result['north'] - result['south'], result['east'] - result['west'])
            
            self.lat_input.setValue(center_lat)
            self.lon_input.setValue(center_lon)
            self.size_input.setValue(size_deg)
            
            self.log(f" Współrzędne pobrane:")
            self.log(f"  Środek: {center_lat:.6f}, {center_lon:.6f}")
            self.log(f"  Rozmiar: {size_deg:.4f}°")
    
    def start_processing(self):
        coordinates = {
            'center_lat': self.lat_input.value(),
            'center_lon': self.lon_input.value(),
            'size_deg': self.size_input.value()
        }
        zoom = self.zoom_input.value()
        
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Potwierdzenie")
        msg.setText("Rozpocząć przetwarzanie?")
        msg.setInformativeText(
            f"Obszar: {coordinates['center_lat']:.6f}, {coordinates['center_lon']:.6f}\n"
            f"Rozmiar: {coordinates['size_deg']:.4f}°\n"
            f"Zoom: {zoom}\n\n"
            "Proces może potrwać kilka minut."
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        
        if msg.exec_() == QMessageBox.Yes:
            self.log("\n" + "="*60)
            self.log("ROZPOCZYNAM PROCES")
            self.log("="*60)
            self.log(f"Czas: {datetime.now().strftime('%H:%M:%S')}")
            self.log("")
            
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress_bar.show()
            self.statusBar().showMessage("Przetwarzanie...")
            
            # Wyczyść poprzednie wyniki
            self.btn_show_orto.setEnabled(False)
            self.btn_show_osm.setEnabled(False)
            self.btn_show_seg.setEnabled(False)
            self.btn_show_ref.setEnabled(False)
            
            self.worker = WorkerThread(coordinates, zoom, "map_ref_creator")
            self.worker.progress.connect(self.log)
            self.worker.finished.connect(self.on_processing_finished)
            self.worker.start()
    
    def stop_processing(self):
        if self.worker and self.worker.isRunning():
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Przerwanie")
            msg.setText("Czy na pewno przerwać?")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            
            if msg.exec_() == QMessageBox.Yes:
                self.log("\n PRZERYWANIE...")
                self.worker.stop()
    
    def on_processing_finished(self, success, message):
        self.progress_bar.hide()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if success:
            self.current_folder = message
            self.statusBar().showMessage(f" Zakończono - {message}")
            
            # Włącz przyciski
            self.btn_show_orto.setEnabled(True)
            self.btn_show_osm.setEnabled(True)
            self.btn_show_seg.setEnabled(True)
            self.btn_show_ref.setEnabled(True)
            
            # Automatycznie pokaż wynik
            self.show_reference()
            
            QMessageBox.information(
                self,
                "Sukces! 🎉",
                f"Proces zakończony!\n\n"
                f"Folder: {message}\n\n"
                f"Użyj przycisków powyżej aby zobaczyć poszczególne etapy."
            )
        else:
            self.statusBar().showMessage(" Błąd")
            QMessageBox.critical(self, "Błąd", f"Wystąpił błąd:\n\n{message}")
    
    def show_ortophoto(self):
        if not self.current_folder:
            return
        
        image_path = os.path.join(self.current_folder, "image_orto.jpg")
        if os.path.exists(image_path):
            self._show_image("📷 Ortofotomapa", image_path, 
                           "Oryginalna ortofotomapa pobrana z Geoportalu")
    
    def show_osm_map(self):
        if not self.current_folder:
            return
        
        viz_path = os.path.join(self.current_folder, "alignment_visualization.png")
        if os.path.exists(viz_path):
            self._show_image("🗺️ Wizualizacja OSM", viz_path,
                           "Dane topograficzne z OpenStreetMap + ortofoto")
    
    def show_segmentation(self):
        if not self.current_folder:
            return
        
        viz_path = os.path.join(self.current_folder, "segmentation_nn_visualization.png")
        if os.path.exists(viz_path):
            self._show_image("🧠 Segmentacja NN", viz_path,
                           "Wynik segmentacji sieci neuronowej")
    
    def show_reference(self):
        if not self.current_folder:
            return
        
        comparison_path = os.path.join(self.current_folder, "reference_map_comparison.png")
        if os.path.exists(comparison_path):
            self._show_image("⭐ Mapa Referencyjna", comparison_path,
                           "Porównanie: NN, OSM i mapa referencyjna")
    
    def _show_image(self, title, path, description):
        # Usuń zakładkę jeśli już istnieje
        for i in range(self.viz_tabs.count()):
            if self.viz_tabs.tabText(i) == title:
                self.viz_tabs.removeTab(i)
                break
        
        # Utwórz nową zakładkę
        tab_widget = QWidget()
        tab_layout = QVBoxLayout(tab_widget)
        
        # Opis
        desc_label = QLabel(description)
        desc_label.setStyleSheet("font-size: 11pt; padding: 5px; background: #f0f0f0;")
        tab_layout.addWidget(desc_label)
        
        # Scroll area dla obrazu
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        label = QLabel()
        pixmap = QPixmap(path)
        
        # Skaluj jeśli za duży
        max_width = 1400
        max_height = 800
        if pixmap.width() > max_width or pixmap.height() > max_height:
            pixmap = pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
        
        scroll.setWidget(label)
        tab_layout.addWidget(scroll)
        
        idx = self.viz_tabs.addTab(tab_widget, title)
        self.viz_tabs.setCurrentIndex(idx)
        
        self.log(f" Wyświetlono: {title}")
    
    def log(self, message):
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _get_leaflet_html(self):
        return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Wybierz region</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css"/>
    <style>
        html, body, #map { height: 100%; margin: 0; padding: 0; }
    </style>
</head>
<body>
    <div id="map"></div>
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <script>
        var map = L.map('map').setView([54.352, 18.646], 13);
        
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
            attribution: '© OpenStreetMap'
        }).addTo(map);
        
        var viewBounds = null;
        
        function updateViewBounds() {
            if (viewBounds) {
                map.removeLayer(viewBounds);
            }
            var bounds = map.getBounds();
            viewBounds = L.rectangle(bounds, {
                color: "#ff7800",
                weight: 2,
                fillOpacity: 0.1
            }).addTo(map);
        }
        
        map.on('moveend', updateViewBounds);
        updateViewBounds();
        
        function getMapBounds() {
            var b = map.getBounds();
            return {
                north: b.getNorth(),
                south: b.getSouth(),
                east: b.getEast(),
                west: b.getWest(),
                zoom: map.getZoom()
            };
        }
    </script>
</body>
</html>
        """


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MapReferenceCreatorGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
