#!/usr/bin/env python3

import sys
import json
import threading
import requests
from xml.etree import ElementTree as ET
import io, base64
import time
import numpy as np
from PIL import Image
import math
import os
import subprocess
from datetime import datetime

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox,
    QFormLayout, QAction, QTreeWidget, QTreeWidgetItem,
    QFileDialog, QTabWidget, QSpinBox, QGroupBox, QDialog, QDialogButtonBox,
    QProgressBar
)
from PyQt5.QtCore import QUrl, pyqtSlot, Qt, QThread, pyqtSignal
from PyQt5.QtWebEngineWidgets import QWebEngineView


class WorkerThread(QThread):
    """Worker thread do przetwarzania w tle"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, coordinates, zoom):
        super().__init__()
        self.coordinates = coordinates
        self.zoom = zoom
        self.should_stop = False
        
    def run(self):
        try:
            # Krok 1: Pobieranie danych
            self.progress.emit("KROK 1/3: Pobieranie ortofotomapy i danych OSM")
            
            test_script = 'test_cli.py'
            if not os.path.exists(test_script):
                test_script = 'test.py'
            
            self.progress.emit(f"Pobieranie danych...")
            
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
                    self.finished.emit(False, "Przerwano")
                    return
                self.progress.emit(line.rstrip())
                
            process.wait()
            
            if process.returncode != 0:
                self.finished.emit(False, f"Błąd pobierania (kod: {process.returncode})")
                return
            
            coord_folders = [d for d in os.listdir('.')
                           if os.path.isdir(d) and d.startswith('lat_')]
            if not coord_folders:
                self.finished.emit(False, "Nie znaleziono folderu")
                return
            
            coord_folder = sorted(coord_folders)[-1]
            self.progress.emit(f"\n Pobrano do: {coord_folder}")
            
            # Krok 2: Segmentacja
            self.progress.emit("KROK 2/3: Segmentacja NN")
            
            segmentation_script = 'segmentation_cli.py'
            if not os.path.exists(segmentation_script):
                segmentation_script = 'segmentation.py'
            
            orto_image = os.path.join(coord_folder, 'image_orto.jpg')
            
            self.progress.emit("Segmentacja (może potrwać kilka minut)...")
            
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
                    self.finished.emit(False, "Przerwano")
                    return
                    
                self.progress.emit(line.rstrip())
                
                if "Processing patch" in line or "Progress:" in line:
                    dot_counter += 1
                    if dot_counter % 10 == 0:
                        self.progress.emit(".....")
                
            process.wait()
            
            if process.returncode != 0:
                self.finished.emit(False, f"Błąd segmentacji (kod: {process.returncode})")
                return
            
            self.progress.emit(f"\n Segmentacja zakończona")
            
            # Krok 3: Łączenie
            self.progress.emit("KROK 3/3: Łączenie map")

            merge_script = 'merge_topo_nn_cli.py'
            if not os.path.exists(merge_script):
                merge_script = 'merge_topo_nn.py'
            
            self.progress.emit("Łączenie OSM + NN...")
            
            process = subprocess.Popen(
                [sys.executable, '-u', merge_script, coord_folder, 'hybrid', '50'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in iter(process.stdout.readline, ''):
                if self.should_stop:
                    process.kill()
                    self.finished.emit(False, "Przerwano")
                    return
                self.progress.emit(line.rstrip())
                
            process.wait()
            
            if process.returncode != 0:
                self.finished.emit(False, f"Błąd łączenia (kod: {process.returncode})")
                return
            
            self.progress.emit(f"\n Zakończono!")
            
            self.finished.emit(True, coord_folder)
            
        except Exception as e:
            import traceback
            self.progress.emit(f"Błąd: {str(e)}\n{traceback.format_exc()}")
            self.finished.emit(False, str(e))
    
    def stop(self):
        self.should_stop = True


class MapProcessor(QtCore.QObject):
    capabilities_parsed = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

    def fetch_and_parse(self, service_url, service_type='WMS'):
        thread = threading.Thread(target=self._fetch_thread, args=(service_url, service_type), daemon=True)
        thread.start()

    def _fetch_thread(self, service_url, service_type):
        try:
            if service_type.upper() == 'WMS':
                url = service_url
                if 'request=GetCapabilities' not in url.lower():
                    sep = '&' if '?' in url else '?'
                    url = f"{url}{sep}SERVICE=WMS&REQUEST=GetCapabilities&VERSION=1.3.0"
            else:
                url = service_url
                if 'request=GetCapabilities' not in url.lower():
                    sep = '&' if '?' in url else '?'
                    url = f"{url}{sep}SERVICE=WMTS&REQUEST=GetCapabilities"

            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            xml = resp.content

            root = ET.fromstring(xml)

            layers = {}
            if service_type.upper() == 'WMS':
                for layer_el in root.findall('.//{http://www.opengis.net/wms}Layer'):
                    name = layer_el.find('{http://www.opengis.net/wms}Name')
                    title = layer_el.find('{http://www.opengis.net/wms}Title')
                    if name is not None and title is not None:
                        layers[name.text] = {
                            'title': title.text,
                            'type': 'WMS'
                        }
            else:
                for layer_el in root.findall('.//{http://www.opengis.net/wmts/1.0}Layer'):
                    title = layer_el.find('{http://www.opengis.net/ows/1.1}Title')
                    identifier = layer_el.find('{http://www.opengis.net/ows/1.1}Identifier')
                    if identifier is not None:
                        layers[identifier.text] = {
                            'title': (title.text if title is not None else identifier.text),
                            'type': 'WMTS'
                        }
            result = {
                'service_url': service_url,
                'service_type': service_type,
                'layers': layers
            }
            self.capabilities_parsed.emit(result)
        except Exception as e:
            self.capabilities_parsed.emit({'error': str(e), 'service_url': service_url, 'service_type': service_type, 'layers': {}})


class MapLoader(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Map Reference Creator")
        self.resize(1200, 800)

        self.processor = MapProcessor()
        self.processor.capabilities_parsed.connect(self.on_capabilities_parsed)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # LEWA STRONA
        left_panel = QVBoxLayout()
        main_layout.addLayout(left_panel, 0)

        # Service URL
        form = QFormLayout()
        self.service_url_input = QLineEdit()
        self.service_type_combo = QComboBox()
        self.service_type_combo.addItems(['WMS', 'WMTS'])
        self.btn_fetch = QPushButton("Fetch GetCapabilities")
        self.btn_fetch.clicked.connect(self.on_fetch_capabilities)
        form.addRow("Service URL:", self.service_url_input)
        form.addRow("Service Type:", self.service_type_combo)
        form.addRow("", self.btn_fetch)
        left_panel.addLayout(form)

        # Coordinates
        coord_layout = QHBoxLayout()
        self.lon_input = QLineEdit()
        self.lon_input.setPlaceholderText("Lon (e.g. 21.0)")
        self.lat_input = QLineEdit()
        self.lat_input.setPlaceholderText("Lat (e.g. 52.0)")
        self.btn_load_at = QPushButton("Load Map at Lon/Lat")
        self.btn_load_at.clicked.connect(self.on_load_at)
        coord_layout.addWidget(self.lon_input)
        coord_layout.addWidget(self.lat_input)
        coord_layout.addWidget(self.btn_load_at)
        left_panel.addLayout(coord_layout)

        # Zoom
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Orto Zoom Level:"))
        self.zoom_spinner = QSpinBox()
        self.zoom_spinner.setRange(10, 19)
        self.zoom_spinner.setValue(18)
        self.zoom_spinner.setToolTip("Higher zoom = more detail, larger image")
        zoom_layout.addWidget(self.zoom_spinner)
        left_panel.addLayout(zoom_layout)

        # Available Layers
        left_panel.addWidget(QLabel("Available Layers:"))
        self.layer_list = QTreeWidget()
        self.layer_list.setHeaderLabels(["Layer name", "Title", "Type"])
        left_panel.addWidget(self.layer_list, 1)

        # Layer buttons
        btns = QHBoxLayout()
        self.btn_preview = QPushButton("Preview Selected")
        self.btn_preview.clicked.connect(self.on_preview_selected)
        self.btn_add = QPushButton("Add Layer to Map")
        self.btn_add.clicked.connect(self.on_add_layer)
        self.btn_remove = QPushButton("Remove Layer")
        self.btn_remove.clicked.connect(self.on_remove_layer)
        btns.addWidget(self.btn_preview)
        btns.addWidget(self.btn_add)
        btns.addWidget(self.btn_remove)
        left_panel.addLayout(btns)

        # Download Area Options
        left_panel.addWidget(QLabel("Download Area Options:"))
        
        self.btn_select_view = QPushButton("Download Current View")
        self.btn_select_coords = QPushButton("Download by Coordinates")
        self.btn_select_all = QPushButton("Download Entire Visible Area")
        
        self.btn_select_view.clicked.connect(self.on_select_view)
        self.btn_select_coords.clicked.connect(self.on_select_coords)
        self.btn_select_all.clicked.connect(self.on_select_all)

        left_panel.addWidget(self.btn_select_view)
        left_panel.addWidget(self.btn_select_coords)
        left_panel.addWidget(self.btn_select_all)

        self.btn_reset = QPushButton("Reset to Initial View")
        self.btn_reset.clicked.connect(self.on_reset_view)
        left_panel.addWidget(self.btn_reset)

        # Processing Pipeline
        processing_group = QGroupBox("Processing Pipeline")
        processing_layout = QVBoxLayout()
        
        self.btn_segment_orto = QPushButton("Segment Ortophoto")
        self.btn_segment_orto.clicked.connect(self.on_segment_orto)
        self.btn_segment_orto.setEnabled(False)
        processing_layout.addWidget(self.btn_segment_orto)
        
        self.btn_process_topo = QPushButton("Process Topographic Map")
        self.btn_process_topo.clicked.connect(self.on_process_topo)
        self.btn_process_topo.setEnabled(False)
        processing_layout.addWidget(self.btn_process_topo)
        
        self.btn_combine = QPushButton("Combine Orto and Topo")
        self.btn_combine.clicked.connect(self.on_combine_maps)
        self.btn_combine.setEnabled(False)
        processing_layout.addWidget(self.btn_combine)
        
        self.btn_export_reference = QPushButton("Export Reference Map")
        self.btn_export_reference.clicked.connect(self.on_export_reference)
        self.btn_export_reference.setEnabled(False)
        processing_layout.addWidget(self.btn_export_reference)
        
        processing_group.setLayout(processing_layout)
        left_panel.addWidget(processing_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        left_panel.addWidget(self.progress_bar)

        # Settings / Advanced
        left_panel.addWidget(QLabel("Settings / Advanced"))
        self.settings_text = QTextEdit()
        self.settings_text.setPlaceholderText("Advanced settings JSON (crs, styles, tileMatrixSet, custom params) ...")
        left_panel.addWidget(self.settings_text, 1)

        # Config buttons
        cfg_layout = QHBoxLayout()
        self.btn_save_cfg = QPushButton("Save Config")
        self.btn_save_cfg.clicked.connect(self.save_config)
        self.btn_load_cfg = QPushButton("Load Config")
        self.btn_load_cfg.clicked.connect(self.load_config)
        cfg_layout.addWidget(self.btn_save_cfg)
        cfg_layout.addWidget(self.btn_load_cfg)
        left_panel.addLayout(cfg_layout)

        # PRAWA STRONA
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel, 1)

        # Tabs
        self.tabs = QTabWidget()
        right_panel.addWidget(self.tabs, 1)

        # Geoportal map
        self.web = QWebEngineView()
        self.web.loadFinished.connect(self.on_geoportal_loaded)
        self.web.setHtml(self._leaflet_html(), QUrl("about:blank"))
        self.tabs.addTab(self.web, "Geoportal")

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("""
            QTextEdit {
                font-family: 'Courier New', monospace;
                font-size: 9pt;
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
        """)
        right_panel.addWidget(self.log, 0)

        # Menu
        self._build_menus()

        # Variables
        self.added_layers = {}
        self.area_tabs = {}
        self.area_data = {}
        self.pending_layers = []
        self.geoportal_ready = False
        self.current_area = None
        self.worker = None

        # Geoportal services
        self.geoportal_services = {
            "BDOT10k (topo)": "https://mapy.geoportal.gov.pl/wss/service/WMTS/guest/wmts/BDOT10k?SERVICE=WMTS&REQUEST=GetCapabilities",
            "ORTO (ortofotomapa)": "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMTS/StandardResolution?SERVICE=WMTS&REQUEST=GetCapabilities"
        }

        for name, url in self.geoportal_services.items():
            self.log.append(f"Autoload: {name}")
            self.processor.fetch_and_parse(url, "WMTS")

    def on_geoportal_loaded(self, ok):
        if ok:
            self.geoportal_ready = True
            self.log.append("Geoportal map ready.")
            for layer_info in self.pending_layers:
                self._add_layer_internal(layer_info['name'], layer_info['type'], layer_info['temporary'])
            self.pending_layers.clear()

    def _build_menus(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        load_action = QAction("Load Map (Geoportal example)", self)
        load_action.triggered.connect(self._demo_load_geoportal)
        file_menu.addAction(load_action)
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        settings_menu = menubar.addMenu("&Settings")
        advanced_action = QAction("Advanced Settings (JSON editor)", self)
        advanced_action.triggered.connect(lambda: self.settings_text.setVisible(not self.settings_text.isVisible()))
        settings_menu.addAction(advanced_action)

        help_menu = menubar.addMenu("&Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(lambda: QtWidgets.QMessageBox.information(self, "About", "Map Reference Creator\nIntegrated system for creating reference maps"))
        help_menu.addAction(about_action)

    def on_fetch_capabilities(self):
        url = self.service_url_input.text().strip()
        s_type = self.service_type_combo.currentText()
        if not url:
            self.log.append("Please enter a service URL.")
            return
        self.log.append(f"Fetching GetCapabilities from {url} ({s_type}) ...")
        self.processor.fetch_and_parse(url, s_type)

    @pyqtSlot(dict)
    def on_capabilities_parsed(self, result):
        if 'error' in result:
            self.log.append(f"Error fetching capabilities: {result['error']}")
            return
        self.layer_list.clear()
        layers = result.get('layers', {})
        for name, info in layers.items():
            item = QTreeWidgetItem([name, info.get('title', ''), info.get('type', '')])
            item.setData(0, QtCore.Qt.UserRole, {'name': name, 'type': info.get('type')})
            self.layer_list.addTopLevelItem(item)
        n = len(layers)
        self.log.append(f"Parsed {n} layers from {result.get('service_url')}")

        for name, info in result.get("layers", {}).items():
            lname = name.lower()
            if "bdot" in lname or "orto" in lname:
                self.log.append(f"Auto-adding {name} layer to map ...")
                self.add_layer_to_map(name, info.get("type", "WMTS"), temporary=False)

    def on_preview_selected(self):
        item = self.layer_list.currentItem()
        if not item:
            self.log.append("Select a layer to preview.")
            return
        data = item.data(0, QtCore.Qt.UserRole)
        name = data['name']
        ltype = data['type']
        self.log.append(f"Previewing {name} ({ltype}) ...")
        self.add_layer_to_map(name, ltype, temporary=True)

    def on_add_layer(self):
        item = self.layer_list.currentItem()
        if not item:
            self.log.append("Select a layer to add.")
            return
        data = item.data(0, QtCore.Qt.UserRole)
        name = data['name']
        ltype = data['type']
        self.log.append(f"Adding layer {name} ({ltype}) to map ...")
        self.add_layer_to_map(name, ltype, temporary=False)

    def on_remove_layer(self):
        item = self.layer_list.currentItem()
        if not item:
            self.log.append("Select an added layer to remove (from left list).")
            return
        name = item.text(0)
        js = f"removeLayer('{name}');"
        self.web.page().runJavaScript(js)
        self.log.append(f"Requested removal of {name}")

    def on_load_at(self):
        try:
            lon = float(self.lon_input.text().strip())
            lat = float(self.lat_input.text().strip())
        except Exception:
            self.log.append("Invalid lat/lon values.")
            return
        js = f"setMapCenter({lat}, {lon}, 12);"
        self.web.page().runJavaScript(js)
        self.log.append(f"Centered map at lat={lat}, lon={lon}")

    def on_select_view(self):
        self.web.page().runJavaScript("getMapBounds();", self.handle_map_bounds)

    def on_reset_view(self):
        self.web.page().runJavaScript("setMapCenter(52.0, 21.0, 6);")
        self.log.append("Reset view to default position.")
    
    def on_select_coords(self):
        try:
            lat = float(self.lat_input.text().strip())
            lon = float(self.lon_input.text().strip())
            north = lat + 0.01
            south = lat - 0.01
            east = lon + 0.01
            west = lon - 0.01
            bounds = {'north': north, 'south': south, 'east': east, 'west': west}
            self.download_area(bounds)
        except Exception as e:
            self.log.append(f"Error in Use Coordinates: {e}")

    def on_select_all(self):
        def handle(bounds):
            if not bounds:
                self.log.append("Could not read map bounds.")
                return
            self.log.append(f"Downloading entire visible area: {bounds}")
            self.download_area(bounds)

        self.web.page().runJavaScript("getMapBounds();", handle)

    def handle_map_bounds(self, bounds):
        if not bounds: 
            self.log.append("Could not read bounds.")
            return
        self.log.append(f"Downloading current view: {bounds}")
        self.download_area(bounds)

    def download_area(self, bounds):
        lat_diff = abs(bounds['north'] - bounds['south'])
        lon_diff = abs(bounds['east'] - bounds['west'])
        
        if lat_diff > 0.1 or lon_diff > 0.1:
            self.log.append(f"Area too large!")
            self.log.append(f"Current size: {lat_diff:.4f}° × {lon_diff:.4f}°")
            
            reply = QtWidgets.QMessageBox.question(
                self, 
                'Large Area Warning',
                f'Area is very large ({lat_diff:.4f}° × {lon_diff:.4f}°).\n'
                'Processing may take very long or fail.\n\nContinue?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            
            if reply == QtWidgets.QMessageBox.No:
                return
        
        area_key = f"{bounds['north']:.4f}_{bounds['south']:.4f}_{bounds['east']:.4f}_{bounds['west']:.4f}"
        
        if area_key in self.area_tabs:
            self.log.append(f"Area already downloaded: {area_key}")
            tab_widget = self.area_tabs[area_key]
            self.tabs.setCurrentWidget(tab_widget)
            self.current_area = area_key
            return

        # Create tab
        tab_widget = QTabWidget()
        self.area_tabs[area_key] = tab_widget
        self.area_data[area_key] = {
            'bounds': bounds,
            'folder': None
        }
        self.current_area = area_key
        
        self.tabs.addTab(tab_widget, f"Area {len(self.area_tabs)}")
        self.tabs.setCurrentWidget(tab_widget)

        # Start processing
        center_lat = (bounds['north'] + bounds['south']) / 2
        center_lon = (bounds['east'] + bounds['west']) / 2
        size_deg = max(bounds['north'] - bounds['south'], bounds['east'] - bounds['west'])
        
        coordinates = {
            'center_lat': center_lat,
            'center_lon': center_lon,
            'size_deg': size_deg
        }
        
        self.start_processing(coordinates)

    def start_processing(self, coordinates):
        self.log.append("STARTING PROCESSING")
        
        self.progress_bar.show()
        
        self.worker = WorkerThread(coordinates, self.zoom_spinner.value())
        self.worker.progress.connect(self.log.append)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.start()

    def on_processing_finished(self, success, message):
        self.progress_bar.hide()
        
        if success:
            self.area_data[self.current_area]['folder'] = message
            
            # Enable buttons
            self.btn_segment_orto.setEnabled(True)
            self.btn_process_topo.setEnabled(True)
            self.btn_combine.setEnabled(True)
            self.btn_export_reference.setEnabled(True)
            
            # Auto-show results
            self.show_all_results()
            
            QtWidgets.QMessageBox.information(self, "Success", f"Processing completed!\n\nFolder: {message}")
        else:
            QtWidgets.QMessageBox.critical(self, "Error", f"Processing failed:\n\n{message}")

    def show_all_results(self):
        """Automatycznie pokaż wszystkie wyniki"""
        folder = self.area_data[self.current_area].get('folder')
        if not folder:
            return
        
        tab_widget = self.area_tabs[self.current_area]
        
        # Ortophoto
        orto_path = os.path.join(folder, "image_orto.jpg")
        if os.path.exists(orto_path):
            self._show_image_in_tab(tab_widget, "Ortophoto", orto_path)
        
        # OSM Topo
        osm_path = os.path.join(folder, "segmentation_mask.png")
        if os.path.exists(osm_path):
            self._show_image_in_tab(tab_widget, "OSM Topo", osm_path)
        
        # Segmentation
        seg_path = os.path.join(folder, "segmentation_nn_colored.png")
        if os.path.exists(seg_path):
            self._show_image_in_tab(tab_widget, "Segmentation", seg_path)
        
        # Reference Map
        ref_path = os.path.join(folder, "reference_map_colored.png")
        if os.path.exists(ref_path):
            self._show_image_in_tab(tab_widget, "Reference Map", ref_path)

    def _show_image_in_tab(self, tab_widget, title, image_path):
        """Pokaż obraz w zakładce"""
        with open(image_path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')
        
        ext = os.path.splitext(image_path)[1].lower()
        mime = 'image/jpeg' if ext == '.jpg' else 'image/png'
        
        web = QWebEngineView()
        html = f"<html><body style='margin:0;padding:0;'><img src='data:{mime};base64,{img_data}' style='width:100%;height:100%;object-fit:contain;'></body></html>"
        web.setHtml(html)
        
        tab_widget.addTab(web, title)

    def on_segment_orto(self):
        """Show Ortophoto"""
        if not self.current_area:
            return
        
        folder = self.area_data[self.current_area].get('folder')
        if folder:
            tab_widget = self.area_tabs[self.current_area]
            
            orto_path = os.path.join(folder, "image_orto.jpg")
            if os.path.exists(orto_path):
                for i in range(tab_widget.count()):
                    if tab_widget.tabText(i) == "Ortophoto":
                        tab_widget.setCurrentIndex(i)
                        return

    def on_process_topo(self):
        """Show OSM Topo"""
        if not self.current_area:
            return
        
        folder = self.area_data[self.current_area].get('folder')
        if folder:
            tab_widget = self.area_tabs[self.current_area]
            
            for i in range(tab_widget.count()):
                if tab_widget.tabText(i) == "OSM Topo":
                    tab_widget.setCurrentIndex(i)
                    return
    
    def on_combine_maps(self):
        """Show Reference Map"""
        if not self.current_area:
            return
        
        folder = self.area_data[self.current_area].get('folder')
        if folder:
            tab_widget = self.area_tabs[self.current_area]
            
            for i in range(tab_widget.count()):
                if tab_widget.tabText(i) == "Reference Map":
                    tab_widget.setCurrentIndex(i)
                    return
    
    def on_export_reference(self):
        """Export Reference Map"""
        if not self.current_area:
            self.log.append("No area selected.")
            return
        
        folder = self.area_data[self.current_area].get('folder')
        if not folder:
            return
        
        ref_path = os.path.join(folder, "reference_map.png")
        if not os.path.exists(ref_path):
            self.log.append("Reference map not found")
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "Export Reference Map", "", "PNG files (*.png);;All files (*)")
        if path:
            import shutil
            shutil.copy(ref_path, path)
            
            # Save metadata
            bounds = self.area_data[self.current_area]['bounds']
            metadata = {
                'bounds': bounds,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'area_key': self.current_area,
                'zoom_level': self.zoom_spinner.value()
            }
            
            meta_path = path.rsplit('.', 1)[0] + '_metadata.json'
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.log.append(f"Reference map exported to {path}")
            self.log.append(f"Metadata saved to {meta_path}")

    def add_layer_to_map(self, layer_name, layer_type, temporary=False):
        if not self.geoportal_ready:
            self.pending_layers.append({'name': layer_name, 'type': layer_type, 'temporary': temporary})
            self.log.append(f"Queued layer {layer_name} (map not ready yet)")
            return
        self._add_layer_internal(layer_name, layer_type, temporary)

    def _add_layer_internal(self, layer_name, layer_type, temporary):
        if layer_type.upper() == 'WMS':
            js = (
                "addWMSLayer({"
                f"layerName: '{layer_name}', "
                f"url: '{self.service_url_input.text().strip()}', "
                f"options: {{layers: '{layer_name}', format: 'image/png', transparent: true}}"
                "}, " + ("true" if temporary else "false") + ");"
            )
        else:
            js = (
                "addWMTSLayer({"
                f"layerName: '{layer_name}', "
                f"url: '{self.service_url_input.text().strip()}', "
                "options: {tilematrixSet: 'GoogleMapsCompatible', format: 'image/png'}"
                "}, " + ("true" if temporary else "false") + ");"
            )
        self.web.page().runJavaScript(js)
        self.log.append(f"Added layer {layer_name} as {layer_type}")

    def _demo_load_geoportal(self):
        example_wmts = "https://mapy.geoportal.gov.pl/wss/service/WMTS/guest/wmts/v1/BDOT10k"
        self.service_url_input.setText(example_wmts)
        self.service_type_combo.setCurrentText("WMTS")
        self.on_fetch_capabilities()

    def save_config(self):
        cfg = {
            'service_url': self.service_url_input.text(),
            'service_type': self.service_type_combo.currentText(),
            'settings': self.settings_text.toPlainText(),
            'zoom_level': self.zoom_spinner.value()
        }
        path, _ = QFileDialog.getSaveFileName(self, "Save config", "", "JSON files (*.json);;All files (*)")
        if path:
            with open(path, 'w', encoding='utf8') as f:
                json.dump(cfg, f, indent=2)
            self.log.append(f"Saved config to {path}")

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load config", "", "JSON files (*.json);;All files (*)")
        if path:
            with open(path, 'r', encoding='utf8') as f:
                cfg = json.load(f)
            self.service_url_input.setText(cfg.get('service_url', ''))
            self.service_type_combo.setCurrentText(cfg.get('service_type', 'WMS'))
            self.settings_text.setPlainText(cfg.get('settings', ''))
            if 'zoom_level' in cfg:
                self.zoom_spinner.setValue(cfg['zoom_level'])
            self.log.append(f"Loaded config from {path}")

    def _leaflet_html(self):
        return """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Embedded Leaflet Map</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css"/>
<style>
  html, body, #map { height: 100%; margin:0; padding:0; }
</style>
</head>
<body>
<div id="map"></div>
<script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
<script>
  var map = L.map('map').setView([52.0, 21.0], 6);
  var base = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 19});
  base.addTo(map);

  window._addedLayers = {};

  function setMapCenter(lat, lon, zoom) {
    map.setView([lat, lon], zoom||12);
  }

  function addWMSLayer(opts, temporary) {
    try {
      var id = opts.layerName;
      var layer = L.tileLayer.wms(opts.url, opts.options || {});
      layer.addTo(map);
      if (temporary) {
        layer.setOpacity(0.6);
      }
      window._addedLayers[id] = layer;
      console.log('Added WMS', id);
    } catch (e) {
      console.error(e);
    }
  }

  function addWMTSLayer(opts, temporary) {
    try {
      var id = opts.layerName;
      var tpl = opts.url + "?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=" + id + "&TILEMATRIXSET=GoogleMapsCompatible&FORMAT=image/png&TileMatrix={z}&TileRow={y}&TileCol={x}";
      var layer = L.tileLayer(tpl, {tileSize: 256, maxZoom: 18});
      layer.addTo(map);
      if (temporary) layer.setOpacity(0.6);
      window._addedLayers[id] = layer;
      console.log('Added WMTS', id);
    } catch(e) {
      console.error(e);
    }
  }

  function removeLayer(id) {
    var l = window._addedLayers[id];
    if (l) {
      map.removeLayer(l);
      delete window._addedLayers[id];
      console.log('Removed', id);
    }
  }

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
    win = MapLoader()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()