import os
import sys
import pickle
import smtplib
import logging
import configparser
from typing import Dict, Any
import traceback

import pandas as pd
import PyPDF2
import re
from PyQt6.QtCore import (QAbstractTableModel, Qt, QVariant, QThreadPool, 
                          QRunnable, pyqtSignal, QObject, QPropertyAnimation, QEasingCurve)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout,
                             QWidget, QPushButton, QFileDialog, QMessageBox,
                             QInputDialog, QLineEdit, QComboBox, QTableView,
                             QProgressBar, QStatusBar, QShortcut, QStyle, QGraphicsOpacityEffect)
from PyQt6.QtGui import QKeySequence, QIcon, QColor
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pyzbar import pyzbar
from PIL import Image
import cv2

# Setup logging
logging.basicConfig(filename='packing_list_processor.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    """
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)

class Worker(QRunnable):
    """
    Worker thread for running background tasks.
    """
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()

class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid():
            if role == Qt.ItemDataRole.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
            elif role == Qt.ItemDataRole.BackgroundRole:
                if self._data.iloc[index.row(), self._data.columns.get_loc('Checked')]:
                    return Qt.GlobalColor.green  # Highlight checked rows in green
        return QVariant()

    def headerData(self, section, orientation, role):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._data.columns[section]
        return QVariant()

class PackingListProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Packing List Processor")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)

        self.create_buttons()
        self.create_status_bar()

        self.project_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.current_project: str = ""

        self.threadpool = QThreadPool()

        self.create_shortcuts()
        self.setup_drag_drop()

        # Load style sheet
        with open('style.qss', 'r') as f:
            self.setStyleSheet(f.read())

        # Set up animations
        self.setup_animations()

    def setup_animations(self):
        self.fade_effect = QGraphicsOpacityEffect()
        self.central_widget.setGraphicsEffect(self.fade_effect)
        self.fade_animation = QPropertyAnimation(self.fade_effect, b"opacity")
        self.fade_animation.setDuration(500)
        self.fade_animation.setStartValue(0)
        self.fade_animation.setEndValue(1)
        self.fade_animation.setEasingCurve(QEasingCurve.InOutCubic)
        self.fade_animation.start()

    def create_buttons(self):
        """Create and set up the main buttons."""
        button_data = [
            ("create_new", "Create New Project", self.create_new_project),
            ("load_existing", "Load Existing Project", self.load_existing_project),
            ("save", "Save Project", self.save_project),
            ("scan", "Start Scanning", self.start_scanning),
            ("email", "Send Status Email", self.send_status_email),
            ("toggle_theme", "Toggle Dark Mode", self.toggle_dark_mode)
        ]

        for name, text, callback in button_data:
            button = QPushButton(text)
            button.clicked.connect(callback)
            button.setObjectName(name)
            icon = self.style().standardIcon(getattr(QStyle.StandardPixmap, f"SP_{name.upper()}"))
            button.setIcon(icon)
            self.layout.addWidget(button)
            self.add_button_hover_effect(button)

    def add_button_hover_effect(self, button):
        button.setStyleSheet("""
            QPushButton { transition: all 0.3s ease; }
            QPushButton:hover { transform: scale(1.05); }
        """)

    def create_buttons(self):
        """Create and set up the main buttons."""
        button_data = [
            ("create_new", "Create New Project", self.create_new_project),
            ("load_existing", "Load Existing Project", self.load_existing_project),
            ("save", "Save Project", self.save_project),
            ("scan", "Start Scanning", self.start_scanning),
            ("email", "Send Status Email", self.send_status_email),
            ("toggle_theme", "Toggle Dark Mode", self.toggle_dark_mode)
        ]

        for name, text, callback in button_data:
            button = QPushButton(text)
            button.clicked.connect(callback)
            button.setObjectName(name)
            icon = self.style().standardIcon(getattr(QStyle.StandardPixmap, f"SP_{name.upper()}"))
            button.setIcon(icon)
            self.layout.addWidget(button)

    def create_status_bar(self):
        """Create and set up the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.progress_bar.hide()

    def create_shortcuts(self):
        """Create keyboard shortcuts for common actions."""
        shortcuts = [
            (QKeySequence.New, self.create_new_project),
            (QKeySequence.Open, self.load_existing_project),
            (QKeySequence.Save, self.save_project),
            ("Ctrl+B", self.start_scanning),
            ("Ctrl+E", self.send_status_email),
            ("Ctrl+D", self.toggle_dark_mode)
        ]

        for key, callback in shortcuts:
            QShortcut(key, self).activated.connect(callback)

    def setup_drag_drop(self):
        """Set up drag and drop functionality."""
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        for file in files:
            if file.lower().endswith('.pdf'):
                self.process_dropped_pdf(file)

    def process_dropped_pdf(self, file_path):
        """Process a dropped PDF file."""
        try:
            df = self.extract_data(file_path)
            pdf_name = os.path.basename(file_path)
            if not self.current_project:
                self.current_project = "Dropped_PDFs"
            if self.current_project not in self.project_data:
                self.project_data[self.current_project] = {}
            self.project_data[self.current_project][pdf_name] = df
            self.create_tab_for_pdf(self.current_project, pdf_name, df)
            self.status_bar.showMessage(f"Processed dropped file: {pdf_name}", 5000)
        except Exception as e:
            logging.error(f"Error processing dropped file {file_path}: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to process dropped file: {str(e)}")

    def create_new_project(self):
        """Create a new project."""
        project_name, ok = QInputDialog.getText(self, "New Project", "Enter project name:")
        if ok and project_name:
            folder = QFileDialog.getExistingDirectory(self, "Select PackingLists Folder")
            if folder:
                self.current_project = project_name
                worker = Worker(self.process_packing_lists, folder, project_name)
                worker.signals.finished.connect(self.on_project_load_complete)
                worker.signals.error.connect(self.on_worker_error)
                self.threadpool.start(worker)
                self.status_bar.showMessage("Processing packing lists...", 5000)

    def load_existing_project(self):
        """Load an existing project."""
        load_file, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "Project Files (*.pkl)")
        if load_file:
            worker = Worker(self.load_project, load_file)
            worker.signals.finished.connect(self.on_project_load_complete)
            worker.signals.error.connect(self.on_worker_error)
            self.threadpool.start(worker)
            self.status_bar.showMessage("Loading project...", 5000)

    def load_project(self, load_file):
        """Load a project from a file."""
        try:
            with open(load_file, 'rb') as f:
                loaded_data = pickle.load(f)
            
            self.project_data = loaded_data
            self.current_project = os.path.splitext(os.path.basename(load_file))[0]
            
            # Clear existing tabs
            self.tab_widget.clear()
            
            # Recreate tabs for the loaded project
            for project_name, pdfs in self.project_data.items():
                for pdf_name, df in pdfs.items():
                    self.create_tab_for_pdf(project_name, pdf_name, df)
            
            return "Project loaded successfully"
        except Exception as e:
            logging.error(f"Error loading project: {str(e)}")
            raise

    def on_project_load_complete(self):
        """Handle completion of project loading."""
        self.status_bar.showMessage("Project loaded successfully", 5000)
        QMessageBox.information(self, "Load Successful", f"Project '{self.current_project}' loaded successfully.")

    def on_worker_error(self, error_info):
        """Handle worker thread errors."""
        exctype, value, _ = error_info
        logging.error(f"Worker thread error: {exctype}, {value}")
        QMessageBox.critical(self, "Error", f"An error occurred: {str(value)}")

    def save_project(self):
        """Save the current project."""
        if not self.project_data:
            QMessageBox.warning(self, "No Project", "There is no project data to save.")
            return

        if self.current_project:
            default_name = f"{self.current_project}.pkl"
        else:
            default_name = "project.pkl"

        save_file, _ = QFileDialog.getSaveFileName(self, "Save Project", default_name, "Project Files (*.pkl)")
        if save_file:
            worker = Worker(self.save_project_file, save_file)
            worker.signals.finished.connect(lambda: self.status_bar.showMessage("Project saved successfully", 5000))
            worker.signals.error.connect(self.on_worker_error)
            self.threadpool.start(worker)
            self.status_bar.showMessage("Saving project...", 5000)

    def save_project_file(self, save_file):
        """Save the project to a file."""
        try:
            with open(save_file, 'wb') as f:
                pickle.dump(self.project_data, f)
            return "Project saved successfully"
        except Exception as e:
            logging.error(f"Error saving project: {str(e)}")
            raise

    def process_packing_lists(self, folder, project_name):
        """Process all PDF files in the given folder."""
        pdf_files = [f for f in os.listdir(folder) if f.lower().endswith('.pdf')]
        self.project_data[project_name] = {}

        for pdf_file in pdf_files:
            file_path = os.path.join(folder, pdf_file)
            df = self.extract_data(file_path)
            self.project_data[project_name][pdf_file] = df
            self.create_tab_for_pdf(project_name, pdf_file, df)

    def extract_data(self, pdf_file):
        """Extract data from a PDF file."""
        with open(pdf_file, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            all_text = ''
            for page in reader.pages:
                all_text += page.extract_text()

        # Extract relevant information
        sales_order_match = re.search(r'SALES ORDER NUMBER\n(\d+)', all_text)
        sales_order = sales_order_match.group(1) if sales_order_match else 'N/A'

        purchase_order_match = re.search(r'PURCHASE ORDER NO\n([\w-]+)', all_text)
        purchase_order = purchase_order_match.group(1) if purchase_order_match else 'N/A'

        # Extract item details
        items = []
        item_pattern = re.compile(r'(\d+)\s+([\w\s]+)\s+(\d+\.\d+)\s+([\d\s]+)')
        for match in item_pattern.finditer(all_text):
            item_number, description, quantity, box_numbers = match.groups()
            box_numbers = box_numbers.split()
            for box in box_numbers:
                items.append({
                    'Item Number': item_number,
                    'Description': description.strip(),
                    'Quantity': float(quantity),
                    'Box Number': box,
                    'Checked': False
                })

        # Create DataFrame
        df = pd.DataFrame(items)
        df['Sales Order'] = sales_order
        df['Purchase Order'] = purchase_order
        
        return df

    def create_tab_for_pdf(self, project_name, pdf_name, df):
        """Create a new tab for a PDF file."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        # Create a container widget for the table and search box
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container.setStyleSheet("""
            QWidget {
                background-color: #fff8e7;
                border-radius: 10px;
                padding: 10px;
            }
        """)

        table_view = QTableView()
        model = PandasModel(df)
        table_view.setModel(model)
        table_view.setAlternatingRowColors(True)
        table_view.setStyleSheet("""
            QTableView {
                border: none;
                gridline-color: #d2b48c;
            }
            QHeaderView::section {
                background-color: #d2b48c;
                color: #4a3728;
                padding: 5px;
                border: 1px solid #8b4513;
            }
        """)
        container_layout.addWidget(table_view)

        # Add search functionality
        search_box = QLineEdit()
        search_box.setPlaceholderText("Search items...")
        search_box.textChanged.connect(lambda text: self.search_items(text, table_view))
        search_box.setStyleSheet("""
            QLineEdit {
                border: 2px solid #d2b48c;
                border-radius: 15px;
                padding: 5px 10px;
                background-color: #fff8e7;
                color: #4a3728;
            }
        """)
        container_layout.addWidget(search_box)

        tab_layout.addWidget(container)
        self.tab_widget.addTab(tab, pdf_name)

        # Add fade-in animation for the new tab
        opacity_effect = QGraphicsOpacityEffect(tab)
        tab.setGraphicsEffect(opacity_effect)
        fade_in = QPropertyAnimation(opacity_effect, b"opacity")
        fade_in.setDuration(500)
        fade_in.setStartValue(0)
        fade_in.setEndValue(1)
        fade_in.setEasingCurve(QEasingCurve.InOutCubic)
        fade_in.start()

    def search_items(self, text, table_view):
        """Search for items in the table view."""
        model = table_view.model()
        for row in range(model.rowCount()):
            match = False
            for column in range(model.columnCount()):
                item = model.data(model.index(row, column), Qt.ItemDataRole.DisplayRole)
                if text.lower() in str(item).lower():
                    match = True
                    break
            table_view.setRowHidden(row, not match)

    def start_scanning(self):
        """Start the barcode scanning process."""
        if not self.project_data:
            QMessageBox.warning(self, "No Project", "Please create or load a project first.")
            return

        worker = Worker(self.scan_barcodes)
        worker.signals.finished.connect(lambda: self.status_bar.showMessage("Scanning completed", 5000))
        worker.signals.error.connect(self.on_worker_error)
        self.threadpool.start(worker)
        self.status_bar.showMessage("Scanning barcodes...", 5000)

    def scan_barcodes(self):
        """Scan barcodes using the camera."""
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                raise Exception("Failed to capture frame from camera.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            barcodes = pyzbar.decode(gray)

            for barcode in barcodes:
                barcode_data = barcode.data.decode('utf-8')
                
                # Draw a rectangle around the barcode
                (x, y, w, h) = barcode.rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Put the barcode data on the image
                cv2.putText(frame, barcode_data, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                for project_name, pdfs in self.project_data.items():
                    for pdf_name, df in pdfs.items():
                        if barcode_data in df['Box Number'].values:
                            df.loc[df['Box Number'] == barcode_data, 'Checked'] = True
                            self.update_table_view(project_name, pdf_name)
                            self.status_bar.showMessage(f"Box {barcode_data} has been scanned and marked as checked.", 3000)

            cv2.imshow("Barcode Scanner", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def update_table_view(self, project_name, pdf_name):
        """Update the table view after scanning a barcode."""
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == pdf_name:
                tab = self.tab_widget.widget(i)
                table_view = tab.findChild(QTableView)
                if table_view:
                    df = self.project_data[project_name][pdf_name]
                    model = PandasModel(df)
                    table_view.setModel(model)
                    
                    # Highlight the newly checked row
                    for row in range(model.rowCount()):
                        if model.data(model.index(row, model.columnCount() - 1), Qt.ItemDataRole.DisplayRole) == 'True':
                            table_view.selectRow(row)
                            table_view.scrollTo(model.index(row, 0))
                            break
                break

    def update_table_view(self, project_name, pdf_name):
        """Update the table view after scanning a barcode."""
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == pdf_name:
                tab = self.tab_widget.widget(i)
                table_view = tab.findChild(QTableView)
                if table_view:
                    df = self.project_data[project_name][pdf_name]
                    model = PandasModel(df)
                    table_view.setModel(model)
                break

    def send_status_email(self):
        """Send a status email."""
        if not self.project_data:
            QMessageBox.warning(self, "No Project", "Please create or load a project first.")
            return

        recipient, ok = QInputDialog.getText(self, "Email Recipient", "Enter recipient email address:")
        if ok and recipient:
            subject = f"Project Status: {self.current_project}"
            body = self.generate_status_summary()
            
            worker = Worker(self.send_email, recipient, subject, body)
            worker.signals.finished.connect(lambda: self.status_bar.showMessage("Email sent successfully", 5000))
            worker.signals.error.connect(self.on_worker_error)
            self.threadpool.start(worker)
            self.status_bar.showMessage("Sending email...", 5000)

    def generate_status_summary(self):
        """Generate a summary of the project status."""
        summary = f"<h2>Project Status for {self.current_project}</h2>"
        
        for pdf_name, df in self.project_data[self.current_project].items():
            total_items = len(df)
            checked_items = df['Checked'].sum()
            completion_percentage = (checked_items / total_items) * 100 if total_items > 0 else 0
            
            summary += f"<h3>PDF: {pdf_name}</h3>"
            summary += f"<p>Total Items: {total_items}<br>"
            summary += f"Checked Items: {checked_items}<br>"
            summary += f"Completion: {completion_percentage:.2f}%</p>"
            
            summary += self.generate_progress_bar(completion_percentage)
            
            if completion_percentage < 100:
                missing_items = df[~df['Checked']]
                summary += "<h4>Missing Items:</h4><ul>"
                for _, row in missing_items.iterrows():
                    summary += f"<li>{row['Description']} (Box: {row['Box Number']})</li>"
                summary += "</ul>"
            
            summary += "<hr>"
        
        return summary

    def generate_progress_bar(self, percentage):
        """Generate an HTML progress bar."""
        return f"""
        <div style="background-color: #f0f0f0; border-radius: 13px; padding: 3px; width: 300px;">
            <div style="background-color: #8b4513; width: {percentage}%; height: 20px; border-radius: 10px;">
                <div style="color: white; text-align: center; padding-top: 2px;">{percentage:.1f}%</div>
            </div>
        </div>
        """

    def show_status_summary(self):
        """Show the status summary in a message box."""
        summary = self.generate_status_summary()
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Project Status Summary")
        msg_box.setText(summary)
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #f5e8d3;
            }
            QLabel {
                color: #4a3728;
            }
        """)
        msg_box.exec()

    def send_email(self, recipient, subject, body):
        """Send an email with the project status."""
        try:
            sender_email = config.get('Email', 'sender_email')
            password = config.get('Email', 'password')
            smtp_server = config.get('Email', 'smtp_server')
            smtp_port = config.getint('Email', 'smtp_port')
            
            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = recipient
            message["Subject"] = subject
            message.attach(MIMEText(body, "plain"))
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, password)
                server.sendmail(sender_email, recipient, message.as_string())
            
            return "Email sent successfully"
        except Exception as e:
            logging.error(f"Error sending email: {str(e)}")
            raise

    def toggle_dark_mode(self):
        """Toggle between light and dark mode."""
        if self.palette().color(self.backgroundRole()).lightness() > 128:
            # Switch to dark mode
            self.setStyleSheet("""
                QWidget { background-color: #2b2b2b; color: #f5e8d3; }
                QPushButton { background-color: #8b4513; border: 1px solid #a0522d; color: #f5e8d3; }
                QTableView { background-color: #3a3a3a; alternate-background-color: #2b2b2b; }
                QTabWidget::pane { border: 1px solid #a0522d; background-color: #2b2b2b; }
                QTabBar::tab { background-color: #8b4513; color: #f5e8d3; }
                QTabBar::tab:selected { background-color: #2b2b2b; }
                QStatusBar { background-color: #8b4513; color: #f5e8d3; }
                QProgressBar { border: 2px solid #a0522d; background-color: #2b2b2b; }
                QProgressBar::chunk { background-color: #8b4513; }
                QLineEdit, QComboBox { background-color: #3a3a3a; border: 1px solid #a0522d; color: #f5e8d3; }
            """)
        else:
            # Switch to light mode
            with open('style.qss', 'r') as f:
                self.setStyleSheet(f.read())
        
        self.fade_animation.start()
        self.status_bar.showMessage("Theme changed", 3000)

// Update the main function to handle Windows-specific settings
def main():
    """Main function to run the application."""
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Use Fusion style for better cross-platform consistency
    
    # Set the default font for the entire application
    font = app.font()
    font.setPointSize(10)  # Adjust the size as needed
    app.setFont(font)
    
    window = PackingListProcessor()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()