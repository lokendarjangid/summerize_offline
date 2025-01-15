import sys
import socket
from PyQt5.QtWidgets import QApplication, QMessageBox
from main_app import PDFSummarizerApp # Import the main application
def is_already_running():
    """Check if the application is already running using a socket."""
    try:
        # Attempt to bind to a specific port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 65432))  # Use an unused port
        return sock
    except socket.error:
        return None

if __name__ == "__main__":
    lock = is_already_running()
    if not lock:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Warning")
        msg.setText("The application is already running.")
        msg.exec_()
        sys.exit(1)

    app = QApplication(sys.argv)
    window = PDFSummarizerApp()
    window.show()
    sys.exit(app.exec_())
