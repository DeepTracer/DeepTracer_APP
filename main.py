import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from qfluentwidgets import setTheme, Theme
from ui.main_window import MainWindow
import os
# OpenCV에 딸려오는 Qt 플러그인이 PyQt5와 충돌하는 문제 방지
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ["QT_PLUGIN_PATH"] = ""
 
def main():
    # 고DPI 지원
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
 
    app = QApplication(sys.argv)
    app.setApplicationName("DeepTracer")
    app.setApplicationVersion("1.0.0")
 
    setTheme(Theme.DARK)
 
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
 
 
if __name__ == "__main__":
    main()
 