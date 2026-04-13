from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication

from qfluentwidgets import (
    FluentWindow,
    FluentIcon as FIF,
    NavigationItemPosition,
    SplashScreen,
)

from ui.pages.home_page     import HomePage
from ui.pages.result_page   import ResultPage
from ui.pages.timeline_page import TimelinePage
from ui.pages.settings_page import SettingsPage


class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepTracer")
        self.resize(1280, 800)
        self.setMinimumSize(1024, 640)

        # 스플래시 화면
        self.splashScreen = SplashScreen(self.windowIcon(), self)
        self.splashScreen.setIconSize(QSize(102, 102))
        self.show()

        # 페이지 생성
        self.home_page     = HomePage(self)
        self.result_page   = ResultPage(self)
        self.timeline_page = TimelinePage(self)
        self.settings_page = SettingsPage(self)

        self._init_navigation()
        self._connect_signals()

        self.splashScreen.finish()

    # ─────────────────────────────────────────────
    def _init_navigation(self):
        self.addSubInterface(self.home_page,     FIF.HOME,     "홈",          NavigationItemPosition.TOP)
        self.addSubInterface(self.result_page,   FIF.DOCUMENT, "분석 결과",    NavigationItemPosition.TOP)
        self.addSubInterface(self.timeline_page, FIF.HISTORY,  "타임라인",     NavigationItemPosition.TOP)
        self.addSubInterface(self.settings_page, FIF.SETTING,  "설정",        NavigationItemPosition.BOTTOM)

        self.navigationInterface.setCurrentItem(self.home_page.objectName())

    # ─────────────────────────────────────────────
    def _connect_signals(self):
        # 홈 → 분석 완료 시 결과 페이지로 이동
        self.home_page.analysis_finished.connect(self._on_analysis_finished)

    def _on_analysis_finished(self, integrated: dict, plate_video_path: str):
        self.result_page.load_result(integrated, plate_video_path)
        self.timeline_page.load_timeline(integrated.get("timeline", []))
        # 결과 페이지로 자동 이동
        self.switchTo(self.result_page)