복사

@echo off
title DeepTracer 빌드
echo ================================
echo  DeepTracer PyInstaller 빌드
echo ================================
 
echo [1/3] 의존성 설치 확인...
pip install pyinstaller PySide6 PySide6-Fluent-Widgets ^
    opencv-python torch torchvision ultralytics ^
    lmdb pillow natsort nltk six
 
echo.
echo [2/3] 이전 빌드 삭제...
if exist dist\DeepTracer rmdir /s /q dist\DeepTracer
if exist build rmdir /s /q build
 
echo.
echo [3/3] PyInstaller 빌드 시작...
pyinstaller deeptracer.spec --clean
 
echo.
if exist dist\DeepTracer\DeepTracer.exe (
    echo ✅ 빌드 성공: dist\DeepTracer\DeepTracer.exe
) else (
    echo ❌ 빌드 실패
)
pause
 