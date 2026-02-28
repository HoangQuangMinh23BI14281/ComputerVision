@echo off
cls
chcp 65001 >nul
echo ========================================================
echo          SROIE OCR - Pure PyTorch Pipeline
echo          DBNet++ (Detection) + SVTR (Recognition)
echo ========================================================
echo.
echo  [1] Install Dependencies (PyTorch ^& Requirements)
echo  [2] Prepare Dataset 
echo  [3] Train Detection Model (DBNet++)
echo  [4] Train Recognition Model (SVTR)
echo  [5] Run Inference / Test
echo  [6] Exit
echo.
set /p choice="Select option [1-6]: "

if "%choice%"=="1" goto SETUP
if "%choice%"=="2" goto PREPDATA
if "%choice%"=="3" goto TRAINDET
if "%choice%"=="4" goto TRAINREC
if "%choice%"=="5" goto INFERENCE
if "%choice%"=="6" exit /b
echo Invalid option.
pause
exit /b

:: ============================================================
:: [1] SETUP ENVIRONMENT
:: ============================================================
:SETUP
cls
echo ========================================================
echo          SETUP PYTORCH ENVIRONMENT
echo ========================================================
echo.

if exist .venv (
    echo [INFO] Virtual environment '.venv' already exists. We will reuse it.
    echo.
) else (
    echo [INFO] Creating new virtual environment '.venv' ^(Python 3.12^)...
    echo --------------------------------------------------------
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo [WARN] Python 3.12 via 'py' Launcher failed. Trying generic 'python'...
        python -m venv .venv
    )
    echo [OK] Environment created.
    echo.
)

echo [INFO] Activating environment...
call .venv\Scripts\activate

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo [INFO] Installing PyTorch with CUDA 12.1 support...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo [INFO] Installing required libraries (OpenCV, Albumentations, Shapely)...
python -m pip install opencv-python albumentations shapely tqdm matplotlib pyclipper

echo.
echo ========================================================
echo [SUCCESS] PyTorch Setup Complete! 
echo ========================================================
pause
goto :EOF

:: ============================================================
:: [2] PREPARE DATASET
:: ============================================================
:PREPDATA
cls
echo ========================================================
echo          PREPARING DATASET
echo ========================================================
echo.
call .venv\Scripts\activate
echo Running src/dataset/prep_data.py...
python src/dataset/prep_data.py
echo.
pause
goto :EOF

:: ============================================================
:: [3] TRAIN DETECTION (DBNet++)
:: ============================================================
:TRAINDET
cls
echo ========================================================
echo          TRAIN DETECTION - DBNet++
echo ========================================================
echo.
call .venv\Scripts\activate
set PYTHONPATH=%cd%
python src/train_det.py
pause
goto :EOF

:: ============================================================
:: [4] TRAIN RECOGNITION (SVTR)
:: ============================================================
:TRAINREC
cls
echo ========================================================
echo          TRAIN RECOGNITION - SVTR
echo ========================================================
echo.
call .venv\Scripts\activate
set PYTHONPATH=%cd%
python src/train_rec.py
pause
goto :EOF


:: ============================================================
:: [5] INFERENCE
:: ============================================================
:INFERENCE
cls
echo ========================================================
echo          RUN INFERENCE
echo ========================================================
echo.
call .venv\Scripts\activate
set PYTHONPATH=%cd%
set test_img=
set /p test_img="Enter path to test image (Leave empty for random): "
if "%test_img%"=="" (
    python inference.py
) else (
    python inference.py --image "%test_img%"
)
pause
goto :EOF
