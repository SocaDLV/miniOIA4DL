@echo off
set "folder=unit_tests\"
set "PYTHON_EXE="C:\Users\Ivan\AppData\Local\Programs\Python\Python311\python.exe""

cd ..
for %%f in (%folder%*.py) do (
    set PYTHONPATH=.
    %PYTHON_EXE% "%folder%%%~nxf"
)
pause