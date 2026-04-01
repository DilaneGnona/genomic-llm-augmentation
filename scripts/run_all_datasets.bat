@echo off
set "PYTHON=python"

REM 确保在正确的工作目录中运行
cd /d c:\Users\OMEN\Desktop\experiment_snp

echo Starting modeling for all datasets...
echo =

REM 为每个数据集运行建模管道
echo Running for pepper dataset...
echo ------------------------
%PYTHON% scripts\unified_modeling_pipeline.py --dataset pepper
if %errorlevel% neq 0 (
    echo ERROR: Failed to run modeling for pepper dataset
    goto end
)

echo. 
echo Running for pepper_10611831 dataset...
echo ------------------------
%PYTHON% scripts\unified_modeling_pipeline.py --dataset pepper_10611831
if %errorlevel% neq 0 (
    echo ERROR: Failed to run modeling for pepper_10611831 dataset
    goto end
)

echo. 
echo Running for ipk_out_raw dataset...
echo ------------------------
%PYTHON% scripts\unified_modeling_pipeline.py --dataset ipk_out_raw
if %errorlevel% neq 0 (
    echo ERROR: Failed to run modeling for ipk_out_raw dataset
    goto end
)

echo. 
echo All modeling tasks completed successfully!
echo Check the results in 03_modeling_results directory.
goto finish

:end
echo. 
echo Modeling tasks aborted due to errors.

:finish
echo. 
echo Press any key to exit...
pause > nul