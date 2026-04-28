@echo off
cd /d C:\Data\Development\Python\McTheoryApps\praxis
.venv\Scripts\python.exe -m engines.sentiment_tracker collect >> data\sentiment_collector.log 2>&1
