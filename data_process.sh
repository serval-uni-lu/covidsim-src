#! /bin/bash
pip install -r requirements.txt
mkdir -p /covidsim-data/data/processed
python3 -m src.datasets.$1_process
