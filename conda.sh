#!/bin/bash

conda create -y -n sketchKeras
conda install -y -n sketchKeras python=3.7 cudatoolkit=10.0 tensorflow-gpu=1.13.1 keras
conda install -y -n sketchKeras -c conda-forge opencv tqdm
