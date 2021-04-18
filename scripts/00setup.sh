#!/bin/bash

directory="${0%/*}"

sudo apt-get install -y python3-venv
python3 -m venv "$directory"/venv

source "$directory"/venv/bin/activate
pip3 install -r "$directory"/requirements.txt
deactivate