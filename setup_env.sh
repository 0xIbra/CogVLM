#!/bin/bash

pip3 install -r requirements.txt
pip3 install -r api/requirements.txt --ignore-installed

python3 -m spacy download en