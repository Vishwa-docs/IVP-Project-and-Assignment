#!/bin/bash

venv_name=".venv"

if [ -d "$venv_name" ]; then
  echo "Virtual environment '$venv_name' found."
  source "$venv_name/bin/activate"
else
  echo "Virtual environment '$venv_name' not found. Creating..."
  python3 -m venv "$venv_name"
  source "$venv_name/bin/activate"
fi

if [ ! -f "requirements.txt" ]; then
  echo "requirements.txt not found. Skipping installation."
else
  echo "Installing libraries from requirements.txt..."
  pip install -r requirements.txt
fi

echo "Virtual environment '$venv_name' is now active."