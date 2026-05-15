#!/bin/bash
source venv/bin/activate
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
python pangea_inference_bridge.py
