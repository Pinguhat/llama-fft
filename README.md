# Llama FFT Block-Circulant Project

This project explores the use of FFT-based block-circulant linear layers
to reduce computational complexity and memory usage in Large Language Models,
specifically Llama 2.

## Goals
- Analyze linear layers in Llama 2
- Replace dense matrix multiplications with block-circulant structures
- Use FFT to accelerate matrix-vector multiplication
- Evaluate theoretical and practical savings

## Project Structure
llama-fft/
├── src/          # Core implementation
├── tests/        # Unit tests
├── notebooks/    # Experiments and analysis
├── docs/         # Documentation / seminar material


## Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Run
python src/run.py

## Author
Lukas


