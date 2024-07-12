# MuRAG

## Quick Start

```
conda create -n mirg python=3.11 -y
conda activate
conda env config vars set GEMINI_KEY=<YOUR_GEMINI_KEY_HERE>
pip install -r requirements.txt
python datasets/muMuQA/prepare.py
python main.py
```

## Model Downloads

### Dacl10k Model

Download the following model and move into `ml_models/model_weights/` directory.

https://huggingface.co/spaces/phiyodr/dacl-challenge/blob/main/runs/2023-08-31_rich-paper-12/best_model_cpu.pth
