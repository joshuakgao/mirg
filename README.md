# MuRAG

## Quick Start

```
conda create -n mirg python=3.9 -y
conda activate mirg
conda env config vars set GEMINI_KEY=<YOUR_GEMINI_KEY_HERE>
conda install pytorch torchvision pytorch-cuda=12.1 faiss-gpu=1.8.0 -c pytorch -c nvidia -y
pip install -r requirements.txt

# install Ollama with: https://ollama.com/download
curl -fsSL https://ollama.com/install.sh | sh  # only for linux
ollama serve
```

## Model Downloads

### Dacl10k Model

Download the following model and move into `ml_models/model_weights/` directory.

https://huggingface.co/spaces/phiyodr/dacl-challenge/blob/main/runs/2023-08-31_rich-paper-12/best_model_cpu.pth
