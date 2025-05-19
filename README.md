# HiDream-I1 4Bit Quantized Model

This repository is a fork of `HiDream-I1` quantized to 4 bits, allowing the full model to run in less than 16GB of VRAM. 

The original repository can be found [here](https://github.com/HiDream-ai/HiDream-I1).

> `HiDream-I1` is a new open-source image generative foundation model with 17B parameters that achieves state-of-the-art image generation quality within seconds.

![HiDream-I1 Demo](assets/demo.jpg)

![image](https://github.com/user-attachments/assets/d4715fb9-efe1-40c3-bd4e-dfd626492eea)

## Models

We offer both the full version and distilled models. The parameter size are the same, so they require the same amount of GPU memory to run. However, the distilled models are faster because of reduced number of inference steps.

| Name            | Min VRAM | Steps | HuggingFace                                                                                                                  |
|-----------------|----------|-------|------------------------------------------------------------------------------------------------------------------------------|
| HiDream-I1-Full | 16 GB    | 50    | 🤗 [Original](https://huggingface.co/HiDream-ai/HiDream-I1-Full) / [NF4](https://huggingface.co/azaneko/HiDream-I1-Full-nf4) |
| HiDream-I1-Dev  | 16 GB    | 28    | 🤗 [Original](https://huggingface.co/HiDream-ai/HiDream-I1-Dev) / [NF4](https://huggingface.co/azaneko/HiDream-I1-Dev-nf4)   |
| HiDream-I1-Fast | 16 GB    | 16    | 🤗 [Original](https://huggingface.co/HiDream-ai/HiDream-I1-Fast) / [NF4](https://huggingface.co/azaneko/HiDream-I1-Fast-nf4) |

## Hardware Requirements

- GPU Architecture: NVIDIA `>= Ampere` (e.g. A100, H100, A40, RTX 3090, RTX 4090)
- GPU RAM: `>= 16 GB`
- CPU RAM: `>= 16 GB`

## Quick Start

Simply run:

```
pip install -r requirements.txt
pip install hdi1 --no-build-isolation
```

> [!NOTE]
> It's recommended that you start a new python environment for this package to avoid dependency conflicts.  
> To do that, you can use `conda create -n hdi1 python=3.12` and then `conda activate hdi1`.  
> Or you can use `python3 -m venv venv` and then `source venv/bin/activate` on Linux or `venv\Scripts\activate.bat` in Windows command prompt (`venv\Scripts\activate.ps1` in Windows Powershell).

### Command Line Interface

Then you can run the module to generate images:

``` python 
python -m hdi1 "A cat holding a sign that says 'hello world'"

# or you can specify the model
python -m hdi1 "A cat holding a sign that says 'hello world'" -m fast
```

> [!NOTE]
> The inference script will try to automatically download `meta-llama/Llama-3.1-8B-Instruct` model files. You need to [agree to the license of the Llama model](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on your HuggingFace account and login using `huggingface-cli login` in order to use the automatic downloader.

### Web Dashboard

We also provide a web dashboard for interactive image generation. You can start it by running:

``` python
python -m hdi1.web
```

![Screenshot 2025-04-08 200120](https://github.com/user-attachments/assets/0c464033-5619-489d-b9de-fef5a7119cfc)

## License

The code in this repository and the HiDream-I1 models are licensed under [MIT License](./LICENSE).
