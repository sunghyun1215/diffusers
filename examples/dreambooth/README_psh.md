# DreamBooth training example

## 환경설정

### 라이브러리 버전

CUDA : 11.8

Pytorch : 2.2.1+cu118

Nvidia Drvier : 525.147.05

### 파이썬 가상환경 생성

프로젝트를 진행할 폴더 내에서 다음과 같은 명령어로 가상환경을 구성:

- 생성

  ``` bash
  python3 -m venv .env
  ```

- 활성화

  ```bash
  source .env/bin/activate
  ```
- cuda11.8 버전에 해당하는 pytorch 설치 (홈페이지)
- networkx 설치

  ```bash
  pip3 install networkx
  ```  

### Diffusers Installation

**Install with conda(필수 아닌 것 같음)**

가상환경 활성화 후 다음과 같음 명령어 수행 

```bash
conda install -c conda-forge diffusers
```



**Install from source**

Accelerate Install

```bash
pip3 install accelerate
```



Source Code Install

````bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
git pull
````



In diffusers folder,

```bash
pip3 install -e ".[torch]"
```



## Training

### Accelerate 초기화

```bash
accelerate config default
```



### Fine-tuning

환경변수 설정

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path_to_training_images"
export OUTPUT_DIR="path_to_saved_model"
```

- 학습 데이터는 'path_to_training_images' 폴더에 저장
- 학습 결과 모델은 'path_to_saved_model' 폴더에 저장됨



Test script

```bash
accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400
```

- sks : 학습에 사용되는 특수 식별자



## Inference

### StableDiffusion 파이프라인 사용

Source Code:

```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "path_to_saved_model"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A photo of sks dog in a bucket"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("dog-bucket.png")
```

