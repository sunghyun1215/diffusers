from diffusers import StableDiffusionPipeline
import torch

model_id = "path_to_saved_model"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A photo of sks men shaking hands"
image = pipe(prompt, num_inference_steps=50, guidance_scale=6.0).images[0]

image.save("thermal_test.png")
