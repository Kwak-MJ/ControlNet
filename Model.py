from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import PNDMScheduler
import torch

def load_model():
    controlnet_model = "lllyasviel/sd-controlnet-canny"
    sd_model = "Lykon/DreamShaper"

    controlnet = ControlNetModel.from_pretrained(
        controlnet_model,
        torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        sd_model, controlnet=controlnet, torch_dtype=torch.float16
    )

    pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    return pipe
