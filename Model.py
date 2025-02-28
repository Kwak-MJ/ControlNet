from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, HeunDiscreteScheduler, DDPMScheduler, EulerDiscreteScheduler, KDPM2DiscreteScheduler, EulerAncestralDiscreteScheduler, DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler, UniPCMultistepScheduler, LMSDiscreteScheduler, PNDMScheduler, DPMSolverSinglestepScheduler
import torch

def load_model(mode:str):
    if mode == "canny":
        controlnet_model = "lllyasviel/sd-controlnet-canny"
    elif mode == "pose":
        controlnet_model = "fusing/stable-diffusion-v1-5-controlnet-openpose"
    else:
        raise ValueError(f"Invalid mode: {mode}")

    sd_model = "Lykon/DreamShaper"
    # sd_model = "runwayml/stable-diffusion-v1-5"

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
