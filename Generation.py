import torch
import os
import tqdm

from Model import load_model
from DataLoader import load_image, load_image_canny

def generation(args):
  mode = args.mode
  prompt = args.prompt
  num_steps = args.num_steps
  seed = args.seed
  data_dir = args.data_dir
  result_dir = args.result_dir
  
  pipe = load_model(mode)

  if not os.path.exists(result_dir):
    os.mkdir(result_dir)
  
  for img in tqdm.tqdm(os.listdir(data_dir)):
    if img.endswith(".jpg") or img.endswith(".png"):
      img_path = os.path.join(data_dir, img)
      if mode == "canny":
        loaded_img = load_image_canny(img_path)
      elif mode=="pose":
        loaded_img = load_image(img_path) # 아직 수정 필요
      else:
        raise ValueError(f"Invalid mode: {mode}")
        
      out_image = pipe(
          prompt,
          num_inference_steps=num_steps,
          generator=torch.manual_seed(seed),
          image=loaded_img
      ).images[0]
      
      out_image.save(os.path.join(result_dir, img.split(".")[0]+"_out.jpg"))
