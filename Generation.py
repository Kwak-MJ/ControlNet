import torch
import os
import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from Model import load_model
from DataLoader import load_image, load_image_canny

def generation(args):
  type = args.type
  seed = args.seed
  data_dir = args.data_dir
  img_path = args.img_path
  result_dir = args.result_dir
  
  pipe = load_model()

  if not os.path.exists(result_dir):
    os.mkdir(result_dir)

  if type == "electric":
    prompt = "masterpiece, best quality, ultra-detailed, illustration, 8k, realistic person with a yellow lightning and sparks"
  elif type == "fire":
    prompt = "masterpiece, best quality, ultra-detailed, illustration, 8k, realistic human with red fire and flames"
  elif type == "water":
    prompt = "masterpiece, best quality, ultra-detailed, illustration, 8k, realistic human of water type"
  elif type == "grass":
    prompt = "masterpiece, best quality, ultra-detailed, illustration, 8k, realistic human with grass, lawn, plant and tree type"
  elif type == "ground":
    prompt = "masterpiece, best quality, ultra-detailed, illustration, 8k, realistic human of mud, dirt and soil type"
  else:
    print("Type Error")
    return
  
  if data_dir != "None":
      for img in tqdm.tqdm(os.listdir(data_dir)):
        if img.endswith(".jpg") or img.endswith(".png"):
          image_path = os.path.join(data_dir, img)

          original_img = load_image(image_path)
          canny_img = load_image_canny(image_path)
            
          out_img = pipe(
              prompt,
              num_inference_steps=30,
              generator=torch.manual_seed(seed),
              image=canny_img
          ).images[0]
  elif data_dir == "None" and img_path != "None":
      img = img_path.split("/")[-1]
      image_path = img_path
      original_img = load_image(image_path)
      canny_img = load_image_canny(image_path)

      out_img = pipe(
              prompt,
              num_inference_steps=30,
              generator=torch.manual_seed(seed),
              image=canny_img
          ).images[0]
  else:
      print("Path Error")
      return


  # 원본, canny, output 이미지 합쳐서 저장
  (Image.fromarray(np.concatenate([original_img, canny_img, out_img], axis=1))).save(os.path.join(result_dir, img.split(".")[0]+"_out.png"))

