## 1. Preparation
pip install -qq opencv-contrib-python diffusers transformers git+https://github.com/huggingface/accelerate.git

## 2. Modify hyperparameters and run on the terminal
python main.py --mode canny --prompt "make animation character face" --num_steps 30 --seed 10 --data_dir C:/AI_Study/YAI/Controlnet/datasets --result_dir C:/AI_Study/YAI/Controlnet/result

### * params *
- mode: canny, pose (아직 pose는 구현 안됨)
- prompt: 괄호 안에 원하는 그림 text로 입력
- num_steps, seed: 결과 달라짐
- data_dir: 이미지들이 있는 폴더 (이미지들 바로 상위여야함)
- result_dir: 생성되는 결과 이미지 저장 할 폴더 (없으면 자동 생성됨)
