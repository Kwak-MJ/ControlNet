## 1. Preparation
requirement.txt 설치

## 2. Terminal command
python main.py --type "electric" --seed 10 --data_dir "None" --img_path C:/AI_Study/YAI/Controlnet/datasets/0016.jpg --result_dir C:/AI_Study/YAI/Controlnet/result_test

### * params *
- type: 생성 원하는 이미지 타입
- seed: NSFW 결과 나오면 조절
- data_dir: 여러 이미지 한 번에 처리할 때, 이미지들이 있는 폴더 (이미지들 바로 상위여야함, 이때 --img_path는 None)
- image_path: 이미지 개별적으로 처리할 때, 이미지 경로 입력 (--data_dir에는 "None" 넣어줘야함)
- result_dir: 생성되는 결과 이미지 저장 할 폴더 (없으면 자동 생성됨)
