## 1. Preparation
- pip install -r requirements.txt
- input image 사이즈 256x256 맞추기 (resize.py 사용)

## 2. Terminal command
- Classification (ResNet)
python analysis.py --folder=human --model=models/classifier.pth  -- 폴더 내 모든
python inference.py --image=human/sample.jpg --model=models/classifier.pth  -- 단일 이미지

- Generation (ControlNet)
python main.py --type "electric" --seed 20 --data_dir=human --img_path=None --result_dir=result --폴더 내 모든
python main.py --type "electric" --seed 10 --data_dir=None --img_path=human/sample.jpg --result_dir=result -- 단일 이미지

### * params *
1. Classification
   - folder: 폴더 내 모든 이미지 분류할 때
   - image: 특정 이미지만 분류할 때
   - model: 분류 모델 경로

2. Generation
- type: 생성 원하는 이미지 타입
- seed: NSFW 결과 나오면 조절
- data_dir: 여러 이미지 한 번에 처리할 때, 이미지들이 있는 폴더 (이미지들 바로 상위여야함, 이때 --img_path는 None)
- image_path: 이미지 개별적으로 처리할 때, 이미지 경로 입력 (--data_dir에는 "None" 넣어줘야함)
- result_dir: 생성되는 결과 이미지 저장 할 폴더 (없으면 자동 생성됨)
