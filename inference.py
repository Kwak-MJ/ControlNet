import torch
from model import build_model
from util import load_image, get_transforms
import argparse

CLASS_NAMES = ['water', 'fire', 'electric', 'earth', 'grass']

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transforms(train=False)
    image = load_image(image_path, transform=transform)
    image = image.unsqueeze(0)  # 배치 차원 추가
    image = image.to(device)

    model = load_model(model_path)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0)
    
    # 각 클래스에 대한 확률을 백분율로 변환
    percentages = {CLASS_NAMES[i]: float(probabilities[i] * 100) for i in range(len(CLASS_NAMES))}
    top_class = max(percentages, key=percentages.get)
    return top_class, percentages

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='단일 이미지에 대해 분류 예측 수행')
    parser.add_argument('--image', type=str, required=True, help='이미지 파일 경로')
    parser.add_argument('--model', type=str, default='models/classifier.pth', help='학습된 모델 파일 경로')
    args = parser.parse_args()
    
    top_class, percentages = predict(args.image, args.model)
    print("Prediction:", top_class)
    print("Probabilities:")
    for cls, pct in percentages.items():
        print(f"{cls}: {pct:.2f}%")
