import os
import argparse
from inference import predict

def analyze_human_images(human_folder, model_path):
    """
    human 폴더 내의 모든 이미지에 대해 inference를 수행하고, 각 이미지의 클래스 확률 분포를 출력
    """
    image_files = [os.path.join(human_folder, f) for f in os.listdir(human_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    results = {}
    for image_path in image_files:
        top_class, percentages = predict(image_path, model_path)
        results[image_path] = percentages
        print(f"Image: {image_path}")
        for cls, pct in percentages.items():
            print(f"  {cls}: {pct:.2f}%")
        print("-" * 30)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='human 이미지 폴더에 대해 전체 분석 수행')
    parser.add_argument('--folder', type=str, required=True, help='human 이미지 폴더 경로')
    parser.add_argument('--model', type=str, default='models/classifier.pth', help='학습된 모델 파일 경로')
    args = parser.parse_args()
    analyze_human_images(args.folder, args.model)
