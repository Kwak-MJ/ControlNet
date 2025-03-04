from PIL import Image
import os


def resize_images(input_folder, output_folder, size=(256, 256)):
    # 출력 폴더 생성 (없으면 자동 생성)
    os.makedirs(output_folder, exist_ok=True)

    # 폴더 내 이미지 파일 처리
    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        try:
            # 이미지 열기
            with Image.open(input_path) as img:
                # RGB로 변환 (일부 PNG는 RGBA 채널이 있어서 변환 필요)
                img = img.convert("RGB")
                # 256x256으로 크기 조정
                img_resized = img.resize(size, Image.LANCZOS)
                # 변환된 이미지 저장
                img_resized.save(output_path)
                print(f"Resized: {file_name} -> {output_path}")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")


# 사용 예시
# 원본 이미지 폴더 경로
input_folder = "C:\\AI_Study\\YAI\\2025yaiCV-toy-project-main\\human"
# 변환된 이미지 저장 폴더
output_folder = "C:\\AI_Study\\YAI\\2025yaiCV-toy-project-main\\human2"
resize_images(input_folder, output_folder)


# def rotate_images(input_folder, output_folder, angle):
#     os.makedirs(output_folder, exist_ok=True)  # 출력 폴더가 없으면 생성

#     for file_name in os.listdir(input_folder):
#         input_path = os.path.join(input_folder, file_name)
#         output_path = os.path.join(output_folder, file_name)

#         try:
#             with Image.open(input_path) as img:
#                 img = img.convert("RGB")  # RGBA → RGB 변환
#                 img_rotated = img.rotate(
#                     angle, expand=True)  # ✅ expand=True 설정
#                 img_rotated.save(output_path)
#                 print(f"Rotated: {file_name} -> {output_path}")

#         except Exception as e:
#             print(f"Error processing {file_name}: {e}")


# # 사용 예시
# rotate_images(input_folder, output_folder, angle=270)
