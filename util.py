import torchvision.transforms as transforms
from PIL import Image

def get_transforms(train=True):
    """
    학습 시에는 augmentation을, inference 시에는 resize 및 normalization 수행.
    """
    if train:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    return transform

def load_image(image_path, transform=None):
    """
    주어진 경로의 이미지를 열고, transform이 주어지면 적용하여 반환
    """
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    return image
