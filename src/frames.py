from PIL import Image
from torchvision import transforms

# Frame preprocessing pipeline

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)


def preprocess_frame(frame):
    """
    Convert a raw RGB frame (HxWxC numpy array) into a 1x3x128x128 tensor.
    """
    img = Image.fromarray(frame)
    tensor = transform(img)
    return tensor.unsqueeze(0)  # shape: [1, 3, 128, 128]
