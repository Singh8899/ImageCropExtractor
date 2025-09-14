import runpod
from model import CropInference


def handler(event):
    """Handler per RunPod Serverless"""
    input_data = event["input"]
    image = input_data["image"]
    return cropper.infer(image)


if __name__ == "__main__":
    cropper = CropInference()
    runpod.serverless.start({"handler": handler})
