import runpod
from model import CropInference
import json

def handler(event):
    #   This function processes incoming requests to your Serverless endpoint.
    #
    #    Args:
    #        event (dict): Contains the input data and request metadata
    #
    #    Returns:
    #       Any: The result to be returned to the client

    # Extract input data
    print(f"Worker Start")
    input = event["input"]

    image = input["image"]

    bboxes = cropper.infer(image)

    return bboxes


# Start the Serverless function when the script is run
if __name__ == "__main__":
    cropper = CropInference()
    runpod.serverless.start({"handler": handler})
