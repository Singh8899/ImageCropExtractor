import runpod
from model import CropInference
import json

cropper = CropInference()
with open("/root/Workspace/Cropper/test_input.json", "r") as f:
    test_input = json.load(f)
input = test_input["input"]
image = input["image"]

bboxes = cropper.infer(image)

# def handler(event):
#     #   This function processes incoming requests to your Serverless endpoint.
#     #
#     #    Args:
#     #        event (dict): Contains the input data and request metadata
#     #
#     #    Returns:
#     #       Any: The result to be returned to the client

#     # Extract input data
#     print(f"Worker Start")
#     input = event["input"]

#     image = input["image"]

#     bboxes = cropper.infer(image)

#     return bboxes


# # Start the Serverless function when the script is run
# if __name__ == "__main__":
#     runpod.serverless.start({"handler": handler})
