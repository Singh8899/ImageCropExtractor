import base64
import json

with open("workerDiego/test_input.json", "w") as f:
    with open("workerDiego/a.jpg", "rb") as f2:
        image_bytes = f2.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        diz = {'input': {'image': image_base64}}
        json.dump(diz, f)