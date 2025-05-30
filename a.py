import json
import base64


with open("/home/jaspinder/Github/ImageCropExtractor/dataset/photo/1.jpg", "rb") as img_file:
    img = img_file.read()
    dict = {"input": {"image": base64.b64encode(img).decode('utf-8')}}
    with open("test.json", "w") as f:
        f.write(json.dumps(dict, indent=4))
    