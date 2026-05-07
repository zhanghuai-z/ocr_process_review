# Please make sure the requests library is installed
# pip install requests
import os
import base64
import requests

API_URL = "https://n6z9feddjca4l7b5.aistudio-app.com/ocr"
TOKEN = "c7bc28d18e194059a9951eae6ef2f534becd8323"

file_path = "<local file path>"
input_filename = os.path.splitext(os.path.basename(file_path))[0]

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

headers = {
    "Authorization": f"token {TOKEN}",
    "Content-Type": "application/json"
}

required_payload = {
    "file": file_data,
    "fileType": <file type>,  # For PDF documents, set `fileType` to 0; for images, set `fileType` to 1
}

optional_payload = {
    "useDocOrientationClassify": False,
    "useDocUnwarping": False,
    "useTextlineOrientation": False,
}

payload = {**required_payload, **optional_payload}

response = requests.post(API_URL, json=payload, headers=headers)

assert response.status_code == 200
result = response.json()["result"]

os.makedirs("output", exist_ok=True)

for i, res in enumerate(result["ocrResults"]):
    print(res["prunedResult"])
    image_url = res["ocrImage"]
    img_response = requests.get(image_url)
    if img_response.status_code == 200:
        # Save image to local
        filename = f"output/{input_filename}_{i}.jpg"
        with open(filename, "wb") as f:
            f.write(img_response.content)
        print(f"Image saved to: {filename}")
    else:
        print(f"Failed to download image, status code: {img_response.status_code}")
