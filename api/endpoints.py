from flask import Blueprint, request, jsonify
from cogvlm_wrapper import ModelWrapper
from io import BytesIO
from PIL import Image
import base64


blueprint = Blueprint('api_bp', __name__)
MODEL = ModelWrapper()


DESIRED_WIDTH = 800

def resize_image(image, width=None, height=None):
    (w, h) = image.size
    if width is None and height is None:
        return image

    if width is None:
        dim = (int(height * w / h), height)
    else:
        dim = (width, int(width * h / w))

    return image.resize(dim, Image.ANTIALIAS)


@blueprint.route('/api/predict', methods=['POST'])
def predict():
    global MODEL
    global DESIRED_WIDTH

    payload = request.get_json()
    if 'image' not in payload:
        return jsonify({'detail': '"image" key must provide base64 image content.'}), 400
    
    if 'prompt' not in payload:
        return jsonify({'detail': '"prompt" key must provide the text prompt.'}), 400
    
    b64_image = payload['image']
    prompt = payload['prompt']

    if 'base64,' in b64_image:
        b64_image = b64_image.partition('base64,')[2]

    try:
        img = Image.open(BytesIO(base64.b64decode(b64_image)))
        if img.size[0] > DESIRED_WIDTH:
            img = resize_image(img, width=DESIRED_WIDTH)

    except Exception:
        return jsonify({'detail': 'could not parse base64 image, verify if base64 is correct.'}), 400

    result = MODEL.predict(prompt, img)

    return jsonify({'data': result})
