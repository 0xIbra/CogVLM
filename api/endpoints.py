from flask import Blueprint, request, jsonify
from cogvlm_wrapper import ModelWrapper
from io import BytesIO
from PIL import Image
import base64


blueprint = Blueprint('api_bp', __name__)

MODEL = ModelWrapper()


@blueprint.route('/api/predict', methods=['POST'])
def predict():
    global MODEL

    payload = request.get_json()
    if 'image' not in payload:
        return jsonify({'detail': '"image" key must provide base64 image content.'}), 400
    
    if 'prompt' not in payload:
        return jsonify({'detail': '"prompt" key must provide the text prompt.'}), 400
    
    b64_image = payload['image']
    prompt = payload['prompt']

    if 'base64,' in b64_image:
        b64_image = b64_image.partition('base64,')[2]

    img = Image.open(BytesIO(base64.b64decode(b64_image)))

    result = MODEL.predict(prompt, img)

    return jsonify({'data': result})
