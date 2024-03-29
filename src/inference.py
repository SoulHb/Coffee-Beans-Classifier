import io
import torchvision.transforms as transforms
import sys
import argparse
from model import Resnet
from flask import Flask, request, jsonify
from PIL import Image
from config import *
sys.path.append("src")

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
app = Flask(__name__)


def load_model():
    """
    Load the pre-trained model for making predictions.

    Returns:
        model: Loaded model.
    """
    model = Resnet()
    model.load_state_dict(torch.load(args.saved_model_path if args.saved_model_path else os.path.join(SAVED_MODEL_FOLDER, MODEL_FILE),
               map_location=DEVICE)['model_state_dict'])
    model.eval()
    return model


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making predictions on an input image.

    Returns:
        JSON: A JSON response containing the predicted class label.
    """
    global model
    model.to(DEVICE)
    file = request.files['file']
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image = transform(image).unsqueeze(0)
    image.to(DEVICE)
    with torch.no_grad():
        output = model(image.to(DEVICE))
    pred = output.argmax().item()
    return jsonify({'prediction': pred})


if __name__=='__main__':
    # Define parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_path", type=str,
                        help='Specify path for chosen model')
    args = parser.parse_args()
    model = load_model()
    app.run(debug=False)
