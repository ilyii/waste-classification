import io
import os
import json
from PIL.Image import Image
import torch
from torch import nn
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
from flask import Flask, request, render_template
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from werkzeug.utils import secure_filename

app = Flask(__name__)
basepath = os.path.dirname(__file__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024

classes = {0: 'Glas', 1: 'Organic', 2: 'Paper', 3: 'Restmuell', 4: 'Wertstoff'}

# Add the id of your folder here (From google drive)
folder_id = '1Z1TjfkEYPUtVBwqZgSV4Om0i2VRTD2qX'
bool_use_upload_feature = False
#net = models.resnet18(pretrained=False)
#num_ftrs = net.fc.in_features
#net.fc = nn.Linear(num_ftrs, 5)

# Our best model: EfficientNet-B3
net = EfficientNet.from_pretrained("efficientnet-b3", num_classes=5)
net.load_state_dict(torch.load('eff_b3_model.pt', map_location='cpu'))
net.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255, interpolation=transforms.InterpolationMode.BICUBIC),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                        ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transforms(image).unsqueeze(0)


def model_predict(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = net(tensor)
    _, y_hat = outputs.max(1)
    softmax = nn.LogSoftmax(1)
    output_ = softmax(outputs)
    output_numpy = torch.exp(output_).detach().numpy()
    probabilities = []
    for x in output_numpy[0]:
        probabilities.append(str(round(x, 3)))
    predicted_idx = y_hat.item()
    return classes[predicted_idx], probabilities


def googleAuth():
    gauth = GoogleAuth()
    # Try to load saved client credentials
    gauth.LoadCredentialsFile("mycreds.txt")
    if gauth.credentials is None:
        # Authenticate if they're not there
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile("mycreds.txt")

    drive = GoogleDrive(gauth)
    return drive


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        # Get the file from post request
        f = request.files['file']
        img_bytes = f.read()
        f.seek(0)

        if bool_use_upload_feature and request.form.to_dict(flat=False)['checkbox'][0] == 'yes':
            f.save(secure_filename(f.filename))
            drive = googleAuth()

            gfile = drive.CreateFile({'parents': [{'id': folder_id}]})
            # Read file and set it as the content of this instance.
            gfile.SetContentFile(secure_filename(f.filename))
            gfile.Upload()  # Upload the file.
            gfile.content.close()
            os.remove(secure_filename(f.filename))

        # Make prediction
        preds, probs = model_predict(img_bytes)
        json_string = json.dumps(probs)
        ret = {"class": preds, "prob": json_string}
        return ret
    return None

@app.route('/images', methods=['GET', 'POST'])
def uploadImages():
    if request.method == 'POST':
        images_list = request.files.getlist('files[]')
        for f in images_list:
            ret = {"got": "" + f.filename}
            if not bool_use_upload_feature:
                return ret
            drive = googleAuth()
            f.save(secure_filename(f.filename))
            gfile = drive.CreateFile({'parents': [{'id': folder_id}]})
            # Read file and set it as the content of this instance.
            gfile.SetContentFile(secure_filename(f.filename))
            gfile.Upload()  # Upload the file.
            gfile.content.close()
            os.remove(secure_filename(f.filename))
            return ret
    return None

@app.route('/example_images.html', methods=['GET'])
def gallery():
    return render_template('example_images.html')

@app.route('/classificate_multiple.html', methods=['GET'])
def load_multiple_class_page():
    return render_template('classificate_multiple.html')


@app.route('/classificate_multiple_predict', methods=['GET', 'POST'])
def multiple_prediction():
    if request.method == 'POST':
        images_list = request.files.getlist('files[]')
        index = request.form.to_dict(flat=False)['index'][0]
        # Get the file from post request
        for f in images_list:
            img_bytes = f.read()
            f.seek(0)

            # Make prediction
            preds, probs = model_predict(img_bytes)
            json_string = json.dumps(probs)
            ret = {"class": preds, "prob": json_string, "indexPredicted": index}
            return ret
    return None

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
if __name__ == "__main__":
    app.run(host='0.0.0.0')
