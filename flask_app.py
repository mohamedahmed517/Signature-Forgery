import os
import cv2
import torch # type: ignore
import socket
import numpy as np
import torch.nn as nn # type: ignore
from PIL import Image
import torch.nn.functional as F # type: ignore
import torchvision.models as models # type: ignore
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from huggingface_hub import hf_hub_download
import torchvision.transforms as transforms # type: ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Flask app
app = Flask(__name__)

# Set up upload folder
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok = True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_img(img_path):
    # Step 1: Convert Img To Grayscale
    gray = img_path.convert("L")
    # Step 2: Convert Grayscale To Numpy Array
    img = np.array(gray)
    # Step 3: Morphological Transformation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,2))
    morphology_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Step 4: Apply Gaussian Blur
    blur = cv2.GaussianBlur(morphology_img, (3,3), 0)
    # Step 5: Thresholding
    _, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Step 6: Find The Bounding Box Around Non_White Pixels
    coords = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(coords)
    # Step 7: Adding Padding To The Bounding Box
    padding = 5
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding
    # Make sure the coordinates are within the image boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)
    # Step 8: Crop And Add Extra White Space
    cropped_image = binary[y:y + h, x:x + w]
    extra_space = np.zeros((cropped_image.shape[0] + 2 * padding, cropped_image.shape[1] + 2 * padding), dtype=np.uint8) * 255
    extra_space[padding:-padding, padding:-padding] = cropped_image
    # Step 9: Resize Image
    corrected = cv2.resize(extra_space,(330,175))
    # Step 10: Convert the numpy array back to PIL image
    resized_image = Image.fromarray(corrected)

    return resized_image

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size = 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size = 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        avg_out = self.fc2(self.relu(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu(self.fc1(max_pool)))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size = 7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size = kernel_size, padding = (kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_pool = torch.mean(x, dim = 1, keepdim = True)
        max_pool, _ = torch.max(x, dim = 1, keepdim = True)
        pool = torch.cat([avg_pool, max_pool], dim = 1)
        attention = self.sigmoid(self.conv(pool))
        return x * attention

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 16, spatial_kernel_size = 7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class SiameseResNet(nn.Module):
    def __init__(self, model_name = "resnet50", pretrained = False):
        super(SiameseResNet, self).__init__()
        self.baseModel = models.resnet50(pretrained = pretrained)
        self.attention1 = CBAM(in_channels = 256)
        self.attention2 = CBAM(in_channels = 1024)

        self.baseModel.conv1 = nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.baseModel.fc = nn.Identity()
    def forward(self, x):
        out = self.baseModel.conv1(x)
        out = self.baseModel.bn1(out)
        out = self.baseModel.relu(out)
        out = self.baseModel.maxpool(out)

        out = self.attention1(self.baseModel.layer1(out))
        out = self.baseModel.layer2(out)

        out = self.attention2(self.baseModel.layer3(out))
        out = self.baseModel.layer4(out)

        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out

class LogisticSiameseRegression(nn.Module):
    def __init__(self, model):
        super(LogisticSiameseRegression, self).__init__()

        self.model = model
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(inplace = True),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.LeakyReLU(inplace = True),
            nn.Linear(256, 1),
            nn.LeakyReLU(inplace = True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward_once(self, x):
        out = self.model(x)
        out = F.normalize(out, p = 2, dim = 1)
        return out

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)

        diff = out1 - out2
        out = self.fc(diff)
        out = self.sigmoid(out)

        return out

siamese_model = SiameseResNet()
siamese_model = nn.DataParallel(siamese_model).to(device)
model_rms = LogisticSiameseRegression(siamese_model).to(device)

repo_id = "mohamed517/logistic_regression_triangular"
filename = "logistic_model_triangular_m09_ashoj3.pth"

model_path = hf_hub_download(repo_id = repo_id, filename = filename)
model_rms.load_state_dict(torch.load(model_path, map_location = torch.device(device)))
model_rms.to(device)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"error": "Please provide both image1 and image2"}), 400

    image1 = request.files['image1']
    image2 = request.files['image2']
    image1_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image1.filename))
    image2_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image2.filename))
    image1.save(image1_path)
    image2.save(image2_path)

    try:
        img1 = preprocess_img(Image.open(image1_path))
        img2 = preprocess_img(Image.open(image2_path))
        transform = transforms.Compose([transforms.Resize((175, 330)), transforms.ToTensor()])
        input1 = transform(img1).unsqueeze(0).to(device)
        input2 = transform(img2).unsqueeze(0).to(device)

        model_rms.eval()
        with torch.no_grad():
            prediction = model_rms(input1, input2)
            pred1 = model_rms.forward_once(input1)
            pred2 = model_rms.forward_once(input2)
            diff = torch.pairwise_distance(pred1, pred2)
            similarity_score = 1 / (1 + diff)
            probability_percentage = prediction.item()

            similarity_score *= 100
            probability_percentage *= 100

        return jsonify({
            "Similarity_Percentage": f"{similarity_score:.2f}%",
            "Probability_Percentage": f"{probability_percentage:.2f}%",
            "Signature_Wasn't_Forged": similarity_score > 80 and probability_percentage > 80
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

if __name__ == '__main__':
    port = find_free_port()
    app.run(host = '0.0.0.0', port = port)