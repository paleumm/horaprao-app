from PIL import Image
from torchvision import transforms
import torch
from ResNet import ResNet50
import streamlit as st
import torchvision.models as models

st.title("Ho-ra-prao Application")
st.write("")

image_up = st.file_uploader("Upload image. (jpg webp) ", type=["jpeg","jpg","webp"])

PATH = 'model.pth'
IMAGE_SIZE = 224
NUM_CLASSES = 2

device = ('cuda' if torch.cuda.is_available() else 'cpu')

model = ResNet50(num_classes=2)
# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, 2)
checkpoint = torch.load(PATH, map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.eval()

def predict(image):

    normalize = transforms.Normalize((0.48232,), (0.23051,))
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize
    ])

    img = Image.open(image)
    batch = torch.unsqueeze(transform(img),0)

    # predict
    model.eval()
    out = model(batch)

    with open('class.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    prob = torch.nn.functional.softmax(out, dim = 1)[0] * 100
    _, indices = torch.sort(out, descending = True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:1]]


if image_up is not None:
    # display image that user uploaded
    image = Image.open(image_up).convert('RGB')
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    # st.write("Just a second ...")
    labels = predict(image_up)

    # predicted class
    for i in labels:
        name = i[0].split()
        st.header(f"Prediction : {name[1]}")
