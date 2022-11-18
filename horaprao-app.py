from PIL import Image
from torchvision import transforms
import torch
from train import ResNetClassifier
import streamlit as st
import torchvision.models as models

st.title("Ho-ra-prao Application")
st.write("")

image_up = st.file_uploader("Upload image. (jpg webp) ", type=["jpeg","jpg","webp"])

PATH = 'horaprao-new.ckpt'
IMAGE_SIZE = 500
NUM_CLASSES = 2

device = ('cuda' if torch.cuda.is_available() else 'cpu')

model = ResNetClassifier(2, 50, 'input/train','input/val')
# model = models.resnet50()
model.load_from_checkpoint(PATH)
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
    st.write("Just a second ...")
    labels = predict(image_up)

    # predicted class
    for i in labels:
        name = i[0].split()
        st.write("Prediction ", name[1], ",   Score: ", i[1])
    # st.write("Prediction (index, name)", labels[0][0], ",   Score: ", labels[0][1])

st.header("Classes")
st.dataframe(["horapa", "kapao"],600)