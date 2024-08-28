import torch
import PIL.Image as Image
from os.path import join
from training.model import CSRNet
from torchvision import transforms
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = CSRNet()
# model = model.cuda()
model = model.to(torch.device('cpu'))

checkpoint = torch.load(join('..', 'dataset', 'PartBmodel_best.pth.tar'), weights_only=False, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])

app = FastAPI()


def predictCount(image):
    img = transform(Image.open(image).convert('RGB'))  # .cuda()
    output = model(img.unsqueeze(0))

    return int(output.detach().cpu().sum().numpy())


@app.get('/')
def read_root():
    return FileResponse('index.html')


@app.post('/')
def upload_file(image: UploadFile):
    return {
        'Image': image.filename,
        'Predicted count: ': predictCount(image.file)
    }
