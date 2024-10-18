import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import os
from imblearn.under_sampling import RandomUnderSampler
from torchinfo import summary

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

BATCH_SIZE = 32
LR = 1e-5
EPOCH = 5

criterion = nn.BCEWithLogitsLoss()
new_model_train = True
model_type = "ResNet"

# 모델 저장 경로 설정
save_model_path = rf"C:/Users/eunyo/Desktop/PythonWorkspace/FOM/FOM_project2/deepfake_vision/results/{model_type}_test.pt"
save_history_path = rf"C:/Users/eunyo/Desktop/PythonWorkspace/FOM/FOM_project2/deepfake_vision/results/{model_type}_history_test.pt"

# CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.Maxpool1 = nn.MaxPool2d(2)

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.Maxpool2 = nn.MaxPool2d(2)

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            nn.Dropout(0.25)
        )
        self.Maxpool3 = nn.MaxPool2d(2)

        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(), 
            nn.Linear(512, 1)
        )
        
        self.weights_initialization()  # 가중치 초기화
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.Maxpool1(x)
        x = self.conv_block2(x)
        x = self.Maxpool2(x)
        x = self.conv_block3(x)
        x = self.Maxpool3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    
    def weights_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

# 데이터 로딩
path = r"D:\FOM_deepfake2"  # base 경로
labels_real = pd.read_csv(path + r"/train/image_labels_real.csv", names=["img_path", "label"])
labels_fake = pd.read_csv(path + r"/train/image_labels_fake.csv", names=["img_path", "label"])
labels_train = pd.concat([labels_real, labels_fake])
labels_val = pd.read_csv(path + r"/val/image_labels.csv", names=["img_path", "label"], header=None, skiprows=1)
labels_test = pd.read_csv(path + r"/test/image_labels.csv", names=["img_path", "label"])

def data_read(path, labels, type):
    img_paths = labels["img_path"].values
    labels = labels["label"].apply(lambda x: 1 if x == "real" else 0)

    x = []
    y = []
    id_check = []  # 인물 ID를 저장할 리스트

    for img_path, label in zip(img_paths, labels):
        person_id = os.path.dirname(img_path)  # 인물별 폴더 이름 추출
        if person_id not in id_check:
            full_path = os.path.join(path, type, img_path)
            with Image.open(full_path) as img:
                img = np.array(img)
                x.append(img)
                y.append(label)
            id_check.append(person_id)

    x = np.array(x)
    y = np.array(y, dtype=np.float32)
    return x, y

x_train0, y_train0 = data_read(path, labels_train, "train")
x_val0, y_val0 = data_read(path, labels_val, "val")

# 언더샘플링
undersample = RandomUnderSampler(sampling_strategy=0.8)
x_train1, y_train1 = undersample.fit_resample(x_train0.reshape(len(x_train0), -1), y_train0)
x_train1 = x_train1.reshape(-1, 128, 128, 3)

x_val1, y_val1 = undersample.fit_resample(x_val0.reshape(len(x_val0), -1), y_val0)
x_val1 = x_val1.reshape(-1, 128, 128, 3)

# 1500개로 샘플링
r = np.arange(x_train1.shape[0])
np.random.shuffle(r)
x_train = x_train1[r][:1500]
y_train = y_train1[r][:1500]

r = np.arange(x_val1.shape[0])
np.random.shuffle(r)
x_val = x_val1[r][:640]
y_val = y_val1[r][:640]

# 데이터셋 및 데이터로더 생성
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

train_DS = Custom_Dataset(x_train, y_train, transform)
train_DL = torch.utils.data.DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True)

val_DS = Custom_Dataset(x_val, y_val, transform)
val_DL = torch.utils.data.DataLoader(val_DS, batch_size=BATCH_SIZE, shuffle=True)

# 모델 학습
if new_model_train:
    model = CNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    loss_history, acc_history = Train(model, train_DL, criterion, optimizer, EPOCH)

    torch.save(model.state_dict(), save_model_path)

    plt.plot(range(1, EPOCH + 1), loss_history, label="Loss")
    plt.legend()
    plt.xlabel('Epoch')
    plt.title("Train Loss")
    plt.grid()
    plt.show()


# 모델 성능 확인
model = CNN().to(DEVICE)
model.load_state_dict(torch.load(save_model_path))  # 저장된 모델 가중치 로드

# 테스트 함수 호출
Test(model, val_DL)
print("Number of trainable parameters:", count_params(model))