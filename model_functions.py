import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image # cv2가 시간 ㄹㅈㄷ로 걸리는 거였구나...
import os
import glob
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        # os.listdir는 파일명만 가져옴 / glob.glob는 최상위 경로까지 다 가져옴
        self.real_li = glob.glob(path+"\\training_real\\*.jpg") # 1
        self.fake_li = glob.glob(path+"\\training_fake\\*.jpg") # 0
        self.img_li = self.real_li + self.fake_li
        self.labels = [1]*len(self.real_li) + [0]*len(self.fake_li)

        self.transform = transform

    def __len__(self):
        return len(self.img_li)

    def __getitem__(self, idx): # 보통 여기서 image open함
        img_path = self.img_li[idx]
        img = Image.open(img_path)
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.float32) # y_hat이 float -> criterion 에서 에러
        # label = label.type(torch.float32)

        if self.transform is not None:
            img = self.transform(img)

        return img, label # return {"img": img, "label": label}
    
def Train(model, train_DL, criterion, optimizer, EPOCH):
    loss_history = []
    acc_history = []
    NoT = len(train_DL.dataset)

    model.train() # train mode로 전환
    for ep in range(EPOCH):
        rloss = 0 # running loss
        rcorrect = 0
        for x_batch, y_batch in train_DL:
            x_batch = x_batch.to(DEVICE) # gpu 사용 (cuda)
            y_batch = y_batch.to(DEVICE)
            # x_batch = x_batch.permute(0, 3, 1, 2) # 개채행렬로
            y_batch = y_batch.unsqueeze(1)
            # inference
            y_hat = model(x_batch)
            # loss 
            loss = criterion(y_hat, y_batch)
            # update
            optimizer.zero_grad() # gradient 누적을 막기 위한 초기화
            loss.backward() # backpropagation
            optimizer.step() # weight update
            #loss accumulation
            loss_b = loss.item() * x_batch.shape[0] # batch loss # BATCH_SIZE를 곱하면 마지막 18개도 32개를 곱하니까!
            # batch loss 는 1/32*loss더한거 -> 여기에 32 곱해서 싹 더해놓고, 마지막에 6만으로 나누
            rloss += loss_b # running loss
            pred = (y_hat >= 0.5)
            corrects_b = torch.sum(pred == y_batch).item()
            rcorrect += corrects_b

        # print loss
        loss_e = rloss/NoT
        accuracy_e = rcorrect/NoT *100
        loss_history += [loss_e]
        acc_history += [accuracy_e]
        print(f"Epoch: {ep+1}, train loss: {round(loss_e,3)}")
        print(f"Epoch: {ep+1}, Train accuracy: {rcorrect}/{NoT} ({round(accuracy_e,3)} %)")
        print("-"*20)
    return loss_history, acc_history

def Test(model, test_DL):
    model.eval()
    with torch.no_grad():
        rcorrect = 0
        for x_batch, y_batch in test_DL:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            # x_batch = x_batch.permute(0, 3, 1, 2) # 개채행렬로
            y_batch = y_batch.unsqueeze(1)
            # inference
            y_hat = model(x_batch)
            # accuracy accumulation
            pred = (y_hat >= 0.5)
            corrects_b = torch.sum(pred == y_batch).item()
            rcorrect += corrects_b
        accuracy_e = rcorrect/len(test_DL.dataset)*100
    print(f"Test accuracy: {rcorrect}/{len(test_DL.dataset)} ({round(accuracy_e,1)} %)")

def Test_plot(model, test_DL):
    model.eval()
    with torch.no_grad():
        x_batch, y_batch = next(iter(test_DL))
        x_batch = x_batch.to(DEVICE)
        y_hat = model(x_batch)
        pred = y_hat.argmax(dim=1)

    x_batch = x_batch.to("cpu")

    plt.figure(figsize=(8,4))
    for idx in range(6):
        plt.subplot(2,3, idx+1, xticks=[], yticks=[])
        plt.imshow(x_batch[idx].permute(1,2,0).squeeze(), cmap="gray") # 컬러일때도 그대로 쓰려고 
        pred_class = test_DL.dataset.classes[pred[idx]]
        true_class = test_DL.dataset.classes[y_batch[idx]]
        plt.title(f"{pred_class} ({true_class})", color = "g" if pred_class==true_class else "r") 

def count_params(model):
    num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num