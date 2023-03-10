#train
import torch
from torch import nn, optim
from tqdm import tqdm
def train(model, is_stitch = False,total_epoch: int=20,data_loader):
   
   total_epoch = 64000//len(data_loader)+100
   # オプティマイザーの定義
   if is_stitch:
     optimizer = optim.Adam(
        params=model.parameters(),
        lr=0.001)
   else:
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=0.05,
        weight_decay = 0.0001,
        momentum = 0.9
    )
    # スケジューラーの定義
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones = [total_epoch//2,(total_epoch)*3//4],
        gamma = 0.2,
        last_epoch=-1, #change step to 64K 
    )

   # 損失関数の定義
   criterion = nn.CrossEntropyLoss()

   model.train()
   i= 0
   for epoch in range(total_epoch):
       accuracy, train_loss = 0.0, 0.0

       # tqdmを用いるとプログレスバーの表示ができる
       for images, labels in tqdm(data_loader):
           optimizer.zero_grad()
           images = images.to(device)
           labels = labels.to(device)

           # モデルからの出力
           out = model(images)
           # lossの算出
           loss = criterion(out, labels)

           loss.backward()
           optimizer.step()
           

           # 推測値
           preds = out.argmax(axis=1)

           train_loss += loss.item()
           # 正答率の算出
           accuracy += torch.sum(preds == labels).item() / len(labels)
           i+=1
           if i == 64000:
             return
       if not is_stitch:  
           scheduler.step()

       # 値の出力
       print(f"epoch: {epoch + 1}")
       print(f"loss: {train_loss / len(data_loader)}")
       print(f"accuracy: {accuracy / len(data_loader)}")
  #  torch.save(model.state_dict(), "/content/drive/MyDrive/CompWeb/FMNISTmodel.pth")