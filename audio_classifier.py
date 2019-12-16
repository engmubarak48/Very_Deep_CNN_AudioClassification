
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio




import torch.nn as nn
import torch.optim as optim





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)






class AudioDataset(Dataset):
    """
    A rapper class for the UrbanSound8K dataset.
    """

    def __init__(self, file_path, audio_paths, folds):
        """
        Args:
            file_path(string): path to the audio csv file
            root_dir(string): directory with all the audio folds
            folds: integer corresponding to audio fold number or list of fold number if more than one fold is needed
        """
        self.audio_file = pd.read_csv(file_path)
        self.folds = folds
        self.audio_paths = glob.glob(audio_paths + '/*' + str(self.folds) + '/*')
    
    

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        
        audio_path = self.audio_paths[idx]
        audio, rate = torchaudio.load(audio_path, normalization=True)
        audio = audio.mean(0, keepdim=True)
        c, n = audio.shape
        zero_need = 160000 - n
        audio_new = F.pad(audio, (zero_need //2, zero_need //2), 'constant', 0)
        audio_new = audio_new[:,::5]
        
        #Getting the corresponding label
        audio_name = audio_path.split(sep='/')[-1]
        labels = self.audio_file.loc[self.audio_file.slice_file_name == audio_name].iloc[0,-2]
        
        return audio_new, labels



def init_weights(m):
    if type(m) == nn.Conv1d or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight.data)

#%%
# M5
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, 80, 4)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, 3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256, 512, 3)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(30) #input should be 512x30 so this outputs a 512x1
        self.fc1 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = self.avgPool(x)
        x = x.permute(0, 2, 1) #change the 512x1 to 1x512
        x = self.fc1(x)
        return x

M5 = Net()
M5.to(device)

# M3
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 256, 80, 4)
        self.bn1 = nn.BatchNorm1d(256)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(256, 256, 3)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(498) #input should be 512x30 so this outputs a 512x1
        self.fc1 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.avgPool(x)
        x = x.permute(0, 2, 1) #change the 512x1 to 1x512
        x = self.fc1(x)
        return x

M3 = Net()
M3.to(device)


# M11
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 80, 4)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(4)
        
        self.conv2 = nn.Conv1d(64, 64, 3) 
        self.conv21 = nn.Conv1d(64, 64, 3) 
        self.bn2 = nn.BatchNorm1d(64)
        self.bn21 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)
        
        self.conv3 = nn.Conv1d(64, 128, 3)
        self.conv31 = nn.Conv1d(128, 128, 3)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn31 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(4)
        
        self.conv4 = nn.Conv1d(128, 256, 3)
        self.conv41 = nn.Conv1d(256, 256, 3)
        self.conv42 = nn.Conv1d(256, 256, 3)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn41 = nn.BatchNorm1d(256)
        self.bn42 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(4)
        
        self.conv5 = nn.Conv1d(256, 512, 3)
        self.conv51 = nn.Conv1d(512, 512, 3)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn51 = nn.BatchNorm1d(512)

        self.avgPool = nn.AvgPool1d(25) #input should be 512x30 so this outputs a 512x1
        self.fc1 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn21(self.conv21(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn31(self.conv31(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn41(self.conv41(x)))
        x = F.relu(self.bn42(self.conv42(x)))
        x = self.pool4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn51(self.conv51(x)))
        x = self.avgPool(x)
        x = x.permute(0, 2, 1) #change the 512x1 to 1x512
        x = self.fc1(x)
        return x

M11 = Net()
M11.to(device)


# M18
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 80, 4)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(4)
        
        self.conv2 = nn.Conv1d(64, 64, 3) 
        self.conv21 = nn.Conv1d(64, 64, 3)
        self.conv22 = nn.Conv1d(64, 64, 3)
        self.conv23 = nn.Conv1d(64, 64, 3)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn21 = nn.BatchNorm1d(64)
        self.bn22 = nn.BatchNorm1d(64)
        self.bn23 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)
        
        self.conv3 = nn.Conv1d(64, 128, 3)
        self.conv31 = nn.Conv1d(128, 128, 3)
        self.conv32 = nn.Conv1d(128, 128, 3)
        self.conv33 = nn.Conv1d(128, 128, 3)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn31 = nn.BatchNorm1d(128)
        self.bn32 = nn.BatchNorm1d(128)
        self.bn33 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(4)
        
        self.conv4 = nn.Conv1d(128, 256, 3)
        self.conv41 = nn.Conv1d(256, 256, 3)
        self.conv42 = nn.Conv1d(256, 256, 3)
        self.conv43 = nn.Conv1d(256, 256, 3)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn41 = nn.BatchNorm1d(256)
        self.bn42 = nn.BatchNorm1d(256)
        self.bn43 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(4)
        
        self.conv5 = nn.Conv1d(256, 512, 3)
        self.conv51 = nn.Conv1d(512, 512, 3)
        self.conv52 = nn.Conv1d(512, 512, 3)
        self.conv53 = nn.Conv1d(512, 512, 3)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn51 = nn.BatchNorm1d(512)
        self.bn52 = nn.BatchNorm1d(512)
        self.bn53 = nn.BatchNorm1d(512)
        
        self.avgPool = nn.AvgPool1d(20) #input should be 512x30 so this outputs a 512x1
        self.fc1 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn21(self.conv21(x)))
        x = F.relu(self.bn22(self.conv22(x)))
        x = F.relu(self.bn23(self.conv23(x)))
        x = self.pool2(x)
        
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn31(self.conv31(x)))
        x = F.relu(self.bn32(self.conv32(x)))
        x = F.relu(self.bn33(self.conv33(x)))
        x = self.pool3(x)
        
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn41(self.conv41(x)))
        x = F.relu(self.bn42(self.conv42(x)))
        x = F.relu(self.bn43(self.conv43(x)))
        x = self.pool4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn51(self.conv51(x)))
        x = F.relu(self.bn52(self.conv52(x)))
        x = F.relu(self.bn53(self.conv53(x)))
#         print(x.shape)
#         print(x.shape[2])
        x = self.avgPool(x)
        x = x.permute(0, 2, 1) #change the 512x1 to 1x512
        x = self.fc1(x)
        return x

M18 = Net()
M18.to(device)

# M34-RES

def res_upsamp(A,m,n):
  upsample = nn.Upsample(size=(m,n), mode='nearest')
  A = torch.unsqueeze(A, 0)
  A = upsample(A)
  A = A.view(-1,m,n)
  return A
  
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 48, 80, 4)
        self.bn1 = nn.BatchNorm1d(48)
        self.pool1 = nn.MaxPool1d(4)
        
        self.conv2 = nn.Conv1d(48, 48, 3, padding = 1)
        self.conv21 = nn.Conv1d(48, 48, 3, padding = 1)
        self.conv22 = nn.Conv1d(48, 48, 3,padding = 1 )
        self.conv23 = nn.Conv1d(48, 48, 3,padding = 1)
        self.conv24 = nn.Conv1d(48, 48, 3,padding = 1)
        self.conv25 = nn.Conv1d(48, 48, 3,padding = 1)
        self.bn2 = nn.BatchNorm1d(48)
        self.bn21 = nn.BatchNorm1d(48)
        self.bn22 = nn.BatchNorm1d(48)
        self.bn23 = nn.BatchNorm1d(48)
        self.bn24 = nn.BatchNorm1d(48)
        self.bn25 = nn.BatchNorm1d(48)
        self.pool2 = nn.MaxPool1d(4)
        
        self.conv3 = nn.Conv1d(48, 96, 3)
        self.conv31 = nn.Conv1d(96, 96, 3)
        self.conv32 = nn.Conv1d(96, 96, 3,padding = 1)
        self.conv33 = nn.Conv1d(96, 96, 3,padding = 1)
        self.conv34 = nn.Conv1d(96, 96, 3,padding = 1)
        self.conv35 = nn.Conv1d(96, 96, 3,padding = 1)
        self.conv36 = nn.Conv1d(96, 96, 3,padding = 1)
        self.conv37 = nn.Conv1d(96, 96, 3,padding = 1)
        self.bn3 = nn.BatchNorm1d(96)
        self.bn31 = nn.BatchNorm1d(96)
        self.bn32 = nn.BatchNorm1d(96)
        self.bn33 = nn.BatchNorm1d(96)
        self.bn34 = nn.BatchNorm1d(96)
        self.bn35 = nn.BatchNorm1d(96)
        self.bn36 = nn.BatchNorm1d(96)
        self.bn37 = nn.BatchNorm1d(96)
        self.pool3 = nn.MaxPool1d(4)
        
        self.conv4 = nn.Conv1d(96, 192, 3)
        self.conv41 = nn.Conv1d(192, 192, 3)
        self.conv42 = nn.Conv1d(192, 192, 3,padding = 1)
        self.conv43 = nn.Conv1d(192, 192, 3,padding = 1)
        self.conv44 = nn.Conv1d(192, 192, 3,padding = 1)
        self.conv45 = nn.Conv1d(192, 192, 3,padding = 1)
        self.conv46 = nn.Conv1d(192, 192, 3,padding = 1)
        self.conv47 = nn.Conv1d(192, 192, 3,padding = 1)
        self.conv48 = nn.Conv1d(192, 192, 3,padding = 1)
        self.conv49 = nn.Conv1d(192, 192, 3,padding = 1)
        self.conv410 = nn.Conv1d(192, 192, 3,padding = 1)
        self.conv411 = nn.Conv1d(192, 192, 3,padding = 1)
        self.bn4 = nn.BatchNorm1d(192)
        self.bn41 = nn.BatchNorm1d(192)
        self.bn42 = nn.BatchNorm1d(192)
        self.bn43 = nn.BatchNorm1d(192)
        self.bn44 = nn.BatchNorm1d(192)
        self.bn45 = nn.BatchNorm1d(192)
        self.bn46 = nn.BatchNorm1d(192)
        self.bn47 = nn.BatchNorm1d(192)
        self.bn48 = nn.BatchNorm1d(192)
        self.bn49 = nn.BatchNorm1d(192)
        self.bn410 = nn.BatchNorm1d(192)
        self.bn411 = nn.BatchNorm1d(192)
        self.pool4 = nn.MaxPool1d(4)
        
        self.conv5 = nn.Conv1d(192, 384, 3, padding = 1)
        self.conv51 = nn.Conv1d(384, 384, 3, padding = 1)
        self.conv52 = nn.Conv1d(384, 384, 3, padding = 1)
        self.conv53 = nn.Conv1d(384, 384, 3, padding = 1)
        self.conv54 = nn.Conv1d(384, 384, 3, padding = 1)
        self.conv55 = nn.Conv1d(384, 384, 3, padding = 1)
        self.bn5 = nn.BatchNorm1d(384)
        self.bn51 = nn.BatchNorm1d(384)
        self.bn52 = nn.BatchNorm1d(384)
        self.bn53 = nn.BatchNorm1d(384)
        self.bn54 = nn.BatchNorm1d(384)
        self.bn55 = nn.BatchNorm1d(384)

        self.avgPool = nn.AvgPool1d(25) #input should be 512x30 so this outputs a 512x1
        self.fc1 = nn.Linear(384, 10)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        residual = x
       
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn21(self.conv21(x)))

        x += residual 
        x = F.relu(self.bn21(x))

        x = F.relu(self.bn22(self.conv22(x)))
        x = F.relu(self.bn23(self.conv23(x)))
        x += residual 
        x = F.relu(self.bn23(x))

        x = F.relu(self.bn24(self.conv24(x)))
        x = F.relu(self.bn25(self.conv25(x)))
        x += residual 
        x = F.relu(self.bn25(x))

        x = self.pool2(x)
        residual = x
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn31(self.conv31(x)))
        residual = res_upsamp(residual,x.shape[1],x.shape[2])
        x += residual 
        x = F.relu(self.bn31(x))

        x = F.relu(self.bn32(self.conv32(x)))
        x = F.relu(self.bn33(self.conv33(x)))
        x += residual 
        x = F.relu(self.bn33(x))
         
        x = F.relu(self.bn34(self.conv34(x)))
        x = F.relu(self.bn35(self.conv35(x)))
        x += residual 
        x = F.relu(self.bn35(x))

        x = F.relu(self.bn36(self.conv36(x)))
        x = F.relu(self.bn37(self.conv37(x)))
        x += residual 
        x = F.relu(self.bn37(x))
        x = self.pool3(x)
        residual = x
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn41(self.conv41(x)))
        residual = res_upsamp(residual,x.shape[1],x.shape[2])
        x += residual 
        x = F.relu(self.bn41(x))

        x = F.relu(self.bn42(self.conv42(x)))
        x = F.relu(self.bn43(self.conv43(x)))
        x += residual 
        x = F.relu(self.bn43(x))

        x = F.relu(self.bn44(self.conv44(x)))
        x = F.relu(self.bn45(self.conv45(x)))
        x += residual 
        x = F.relu(self.bn45(x))

        x = F.relu(self.bn46(self.conv46(x)))
        x = F.relu(self.bn47(self.conv47(x)))
        x += residual 
        x = F.relu(self.bn47(x))

        x = F.relu(self.bn48(self.conv48(x)))
        x = F.relu(self.bn49(self.conv49(x)))
        x += residual 
        x = F.relu(self.bn49(x))

        x = F.relu(self.bn410(self.conv410(x)))
        x = F.relu(self.bn411(self.conv411(x)))
        x += residual 
        x = F.relu(self.bn411(x))

      
        x = self.pool4(x)
        residual = x
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn51(self.conv51(x)))
        residual = res_upsamp(residual,x.shape[1],x.shape[2])
        x += residual 
        x = F.relu(self.bn51(x))

        x = F.relu(self.bn52(self.conv52(x)))
        x = F.relu(self.bn53(self.conv53(x)))
        x += residual 
        x = F.relu(self.bn53(x))

        x = F.relu(self.bn54(self.conv54(x)))
        x = F.relu(self.bn55(self.conv55(x)))
        x += residual 
        x = F.relu(self.bn55(x))
        
        
        x = self.avgPool(x)
        x = x.permute(0, 2, 1) #change the 512x1 to 1x512
        x = self.fc1(x)
        return x
M34_RES = Net()
M34_RES.to(device)



def train(model, epoch):
    model.train()
    correct = 0
    losses_train = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        data = data.requires_grad_() #set requires_grad to True for training
        output = model(data)
        output = output.permute(1, 0, 2) #original output dimensions are batchSizex1x10 
        loss = criterion(output[0], target) #the loss functions expects a batchSizex10 input
        losses_train.append(loss.item())
        pred = output.max(2)[1] 
        correct += pred.eq(target).cpu().sum().item()
        loss.backward()
        optimizer.step()
#        if batch_idx % log_interval == 0: #print training stats
#            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),  
#                  100. * batch_idx / len(train_loader), loss))
    return 100. * correct / len(train_loader.dataset), np.mean(losses_train)



def test(model, epoch):
    model.eval()
    correct = 0
    test_losses = []
    for data, target in test_loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(data)
        output = output.permute(1, 0, 2)
        loss = criterion(output[0], target) #the loss functions expects a batchSizex10 input
        test_losses.append(loss.item())
        pred = output.max(2)[1] # get the index of the max log-probability
        correct += pred.eq(target).cpu().sum().item()
#    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),
#        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset), np.mean(test_losses)


    
import matplotlib.pyplot as plt
import time

t0 = time.time()

from sklearn.model_selection import LeaveOneOut

csv_path = './UrbanSound8K/metadata/UrbanSound8K.csv'
file_path = './UrbanSound8K/audio/'

folds = np.array([1,2,3,4,5,6,7,8,9,10])

loo = LeaveOneOut()
loo.get_n_splits(folds)

networks = [M3, M5, M11, M18, M34_RES]
criterion = nn.CrossEntropyLoss()

for net in range(len(networks)):
    avg_loss_model_train, avg_loss_model_test = [], []
    avg_acc_model_train,  avg_acc_model_test = [], []
    for i, (train_folds, test_folds) in enumerate(loo.split(folds)):
        train_set = AudioDataset(csv_path, file_path, folds[train_folds])
        test_set = AudioDataset(csv_path, file_path, folds[test_folds])
#        print("Train set size: " + str(len(train_set)))
#        print("Test set size: " + str(len(test_set)))
        networks[net].apply(init_weights)
        kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {} #needed for using datasets on gpu
    
        train_loader = torch.utils.data.DataLoader(train_set, batch_size = 32, shuffle = True, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size = 32, shuffle = True, **kwargs)
    
    
        optimizer = optim.Adam(networks[net].parameters(), lr = 0.01, weight_decay = 0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)
        
#        print('Number of parameters: {:,}'.format(sum([p.nelement() for p in M34_RES.parameters()])) )
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        log_interval = 20
        loss_epoch_train = []
        loss_epoch_test = []
        accuracy_epoch_train = []
        accuracy_epoch_test = []
        epoches = 10
        for epoch in range(epoches):
            if epoch == 31:
                print("First round of training complete. Setting learn rate to 0.001.")
            scheduler.step()
            accuracy_train, losses_train = train(networks[net], epoch)
            loss_epoch_train.append(losses_train)
            accuracy_epoch_train.append(accuracy_train)
            accuracy_test, losses_test = test(networks[net], epoch)
            loss_epoch_test.append(losses_test)
            accuracy_epoch_test.append(accuracy_test)
            print('ephoch {} done'.format(epoch+1))
        
        avg_loss_model_train.append(np.mean(loss_epoch_train))
        avg_loss_model_test.append(np.mean(loss_epoch_test))
        avg_acc_model_train.append(np.mean(accuracy_epoch_train))
        avg_acc_model_test.append(np.mean(accuracy_epoch_test))
        ax[0].plot(np.arange(epoches), loss_epoch_train, label = 'train loss for fold {0} and model {1}'.format(i+1, str(net+1)))  # save the figure to file
        ax[0].plot(np.arange(epoches), loss_epoch_test, label = 'test loss for fold {0} and model {1}'.format(i+1, str(net+1)))
        ax[0].set_xscale('log')
        ax[0].set_title('Loss')
        ax[0].set_xlabel('epochs')
        ax[0].set_ylabel('loss')
        ax[0].legend()
#        plt.close(fig)
        ax[1].plot(np.arange(epoches), accuracy_epoch_train, label = 'train accuracy for fold {0} and model {1}'.format(i+1, str(net+1)))  # save the figure to file
        ax[1].plot(np.arange(epoches), accuracy_epoch_test, label = 'test accuracy for fold {0} and model {1}'.format(i+1, str(net+1))) 
        ax[1].set_xscale('log')
        ax[1].set_title('Accuracy')
        ax[1].set_xlabel('epochs')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()
        fig.savefig('plots1/figure_fold:{0}_model:{1}.png'.format(i+1, str(net+1)))
#        plt.close(fig)
    print('Average train loss for Model {}: {}'.format(net+1, np.mean(avg_loss_model_train)))
    print('Average test loss for Model {}: {}'.format(net+1, np.mean(avg_loss_model_test)))
    print('Average train accuracy for Model {}: {}'.format(net+1, np.mean(avg_acc_model_train)))
    print('Average test accuracy for Model {}: {}'.format(net+1, np.mean(avg_acc_model_test)))

    
    
    
    
    
    
    
    
    
