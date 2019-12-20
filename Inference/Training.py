from Models.Classification1 import CNNModel
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as network
import os

def CNNTrain(trainData,valData = None,
             parameters = {'num_epochs':5,'num_classes': 9,'batch_size':50,'learning_rate':0.0001}
             ):
    MODELPATH = './DUMPS/'
    if os.path.exists(MODELPATH) == False:
        os.makedirs(MODELPATH)
    model = CNNModel()
    model.double()
    loader = DataLoader(trainData,batch_size=50,num_workers=3,shuffle=True)
    criterion = network.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=parameters['learning_rate'])
    batchIndex = 0
    for epoch in range(parameters['num_epochs']):
        batchIndex = 0
        for index,data in enumerate(loader):
            trainTensor =  (data['image'])
            labelTensor = data['label']
            outputs = model(Variable(trainTensor))
            loss = criterion(outputs,labelTensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batchIndex = batchIndex + 1
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, parameters['num_epochs'],batchIndex,len(loader), loss.item()))
    torch.save(model,MODELPATH + 'CNNModel.pth')










