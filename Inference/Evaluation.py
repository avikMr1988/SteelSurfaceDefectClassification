import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import numpy as np
import os

def CNNEval(testData):
    MODEL_STORE_PATH = './DUMPS/'
    if os.path.exists(MODEL_STORE_PATH) == False:
        print('Please run Training first')
        return
    model = torch.load(MODEL_STORE_PATH + 'CNNModel.pth')
    model.eval()
    testLoader = DataLoader(testData,batch_size=1)
    predictionList = list()
    outputList = list()
    for image,label in enumerate(testLoader):
        testData = Variable(label['image'])
        with torch.no_grad():
            output = model(testData)
        softmax = torch.exp(output).cpu()
        prob = list(softmax.numpy())
        prediction = np.argmax(prob,axis=1)[0]
        predictionList.append(prediction)
        outputList.append(label['label'][0])
    print(predictionList)
    print(outputList)
    accuracy = accuracy_score(outputList, predictionList)
    torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')
    return accuracy











