import gdown
import PIL
from PIL import Image
import glob
import os
import zipfile
import csv
from pathlib import Path

def DownloadNEU64():
    path = Path(os.getcwd())
    TARPATH = str(path) + '/Data/'
    print(TARPATH)
    os.makedirs(TARPATH, exist_ok=True)
    fileName = 'NEU-CLS-64.zip'
    url = 'https://drive.google.com/uc?id=1UD68IdpVjmN9SoOp77W5UgFKPDQziRPj'
    output = TARPATH + '/' + fileName
    gdown.download(url, output, quiet=False)
    return output
def ExtractFiles(tarPath):
    extractPath = os.path.dirname(tarPath)
    dirName = (os.path.basename(tarPath))[:-4]
    os.makedirs(extractPath, exist_ok=True)
    with zipfile.ZipFile(tarPath, 'r') as zip_ref:
        zip_ref.extractall(extractPath)
    os.remove(tarPath)
    return extractPath + '/' + dirName

def augmentFilesANDPrepeareCSV():
    tarPath = DownloadNEU64()
    filePath = ExtractFiles(tarPath)
    csvTrainFileName = os.path.dirname(filePath) + '/Train.csv'
    csvTestFileName = os.path.dirname(filePath) + '/Test.csv'
    csvValFileName = os.path.dirname(filePath) + '/Val.csv'
    labels = ['cr','gg','in','pa','ps','rp','rs','sc','sp']
    dirs = glob.glob(filePath+'/*')
    trainIndex = 0
    testIndex = 0
    grossTrainList = list()
    grossTestList = list()
    grossValList = list()
    for dir in dirs:
        if os.path.basename(dir) == 'gg' or os.path.basename(dir) == 'sp' or os.path.basename(dir) == 'rp':
            files = glob.glob(dir+'/*')
            fileCount = len(files)
            trainIndex = 0
            testIndex = 0
            valIndex = 0
            for file in files:
                img = Image.open(file)
                out1 = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                out2 = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                out3 = img.rotate(30)
                out4 = img.rotate(-30)

                fileCount = fileCount + 1
                out1.save(dir + '/' + str(fileCount) + '.jpg')
                fileCount = fileCount + 1
                out2.save(dir + '/' + str(fileCount) + '.jpg')
                fileCount = fileCount + 1
                out3.save(dir + '/' + str(fileCount) + '.jpg')
                fileCount = fileCount + 1
                out4.save(dir + '/' + str(fileCount) + '.jpg')

                if trainIndex <= 700:
                    grossTrainList.append([file,str(labels.index(os.path.basename(dir)))])
                    grossTrainList.append([dir + '/' + str(fileCount - 4) + '.jpg', str(labels.index(os.path.basename(dir)))])
                    grossTrainList.append([dir + '/' + str(fileCount - 3) + '.jpg', str(labels.index(os.path.basename(dir)))])
                    grossTrainList.append([dir + '/' + str(fileCount - 2) + '.jpg', str(labels.index(os.path.basename(dir)))])
                    grossTrainList.append([dir + '/' + str(fileCount - 1) + '.jpg', str(labels.index(os.path.basename(dir)))])
                    trainIndex = trainIndex + 5
                elif valIndex <= 100:
                    grossValList.append([file, str(labels.index(os.path.basename(dir)))])
                    grossValList.append([dir + '/' + str(fileCount - 4) + '.jpg', str(labels.index(os.path.basename(dir)))])
                    grossValList.append([dir + '/' + str(fileCount - 3) + '.jpg', str(labels.index(os.path.basename(dir)))])
                    grossValList.append([dir + '/' + str(fileCount - 2) + '.jpg', str(labels.index(os.path.basename(dir)))])
                    grossValList.append([dir + '/' + str(fileCount - 1) + '.jpg', str(labels.index(os.path.basename(dir)))])
                    valIndex = valIndex + 5
                elif testIndex <= 200:
                    grossTestList.append([file, str(labels.index(os.path.basename(dir)))])
                    grossTestList.append([dir + '/' + str(fileCount - 4) + '.jpg', str(labels.index(os.path.basename(dir)))])
                    grossTestList.append([dir + '/' + str(fileCount - 3) + '.jpg', str(labels.index(os.path.basename(dir)))])
                    grossTestList.append([dir + '/' + str(fileCount - 2) + '.jpg', str(labels.index(os.path.basename(dir)))])
                    grossTestList.append([dir + '/' + str(fileCount - 1) + '.jpg', str(labels.index(os.path.basename(dir)))])
                    testIndex = testIndex + 5
                else:
                    break
        elif os.path.basename(dir) == 'in' or os.path.basename(dir) == 'ps' or os.path.basename(dir) == 'sc':
            print(os.path.basename(dir))
            files = glob.glob(dir + '/*')
            trainIndex = 0
            testIndex = 0
            valIndex = 0
            fileCount = len(files)
            for file in files:
                img = Image.open(file)
                out1 = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                out2 = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                fileCount = fileCount + 1
                out1.save(dir + '/' + str(fileCount) + '.jpg')
                fileCount = fileCount + 1
                out2.save(dir + '/' + str(fileCount) + '.jpg')
                if trainIndex < 700:
                    grossTrainList.append([file,str(labels.index(os.path.basename(dir)))])
                    grossTrainList.append([dir + '/' + str(fileCount - 4) + '.jpg', str(labels.index(os.path.basename(dir)))])
                    grossTrainList.append([dir + '/' + str(fileCount - 3) + '.jpg', str(labels.index(os.path.basename(dir)))])
                    trainIndex = trainIndex + 3
                elif valIndex <= 100:
                    grossValList.append([file, str(labels.index(os.path.basename(dir)))])
                    grossValList.append([dir + '/' + str(fileCount - 4) + '.jpg', str(labels.index(os.path.basename(dir)))])
                    grossValList.append([dir + '/' + str(fileCount - 3) + '.jpg', str(labels.index(os.path.basename(dir)))])
                    valIndex = valIndex + 3
                elif testIndex <= 200:
                    grossTestList.append([file, str(labels.index(os.path.basename(dir)))])
                    grossTestList.append([dir + '/' + str(fileCount - 4) + '.jpg', str(labels.index(os.path.basename(dir)))])
                    grossTestList.append([dir + '/' + str(fileCount - 3) + '.jpg', str(labels.index(os.path.basename(dir)))])
                    testIndex = testIndex + 3
                else:
                    break
        else:
            files = glob.glob(dir + '/*')
            fileCount = len(files)
            trainIndex = 0
            testIndex = 0
            valIndex = 0
            for file in files:
                if trainIndex < 700:
                    grossTrainList.append([file,str(labels.index(os.path.basename(dir)))])
                    trainIndex = trainIndex + 1
                elif valIndex < 100:
                    grossValList.append([file, str(labels.index(os.path.basename(dir)))])
                    valIndex = valIndex + 1
                elif testIndex <= 200:
                    grossTestList.append([file, str(labels.index(os.path.basename(dir)))])
                    testIndex = testIndex + 1
                else:
                    break
    with open(csvTrainFileName, 'w', newline='') as file:
        writer = csv.writer(file,delimiter=',', quoting=csv.QUOTE_ALL)
        for i in range(len(grossTrainList)):
            writer.writerow([grossTrainList[i][0],grossTrainList[i][1]])
    with open(csvTestFileName, 'w', newline='') as file:
        writer = csv.writer(file,delimiter=',', quoting=csv.QUOTE_ALL)
        for i in range(len(grossTestList)):
            writer.writerow([grossTestList[i][0],grossTestList[i][1]])
    with open(csvValFileName, 'w', newline='') as file:
        writer = csv.writer(file,delimiter=',', quoting=csv.QUOTE_ALL)
        for i in range(len(grossValList)):
            writer.writerow([grossValList[i][0],grossValList[i][1]])

def DownloadNEU():
    url = 'https://drive.google.com/uc?id=1UD68IdpVjmN9SoOp77W5UgFKPDQziRPj'
    output = '../Data/NEU-CLS-64/NEU-CLS-64.tar'
    gdown.download(url, output, quiet=False)

def DownloadSurfaceDefectGross():
    url = 'https://drive.google.com/uc?id=1UD68IdpVjmN9SoOp77W5UgFKPDQziRPj'
    output = '../Data/NEU-CLS-64/NEU-CLS-64.tar'
    gdown.download(url, output, quiet=False)

def DownloadSurfaceDefectDetect():
    url = 'https://drive.google.com/uc?id=1UD68IdpVjmN9SoOp77W5UgFKPDQziRPj'
    output = '../Data/NEU-CLS-64/NEU-CLS-64.tar'
    gdown.download(url, output, quiet=False)