import Utils.Downloader as dwnld
from Utils.DataSet import SurfaceDataSet
from Inference import Training,Evaluation
import os

def main():

# Initialize Path Variables

    DATAPATH= './Data'
    TRAINFILENAME = 'Train.csv'
    TESTFILENAME = 'Test.csv'
    print('Downloading NEU-CLS-64 data....')

# Check for Presence of Data Set. If Present don't Download

    if os.path.exists(DATAPATH + '/NEU-CLS-64') == True or os.path.exists(DATAPATH + TRAINFILENAME) or os.path.exists(DATAPATH + TESTFILENAME):
        print('Data Set is already prepeared')
    else:
        dwnld.augmentFilesANDPrepeareCSV()

# Create the train and test dataset using the DataSet class of PyTorch

    trainDataset = SurfaceDataSet(DATAPATH + '/' + TRAINFILENAME)
    testDataSet = SurfaceDataSet(DATAPATH + '/' + TESTFILENAME)

# Train a simple CNN Model on the Train Data Set

    Training.CNNTrain(trainDataset)

# Evaluate the model on the test data set and return the accuracy

    accuracy = Evaluation.CNNEval(testDataSet)
    print('The accuracy of prediction on the test data set is {:.4f}' .format(accuracy))

if __name__ == '__main__':
    main()

