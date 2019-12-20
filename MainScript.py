import Utils.Downloader as dwnld
from Utils.DataSet import SurfaceDataSet
from Inference import Training,Evaluation

def main():
    DATAPATH= './Data'
    TRAINFILENAME = 'Train.csv'
    TESTFILENAME = 'Test.csv'
    print('Downloading NEU-CLS-64 data....')
    #dwnld.augmentFilesANDPrepeareCSV()
    #trainDataset = SurfaceDataSet(DATAPATH + '/' + TRAINFILENAME)
    #trainDataset = SurfaceDataSet(DATAPATH + '/' + TRAINFILENAME)
    testDataSet = SurfaceDataSet(DATAPATH + '/' + TESTFILENAME)
    #Training.CNNTrain(trainDataset)
    accuracy = Evaluation.CNNEval(testDataSet)
    print('The accuracy of prediction on the test data set is {:.4f}' .format(accuracy))

if __name__ == '__main__':
    main()

