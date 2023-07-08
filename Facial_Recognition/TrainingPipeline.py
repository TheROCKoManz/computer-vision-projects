from DataPreproc import Preprocess, extractFrames
from Training import TrainModel
def main():
    Targets = ['manasij','ayush'] # list of training targets
    extractFrames.extractFrames()
    Data = Preprocess.preprocess(Targets)
    TrainModel.train(Data)

if __name__ == '__main__':
    main()