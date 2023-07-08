from DataPreproc import Preprocess
from Training import TrainModel
def main():
    Targets = [] # list of training targets
    Data = Preprocess.preprocess(Targets)
    TrainModel.train(Data)

if __name__ == '__main__':
    main()