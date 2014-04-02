__author__ = 'hanskhe'
__author__ = 'juularthur'
import Backprop_skeleton as Bp
from pylab import *

#Class for holding your data - one object for each line in the dataset
class dataInstance:

    def __init__(self,qid,rating,features):
        self.qid = qid #ID of the query
        self.rating = rating #Rating of this site for this query
        self.features = features #The features of this query-site pair.

    def __str__(self):
        return "Datainstance - qid: "+ str(self.qid)+ ". rating: "+ str(self.rating)+ ". features: "+ str(self.features)


#A class that holds all the data in one of our sets (the training set or the testset)
class dataHolder:

    def __init__(self, dataset):
        self.dataset = self.loadData(dataset)

    def loadData(self,file):
        #Input: A file with the data.
        #Output: A dict mapping each query ID to the relevant documents, like this: dataset[queryID] = [dataInstance1, dataInstance2, ...]
        data = open(file)
        dataset = {}
        for line in data:
            #Extracting all the useful info from the line of data
            lineData = line.split()
            rating = int(lineData[0])
            qid = int(lineData[1].split(':')[1])
            features = []
            for elem in lineData[2:]:
                if '#docid' in elem: #We reached a comment. Line done.
                    break
                features.append(float(elem.split(':')[1]))
            #Creating a new data instance, inserting in the dict.
            di = dataInstance(qid,rating,features)
            if qid in dataset.keys():
                dataset[qid].append(di)
            else:
                dataset[qid]=[di]
        return dataset


def runRanker(trainingset, testset):
    #Dataholders for training and testset
    dhTraining = dataHolder(trainingset)
    dhTesting = dataHolder(testset)

    #Creating an ANN instance - feel free to experiment with the learning rate (the third parameter).
    nn = Bp.NN(46,10,0.001)

    trainingPatterns = [] #For holding all the training patterns we will feed the network
    testPatterns = [] #For holding all the test patterns we will feed the network
    for qid in dhTraining.dataset.keys():
        #This iterates through every query ID in our training set
        dataInstance=dhTraining.dataset[qid] #All data instances (query, features, rating) for query qid
        for i in range(len(dataInstance)-1):
            for j in range(i+1,len(dataInstance)):
                if (dataInstance[i].rating != dataInstance[j].rating):
                    if (dataInstance[i].rating>dataInstance[j].rating):
                        trainingPatterns.append((dataInstance[i],dataInstance[j]))
                    else:
                        trainingPatterns.append((dataInstance[j],dataInstance[i]))

    for qid in dhTesting.dataset.keys():
        #This iterates through every query ID in our test set
        dataInstance=dhTesting.dataset[qid]
        
        for i in range(len(dataInstance)-1):
            for j in range(i+1,len(dataInstance)):
                if (dataInstance[i].rating != dataInstance[j].rating):
                    if (dataInstance[i].rating>dataInstance[j].rating):
                        testPatterns.append((dataInstance[i],dataInstance[j]))
                    else:
                        testPatterns.append((dataInstance[j],dataInstance[i]))

    #Check ANN performance before training
    print("first")
    test_error_percent = []
    training_error_percent = []
    test_error_percent.append(nn.countMisorderedPairs(testPatterns))
    training_error_percent.append(nn.countMisorderedPairs(trainingPatterns))
    numIterations = 25
    for i in range(numIterations):
        #Running 25 iterations, measuring testing performance after each round of training.
        #Training
        nn.train(trainingPatterns,iterations=1)
        #Check ANN performance after training.
        print("Iteration #" + str(i))
        test_error_percent.append(nn.countMisorderedPairs(testPatterns))
        training_error_percent.append(nn.countMisorderedPairs(trainingPatterns))

    #TODO: Store the data returned by countMisorderedPairs and plot it, showing how training and testing errors develop.
    #Printing graph with pylab

    #plot(range(1,numIterations+2),test_error_percent, label="Test set")
    #plot(range(1,numIterations+2),training_error_percent, label="Training set")
    #legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    #ylim([0,1])
    #show()
    return test_error_percent, training_error_percent

#Method for calculating the average of 5 runs.
#Initialize empty lists
total_test = [0.0]*26
total_training = [0.0]*26
for i in range(0,5):
    #Run the program 5 times
    test, training = runRanker("../data_sets/train.txt","../data_sets/test.txt")
    for j in range(len(test)):
        #Add results for this run to the lists of results
        total_test[j] += test[j]
        total_training += training[j]

#Average the results
total_test = [x/5 for x in total_test]
total_training = [x/5 for x in total_training]

#Plot graphs
plot(range(1,27),total_test, label="Test set")
plot(range(1,27), total_training, label="Training set")
legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
ylim([0,1])
show()