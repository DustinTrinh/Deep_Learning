# AutoEncoders

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
# Sep ::            => Parameters are separate by ::
# Header = None      => Tell it theres no header, no column names
# Engine = python   => Make it efficent
# encoding          => To deal with special characters in the movie's name
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

#Prepare training and test sets
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max( max(training_set[:, 0]), max(test_set[:, 0]) ))
nb_movies = int(max( max(training_set[:, 1]), max(test_set[:, 1]) ))

# Converting the data into an array with users in lines and movies in columns
def convert (data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        # Get all the movies that had been rated by the "user id" (example: 1) in the training_set
        id_movies = data[:, 1][data[:, 0] == id_users]
        
        # Get all the ratings that had been rated by the "user id" (example: 1) in the training_set
        id_ratings = data[:, 2][data[:, 0] == id_users]
        
        #Create a list with the shape of nb_movies
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Create architecture of the neural networks
#Create a class for Stack Autoencoder, inherite from PArent class Module
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__() #Make sure we inherit all the functions from  parent
        #First full connection 
        #put the movies into the input vector, 2nd input is the nodes in the first hidden layer
        self.fc1 = nn.Linear(nb_movies, 20)
        
        #Second full connection and connect between first and second layer
        #First param is to use the 20 nodes from the above declaration, second param is the nodes in the 2nd hidden layer
        self.fc2 = nn.Linear(20, 10)
        
        #Third full connection and connect between the 2 hidden layer with the third
        #Start to decode
        self.fc3 = nn.Linear(10, 20)

        #Fourth full connection
        #Output layer
        self.fc4 = nn.Linear(20, nb_movies)
        
        #Activation function
        self.activation = nn.Sigmoid()
        
    #Forward propagation functin. X is input vector
    def forward(self, x):
        #Return encoded rector
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
    
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

#Training the SAE
nb_epoch = 200

for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data*mean_corrector)
            s += 1.
            optimizer.step()
    print('Epoch: ' + str(epoch) + ' loss: ' + str(train_loss/s))    
        
#Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[(target == 0).unsqueeze(0)] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += float(np.sqrt(loss.data*mean_corrector))
        s += 1.
print('loss: ' + str(test_loss/s)) 

