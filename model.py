# -*- coding: utf-8 -*-
#importing libraries
#main libraries 
import numpy as np 
import pandas as pd 

# visual libraries
from matplotlib import pyplot as plt
import seaborn as sns
# sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import random
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


#read the  data
dataset=pd.read_csv('cities_predict.csv')

#DATA PREPROCESSING 
#->remove the nomad_score from the dataset 
ratings_data=dataset.iloc[:,:13]#independet values 
label_data=dataset.iloc[:,16:] # dependent value 
complete_data=pd.concat([ratings_data,label_data],axis=1)
#print(ratings_data.columns)
#checking the missing values 
complete_data.isnull().any().sum()

#take top 5 rows of the dataset
#dataset2=complete_data.head(400)
ratings_data=complete_data.iloc[:,:13]#independet values 
label_data=complete_data.iloc[:,-1:]

#function to normalazie the data 
def normalize_data(row):
   amin, amax = min(row), max(row)
   for i, val in enumerate(row):
       row[i] = (val-amin) / (amax-amin)
       
   return row 
#normalazing the dataframe 
norm_data=list()
for i in ratings_data.values:
     i=i.tolist()
     row=normalize_data(i)
     norm_data.append(row) 

#now create a dataframe from the normalize data
ratings_data = pd.DataFrame(data=norm_data ,columns =['cost_of_living', 'freedom_of_speech', 'friendly_to_foreigners', 'fun',
       'happiness', 'healthcare', 'internet', 'nightlife', 'peace','quality_of_life', 'safety', 'traffic_safety', 'walkability'])


#concate the normalized ratings data and label_data
final_data=pd.concat([ratings_data,label_data],axis=1)

#to calculate the euclidian distance
from scipy.spatial import distance

#get neighbors of the selected city
def get_neighbors(train_data,test_row,k_neighbors):
# Find the distance between the selected_city and others in the test instance(city).
  euclidean_distances = train_data.apply(lambda row: distance.euclidean(row, test_row), axis=1)

# Create a new dataframe with distances.
  distance_frame = pd.DataFrame(data={"dist": euclidean_distances, "idx": euclidean_distances.index})
  
  distance_frame.sort_values(by=['dist'], inplace=True)
  
# Find the most similar cities to the city that have selected features
  neighbors=list()
  for i in range(k_neighbors):
     neighbors.append(distance_frame.iloc[i]["idx"])

  return  neighbors


  """
  NOTE:
  A Counter is a dict subclass for counting hashable objects. It is an unordered collection where elements
  are stored as dictionary keys and their counts are stored as dictionary values
  """
from collections import Counter

#given a list  of nearest neighbours for a test case, tally up their classes to vote on test case class
def get_majority(neighbors,k):
   
    k_labels = list()
    for i in range(k):
        k_labels.append(label_data.loc[int(neighbors[i])]["place_slug"])
    
    c = Counter(k_labels)
    
    return c.most_common()[0][0]

#generate a random vector for test
    
def generate_random_city():
 random_list=list()
 for i in range(13):
     if i==6:
         num= random.randint(1, 100)
     else:
         num= random.randint(1, 4)
     random_list.append(num)
 return random_list

#make a recommendation 
def make_recommendation(train_data,test_instance,k_neighbors):    
  neighbors=get_neighbors(train_data,test_instance,k_neighbors)
  
  for i in range(k_neighbors):
      most_similar = final_data.loc[int(neighbors[i])]["place_slug"]
      print("most-similar city:",most_similar)
      
 
  prediction=get_majority(neighbors,k_neighbors)
    
  return prediction

"""
calling the generate_random_city function and receive a random vector
we will take the inputs via FlaskApi in app.py instead of using this random generator
This function is only for checking the the results and accuracy!!!!!!!!
"""
selected_features=generate_random_city()
print("selected features:",selected_features)
#normalize the received vector
selected_features=normalize_data(selected_features)
#convert the list to a dataframe 
selected_city=pd.DataFrame(data=[selected_features],columns =['cost_of_living', 'freedom_of_speech', 'friendly_to_foreigners', 'fun',
       'happiness', 'healthcare', 'internet', 'nightlife', 'peace','quality_of_life', 'safety', 'traffic_safety', 'walkability'])
  


#define  the true vector to compare to the predicted one 
y_true=selected_features
#k -->number of neighbors we want 
k=5

#print recommendation
recommended_city=make_recommendation(ratings_data,selected_city,k)
print("***********") 
print('recommended city:',recommended_city)
print("***********") 
#the evaluation of the algorithm


#checking acccuracy using r2_score
def accuracy_with_r2score():
    """
    NOTE:
    It represents the proportion of variance (of y)
    that has been explained by the independent variables in the model.
    Best possible score is 1.0 
    """
    y_pred= ratings_data[final_data["place_slug"] == recommended_city]
    y_pred=(y_pred.values).tolist()
    from sklearn.metrics import r2_score
    accuracy=r2_score(y_true, y_pred[0])
    print("R2 score for knn is:",accuracy)
    

#checking accuracy using  mean_squared_error
def accuracy_with_rmse():
    """
    NOTE:
    The mean_squared_error function computes mean square error, a risk metric 
    corresponding to the expected value of the squared (quadratic) error or loss.
    """
    y_pred= ratings_data[final_data["place_slug"] == recommended_city]
    y_pred=(y_pred.values).tolist()
    from math import sqrt
    from sklearn.metrics import mean_squared_error
    mean_squared_error=mean_squared_error(y_true,y_pred[0])
    print("Root Mean Square error for knn is: ",sqrt(mean_squared_error))

#calling the accuracy and rmse functions
accuracy_with_r2score()

accuracy_with_rmse()


