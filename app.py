
from flask import Flask, request, jsonify, render_template
from bs4 import BeautifulSoup
import requests
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

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    k=5
    int_features = [int(x) for x in request.form.values()]
    #normalize the received vector
    selected_features=normalize_data(int_features)
    #convert the list to a dataframe 
    selected_city=pd.DataFrame(data=[selected_features],columns =['cost_of_living', 'freedom_of_speech', 'friendly_to_foreigners', 'fun',
       'happiness', 'healthcare', 'internet', 'nightlife', 'peace','quality_of_life', 'safety', 'traffic_safety', 'walkability'])
    
    prediction = make_recommendation(ratings_data,selected_city,k)
    #reshape the string
    word=prediction.split('-')
  
    if("city" in word):
         city=word[0] + "-"+ "city"
    else:
         city=word[0]
    print(city)
    url="https://gezimanya.com/" + city

    # Make a GET request to fetch the raw HTML content
    html_content = requests.get(url).text
    
    # Parse the html content
    soup = BeautifulSoup(html_content, "lxml")
    info = soup.find("article", class_= "node teaser-taxonomy-wrapper node--type-lokasyon-detay node--view-mode-teaser-taxonomy")
    
    
    if(info):
        info_list=[]
        article=info.findAll("p")
        for p in article:
            p = p.text
            info_list.append(p)
        return render_template('index.html', prediction_text='We recommend you to take your next flight to  {}'.format(prediction),city_information=info_list)
    
    else:
        city_info=["Bu sehir hakkında bilgi bulunmamaktadır.",'']
        return render_template('index.html', prediction_text='predicted city {}'.format(prediction),city_information=city_info)
    
        
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

#make a recommendation 
def make_recommendation(train_data,test_instance,k_neighbors):    
  neighbors=get_neighbors(train_data,test_instance,k_neighbors)
  
  for i in range(k_neighbors):
      most_similar = final_data.loc[int(neighbors[i])]["place_slug"]
      print("most-similar city:",most_similar)
      
 
  prediction=get_majority(neighbors,k_neighbors)
      
  return prediction

if __name__ == "__main__":
    app.run(debug=True)

