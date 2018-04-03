
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import itertools
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture


# In[23]:


PATH = "./data/Anuran Calls (MFCCs)/Frogs_MFCCs.csv"


# In[24]:


# metric functions
# --------------------------------------------

def total_sum_of_squares_df(df, centroid = None):
    """ Calculates and returns the TTS of the given DataFrame """
    if centroid is None:
        centroid = find_centroid_df(df)
        
    return total_sum_of_squares(df.as_matrix(), centroid)

def total_sum_of_squares(data, centroid):
    """ Calculates and returns the TTS of the given matrix
    
    Arguments:
      data - Iterable<Iterable>
      centroid - Array
    """        
    total = 0
    
    for row in data:
        for index, value in enumerate(row):
            diff = value - centroid[index]
            diffsq = diff ** 2
            total += diffsq
            
    return total


def find_centroid_df(df):
    """ Calculates and returns the centroid for a DataFrame """
    return df.mean()

# clustering functions
# --------------------------

def get_cluster_indexes(cluster_assignments):
    cluster_slices = {}
    
    for index, assignment in enumerate(assignments):
        if assignment not in cluster_slices:
            cluster_slices[assignment] = list()
            
        cluster_slices[assignment].append(index)
        
    return cluster_slices

def get_cluster_data(df, assignments):
    cluster_indexes = get_cluster_indexes(assignments)
    
    cluster_data = {k: df.iloc[v] for k, v in cluster_indexes.items()}
    
    return cluster_data

def get_clusters(df, assignments):
    """Returns an array of tuples with (<cluster>, <cluster_centroid>, <cluster_data_points>)"""
    
    return [
        (cluster, find_centroid_df(cluster_data), cluster_data) 
        for cluster, cluster_data 
        in get_cluster_data(df, assignments).items()
    ]

# model runners
# ------------------------------------------

def run_gaussian_mixture(model, data):
    model.fit(data)
    return model.predict(data)

def run_kmeans(model, data):
    model.fit(data)
    return model.predict(data)

def run_hclustering(model, data):
    return model.fit_predict(data)

# data functions 
# -------------------------------------------

def clean_data(data):
    # Strip whitespaces from all string values
    # and replace "?" with None,
    # and drop all na rows
    data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)                .replace(["?"], [None])                .dropna()

    data = data.iloc[:,2:22]
    return data

def prepare_data(data):
    return data

def read_data(path):
    dataset = pd.read_csv(path)
    dataset = clean_data(dataset)
    dataset = prepare_data(dataset)
    return dataset


# In[25]:


df = read_data(PATH)


# In[27]:


tss = total_sum_of_squares_df(df)
print("tss for data = %s" % tss)


# In[31]:


models = [
    ("KMeans", lambda k: KMeans(n_clusters=k), run_kmeans),
    ("H-Clustering", lambda k: AgglomerativeClustering(n_clusters=k), run_hclustering),
    ("Gaussian Mixture", lambda k: GaussianMixture(n_components=k, reg_covar=0.001), run_gaussian_mixture)
]


# In[32]:


for model_name, create_model, run_model in models:
    print("-------------------------------")
    print(model_name)
    print("-------------------------------")
    print("")
    for k in range(1,11):
        print("Calculating %s clusters..." % k)
        print("")
        model = create_model(k)
        assignments = run_model(model, df)
        clusters = get_clusters(df, assignments)

        twss = 0
        for cluster, centroid, cluster_slice in clusters:
            cluster_tss = total_sum_of_squares_df(cluster_slice, centroid)
            print("cluster %s | tss = %s | size = %s" % (cluster, cluster_tss, len(cluster_slice)))
            twss += cluster_tss

        print("twss/tss = %s/%s = %s" % (twss, tss, twss / tss))
        print("")

