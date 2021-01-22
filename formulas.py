#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import math


# In[16]:



# In[17]:

# Class to explore data: size, shape, inf, missing values, categorical and nume var, class balance, and clean data base
class ExploringData:
    def __init__(self,database, class_column: int=-1):
        self.database = database
        self.class_column = class_column # data column where the target column is.
        
    def basic_database(self):
        print("Number of elements:" +" "+ str(self.database.size)) 
        print("Dimension:" +" "+ str(self.database.ndim)) # dimension
        print("Rows, Columns:" +" "+ str(self.database.shape)) # rows, columns
        print()
        print("Database information"+" "+ str(self.database.info()))
        print()

    def missing_values(self):
        miss_value = self.database.isnull().sum()
        miss_value_percentage = 100 * (self.database.isnull().sum() / len(self.database))
        miss_value_table = pd.concat([miss_value, miss_value_percentage], axis=1)
        miss_value_table = miss_value_table.rename(columns={0:"Missing Values", 1:"% Missing Values"})
        miss_value_table = miss_value_table[miss_value_table.iloc[:,1]!=0].sort_values("% Missing Values",ascending=False).round(2)
        print(miss_value_table)
        print()
        print(self.database.isna().any())
    
    def categorical_numerical (self):        
        #Store categorical and numerical columns:
        self.numerical_cols = list(set(self.database._get_numeric_data().columns))
        self.categorical_cols= list(set(self.database.columns)-set(self.database._get_numeric_data().columns)) 
        print("Numerical columns are:", str(self.numerical_cols), sep="\n")
        print()
        print("Categorical columns are:", str(self.categorical_cols), sep="\n")
        print()
        
     
    def class_balance(self):
        print("Class label distribution")
        print()
        labels = np.unique(self.database.iloc[:,self.class_column])
        count = self.database.iloc[:,self.class_column].value_counts(normalize=True)
        print(count)
        
    def clean_database(self):
        na_cols = self.database.isna().any()
        na_cols = na_cols[na_cols == True].reset_index()
        na_cols = na_cols["index"].tolist()
        for col in self.database.columns[:-1]:
            if col in na_cols:
                if self.database[col].dtype != "object":
                    self.database[col]= self.database[col].fillna(self.database[col].mean()).round(0)
        print(self.database.isna().any())        


# In[18]:


# What parameters have high correlation to avoid intercolinearity
# I exclude categorical variables
class CorrelationMatrix:
    def __init__(self,database, name_col):
        self.database = database
        self.name_col = name_col # database columns 
    def correlation_matrix(self):
        k = len(self.name_col)
        cols=self.database[self.name_col].corr().index
        cm = self.database[cols].corr()
        plt.figure(figsize=(15,6))
        sns.heatmap(cm, annot=True, cmap="viridis")
    
    


# In[19]:


# Function for plotting features
class Plotting:
    def __init__(self, database, name_cols, cols: int=3):
        self.database = database
        self.name_cols = name_cols
        self.cols = cols # number of charts per row
        self.rows = math.ceil(len(name_cols)/cols) # number of rows  per figure
        
        
    def features_plotting(self):
        fig,ax = plt.subplots(nrows=self.rows, ncols=self.cols, figsize=(50,30))
        self.i = 0
       
        for col in self.name_cols:
            self.database[col].plot(kind="hist", ax=ax[self.i // self.cols][self.i % self.cols], title=col)
            self.i = self.i + 1
        


# In[20]:

# This class summarizes the classification metrics, yields a confusion matrix and plotts RUC and RP curves
#from pandas_ml import ConfusionMatrix
class SummaryMetrics:
    def __init__(self,actual_label, predicted_label, predicted_prob):
        self.actual_label = actual_label
        self.predicted_label = predicted_label
        self.predicted_prob = predicted_prob

    def performance_metrics(self):
        from sklearn import metrics
        from numpy import argmax
        self.acc = metrics.accuracy_score(self.actual_label, self.predicted_label) 
        self.ROC_area = metrics.roc_auc_score(self.actual_label, self.predicted_prob).round(3)
        self.precision, self.recall, self.thresholds_p_r = metrics.precision_recall_curve(self.actual_label,self.predicted_prob)
        self.PR_area = metrics.auc(self.recall,self.precision).round(3)
        self.fpr1,self.tpr1,self.thresholds_roc=metrics.roc_curve(self.actual_label,self.predicted_prob) 
        self.tn, self.fp, self.fn, self.tp = metrics.confusion_matrix(self.actual_label,self.predicted_label).ravel()
        self.tpr = round((self.tp/(self.tp + self.fn)),3)
        self.fpr = round((1-(self.tn/(self.tn + self.fp))),3)
        self.tnr = round((self.tn/(self.tn + self.fp)),3)
        self.fnr = round((1-(self.tp/(self.tp + self.fn))),3)
        self.precision_no = round((self.tn / (self.tn + self.fn)),3)
        self.precision_yes = round((self.tp / (self.tp + self.fp)),3)
        self.recall_no = round((self.tn / (self.tn + self.fp)),3)
        self.recall_yes = round((self.tp / (self.tp + self.fn)),3)
        self.f1_no = round((2 * self.precision_no * self.recall_no / (self.precision_no + self.recall_no)),3)
        self.f1_yes = round((2 * self.precision_yes * self.recall_yes / (self.precision_yes + self.recall_yes)),3)
    
    def confusion_matrix(self):                    
        t_act_no = self.tn + self.fp
        t_act_yes = self.fn + self.tp
        print("Accuracy"+" "+str(self.acc))
        print("ROC area"+" "+str(self.ROC_area))
        print("PR area"+"  "+str(self.PR_area))
        
        self.matrix = pd.DataFrame({"Pre_No_ratio":[self.tnr,self.fnr,"--"],
                                      "Pre_Yes_ratio":[self.fpr,self.tpr,"--"],
                                      "Total_ratio":[(self.tnr+self.fpr),(self.fnr+self.tpr),"--"],
                                      "Pre_No":[self.tn,self.fn,(self.tn+self.fn)],
                                      "Pre_Yes":[self.fp,self.tp,(self.fp+ self.tp)],
                                      "Total_Act":[t_act_no,t_act_yes,(self.tn+self.fn+self.fp+self.tp)],
                                      "Precision":[self.precision_no,self.precision_yes,"--"],
                                      "Recall":[self.recall_no,self.recall_yes,"--"],
                                      "F1":[self.f1_no,self.f1_yes,"--"]},   
                                      index=["No","Yes","T_Pre"])
    

    def plotting_metrics_curves(self):
        # ==== Getting the optimun F1score /tpr /fpr / Threshold ====
        self.f1 = 2 * (self.recall * self.precision) / (self.recall + self.precision)
        ix = np.argmax(self.f1)
        self.opt_threshold = self.thresholds_p_r[ix]
        self.opt_f1 = self.f1[ix]
        self.opt_precision = self.precision[ix]
        self.opt_recall = self.recall[ix]
        # Need to find the index in thresholds1 closest to the opt_threshold
        # and get the correspongind fpr1 and tpr1
        index_thresholds1 = min(range(len(self.thresholds_roc)), key=lambda i: abs(self.thresholds_roc[i]-self.opt_threshold))
        self.opt_fpr1 = self.fpr1[index_thresholds1]
        self.opt_tpr1 = self.tpr1[index_thresholds1]
    
        # ==== Plotting ====
        fig, (ax0, ax1, ax2)=plt.subplots(1, 3, sharey=False, figsize=(18,4))
        lw = 3
        # ==== Plotting the ROC Curve ====
        ax0.plot(self.fpr1, self.tpr1, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % self.ROC_area)
        ax0.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        ax0.scatter(self.opt_fpr1, self.opt_tpr1, marker='o', color='black', label='Best (threshold = %0.2f)'% self.opt_threshold)
        ax0.set_xlim([0.0, 1.0])
        ax0.set_ylim([0.0, 1.05])
        ax0.set(title='ROC Curve', xlabel="FPR", ylabel='TPR')
        ax0.legend(loc="lower right")
        # ==== Plotting the RP Curve ====
        no_skill = len(self.actual_label[self.actual_label==1])/len(self.actual_label)
        ax1.plot(self.recall, self.precision, color='darkorange',lw=lw, label='PR curve (area = %0.2f)' % self.PR_area)
        ax1.plot([0, 1], [no_skill, no_skill], color='navy', lw=lw, linestyle='--')
        ax1.scatter(self.opt_recall, self.opt_precision, marker='o', color='black', label='Best (threshold: P = %0.2f)'% self.opt_threshold)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set(title='PR Curve', xlabel="Recall", ylabel='Precision')
        ax1.legend(loc="upper right")
        # ==== Plotting the predicted proba ====
        ax2.hist(self.predicted_prob, color='darkorange')
        ax2.set(title='Predict Prob', xlabel="Predicted Prob", ylabel='Class 1')   
                   


# In[21]:


# Class Bayes with continuous and categorical features; where the target is a binary classification.
# This class yields a Bayes prediction with a mix of continuous and categorical features.
class BayesM:
    def __init__(self,categorical, continuous):
        self.categorical = categorical
        self.continuous = continuous
        
        
    def fit(self,x,y):    
        # yield the dict with the hist information by feature by class        
        # reindex a data base, when it comes from SKFold, etc.
        x.reset_index(drop=True, inplace = True)
        y.reset_index(drop=True, inplace = True)
        self.x=x
        self.y=y
            
        self.summary_feature = dict()
        self.class_label = list(np.unique(self.y))
        self.columns = self.x.columns
        self.summary=()
        self.count=self.sum=self.mean=self.stdv=0
        self.aggregate = []
        cat_one = 0
        for label in self.class_label:  
            self.features=dict()
            for feature in self.columns:
                if feature not in self.categorical:
                    for index in range(len(self.x)):
                        if self.y.iloc[index]==label:
                            self.aggregate.append(self.x[feature][index])
                    #### metrics for the Gaussian Naive Bayes (asuming normal distribution // continuous variable)
                    self.count= len(self.aggregate)
                    self.mean = np.mean(self.aggregate)
                    self.stdv = np.std(self.aggregate)
                    self.summary =(self.mean,self.stdv,self.count)
                    self.features[feature]=self.summary
                    self.count=self.mean=self.stdv=0
                    self.aggregate.clear()
                  
                else: 
                    #### metrics for the Bernoulli probability
                    for index in range(len(self.x)):
                        if self.y.iloc[index]==label:
                            self.aggregate.append(self.x[feature][index])
                            if self.x[feature][index] == 1: # count the numbers of categorical var = 1
                                cat_one +=1
                                                      
                    self.count = len(self.aggregate)
                    self.pro_feature = cat_one / self.count
                    self.summary=(self.pro_feature)
                    self.features[feature]=self.summary
                    self.aggregate.clear()
                    cat_one = self.count = 0
            self.summary_feature[label]=self.features
        return
    
    # predicted probability
    def predict(self,x_test):
        from math import sqrt
        from math import exp
        from math import pi
        # reindex a data base, when it comes from SKFold, etc.
        x_test.reset_index(drop=True, inplace = True)
        self.x_test=x_test
        self.prob_class = dict()
        for label in self.class_label:
            self.x_pro =self.summary_feature[label]["Account Length"][2]/len(self.y) #use any feature to get the len from Dict
            self.prob_class[label]=self.x_pro
        
        self.columns = self.x_test.columns
        self.prob_test=dict()
        self.predict_prob=dict()
        self.predict1=[]
        self.highest_p=[]
        self.predict_proba1=[]
        self.pro=1
        for index in range(len(self.x_test)):
            self.total =0
            for label in self.class_label:
                for feature in self.columns:
                    #### Gaussian propability
                    if feature not in self.categorical:
                        self.x_p = self.x_test[feature][index]
                        self.mean_test = self.summary_feature[label][feature][0]
                        self.stdev_test = self.summary_feature[label][feature][1]
                        exponent = exp(-((self.x_p - self.mean_test)**2 / (2 * self.stdev_test**2 )))
                        self.normal_p = (1 / (sqrt(2 * pi) * self.stdev_test)) * exponent
                        self.pro *= self.normal_p
                        
                    else:
                        #### Bernoulli probability
                        self.x_p = self.x_test[feature][index]
                        proba = self.summary_feature[label][feature]
                        self.bernoulli_p =  proba * self.x_p + (1 - proba)*(1 - self.x_p)
                        self.pro *= self.bernoulli_p
                #### Naive Bayes probability        
                self.prob_test[label] = self.prob_class[label] * self.pro # P(feature1, feature2, .. featureM | y= class[label])
                self.total += self.prob_test[label] #  P(feature1, feature2, ..., featureM)
                self.pro = 1
            pre_prob =[]
            for i in range(len(self.class_label)):
                self.predict_prob[i]= self.prob_test[i] / self.total # p(y=class[label]|feature1,feature2,...,featureM)
                pre_prob.append(self.predict_prob.get(i))
            self.predict_proba1.append(pre_prob)
                
            # predict class label based on highest probability
            highest_prob = max(self.predict_prob.values())
            self.highest_p.append(highest_prob)
            for key,value in self.predict_prob.items():
                if value == highest_prob:
                    self.predict1.append(key)
        return np.array(self.predict1)
    
    def predict_proba(self,x_test):
        self.predict(x_test)
        return np.array(self.predict_proba1)    
            
        
        


# Class that resamples a data base with an unbalance binary class / labels    
class ClassResample:
    def __init__(self,x,y,minority_class,mayority_class):
        self.x = x
        self.y = y
        self.minority_class = minority_class
        self.mayority_class = mayority_class
        
    def resampling(self):   
        from sklearn.utils import resample
        print("Number of class 1 examples before:", self.x[self.y==self.minority_class].shape[0])
        print("Number of class 0 examples before:", self.x[self.y==self.mayority_class].shape[0])
        self.x_upsampled, self.y_upsampled = resample(self.x[self.y==self.minority_class],
                                                      self.y[self.y==self.minority_class],
                                                      replace=True,
                                                      n_samples = self.x[self.y==self.mayority_class].shape[0],
                                                      random_state = 222 )
        print("Number of class 1 examples after:",self.x_upsampled.shape[0])
        self.x_bal = np.vstack((self.x[self.y==self.mayority_class],self.x_upsampled))
        self.y_bal = np.hstack((self.y[self.y==self.mayority_class],self.y_upsampled))
        print("=======")
        print("The new x_bal shape is:", self.x_bal.shape)
        print("The new y_bal shape is:", self.y_bal.shape)
    
    
    
# In[ ]:

      
        
# In[ ]:





# In[ ]:




