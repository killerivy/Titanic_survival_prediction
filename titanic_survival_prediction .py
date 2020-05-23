#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib','inline')
sns.set(style='white',context='notebook',palette='deep')
import warnings
warnings.filterwarnings('ignore')

#csv files

train= pd.read_csv("/home/killerivy/Downloads/train.csv")
test= pd.read_csv("/home/killerivy/Downloads/test.csv")
IDtest = test["PassengerId"]

#train.head()
#train.info()
#test.info()


# In[2]:


#dealing with missing values in train dataset
train_na=(train.isnull().sum()/len(train))* 100
train_na=train_na.drop(train_na[train_na==0].index).sort_values(ascending=False)[0:30]
miss_train = pd.DataFrame({'Train Missing Ratio': train_na})
miss_train


# In[3]:


#dealing with missing values in test dataset
test_na=(test.isnull().sum()/len(test))* 100
test_na=test_na.drop(test_na[test_na==0].index).sort_values(ascending=False)[0:30]
miss_test = pd.DataFrame({'Test Missing Ratio': test_na})
miss_test


# In[4]:


#fill emptys with NaN
train=(train.fillna(np.nan))
test=(test.fillna(np.nan))


# In[5]:


#Analyze the count of variable by Pclass
ax = sns.countplot(x='Pclass',hue='Survived',data=train)
train[['Pclass','Survived']].groupby(['Pclass']).count().sort_values(by='Survived',ascending=False)


# In[6]:


#survivor prpbability by Pclass
g = sns.barplot(x='Pclass',y="Survived",data=train,ci=None)
#h = sns.barplot(x='Pclass',y="Survived",data=train)
#g=g.set_ylabel("Survival Probability")
train[['Pclass','Survived']].groupby(['Pclass']).mean().sort_values(by='Survived',ascending=False)


# In[7]:


#count number of passengers by gender
ax = sns.countplot(x='Sex',hue="Survived",data=train)
train[['Sex','Survived']].groupby(['Sex']).count().sort_values(by='Survived',ascending=False)


# In[8]:



#train[['Sex','Survived']].groupby(['Sex']).head()
g= sns.barplot(x="Sex",y="Survived",data=train,ci=None)
g=g.set_ylabel("Survival Probability")
train[['Sex','Survived']].groupby(['Sex']).mean().sort_values(by='Survived',ascending=False)


# In[9]:


#AGE
fig=plt.figure(figsize=(10,8),)
axis= sns.kdeplot(train.loc[(train['Survived']==1),'Age'],color='g',shade=True,label="Survived")
axis= sns.kdeplot(train.loc[(train['Survived']==0),'Age'],color='b',shade=True,label="Did not Survived")
plt.xlabel("Passenger age")
plt.ylabel("Frequency")


# In[10]:


sns.lmplot('Age','Survived',data=train)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   


# In[11]:


#analyzing count by siblings and parents
ax= sns.countplot(x='SibSp',hue='Survived',data=train)
train[['SibSp','Survived']].groupby(['SibSp']).count().sort_values(by='Survived',ascending=False)


# In[12]:


#analyze probability of survival by siblings and parents
g= sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar",ci=None,size=7)
g.despine(left=True)
g=g.set_ylabels("Survival probability")
train[['SibSp','Survived']].groupby(['SibSp']).mean().sort_values(by='Survived',ascending =False)


# In[13]:


#AnALYZE the count survivors by parch   parent childern
ax=sns.countplot(x='Parch',hue="Survived",data=train)
train[['Parch','Survived']].groupby(['Parch']).count().sort_values(by='Survived',ascending=False)


# In[14]:


#analyze probabilty by Parch parent children
g=sns.factorplot(x="Parch",y="Survived",data=train,kind="bar",ci=None,size=7,palette="muted")
g.despine(left=True)
g= g.set_ylabels("Survival probaility")
train[['Parch',"Survived"]].groupby(['Parch']).mean().sort_values(by='Survived',ascending=False)


# In[15]:


#analyze survival rate survial by Embarkation
ax=sns.countplot(x='Embarked',hue="Survived",data=train)
train[['Embarked',"Survived"]].groupby(["Embarked"]).count().sort_values(by="Survived",ascending=False)


# In[16]:


#analyze probability by port of Embarkation


# In[17]:


g=sns.factorplot(x="Embarked",y="Survived",data=train,kind='bar',size=7,ci=None,palette='muted')
g.despine(left=True)
g=g.set_ylabels("Survival Probability")
train[["Embarked","Survived"]].groupby(['Embarked']).mean().sort_values(by="Survived",ascending=False)


# In[18]:


#Relatio amaong Pclass GEnder and survial rate
g= sns.catplot(x='Sex',y='Survived',col='Pclass',data=train,kind='bar',ci=None,aspect=.6)


# In[19]:


#Relatio amaong SibSp GEnder and survial rate
g= sns.catplot(x='Sex',y='Survived',col='SibSp',data=train,kind='bar',ci=None,aspect=.6)


# In[20]:


#Relatio amaong Parch GEnder and survial rate
g= sns.catplot(x='Sex',y='Survived',col='Parch',data=train,kind='bar',ci=None,aspect=.6)


# In[21]:


""""What we need to do to process following variables 

PassengerID - No action required

PClass - Have only 3 numerical values. We will use it as it is.

Name - Can be used to create new variable Title by extracting the salutation from name.

Sex - Create dummy variables

Age - Missing value treatment, followed by creating dummy variables

SibSP - Drop the variable

Parch - Drop the variable as most of the values are 0

Ticket - Create dummy variables post feature engineering

Fare - Missing value treatment followed by log normalization

Cabin - Create dummy variables post feature engineering

Embarked - Create dummy variables"""


# In[22]:


#combining train set and test set
train['source']='train'
test['source']='test'
combdata=pd.concat([train,test],ignore_index=True)
combdata


# In[23]:


#as we dont need passenger id
#combdata.drop(labels=["PassengerId"],inplace=True)
#extracting unique values in Pclass
combdata['Pclass'].unique()
#Extracting salutations from name variables
salutation= [i.split(",")[1].split(".")[0].strip() for i in combdata["Name"]]
combdata["Title"]= pd.Series(salutation)
combdata["Title"].value_counts()


# In[24]:


# PassengerID - Drop PassengerID
combdata.drop(labels = ["PassengerId"], axis = 1, inplace = True)


# In[25]:


combdata


# In[26]:


combdata['Title']= combdata['Title'].replace('Mlle','Miss')
combdata['Title']= combdata['Title'].replace(['Mme','Lady','Ms'],'Mrs')
combdata.Title.loc[(combdata.Title!= 'Master') & (combdata.Title != 'Mr')&(combdata.Title !='Miss')&(combdata.Title!="Mrs")]="Others"
combdata["Title"].value_counts()


# In[27]:


#finding probability of above
combdata[['Title','Survived']].groupby(["Title"]).mean()


# In[28]:


combdata.head()


# In[29]:


# Create dummy variable 
combdata = pd.get_dummies(combdata, columns = ["Title"])


# In[30]:


#checking Fare
#missing values
combdata["Fare"].isnull().sum()


# In[31]:


#only one value missing so we will fill the same with median
combdata["Fare"] = combdata["Fare"].fillna(combdata["Fare"].median())
combdata['Fare']


# In[32]:


sns.distplot(combdata['Fare'])


# In[33]:


combdata['Fare-bin'] = pd.qcut(combdata.Fare,5,labels=[1,2,3,4,5]).astype(int)


# In[34]:


combdata[['Fare-bin','Survived']].groupby(['Fare-bin'],as_index=False).mean()


# In[35]:


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
combdata_temp = combdata[['Age','Title_Master','Title_Miss','Title_Mr','Title_Mrs','Title_Others','Fare-bin','SibSp']]

X  = combdata_temp.dropna().drop('Age', axis=1)
Y  = combdata['Age'].dropna()
holdout = combdata_temp.loc[np.isnan(combdata.Age)].drop('Age', axis=1)

regressor = RandomForestRegressor(n_estimators = 300)
#regressor = GradientBoostingRegressor(n_estimators = 500)
regressor.fit(X, Y)
y_pred = np.round(regressor.predict(holdout),1)
combdata.Age.loc[combdata.Age.isnull()] = y_pred

combdata.Age.isnull().sum(axis=0) 
bins = [ 0, 4, 12, 18, 30, 50, 65, 100] # This is somewhat arbitrary...
age_index = (1,2,3,4,5,6,7)

combdata['Age-bin'] = pd.cut(combdata.Age, bins, labels=age_index).astype(int)
combdata[['Age-bin', 'Survived']].groupby(['Age-bin'],as_index=False).mean()


# In[36]:


combdata['Sex']= combdata['Sex'].map({"male":0,"female":1})


# In[37]:


#creating family size using Parch and SibSp
combdata['Fsize']= combdata['SibSp']+combdata['Parch']+1
#add 1 for self
#analize probability
combdata[['Fsize','Survived']].groupby(['Fsize'],as_index=False).mean()


# In[38]:


#it is unlikely for families with more than 4 members to survive 
combdata.Fsize = combdata.Fsize.map(lambda x: 0 if x> 4 else x)
g=sns.factorplot(x='Fsize',y='Survived',data=combdata,kind='bar',size=7)
g.despine(left=True)
g = g.set_ylabels("Survival Probability")
g = g.set_xlabels("Family Size")
combdata[['Fsize', 'Survived']].groupby(['Fsize']).mean().sort_values(by='Survived',ascending=False)


# In[39]:


# SibSp - Drop the variable
combdata = combdata.drop(labels='SibSp', axis=1)
# Parch - Drop the variable
combdata = combdata.drop(labels='Parch', axis=1)


# In[40]:


''''Tickets are of 2 types here.

Type 1 has only number and Type 2 is a combination of some code followed by the number. Let's extract the first digit and compare it with survival probability.'''
combdata.Ticket= combdata.Ticket.map(lambda x: x[0])
combdata[['Ticket','Survived']].groupby(['Ticket'],as_index=False).mean()


# In[41]:


#number of people for each type of tickets
combdata['Ticket'].value_counts()
#Most of these tickets belong to category 1, 2, 3, S, P, C.


# In[42]:


combdata.Ticket=combdata.Ticket.replace(['A','W','F','L','5','6','7','8','9'], '4')
combdata[['Ticket','Survived']].groupby(['Ticket'],as_index=False).mean()


# In[43]:


# Create dummy variables
combdata = pd.get_dummies(combdata, columns = ["Ticket"], prefix="T")
combdata.head()


# In[44]:


#replace missing cabin number by Unkown Ua

combdata["Cabin"]= pd.Series([i[0] if not pd.isnull(i) else 'U' for i in combdata['Cabin']])


# In[45]:


g=sns.factorplot(x="Cabin",y="Survived",data=combdata,kind="bar",ci=None,size=7,order=['A','B','C','D','E','F','G','T','U'])
g.despine(left=True)
g=g.set_ylabels("survival Probability")


# In[46]:


combdata = combdata.drop(labels="Cabin",axis=1)
combdata = combdata.drop(labels="Embarked",axis=1)
combdata = combdata.drop(labels=['Age','Fare','Name'],axis=1)


# In[47]:


###CREATING MODEL


# In[48]:


from sklearn.svm import SVC
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,learning_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,ExtraTreesClassifier, VotingClassifier


# In[49]:


##Separate train dataset and test data set using the index variable 'Source'

train= combdata.loc[combdata['source']=="train"]
test= combdata.loc[combdata['source']=="test"]
test.drop(labels=['Survived'],axis=1,inplace=True)
test.drop(labels=['source'],axis=1,inplace=True)
train.drop(labels=['source'],axis=1,inplace=True)
test


# In[50]:


test


# In[51]:


train['Survived'] = train['Survived'].astype(int)
y_train= train["Survived"]
x_train= train.drop(labels=["Survived"],axis=1)


# In[52]:


x_train


# In[53]:


# Cross validate model with Kfold stratified cross val
kfold= StratifiedKFold(n_splits=10)


# In[54]:


kfold


# In[55]:


classifiers = []
classifiers.append(KNeighborsClassifier())
classifiers.append(LinearDiscriminantAnalysis())
classifiers.append(SVC(random_state=2))
classifiers.append(MLPClassifier(random_state=2))
classifiers.append(ExtraTreesClassifier(random_state=2))
classifiers.append(LogisticRegression(random_state=2))
classifiers.append(DecisionTreeClassifier(random_state=2))
classifiers.append(RandomForestClassifier(random_state=2))
classifiers.append(GradientBoostingClassifier(random_state=2))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=2),random_state=2,learning_rate=0.1))


# In[56]:


cv_results=[]
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier,x_train,y= y_train,scoring="accuracy",cv=kfold,n_jobs=4))

cv_means=[]
cv_std=[]
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res=pd.DataFrame({"CrossValMeans":cv_means,"Crossvalerrors":cv_std,
                     "Algorithm":["SVC",
                                   "AdaBoost",
                                    "ExtraTrees",
                                    "KNeighboors",
                                    "DecisionTree",
                                    "RandomForest",
                                    "GradientBoosting",
                                    "LogisticRegression",
                                    "MultipleLayerPerceptron",
                                    "LinearDiscriminantAnalysis"]})
g=sns.barplot("CrossValMeans","Algorithm",data=cv_res,ci=None,**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g=g.set_title("Cross Validation scores")
cv_res

cv_res.groupby("CrossValMeans")
cv_res
# In[57]:



cv_res.sort_values(by=['CrossValMeans','Crossvalerrors'],ascending=False)


# In[58]:


#adaBoost
DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
                  "base_estimator__splitter" :   ["best", "random"],
                  "algorithm": ["SAMME","SAMME.R"],
                  "n_estimators" :[1,2],
                  "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsadaDTC.fit(x_train,y_train)
ada_best = gsadaDTC.best_estimator_
gsadaDTC.best_score_


# In[59]:


#ExtraTrees 
ExtC = ExtraTreesClassifier()

## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}

gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsExtC.fit(x_train,y_train)
ExtC_best = gsExtC.best_estimator_

# Best score
gsExtC.best_score_


# In[60]:


# RFC Parameters tunning 
RFC = RandomForestClassifier()

## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}
gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsRFC.fit(x_train,y_train)
RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_


# In[61]:


# Gradient boosting 
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }
gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsGBC.fit(x_train,y_train)
GBC_best = gsGBC.best_estimator_

# Best score
gsGBC.best_score_


# In[62]:


### SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}
gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsSVMC.fit(x_train,y_train)
SVMC_best = gsSVMC.best_estimator_

# Best score
gsSVMC.best_score_


# In[63]:


# Feature importance
nrows = ncols = 2
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,15))

names_classifiers = [("AdaBoosting", ada_best),("ExtraTrees",ExtC_best),
                     ("RandomForest",RFC_best),("GradientBoosting",GBC_best)]

nclassifier = 0
for row in range(nrows):
    for col in range(ncols):
        name = names_classifiers[nclassifier][0]
        classifier = names_classifiers[nclassifier][1]
        indices = np.argsort(classifier.feature_importances_)[::-1][:40]
        g = sns.barplot(y=x_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h',ax=axes[row][col])
        g.set_xlabel("Relative importance",fontsize=12)
        g.set_ylabel("Features",fontsize=12)
        g.tick_params(labelsize=9)
        g.set_title(name + " feature importance")
        nclassifier += 1


# In[64]:


# Concatenate all classifier results
test_Survived_RFC = pd.Series(RFC_best.predict(test), name="RFC")
test_Survived_ExtC = pd.Series(ExtC_best.predict(test), name="ExtC")
test_Survived_SVMC = pd.Series(SVMC_best.predict(test), name="SVC")
test_Survived_AdaC = pd.Series(ada_best.predict(test), name="Ada")
test_Survived_GBC = pd.Series(GBC_best.predict(test), name="GBC")

ensemble_results = pd.concat([test_Survived_RFC,test_Survived_ExtC,test_Survived_AdaC,test_Survived_GBC, test_Survived_SVMC],axis=1)
g= sns.heatmap(ensemble_results.corr(),annot=True)


# In[65]:


# Use voting classifier to combine the prediction power of all models
votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)
votingC = votingC.fit(x_train, y_train)


# In[69]:


# Predict and export the results
test_Survived = pd.Series(votingC.predict(test), name="Survived")
results = pd.concat([IDtest,test_Survived],axis=1)
results.to_csv("/home/killerivy/Downloads/Final Submission File.csv",index=False)


# In[ ]:





# In[ ]:





# In[ ]:




