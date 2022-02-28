import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os
data=pd.read_csv("HCV-Egy-Data.csv")
data.columns=['Age ', 'Gender', 'BMI', 'Fever', 'Nausea/Vomting','Headache ',
       'Diarrhea','Fatigue','Jaundice',
       'Epigastric_pain ', 'WBC','RBC','HGB', 'Plat','AST_1','ALT_1',
       'ALT_4', 'ALT_12','ALT_24','ALT_36','ALT 48','ALT_after_24w',
       'RNA_Base','RNA 4','RNA_12', 'RNA_EOT','RNA_EF',
       'Baseline_histological_Grading','Baselinehistological_staging']
data.columns = data.columns.str.strip()
# data.describe()
data_cat=data[['Gender','Fever','Nausea/Vomting','Headache','Fatigue','Jaundice','Diarrhea','Epigastric_pain',"Baselinehistological_staging"]]
data_cat=data_cat.astype('category')
data_cat['Gender'].replace([1,2],['Male','Female'],inplace=True)
Symptoms_cols=data_cat[["Fever","Nausea/Vomting","Headache","Fatigue","Jaundice","Diarrhea",'Epigastric_pain']]
Symptoms_cols_values = data_cat[["Fever","Nausea/Vomting","Headache","Fatigue","Jaundice","Diarrhea",'Epigastric_pain']].values
unique_values =  np.unique(Symptoms_cols_values)

data_cat['Fever'].replace([1,2],['Absent','Present'],inplace=True)
data_cat['Nausea/Vomting'].replace([1,2],['Absent','Present'],inplace=True)
data_cat['Headache'].replace([1,2],['Absent','Present'],inplace=True)
data_cat['Fatigue'].replace([1,2],['Absent','Present'],inplace=True)
data_cat['Jaundice'].replace([1,2],['Absent','Present'],inplace=True)
data_cat['Diarrhea'].replace([1,2],['Absent','Present'],inplace=True)
data_cat['Epigastric_pain'].replace([1,2],['Absent','Present'],inplace=True)
data_cat['Baselinehistological_staging'].replace([1,2,3,4],['Portal Fibrosis','Few Septa','Many Septa','Cirrhosis'],inplace=True)

#check Histological Grading and Staging across each Gender
sns.countplot(x=data['Baseline_histological_Grading'],hue=data_cat['Gender'],palette="Dark2")
plt.title("Gender Chart for Histological Grading")
plt.xlabel("Histological Grading")
plt.legend(bbox_to_anchor=(1,1))
plt.show()

#Histological Stages across each Gender
sns.countplot(x=data_cat['Baselinehistological_staging'],hue=data_cat['Gender'],palette="viridis")
plt.legend(bbox_to_anchor=(1,1))
plt.title("Gender Chart for Histological Staging")
plt.xlabel("Histological Staging")
plt.show()

#Age distrubtion across the dataset

sns.distplot(data.Age,bins=10,label="Age",color="green",rug=True)
plt.yticks([])
plt.title("Age Distribution")
plt.legend()
plt.show()

#AST vs Stages
plt.figure(figsize=(10,5))
sns.swarmplot(y=data['AST_1'],x=data_cat['Baselinehistological_staging'],hue=data_cat.Gender,palette="coolwarm")
plt.legend(bbox_to_anchor=(1,1))
plt.xlabel("Stages")
plt.ylabel("AST")
plt.title("The distribution of the AST Enzyme Across each Stage ")
plt.show()
# plt.savefig("ASTstages.png")



# ALT vs Stages:
plt.figure(figsize=(10,5))
sns.swarmplot(y=data['ALT_1'],x=data_cat['Baselinehistological_staging'],hue=data_cat.Gender,palette="crest")
plt.legend(bbox_to_anchor=(1,1))
plt.xlabel("Stages")
plt.ylabel("AST")
plt.title("The distribution of the ALT Enzyme Across each Stage ")
plt.show()


#AST VS HGB(Blood Hemoglobin level) across Gender:
plt.figure(figsize=(10,5))

sns.swarmplot(y=data['AST_1'],x=data['HGB'],hue=data_cat.Gender,palette="ocean")

plt.legend(bbox_to_anchor=(1,1))
plt.xlabel("HGB Level")
plt.ylabel("AST")
plt.title("The distribution of the AST Enzyme Across each HGB Level ")

plt.show()
# plt.savefig('asthgb.png')

#Dashboard
sns.set_style("darkgrid")
fig,axis=plt.subplots(4,2,figsize=(10,20))
k1=sns.countplot(x=data['Baseline_histological_Grading'],hue=data_cat['Gender'],palette="Dark2",ax=axis[0,0])
k2=sns.countplot(x=data_cat['Baselinehistological_staging'],hue=data_cat['Gender'],palette="viridis",ax=axis[0,1])
k3=sns.distplot(data['Age'],bins=10,label="Age",color="green",rug=True,ax=axis[1,0])
k4=sns.swarmplot(y=data['AST_1'],x=data_cat['Baselinehistological_staging'],hue=data_cat.Gender,palette="coolwarm",ax=axis[1,1])
k5=RBCASTplt=sns.kdeplot(data['RBC'],data['AST_1'],cmap="Reds",shade=True,shade_lowest=False,ax=axis[2,0])
k6=sns.swarmplot(y=data['ALT_after_24w'],x=data['HGB'],hue=data_cat.Gender,palette="ocean",ax=axis[2,1])
k7=HGBStagingPlt=sns.countplot(data['HGB'],hue=data_cat['Baselinehistological_staging'],palette="twilight",ax=axis[3,0])
k8=sns.distplot(data['Plat'],bins=8,axlabel="Platelets",rug=True,color='red',ax=axis[3,1])
plt.show()
