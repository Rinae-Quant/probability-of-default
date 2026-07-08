import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data = pd.read_csv(r"C:\Users\rinae\Music\PROJECTS\Credit Analysis_ Probability Of Default\Code and Excel data file\loan_data.csv")

print(data.head())
print(data.shape)                      #print out the size of the data(number of rows,number of rows)
print(data.columns)
print(data.info())                     #checks the kind of a data that is there in the columns


#this one is important for data analysis since it checks for missing values
print(data.isnull().sum())             #In our case we have found 895 nulls in history lenth and 3116 in loan interest rate

print(data.info()) 


#since we had like the missing numbers we are filling in the missing values with the medians
data["person_emp_length"] = data["person_emp_length"].fillna(data["person_emp_length"].median())
data["loan_int_rate"] = data["loan_int_rate"].fillna(data["loan_int_rate"].median())

#verification that indeed it has changes
print(data.isnull().sum())


#checking for unrealistic numbers where i check the maximum values and the minimum values
print(data.describe())


#check the duplicates
print(data.duplicated().sum())
print(data[data.duplicated()].head())


duplicate_rows = data[data.duplicated(keep=False)]                #show both their original and their duplicates
print(duplicate_rows.sort_values(by=list(data.columns)).head(10)) #sort them to keep them close to each other


data = data.drop_duplicates()

print(data.duplicated().sum())                                     #it should be zero


print(data["loan_status"].value_counts())

#they should be 100
print(data["loan_status"].value_counts(normalize=True) * 100)

#check for reasonable values

#we assume employment cannot exceed (age - 15).
data = data[data["person_emp_length"] <= data["person_age"] - 15]

#first we need to find how many year are greater than 100
print(data[data["person_age"] > 100].shape[0])                     #since it is only 5 of them we can remove them 
# Remove applicants older than 100
data = data[data["person_age"] <= 100]


#now we are done with the data cleaning
#Exploratory Data Analysis part
data["person_age"].hist()
plt.title("Person Age Distribution")
plt.xlabel("Person Age")
plt.ylabel("Number of Applicants")
# Display the graph and you can see from the graph that the age that people have loan are in their twenties
plt.show()  

data["loan_amnt"].hist()
plt.title("Loan Amount Distribution") #highest loans in their 5000-10000
plt.xlabel("Loan Amount")
plt.ylabel("Number of Applicants")
plt.show()
data["loan_int_rate"].hist()

plt.title("Interest Rate Distribution")  #where as here most people are around 12.5%
plt.xlabel("Interest Rate")
plt.ylabel("Number of Applicants")
plt.show()

# exploring the categorical data
print(data["loan_grade"].value_counts())
print(data.shape)
print(data["person_home_ownership"].value_counts())
print(data["loan_intent"].value_counts())
print(data["cb_person_default_on_file"].value_counts())
print(data.head)


#categorical data 
data["loan_grade"].value_counts().plot(kind="bar")
plt.title("Distribution of Loan Grades")
plt.xlabel("Loan Grade")
plt.ylabel("Number of Applicants")
plt.show()

data["person_home_ownership"].value_counts().plot(kind="bar")
plt.title("Distribution of Home Ownership")
plt.xlabel("Home Ownership")
plt.ylabel("Number of Applicants")
plt.show()
data["loan_intent"].value_counts().plot(kind="bar")
plt.title("Distribution of Loan Inten")
plt.xlabel("Loan intent")
plt.ylabel("Number of Applicants")
plt.show()
data["cb_person_default_on_file"].value_counts().plot(kind="bar")
plt.title("Distribution of person_default_on_file")
plt.xlabel("cb_person_default_on_file")
plt.ylabel("Number of Applicants")
plt.show()

#The correlation between the variables
print(pd.crosstab(data["loan_grade"], data["loan_status"]))
print(pd.crosstab(data["person_home_ownership"], data["loan_status"]))

print(pd.crosstab(data["loan_intent"], data["loan_status"]))

print(pd.crosstab(data["cb_person_default_on_file"], data["loan_status"]))

#CORRELATION
correlation = data.corr(numeric_only=True)

print(correlation)
#changing of the categorical values to numerical inputs
######################################################
# MACHINE LEARNING PREPROCESSING
######################################################



#seaparate the features and the data(feature is the information we use to predict whether someone will default)
x = data.drop("loan_status", axis=1)

y = data["loan_status"]

#remember why y is there on the dataset it is because we are using previous information

#since the language cannot understand the categorical variables we need to encode them
#convert them into dummies 001,010

#Convert categorical variables into numerical variables
x = pd.get_dummies(x, drop_first=True) 
print(x.head())


print(x.info())



#separating the data randomly
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.20, random_state=42)
#checking if it worked
print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)



#after spliting we scale note that logaritmic functon uses log function
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)





############################################
#   logistic model 
############################################

from sklearn.linear_model import LogisticRegression  # add the logistic regression

#create the model and this model hasnt seen any dat,havent leant anything,it isnt ready to make predictions
model = LogisticRegression(random_state=42)

random_state=42
#train the model


model.fit(X_train_scaled, y_train)
#first type :this one gives you the 0 not default or defaulted
y_pred = model.predict(X_test_scaled)
print(y_pred[:10])


#the second type which is to predict the probability
y_prob = model.predict_proba(X_test_scaled)# Keep only the probability of default (loan_status = 1)
pd_probability = y_prob[:, 1]

print(pd_probability[:10])

#now we go to the valuation how well the model works comparing it with the actual truth
#how well the model works 
from sklearn.metrics import (  confusion_matrix,accuracy_score,precision_score, recall_score, f1_score, classification_report,roc_auc_score)
cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

precision = precision_score(y_test, y_pred)

print("Precision:", precision)

recall = recall_score(y_test, y_pred)

print("Recall:", recall)

f1 = f1_score(y_test, y_pred)

print("F1 Score:", f1)

print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, pd_probability)

print("ROC-AUC:", auc)


###########################################


#end of the LOGISTIC REGRESSION MODEL     


###########################################

# =====================================
# SAVE THE TRAINED MODEL
# =====================================

import os
import joblib


project_folder = os.path.dirname(os.path.abspath(__file__))

# Save everything into my folder
joblib.dump(model, os.path.join(project_folder, "credit_model.pkl"))
joblib.dump(scaler, os.path.join(project_folder, "scaler.pkl"))
joblib.dump(X_train.columns, os.path.join(project_folder, "model_columns.pkl"))

print("Files saved successfully!")
print("Saved to:", project_folder)