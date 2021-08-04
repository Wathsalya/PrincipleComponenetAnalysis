import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

mat = sio.loadmat('PCA.mat')#load the mat file where X= 3+1+8+1= 13
#print(mat)#for key in mat:print(key)

# Extracting specific keys from dictionary- extract the ['DataMatrix']
res = dict((k, mat[k]) for k in ['DataMatrix']
                                        if k in mat)
print("The filtered dictionary is : " + str(res))
p=str(res)
l=res.values()
print(res.values())
data = list(res.values())
an_array = np.array(data)
#print(an_array.shape)
#print(type(an_array)
print(an_array)


# Creating a Dataframe with the targeted dataset
A=res['DataMatrix']
print('Type of targeted dataset is :',type(A))
lables= [_ for _ in 'ABCDEFGHIJKLMNOPQRSTUVWXYZab'] # Labling to the columns
names = [_ for _ in range(41)]     #Labling the raws
df = pd.DataFrame(A, index=names, columns=lables)# Creating a pandas Dataframe using pandas
df.to_csv('df.csv', index=True, header=True, sep=' ')
print(df)
print(df.shape)



#Standardize the Data to preprocess
from sklearn.preprocessing import StandardScaler
features = ['A', 'B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['b']].values
# Standardizing the features
x = StandardScaler().fit_transform(x) #Preprocessing the Data


#Applying PCA to the extracted dataframe
from sklearn.decomposition import PCA
pca=PCA(n_components=3) #Creatingg 3 principal components
pca.fit(x)
X_pca=pca.transform(x)

# Creating a pandas Dataframe
principalDf = pd.DataFrame(data = X_pca , columns = ['principal component 1', 'principal component 2','principal component 3'])
#lst2 = [item[0] for item in X_pca]
print(type(principalDf))
print("shape of principalDF", principalDf.shape)

#Deteting and Removing Outliers in principal component by Interquartile Range(IQR) Method

#Removing outliers in 'principal component 1'
first_column = principalDf.iloc[:, 0] #get first column of `principalDfdf`
#sns.boxplot(data=principalDf,x=first_column)
#plt.show()

Q1=np.quantile(first_column,0.25)
Q3=np.quantile(first_column,0.75)
IQR=Q3-Q1
print('IOR of principal component 1 ', IQR)
v=1.5*IQR
Lower_limit = Q1 - v
Upper_limit = Q3 + v
print(Lower_limit, Upper_limit)

principalDf = principalDf[first_column< Upper_Whisker]
print("New shape of principalDf after Removing outliers in 'principal component 1' is ", principalDf.shape)


#Removing outliers in 'principal component 2'
second_column = principalDf.iloc[:, 1] #get second column of `principalDfdf`
#sns.boxplot(data=principalDf,x=second_column)
#plt.show()
Q1=np.quantile(second_column,0.25)
Q3=np.quantile(second_column,0.75)
IQR=Q3-Q1
print('IOR of principal component 2 ',IQR)
v=1.5*IQR
Lower_limit = Q1 - v
Upper_limit = Q3 + v
print(Lower_limit, Upper_limit)

principalDf = principalDf[second_column< Upper_limit]
print("New shape of principalDf after Removing outliers in 'principal component 2' is ", principalDf.shape)


#let's check the shape of Newdataset
print("Shape of Dimentionally reduced dataset is", principalDf.shape)

#Creating new dataframe after removing outliers with the taret value 'b'
finalDf = pd.concat([principalDf, df[['b']]], axis = 1)
#print(finalDf)

#Finding the varience ratio of each of principal components
ex_variance=np.var(X_pca,axis=0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
print('Variance ratio of each of principal components are',ex_variance_ratio)
print('Sum of Variance ratio of PCA1 and PCA2 is',ex_variance_ratio[0]+ex_variance_ratio[1])


#Visualize 2D Projection of PCA with 'principal component 1' and 'principal component 2'
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('PCA with 2 principal Components ', fontsize = 20)
bs = [1, 2, 3, 4]
colors = ['red', 'green', 'blue', 'black']
for b, color in zip(bs,colors):
    indicesToKeep = finalDf['b'] == b
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(bs)
ax.grid()
fig.savefig("PCA-without outlier.png")
fig.show("PCA-without outlier.png")
plt.show()













