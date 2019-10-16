import statistics
import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor

non_na_indices=[]
actual_indices=[]


#Function used earlier for processing data. Got decent rmse for ridge regression. Upon changing the technique got better results by
#keeping values and using random forests
def process_data(df):
    print(df.shape)
    subset_columns = ['Gender','Size of City','University Degree','Hair Color']
    for column in subset_columns:
        #print(column)
        column_values = df[column]
        new_column_values = []
        if column == 'Gender':
            for gender in column_values:
                if pd.isna(gender):
                    new_column_values.append('unknown')
                # if gender=='unknown' or gender=='0':
                #     new_column_values.append(np.nan)
                else:
                    new_column_values.append(gender)

    #     elif column=='Size of City':
    #         for size in column_values:
    #             if size > 2852057:
    #                 new_column_values.append(np.nan)
    #             else:
    #                 new_column_values.append(size)
    #
    #     elif column=='University Degree':
    #         for degree in column_values:
    #             if degree=='0':
    #                 new_column_values.append(np.nan)
    #             elif degree =='No':
    #                 new_column_values.append('no_degree')
    #             else:
    #                 new_column_values.append(degree)
    #
    #     elif column=='Hair Color':
    #         for color in column_values:
    #             if color=='0' or color=='unknown':
    #                 new_column_values.append(np.nan)
    #             else:
    #                 new_column_values.append(color)
    #
        df[column]= new_column_values
    #
    # df.dropna(how='any', inplace=True)
    print(df.shape)
    return df

#Function used earlier for processing data. Got decent rmse for ridge regression. Upon changing the technique got better results by
#keeping values and using random forests

def process_data_test(df):
    #print(df.columns.values)
    subset_columns = ['Gender','Size of City','University Degree','Hair Color']
    for column in subset_columns:
        #print(column)
        column_values = df[column]
        new_column_values = []
        if column == 'Gender':
            for gender in column_values:
                if gender=='unknown' or gender=='0':
                    new_column_values.append(np.nan)
                else:
                    new_column_values.append(gender)

        elif column=='Size of City':
            for size in column_values:
                if size > 2838836:
                    new_column_values.append(np.nan)
                else:
                    new_column_values.append(size)

        elif column=='University Degree':
            for degree in column_values:
                if degree=='0':
                    new_column_values.append(np.nan)
                elif degree =='No':
                    new_column_values.append('no_degree')
                else:
                    new_column_values.append(degree)

        elif column=='Hair Color':
            for color in column_values:
                if color=='0' or color=='unknown':
                    new_column_values.append(np.nan)
                else:
                    new_column_values.append(color)

        df[column]= new_column_values

    global actual_indices
    actual_indices= df['Instance']
    df.dropna(how='any', inplace=True)
    global non_na_indices
    non_na_indices = df['Instance']
    return df



#Target econder function. Calculates the means for all values of categorical variable against the target variable
def target_encoder(data_frame, column, target_variable, suitable_weight=50):
    mean_of_dataframe = statistics.mean(data_frame[target_variable])
    grouped= data_frame.groupby(column)[target_variable]
    x= grouped.agg(['count', 'mean'])
    resultant_counts = x['count']
    resultant_means = x['mean']
    smoothing = ((resultant_counts * resultant_means) + (suitable_weight * mean_of_dataframe)) / (resultant_counts + suitable_weight)
    result = data_frame[column].map(smoothing)
    return result


os.chdir('C:/Users/DELL PC/Desktop/ALL IN ONE/IRELAND - PERSONAL/STUDY DOCS/MACHINE LEARNING/COMPETETION 1/tcdml1920-income-ind (1)')
training_data = pd.read_csv("tcd ml 2019-20 income prediction training (with labels).csv")

training_data = pd.DataFrame(training_data)
training_data = training_data[['Year of Record','Gender','Age','Country','Size of City','Profession','University Degree','Hair Color','Body Height [cm]','Income in EUR']]

#Imputing the missing values accordingly
training_data['Year of Record']= training_data['Year of Record'].fillna(statistics.mean(training_data['Year of Record']))
training_data['Gender']= training_data['Gender'].replace(pd.np.nan,'unknown')
training_data['Gender']= training_data['Gender'].replace('0','unknown')
training_data['Age'] = training_data['Age'].fillna(statistics.mean(training_data['Age']))
training_data['Size of City']= training_data['Size of City'].fillna(statistics.mean(training_data['Size of City']))
training_data['University Degree'] = training_data['University Degree'].replace(pd.np.nan, '0')
training_data['Hair Color'] = training_data['Hair Color'].replace(pd.np.nan, 'Unknown')

training_data.dropna(how='any', inplace=True)


#Using an additional column 'label' to separate the training and testing data
training_label = []
for x in range(len(training_data['Gender'])):
    training_label.append('train')
training_data['label']= training_label

test_data = pd.read_csv("tcd ml 2019-20 income prediction test (without labels).csv")
test_data = pd.DataFrame(test_data)
test_data=test_data.rename(columns={"Income":"Income in EUR"})
test_data = test_data[['Instance','Year of Record','Gender','Age','Country','Size of City','Profession','University Degree','Hair Color','Body Height [cm]']]

test_data['Gender']= test_data['Gender'].replace(pd.np.nan,'unknown')
test_data['Gender']= test_data['Gender'].replace('0','unknown')
test_data['Year of Record']= test_data['Year of Record'].fillna(statistics.mean(test_data['Year of Record']))
test_data['Age']= test_data['Age'].fillna(statistics.mean(test_data['Age']))
test_data['Size of City']= test_data['Size of City'].fillna(statistics.mean(test_data['Size of City']))
test_data['Body Height [cm]']= test_data['Body Height [cm]'].fillna(statistics.mean(test_data['Body Height [cm]']))
test_data['University Degree'] = test_data['University Degree'].replace(pd.np.nan, '0')
test_data['Hair Color'] = test_data['Hair Color'].replace(pd.np.nan, 'Unknown')


#Making two indices list. Should there be a case where the test data has any rows removed, these lists can help impute the income
#with the mean predicted income
actual_indices= test_data['Instance']
test_data.dropna(how='any', inplace=True)
non_na_indices = test_data['Instance']

income_na_column=[]
test_data = test_data[['Year of Record','Gender','Age','Country','Size of City','Profession','University Degree','Hair Color','Body Height [cm]']]
testing_label=[]

#Using the additional column 'label'. For incomes, just imputed zeros values over here for facilitation in target encoding
for x in range(len(test_data['Gender'])):
    testing_label.append('test')
    income_na_column.append(0)
test_data['Income in EUR']= income_na_column
test_data['label']= testing_label

#Combining the training and testing data
final_data = training_data.append(test_data)


#Creating dummy variable for columns over the whole data. This is done to avoid any clashes were the two datasets were separate.
#The label column is used to separate the whole dataset later
dummies_gender = pd.get_dummies(final_data['Gender'])

#Used dummies for country and profession earlier, but got better results on target encoding these in the end.
#dummies_country = pd.get_dummies(final_data['Country'])
#dummies_profession = pd.get_dummies(final_data['Profession'])
dummies_degree = pd.get_dummies(final_data['University Degree'])
dummies_hair_color = pd.get_dummies(final_data['Hair Color'])





#Combining the dummies and the normal columns
final_data = pd.concat([final_data['label'],final_data['Year of Record'],dummies_gender, final_data['Age'], final_data['Country'], final_data['Size of City'], final_data['Profession'], dummies_degree, dummies_hair_color, final_data['Body Height [cm]'], final_data['Income in EUR']], axis=1)


#Target encoded the profession and country variables since they were of relatively high cardinality. Gave more weight to profession.
final_data['Profession']= target_encoder(final_data, 'Profession', 'Income in EUR', 55)
final_data['Country']= target_encoder(final_data, 'Country', 'Income in EUR', 40)


print(final_data.shape[0])
print(final_data.shape[1])




#Following commented code to test the results on local machine. Not used finally

# final_training_data= final_data[final_data['label']=='train']
# print(final_training_data.shape)
# X_train, X_test, y_train, y_test = train_test_split(final_training_data.iloc[:,1:final_training_data.shape[1]-2], final_training_data.iloc[:,final_training_data.shape[1]-1], test_size=0.9)
#
#
#
# regr= RandomForestRegressor(n_estimators=1000)
#
#
# regr.fit(X_train, y_train)
# y_pred= regr.predict(X_test)
# print(math.sqrt(mean_squared_error(y_pred,y_test)))


X_train= final_data[final_data['label']=='train'].iloc[:,1:final_data.shape[1]-2]
Y_train=final_data[final_data['label']=='train'].iloc[:,final_data.shape[1]-1]
X_test= final_data[final_data['label']=='test'].iloc[:,1:final_data.shape[1]-2]
regr = RandomForestRegressor(n_estimators=1000)
regr.fit(X_train, Y_train)
Y_pred=regr.predict(X_test)
print(Y_pred)

counter=0
final_Y_Pred=[]
mean_pred = statistics.mean(Y_pred)
print(mean_pred)



#Code to impute any missing income values with mean predicted income.
actual_indices= list(actual_indices)
non_na_indices= list(non_na_indices)
for instance in actual_indices:
    if instance in non_na_indices:
        final_Y_Pred.append(Y_pred[counter])
        counter = counter + 1
    else:
        final_Y_Pred.append(mean_pred)
final_Y_Pred=pd.Series(final_Y_Pred)
actual_indices= pd.Series(actual_indices)
df = pd.concat([actual_indices,final_Y_Pred], axis=1)
df.to_csv(r'predicted_incomes.csv')






