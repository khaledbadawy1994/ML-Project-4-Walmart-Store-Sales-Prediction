# ML-Project-4-Walmart-Store-Sales-Prediction

import numpy as np      # To use np.arrays
import pandas as pd     # To use dataframes
from pandas.plotting import autocorrelation_plot as auto_corr

# To plot
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib as mpl
import seaborn as sns

from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose as season
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.linear_model import LinearRegression

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/Walmart.csv')

#df.drop(['car name'], axis=1, inplace=True)
display(df.head())

original_df = df.copy(deep=True)

print('\n\033[1mInference:\033[0m The Datset consists of {} features & {} samples.'.format(df.shape[1], df.shape[0]))

df.head()

df.Date

import datetime as dt
str = '01,01,2017'
datetime_value = dt.datetime.strptime(str,'%d,%m,%Y')
print(datetime_value) # prints: 2017-01-01 00:00:00

# Reframing the columns

df['Date'] = dt.datetime.strptime(str,'%d,%m,%Y')

df['weekday'] = df.Date.dt.weekday
df['month'] = df.Date.dt.month
df['year'] = df.Date.dt.year

# df['Monthly_Quarter'] = df.month.map({1:'Q1',2:'Q1',3:'Q1',4:'Q2',5:'Q2',6:'Q2',7:'Q3',
#                                       8:'Q3',9:'Q3',10:'Q4',11:'Q4',12:'Q4'})

df.drop(['Date'], axis=1, inplace=True)#,'month'

target = 'Weekly_Sales'
features = [i for i in df.columns if i not in [target]]
original_df = df.copy(deep=True)

df.head()

#Checking the dtypes of all the columns

df.info()

#Checking number of unique rows in each feature

df.nunique().sort_values()

#Checking number of unique rows in each feature

nu = df[features].nunique().sort_values()
nf = []; cf = []; nnf = 0; ncf = 0; #numerical & categorical features

for i in range(df[features].shape[1]):
    if nu.values[i]<=45:cf.append(nu.index[i])
    else: nf.append(nu.index[i])

print('\n\033[1mInference:\033[0m The Datset has {} numerical & {} categorical features.'.format(len(nf),len(cf)))

#Checking the stats of all the columns

display(df.describe())

Inference: The stats seem to be fine, let us do further analysis on the Dataset

df.head() # last ready data set

df.shape

Store & Department Numbers

df['Store'].nunique() # number of different values

Now, I will look at the average weekly sales for each store to see if there is any weird values or not. There are 45 stores for stores.

store_dept_table = pd.pivot_table(df, index='Store',
                                  values='Weekly_Sales', aggfunc=np.mean)
display(store_dept_table)

df = df.loc[df['Weekly_Sales'] > 0]

df.shape # new data shape

#Check for columns

print(df.columns.to_list())
['Store', 'Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'weekday', 'month', 'year']

#Checking for top 10 largest Sales

df[['Store', 'Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price']].nlargest(10,'Weekly_Sales')

Store 14 has the most weekly_sales, while store 20,10,13,4 have are in top 10 twice

Convert the numerical columns to categorical

df['Holiday_Flag'] = pd.Categorical(df.Holiday_Flag)
df.Holiday_Flag.dtype

#Checking if there is Nulls
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    #print('{} - {}%'.format(col,pct_missing))
    print(col,pct_missing)

df['Holiday_Flag']

#Changeing Temperature from F to C
df['Temperature'] = df['Temperature'].apply(lambda x: round((x-32)*.5556,2))
df['Temperature']

#Check how many store this file contains
df['Store'].nunique()

#Select the row with biggest weekly sales
df[df['Weekly_Sales'] == df['Weekly_Sales'].max()]

df['Holiday_Flag'].unique()

df['Holiday_Flag'].value_counts()

df.head()

df.head()

# group the store number and year with the summation of the weekly sales and store it in new variable df1
df1 = df.groupby(['Store', 'year']).aggregate({
    'Weekly_Sales': 'sum'}).reset_index().sort_values(by = 'Weekly_Sales', ascending = False)

#print columns
df1.columns
Index(['Store', 'year', 'Weekly_Sales'], dtype='object')

#change name of the columns
new_cols_df1 = ['Store_Num', 'Year', 'Annual_Sales']

#update column
df1.columns = new_cols_df1

#sort data decendingly
df1.sort_values(by = ['Year', 'Annual_Sales'], ascending = False, inplace  = True)

df1.head()

df.describe().style.background_gradient(cmap = 'YlGnBu')

df.shape

for i in df.columns:
    print(f'{i}: {df[i].nunique()}')

numericalData =df.select_dtypes(include = ['float64', 'int64']).columns
print(f"\nNumerical Features:\n{numericalData}")

categoricalData = df.select_dtypes(include = ['object']).columns
print(f"\nCategorical Features:\n{categoricalData}")

#Treatment Of Outliers In Dataset df

#Checking Outliers

#checking outliers with the help of Boxplot

pno = 1
plt.figure(figsize=(15,10))
for i in ["Weekly_Sales","Temperature","Fuel_Price","CPI","Unemployment"]:
        if pno<=5:
            plt.subplot(3,2,pno);
            pno+=1
            sns.boxplot(df[i]);
            plt.xlabel(i);

we can see here that 3 columns "Weekly sale" , "Temperature" and "Unemployment has outliers.

Treating Outliers

#treating outliers with the help of upper whisker and lower whisker.

def outlier_treatment():
    l = ["Weekly_Sales","Temperature","Unemployment"]
    for i in l:
        x = np.quantile(df[i],[0.25,0.75])
        iqr = x[1]-x[0]
        uw = x[1]+1.5*iqr
        lw = x[0]-1.5*iqr
        df[i]  = np.where(df[i]>uw,uw,(np.where(df[i]<lw,lw,df[i])))

outlier_treatment()
Checking Outliers In features after treatment

pno = 1
plt.figure(figsize=(15,10))
for i in ["Weekly_Sales","Temperature","Fuel_Price","CPI","Unemployment"]:
        if pno<=5:
            plt.subplot(3,2,pno);
            pno+=1
            sns.boxplot(df[i]);
            plt.xlabel(i);

def find_outlier_rows(df, col, level='both'):
    """
    Finds the rows with outliers in a given column of a dataframe.

    This function takes a dataframe and a column as input, and returns the rows
    with outliers in the given column. Outliers are identified using the
    interquartile range (IQR) formula. The optional level parameter allows the
    caller to specify the level of outliers to return, i.e., lower, upper, or both.

    Args:
        df: The input dataframe.
        col: The name of the column to search for outliers.
        level: The level of outliers to return, i.e., 'lower', 'upper', or 'both'.
               Defaults to 'both'.

    Returns:
        A dataframe containing the rows with outliers in the given column.
    """
    # compute the interquartile range
    iqr = df[col].quantile(0.75) - df[col].quantile(0.25)

    # compute the upper and lower bounds for identifying outliers
    lower_bound = df[col].quantile(0.25) - 1.5 * iqr
    upper_bound = df[col].quantile(0.75) + 1.5 * iqr

    # filter the rows based on the level of outliers to return
    if level == 'lower':
        return df[df[col] < lower_bound]
    elif level == 'upper':
        return df[df[col] > upper_bound]
    else:
        return df[(df[col] > upper_bound) | (df[col] < lower_bound)]

def count_outliers(df):
    """
    This function takes in a DataFrame and returns a DataFrame containing the count and
    percentage of outliers in each numeric column of the original DataFrame.

    Input:
        df: a Pandas DataFrame containing numeric columns

    Output:
        a Pandas DataFrame containing two columns:
        'outlier_counts': the number of outliers in each numeric column
        'outlier_percent': the percentage of outliers in each numeric column
    """
    # select numeric columns
    df_numeric = df.select_dtypes(include=['int', 'float'])

    # get column names
    columns = df_numeric.columns

    # find the name of all columns with outliers
    outlier_cols = [col for col in columns if len(find_outlier_rows(df_numeric, col)) != 0]

    # dataframe to store the results
    outliers_df = pd.DataFrame(columns=['outlier_counts', 'outlier_percent'])

    # count the outliers and compute the percentage of outliers for each column
    for col in outlier_cols:
        outlier_count = len(find_outlier_rows(df_numeric, col))
        all_entries = len(df[col])
        outlier_percent = round(outlier_count * 100 / all_entries, 2)

        # store the results in the dataframe
        outliers_df.loc[col] = [outlier_count, outlier_percent]

    # return the resulting dataframe
    return outliers_df

# count the outliers in sales dataframe
count_outliers(df).sort_values('outlier_counts', ascending=False)

# view the summary statistics of unemployment rate
find_outlier_rows(df, 'Unemployment')['Unemployment'].describe()

# Final checking the data

df.isnull().sum()

df.describe()

df.info()

df.shape

df[df.duplicated()]

*Univariant ,bivariante Analysis and data correlation *

# Plot weekly sales distribution
sns.histplot(df['Weekly_Sales'], kde=True)
plt.title('Weekly Sales Distribution')
plt.show()

It's an exponential distribution. We should use log function on Weekly Sales. Also, in this section, we extract the Feature of the data to introduce to the algorithm.

### Plot log of the target of samples ('Weekly_Sales' column) for finding distribution
plt.figure(figsize=(20,12))
sns.distplot(np.log(df['Weekly_Sales']))

### Plot Sqrt of the target of samples ('Weekly_Sales' column) for finding distribution
plt.figure(figsize=(20,12))
sns.distplot(np.sqrt(df['Weekly_Sales']))

#visualisation
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(np.sqrt(df['Weekly_Sales']), bins=15);

numericalData =df.select_dtypes(include = ['float64', 'int64']).columns
print(f"\nNumerical Features:\n{numericalData}")

categoricalData = df.select_dtypes(include = ['object']).columns
print(f"\nCategorical Features:\n{categoricalData}")

numericalData = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment','weekday', 'month','year']

plt.figure(figsize = (10, 10))

for i in numericalData:
    plt.subplot(3, 2, numericalData.index(i) + 1 % 6)
    sns.histplot(data = df, x = i, kde = True, bins = 15, color = 'c')
    plt.title(f"Distribution of {i}")
    plt.xlabel(i)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

holidayFlagCounts = df['Holiday_Flag'].value_counts()

totalHolidays = holidayFlagCounts.get(1, 0)
totalNonHolidays = holidayFlagCounts.get(0, 0)

plt.figure(figsize = (10, 5))

plt.subplot(1, 2, 1)
bars = plt.bar(['Holidays', 'Non-Holidays'], [totalHolidays, totalNonHolidays], color = ['lightgreen', 'lightcoral'])
plt.title("Total Number of Holidays and Non-Holidays")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha = 'center', va = 'bottom')

plt.subplot(1, 2, 2)
plt.pie(holidayFlagCounts, labels = holidayFlagCounts.index, autopct = '%1.1f%%', startangle = 90, colors = ['lightcoral', 'lightgreen'])
plt.title("Percentage Distribution of Holidays")

plt.show()

totalSalesByStore = df.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending = False)

plt.figure(figsize = (12, 6))
totalSalesByStore.plot(kind = 'bar', color = 'skyblue')

plt.title('Total Sales by Store')
plt.xlabel('Store Number')
plt.ylabel('Total Sales')
plt.xticks(rotation = 0)
plt.show()

highestSalesStore = totalSalesByStore.idxmax()
highestSalesValue = totalSalesByStore.max()
print(f"Highest Sales Store: {highestSalesStore}, Total Sales: ${highestSalesValue:,.0f}")

lowestSalesStore = totalSalesByStore.idxmin()
lowestSalesValue = totalSalesByStore.min()
print(f"Lowest Sales Store : {lowestSalesStore}, Total Sales: ${lowestSalesValue:,.0f}")

Highest Sales Store: 20, Total Sales: 299,066,335LowestSalesStore:33,TotalSales:37,160,222

correlationMap = df.select_dtypes(include = ['number']).corr()

plt.figure(figsize = (8, 6))
sns.heatmap(correlationMap, annot = True, cmap = 'coolwarm', fmt = '.2f', linewidths = .5)
plt.title('Correlation Heatmap')
plt.show()

df.head(1)

If the yearly sales show a seasonal trend, when and what could be the reason?

sales_date = df[['year','Weekly_Sales']]
sales_date.set_index('year',inplace=True)
sales_date.head()

plt.figure(figsize=(20,5))
sns.lineplot(data=sales_date,palette='tab20')

Lets check the sales by holiday season.

plt.figure(figsize=(20,5))
sns.lineplot(data=df,x='year',y='Holiday_Flag')

We can clearly see there is a seasonality component in yearly sales. Whole year sales is average. But at the end of the year there is an exponential hike in the sales.

The Major Reason is during this time of the year, the holiday season begins.

So, the spike in the sales overlaps with the holiday season.

As we know Walmart is very famous in USA, and there is a holiday season during christmans and new year, there might be a lot of offers being given by brands in holiday season. This explains the sudden spike in the sales at the end of the year.

Does temperature affect the weekly sales in any manner?

plt.figure(figsize=(20,5))
#sns.lineplot(data=df,x='year',y='Weekly_Sales')
sns.lineplot(data=df,x='year',y='Temperature')

plt.figure(figsize=(20,5))
sns.lineplot(data=sales_date,palette='tab20')

The only noted effect that can be seen is again of the holiday season. Holiday season are marked with winters and snow, that increases the needed clothing and stuff. Other than this there is no such clear trend of shopping related with temprature.

How is the Consumer Price index affecting the weekly sales of various stores?

df.head(1)

plt.figure(figsize=(20,5))
sns.lineplot(data=df,x='year',y='CPI',color='green')

plt.figure(figsize=(20,5))
sns.lineplot(data=sales_date)

Although there is inflation over time represented by increasing CPI over the time period. There is no upward or downward trend followed by weekly sales.

Lets check the store with maximum average sales over the given period.

df.head(1)

average_store_sales=df.groupby('Store')['Weekly_Sales'].agg('mean')

ave_sales=pd.DataFrame(average_store_sales)

ave_sales['Weekly_Sales']=ave_sales['Weekly_Sales']/(ave_sales['Weekly_Sales'].max()-ave_sales['Weekly_Sales'].min())

ave_sales

plt.figure(figsize=(20,5))
sns.barplot(data=ave_sales,x=ave_sales.index,y='Weekly_Sales',palette='Dark2')

ave_sales.sort_values('Weekly_Sales',ascending=False).head(5)

Top performing 5 stores are --> store number 20, 4, 14, 13, 2

** The worst performing store, and how significant is the difference between the highest and lowest performing stores.**

ave_sales.sort_values('Weekly_Sales').head(5)

Worst performing stores are 33, 44, 5, 36, 38.

Significant difference in highest and lowest performing store.

ave_sales_2=pd.DataFrame(average_store_sales)

(ave_sales_2.loc[33][0]/ave_sales_2.loc[20][0])*100

Lowest performing store's sales only accounts for 12% of sales done by top performing store on average.

plt.figure(figsize=[8,4])
sns.distplot(df['Weekly_Sales'], color='g',hist_kws=dict(edgecolor="black", linewidth=2), bins=30)
plt.show()

g = sns.pairplot(df)
g.map_upper(sns.kdeplot, levels=4, color=".2")
plt.show()

plt.pie(df['year'].value_counts().values,labels =df['year'].value_counts().index,
       autopct='%1.2f%%',shadow=True,colors=['gold','red',"silver"])
plt.title('Annual Sales')
plt.show()

plt.figure(figsize=(7,4))

df.groupby('Store')['Weekly_Sales'].sum().plot(kind='bar')
plt.title('Year-Wise Sales')

plt.show()

Store number 4 and 20 have highest weekly sales.

plt.figure(figsize=(7,4))

df.groupby('month')['Weekly_Sales'].sum().plot(kind='bar',color='Orange')
plt.title('Month-wise Sales')

plt.show()

plt.figure(figsize=(7,4))

df.groupby('year')['Weekly_Sales'].sum().plot(kind='bar')
plt.title('Year-Wise Sales')

plt.show()

n = 1
cols =["Weekly_Sales","Temperature","Fuel_Price","CPI","Unemployment"]
plt.figure(figsize=(15,10))
for i in cols:
        if n<=5:
            plt.subplot(3,2,n);
            n+=1
            sns.boxplot(x = df[i])
            plt.xlabel(i)

l = ["Weekly_Sales","Temperature","Unemployment"]
def outlier_removal(l):
    for i in l:
        Q1  = df[i].quantile(0.25)
        Q3  = df[i].quantile(0.75)
        IQR = Q3-Q1
        Uper = Q3+1.5*IQR
        lower = Q1-1.5*IQR
        df[i]  = np.where(df[i]>Uper,Uper,(np.where(df[i]<lower,lower,df[i])))
outlier_removal(l)

n = 1
cols =["Weekly_Sales","Temperature","Unemployment"]
plt.figure(figsize=(15,3))
for i in cols:
        if n<=3:
            plt.subplot(1,3,n);
            n+=1
            sns.boxplot(x = df[i])
            plt.xlabel(i)

df.head

df2 = df.copy()
df2.head()

days = {0:'Sunday',1:'Monday',2:'Tuesday',3:'Wednesday',4:'Thursday',5: 'Friday',6:'Saturday'}
df2['Day'] = df2['weekday'].map(days)
months={1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September',10:'October',11:'Novemenber',12:'December'}
df2['Month']= df2['month'].map(months)
df2.head()

plt.figure(figsize=(15,5))
plt.subplot(1, 3, 2)
plt.pie(df2['Day'].value_counts().values, labels =df2['Day'].value_counts().index,
autopct = '%1.2f%%', shadow=True,colors=["Orange",'tomato', 'cornflowerblue', 'gold', 'orchid', 'green',"#77BFE2"])
plt.title('Day-wise Distribution')
plt.grid()

plt.subplot(1, 3, 1)
plt.pie(df2['Month'].value_counts().values, labels =df2['Month'].value_counts().index,
autopct = '%1.2f%%',startangle=90, shadow=True,colors=sns.color_palette('Set2'))
plt.title('Month-wise Distribution')
plt.grid()

plt.subplot(1, 3, 3)
df3 = df2.groupby('Holiday_Flag')['Weekly_Sales'].sum().reset_index()
plt.pie(df2['year'].value_counts().values, labels =df2['year'].value_counts().index,
autopct = '%1.2f%%',startangle=90, shadow=True,colors=sns.color_palette('Set2'),labeldistance=1.1)
plt.title('Year-wise Distribution')
plt.grid()

plt.show()

plt.figure(figsize=(15,12))
monthly_sales = pd.pivot_table(df2, index = "Month", columns = "year", values = "Weekly_Sales")
monthly_sales.plot()
plt.title('Yearly Sales')
plt.show()

print('Minimum Sales in the Walmart: ',df2.groupby('Store')['Weekly_Sales'].sum().min())
print('Maximum Sales in the Walmart: ',df2.groupby('Store')['Weekly_Sales'].sum().max())

df3 = df2.groupby('Holiday_Flag')['Weekly_Sales'].sum().reset_index()
plt.pie(df3['Weekly_Sales'],labels= ['Normal Week','Special Holiday Week'],
autopct='%1.2f%%',startangle=90,explode=[0,0.2],shadow=True,colors=['gold','pink'])
plt.show()

from numpy import mean
t = 1
plt.figure(figsize=(20,15))
for i in ["Weekly_Sales","Temperature","Fuel_Price","CPI","Unemployment"]:
        if t<=5:
            plt.subplot(3,2,t)
            ax = sns.barplot(data = df2 , x = "Holiday_Flag" ,y = i  , hue = df.Holiday_Flag ,estimator=mean);
            t+=1

            for i in ax.containers:     #to set a label on top of the bars.
                ax.bar_label(i,)

n = 1
plt.figure(figsize=(20,15))
for i in ["Weekly_Sales","Temperature","Fuel_Price","CPI","Unemployment"]:
        if n<=5:
            plt.subplot(5,1,n)
            ax = sns.lineplot(data = df2 , x = "year" ,y = i  , hue = df.Holiday_Flag );
            plt.xticks(rotation = 90)
            n+=1

sns.pairplot(df2 , hue = "Holiday_Flag" );
plt.title("Distribution and relation of all attributes on Holiday and Normal Week");

plt.figure(figsize=(7,4))

sns.countplot(x= df2.Holiday_Flag)
plt.title('Holiday')

plt.show()

df2.Holiday_Flag.value_counts()

n = 1
plt.figure(figsize=(15,10))
for i in ['Weekly_Sales','Temperature', 'Fuel_Price','CPI', 'Unemployment','year']:
        if n<=6:
            plt.subplot(3,2,n);
            n+=1
            sns.kdeplot(x = df2[i])
            plt.xlabel(i)

** Top 15 Cities w.r.t Sales**

df.hist(figsize=(40,20))

Inference :

the number of transactions occurred almost evenly across various stores and years. The distribution of weekly_sales right-skewed. The distribution of temperature is approximately normal,a little bit left-skewed. The distribution of fuel_price is bi-modal. CPI formed two clusters. = unemployment rate is near normally distributed.

# Sales over months

fig, ax = plt.subplots(figsize=(20, 5))
sns.lineplot(x=df.month, y=(df.Weekly_Sales/1e6))
plt.xlabel('months')
plt.ylabel('Weekly Sales (in million USD)')
plt.title('Weekly Sales Trend',fontdict={'fontsize': 16, 'color':'red'}, pad=5)

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

plt.show()

Inference : Sales grows in the month of :

November December

df2.head()

holiday = df2[df2['Holiday_Flag']==1]
non_holiday = df2[df2['Holiday_Flag']!=1]

plt.boxplot(holiday['Weekly_Sales'])

plt.boxplot(non_holiday['Weekly_Sales'])

fig, ax = plt.subplots(figsize=(15,15))
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap ="YlGnBu")
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':14}, pad=12);

df1 = df.groupby(['Store', 'year']).aggregate({
    'Weekly_Sales': 'sum'}).reset_index().sort_values(by = 'Weekly_Sales', ascending = False)

#print columns
df1.columns
Index(['Store', 'year', 'Weekly_Sales'], dtype='object')

new_cols_df1 = ['Store_Num', 'Year', 'Annual_Sales']

#update column
df1.columns = new_cols_df1

#sort data decendingly
df1.sort_values(by = ['Year', 'Annual_Sales'], ascending = False, inplace  = True)

df1.head()

df_std = df.groupby(['Store']).aggregate({
    'Weekly_Sales': 'std'}).reset_index()

df_std.columns = ['Store', 'STD_of_Weekly_Sales']
df_std.sort_values(by = 'STD_of_Weekly_Sales', inplace = True, ascending = False)

print("\nTOP 5 Stores with Weekly Sales varying a lot\n")
df_std.head()

df_std.set_index('Store').head(5).plot(kind = 'bar')
plt.xlabel("Stores")
plt.ylabel("STD of Weekly Sales")
plt.title("TOP 5 Stores with Weekly Sales varying a lot")
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()

Store 14 has the maximum standard deviation #2

Some holidays have a negative impact on sales. Find out holidays that have higher sales than the mean sales in the non-holiday season for all stores together¶

mean_sales_non_holiday = df[(df['Holiday_Flag'] == 0)]['Weekly_Sales'].mean()
print(mean_sales_non_holiday)

mean_sales_holiday = df[(df['Holiday_Flag'] == 1)]['Weekly_Sales'].mean()
print(mean_sales_holiday)

fig, ax = plt.subplots(figsize=[8,6])
sns.barplot(data = df, x = df['Holiday_Flag'], y = df['Weekly_Sales'])
ax.set_title("Holyday Vs. Weekly_Sales", fontsize = 16);

We conclude that the weekly sales increases when it's holyday

#Temperature Vs. Weekly_Sales

Temperature in Categories

bins=[-2.06,20,40,60,80,100.14]
labels =['< 20','From 20 To 40','From 40 To 60','From 60 To 80','> 80']
df['Temperature Category'] = pd.cut(df['Temperature'], bins, labels = labels)
df.head()

temp = df['Temperature Category']
temp

fig, ax = plt.subplots(figsize=[10,6])
labels = ['< 20','From 20 To 40','From 40 To 60','From 60 To 80','> 80']
colors = ['tomato', "deepskyblue", "gold", "limegreen", "gray"]
ax.pie(x = temp.value_counts(), autopct = "%.1f%%", explode = [0.05]*5,
       colors = colors, labels = labels, pctdistance = 0.5)
ax.set_title("Temp in Fahrenhite", fontsize = 16);
plt.show()

from the above chart we conclude that the most common / repeated degree is between 60 and 80 C

#Temp in F Vs. Weekly_Sales

fig, ax = plt.subplots(figsize=(10,5))
x = temp
y = df['Weekly_Sales']
sns.stripplot(data = df , x = "Temperature Category", y = "Weekly_Sales", ax = ax)
ax.set_title("Temp in F Vs. Weekly_Sales", fontsize = 16);

From the above Chart we can conclude that :

the Sales was at it's top when the temperature ranges between 20 F and 60 F the lowest Sales scored when the temperature is less than 20 F

#CPI Vs. Weekly_Sales

CPI(Prevailing Consumer Price Index)

CPI = df['CPI']

CPI1 = CPI.round(0)

CPI1.dropna()

CPI1 =  CPI1.drop_duplicates()
CPI1

CPI = df[["CPI"]]
CPI.boxplot()
plt.show()

From the above the plot we can conclude :

The Min CPI scored is approximately 125 The Max CPI scored is approximately 225 The Mean CPI scored is approximately 185

#The relation between CPI Vs. Weekly Sales

plt.figure(figsize=(8,6))
sns.scatterplot(x = df["CPI"], y = df["Weekly_Sales"])
plt.show()

From the above plot we can conclude that :

The Majority of sales have a CPI greater than 180 , In other words Most of the sales have a CPI greater than the mean which is 185

Fuel price Vs. Weekly_Sales:

Fuel Price

bins=[2.472,3.5,4.468]
labels =['From 2.4 to 3.5','From 3.5 To 4.4']
df['FuelPriceCategory'] = pd.cut(df['Fuel_Price'],bins,labels =labels)
df.head()

fuel = df["FuelPriceCategory"]

fuel1 = fuel.drop_duplicates()

fuel1 = fuel1.dropna()

fuel1.head()

fuel2 = df[["Fuel_Price"]]
fuel2.boxplot()
plt.show()

From the above plot we can conclude that:

The lowest fuel price scored is approximately 2.4 The highest fuel price scored is approximately 4.4 The mean fuel price scored is approximately 3.4

#The relation between Fuel Price and Weekly Sales

fig, ax = plt.subplots(figsize=(10,5))
x = fuel1
y = df['Weekly_Sales']
sns.stripplot(data = df , x = "FuelPriceCategory", y = "Weekly_Sales", ax = ax)
ax.set_title("Fuel_Price Vs. Weekly_Sales", fontsize = 16);

From the above plot we can conclude that:

The Sales was at it's top when the fuel price was less than 3.5 We can say that the highest sales is when the fuel price is less than the mean which is 2.4 The highest fuel price, The lowest sales.

Relation between the Fuel_Price and the top-5 stores' weekly_sales

From the above plot we can conclude that :

When the fuel price was between 2.4 and 3.5 the weekly sales reached it's highest value and this was achieved in 2011

4- Unemployment Vs. Weekly_Sales: (Prevailing unemployment rate)

#Unemployment

sns.set(style="darkgrid")
sns.relplot(x="Unemployment", y="Weekly_Sales", data=df);

From the above the plot we can conclude :

The majority of sales has an Unemployment rate betwwen 5 and 10. the highest Unemployment rate, The lowest sales. So we now will filter this data

empo = df['Unemployment']

empo = empo.drop_duplicates().T

empo1 = empo.round(0)

sns.set(style = "darkgrid")
sns.lineplot(x = empo1, y = "Weekly_Sales", data = df);

From the above the plot we can conclude:

The unemployment takes negative scale which affect on the weekly sales

columns = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'Unemployment', 'CPI']
plt.figure(figsize=(18, 20))
for i,col in enumerate(columns):
    plt.subplot(3, 2, i+1)
    sns.histplot(data = df, x = col, kde = True, bins = 15, color = 'r')
plt.show()

Conclusion:

The distribution of Weekly_Sales is right skewed, this is normal because the weekly sales may be high in some time. Temperature and Unemployment have normal distribution. CPI and Fuel_Price have bimodal distribution.

fig, ax = plt.subplots(1, 2, figsize = (14, 6))
sns.countplot(data = df, x = 'Holiday_Flag', ax = ax[0])

ax[1].pie(df['Holiday_Flag'].value_counts().values,
          labels = ['Not Holidays', 'Holidays'],
          autopct = '%1.2f%%')

plt.show()

Conclusion:

Days of no holiday are the most frequent than days of holiday in the dataset with a percentage of 93 % and this is normal.

#year

df['year'].value_counts()

fig, ax = plt.subplots(1, 2, figsize = (14, 6))
sns.countplot(data = df, x = 'year', ax = ax[0])
ax[1].pie(df['year'].value_counts().values,
          labels = df['year'].value_counts().index,
          autopct = '%1.1f%%')
plt.show()

Conclusion:

2011 is the most frequent in the dataset because most of the weekly sales were recorded during this year.

#Months

df['month'].value_counts()

plt.figure(figsize=(16, 6))
sns.countplot(data = df, x = 'month')
plt.xlabel('month')
plt.show()

Conclusion:

April and July are the most frequent in the dataset because most of the weekly sales were recorded in these months.

df['weekday'].value_counts()

plt.figure(figsize=(4, 6))
sns.countplot(data = df, x = 'weekday')
plt.xlabel('Day')
plt.show()

Conclusion:

Friday is frequented in each row.

weekly_sales & is_holiday

df.groupby('Holiday_Flag')['Weekly_Sales'].mean()

plt.figure(figsize = (12, 6))
sns.barplot(data = df,
            x = 'Holiday_Flag',
            y = 'Weekly_Sales',
            estimator = np.mean,
            ci = False)

# Add labels and title
plt.title('Average Sales by Holidays')
plt.xlabel('Holiday_Flag', size = 12)
plt.ylabel('Weekly_Sales', size = 12)
plt.show()

Conclusion:

The rate of sales on holidays is higher than on other days.

df.groupby('Holiday_Flag')['Weekly_Sales'].sum()

plt.figure(figsize = (12, 6))
sns.barplot(data = df,
            x = 'Holiday_Flag',
            y = 'Weekly_Sales',
            estimator = np.sum,
            ci = False)

# Add labels and title
plt.title('Total Sales by Holidays')
plt.xlabel('Is Holiday', size = 12)
plt.ylabel('Total Sales', size = 12)
plt.show()

Conclusion:

Total sales on holidays are lower than on other days, which is normal because the number of holidays is very small compared with the number of other days.

weekly_sales & store

gb_store = df.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending = False)
gb_store

plt.figure(figsize = (18, 6))
sns.barplot(data = df,
            x = 'Store',
            y = 'Weekly_Sales',
            order = gb_store.index,
            ci = False)

# Add labels and title
plt.title('Total Sales in each Store', size = 20)
plt.xlabel('Store', size = 15)
plt.ylabel('Total Sales', size = 15)
plt.show()

Conclusion:

There is a high variance in weekly sales from one store to another. Store No. 20 has the highest sales from any store with 301,397,792 followed by Store No. 4 with 299,543,953 and Store No. 33 comes last with 37,160,222$.

#weekly_sales & temperature

plt.figure(figsize = (14, 5))
sns.scatterplot(data = df,
                x = 'Temperature',
                y = 'Weekly_Sales',
                edgecolor = "black")

# Add labels and title
plt.title('Sales by Temperature', size = 20)
plt.xlabel('Temperature', size = 15)
plt.ylabel('Sales', size = 15)
plt.show()


Conclusion:

Sales are not affected by changes in temperature.

weekly_sales & cpi

plt.figure(figsize = (14, 5))
sns.scatterplot(data = df,
                x = 'CPI',
                y = 'Weekly_Sales',
                color = '#8de5a1',
                edgecolor = "black")

# Add labels and title
plt.title('Sales by Consumer Price Index', size = 20)
plt.xlabel('CPI', size = 15)
plt.ylabel('Sales', size = 15)
plt.show()

Conclusion:

Consumer Price Index (CPI) does not affect sales. And based on the distribution of average consumer prices in the above figure, customers can be divided into two categories: customers who pay from 120 to 150 (Middle-class customers). customers who pay from 180 to 230 (High-class customers).

weekly_sales & unemployment

plt.figure(figsize = (14, 5))
sns.scatterplot(data = df,
                x = 'Unemployment',
                y = 'Weekly_Sales',
                color = 'blue',
                edgecolor = 'black')

# Add labels and title
plt.title('Sales by Unemployment Rate', size = 20)
plt.xlabel('Unemployment Rate', size = 15)
plt.ylabel('Sales', size = 15)
plt.show()

What are the total sales in each year?

df.groupby('year')['Weekly_Sales'].sum().sort_values(ascending = False)

plt.figure(figsize = (14, 6))
sns.barplot(data = df,
            x = 'year',
            y = 'Weekly_Sales',
            estimator = np.sum,
            ci = False)

# Add labels and title
plt.title('Total Sales in each Year')
plt.xlabel('Year', size = 13)
plt.ylabel('Total Sales', size = 13)
plt.show()

Conclusion:

Total sales in 2011 were the highest, with 2,448,200,007$.

What are the total sales in each month?

df.groupby('month')['Weekly_Sales'].sum().sort_values(ascending = False)

plt.figure(figsize = (14, 6))
sns.barplot(data = df,
            x = 'month',
            y = 'Weekly_Sales',
            estimator = np.sum,
            ci = False)

# Add labels and title
plt.title('Total Sales in each Month')
plt.xlabel('Month', size = 13)
plt.ylabel('Total Sales', size = 13)
plt.show()

plt.figure(figsize = (18, 5))
sns.lineplot(data = df,
            x = 'month',
            y = 'Weekly_Sales',
            estimator = np.sum)

# Add labels and title
plt.title('Total Sales in each Month')
plt.xlabel('Month', size = 13)
plt.ylabel('Total Sales', size = 13)
plt.show()

Conclusion:

Total sales for all years in April are the highest from any month, with 650,000,977$.

What are the total sales in each year regarding the month?

pd.pivot_table(data = df,
               index = 'year',
               columns = 'month',
               values = 'Weekly_Sales',
               aggfunc = 'sum')

plt.figure(figsize = (16, 6))
sns.barplot(data = df,
            x = 'year',
            y = 'Weekly_Sales',
            hue = 'month',
            estimator = np.sum,
            ci = False)

# Add labels and title
plt.title('Total Sales for each Month in each Year')
plt.xlabel('Year', size = 18)
plt.ylabel('Total Sales', size = 18)

plt.show()

Conclusion:

Total sales in December 2010, 2011 are the highest in the three years, where:

In 2010, total sales in December were the highest. In 2011, total sales in April were the highest. In 2012, total sales in June were the highest.

What are the total sales in each year regarding the day of the week

pd.pivot_table(data = df,
               index = 'year',
               columns = 'weekday',
               values = 'Weekly_Sales',
               aggfunc = 'sum')

plt.figure(figsize = (16, 6))
sns.barplot(data = df,
            x = 'year',
            y = 'Weekly_Sales',
            hue = 'weekday',
            estimator = np.sum,
           ci = False)

# Add labels and title
plt.title('Total Sales For Each Day In Each Year')
plt.xlabel('Year', size = 15)
plt.ylabel('Total Sales', size = 15)
plt.show()

Conclusion:

Total weekly sales in Fridays 2011 were the highest in the three years, where: In 2010, total sales on wednesdays were the highest . In 2011, total sales on wednesdays were the highest . In 2012, total sales on wednesdays were the highest .

What happened to the total sales over time?

years = ['2010', '2011', '2012']
colors = ['red', 'black', 'blue']

plt.figure(figsize = (18, 5))
for i, year in enumerate(years):
    sns.lineplot(data = df[df['year'] == int(year)],
                 x = 'month',
                 y = 'Weekly_Sales',
                 estimator = np.sum,
                 color = colors[i],
                 label = year)

# Add labels and title
plt.title(f'Total Sales Over Time', size = 22)
plt.xlabel('Month', size = 20)
plt.ylabel('Total Sales', size = 20)

# Add a legend
plt.legend()

# Show the plot
plt.show()

Conclusion:

Sales are similar in most months, but they increased at the end of 2010 and 2011 and decreased at the end of 2012.

What happened to the average sales over time?
]
plt.figure(figsize = (18, 5))
for i, year in enumerate(years):
    sns.lineplot(data = df[df['year'] == int(year)],
                 x = 'month',
                 y = 'Weekly_Sales',
                 estimator = np.mean,
                 color = colors[i],
                 label = year)

# Add labels and title
plt.title(f'Total Sales Over Time', size = 22)
plt.xlabel('Month', size = 20)
plt.ylabel('Total Sales', size = 20)

# Add a legend
plt.legend()

# Show the plot
plt.show()

Conclusion:

Sales rates are similar in most months, but they increased at the end of 2010 and 2011 and decreased at the end of 2012.

What happens to the unemployment rate over time?

plt.figure(figsize = (18, 5))
for i, year in enumerate(years):
    sns.lineplot(data = df[df['year'] == int(year)],
                 x = 'month',
                 y = 'Unemployment',
                 estimator = np.mean,
                 color = colors[i],
                 label = year)

# Add labels and title
plt.title(f'Unemployment Rate Over Time', size = 22)
plt.xlabel('Date', size = 20)
plt.ylabel('Unemployment Rate', size = 20)

# Add a legend
plt.legend()

# Show the plot
plt.show()

Conclusion:

The unemployment rate decreases over time.

Are there any seasonality trends in the dataset

# create the pivot table
pivot_table = df.pivot_table(index='month', columns='year', values='Weekly_Sales')
# display the pivot table
pivot_table

# plot the average sales
fig, ax = plt.subplots(figsize=(20, 6))
sns.set_palette("bright")
sns.lineplot(x=pivot_table.index, y=pivot_table[2010]/1e6, ax=ax, label='2010')
sns.lineplot( x=pivot_table.index, y=pivot_table[2011]/1e6, ax=ax, label='2011')
sns.lineplot( x=pivot_table.index, y=pivot_table[2012]/1e6, ax=ax, label='2012')
plt.ylabel('Average weekly sales (in millions USD)')
plt.title('Average Sales Trends for 2010, 2011 & 2012', fontdict ={'fontsize':16,
                                                                   'color':'red',
                                                                   'horizontalalignment': 'center'},
                                                                   pad=12)
# Add a legend
plt.legend()
plt.show()

Which stores had the highest and lowest average revenues over the years?

def plot_top_and_bottom_stores(df, col):
    """
    Plot the top and bottom 5 stores based on their average weekly sales.

    Parameters:
    df (pandas DataFrame): The dataframe containing the sales data.
    col (str): The name of the column to group the data by.

    Returns:
    None
    """
    # Group the data by the specified column and sort it by sales in descending order
    df = df.groupby(col).mean().sort_values(by='Weekly_Sales', ascending=False)

    # Select the top 5 and bottom 5 products
    top_stores = df.head(5)
    bottom_stores = df.tail(5)

    # Set the color palette
    sns.set_palette("bright")

    # Create a bar chart of the top 5 products
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_stores.index, y=top_stores['Weekly_Sales']/1e6, order=top_stores.index)
    plt.title('Top 5 Stores by Average Sales')
    plt.ylabel('Average weekly sales (millions USD)')
    plt.show()

    # Create a bar chart of the bottom 5 products
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=bottom_stores.index, y=bottom_stores['Weekly_Sales']/1e6, order=bottom_stores.index)
    plt.title('Bottom 5 Stores by Average Sales')
    plt.ylabel('Average weekly sales (millions USD)')
    plt.show()
** What happens to the fuel price rate over time?**

plt.figure(figsize = (18, 5))
for i, year in enumerate(years):
    sns.lineplot(data = df[df['year'] == int(year)],
                 x = 'month',
                 y = 'Fuel_Price',
                 estimator = np.mean,
                 color = colors[i],
                 label = year)

# Add labels and title
plt.title(f'Fuel Price Over Time', size = 22)
plt.xlabel('Date', size = 20)
plt.ylabel('Fuel Price', size = 20)

# Add a legend
plt.legend()

# Show the plot
plt.show()

Conclusion:

The fuel price increases over time in 2010 & 2012. The fuel price decreases over time in 2011.

How does non-holiday weekly sales compared to holiday weekly sales?

# filter out non-holiday and holiday weekly sales
non_holiday_sales = df[df['Holiday_Flag'] == 0]
holiday_sales = df[df['Holiday_Flag'] == 1]

# plot box plots of non-holiday and holiday weekly sales
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=[holiday_sales['Weekly_Sales']/1e6, non_holiday_sales['Weekly_Sales']/1e6])
plt.ylabel('Weekly sales in million USD')
plt.xlabel('Week type')
plt.title('Box plots of non-holiday and holiday weekly sales')
plt.show()

We can see that both holiday and non-holiday weekly sales have similar spread. However, the bigger sales happen during the holiday weeks.

df.head()

# Heatmap to explain the correlation between the features
plt.figure(figsize = (20, 12))
sns.heatmap(df[['Weekly_Sales','Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment' , 'weekday' ,'month' ,'year']].corr(), annot = True, cmap='viridis')
plt.show()

# histograms
df.hist(figsize=(30,20));

Fuel Price, CPI , Unemployment , Temperature Effects

fuel_price = pd.pivot_table(df, values = "Weekly_Sales", index= "Fuel_Price")
fuel_price.plot()

temp = pd.pivot_table(df, values = "Weekly_Sales", index= "Temperature")
temp.plot()

CPI = pd.pivot_table(df, values = "Weekly_Sales", index= "CPI")
CPI.plot()

unemployment = pd.pivot_table(df, values = "Weekly_Sales", index= "Unemployment")
unemployment.plot()

From graphs, it is seen that there are no significant patterns between CPI, temperature, unemployment rate, fuel price vs weekly sales. There is no data for CPI between 140-180 also.

plt.figure(figsize=[10,4])
sns.distplot(df[target], color='b',hist_kws=dict(edgecolor="black", linewidth=2), bins=30)
plt.title('Weekly_Sales')
plt.show()

*numerical *

print('\033[1mNumeric Features Distribution'.center(130))

n=4

clr=['r','g','b','g','b','r']

plt.figure(figsize=[15,6*math.ceil(len(nf)/n)])
for i in range(len(nf)):
    plt.subplot(math.ceil(len(nf)/3),n,i+1)
    sns.distplot(df[nf[i]],hist_kws=dict(edgecolor="black", linewidth=2), bins=10, color=list(np.random.randint([255,255,255])/255))
plt.tight_layout()
plt.show()

plt.figure(figsize=[15,6*math.ceil(len(nf)/n)])
for i in range(len(nf)):
    plt.subplot(math.ceil(len(nf)/3),n,i+1)
    df.boxplot(nf[i])
plt.tight_layout()
plt.show()

from numpy import mean

pno = 1
plt.figure(figsize=(20,18))
for i in ["Weekly_Sales","Temperature","Fuel_Price","CPI","Unemployment"]:
        if pno<=5:
            plt.subplot(3,2,pno)
            ax = sns.barplot(data = df , x = "Holiday_Flag" ,y = i  , hue = df.Holiday_Flag ,estimator=mean);
            pno+=1

            for i in ax.containers:     #to set a label on top of the bars.
                ax.bar_label(i,)

sns.distplot(df['Weekly_Sales'])

sns.displot(df['Temperature'],kde=True,color = 'r')#, ax=axes[1])
sns.displot(df['Fuel_Price'],kde=True)
sns.displot(df['Unemployment'],kde=False)

df.groupby('year')['Weekly_Sales'].sum()

plt.pie(df.groupby('year')['Weekly_Sales'].sum(),labels=df['year'].unique(),normalize=True,autopct='%1.2f%%',colors=['hotpink','green','violet'])
plt.title('Annual Sales')

df2 = df.groupby('weekday')['Weekly_Sales'].sum().reset_index()
df2.head(10)

plt.figure(figsize=(10,8))
plt.pie(df2['Weekly_Sales'],labels= df2['weekday'],autopct='%1.2f%%', normalize=True)

df4 = df.groupby('month')['Weekly_Sales'].sum().reset_index()

df4.head()

plt.figure(figsize=(10,10))
plt.pie(df4['Weekly_Sales'],labels=df4['month'],normalize=True,autopct='%1.2f%%')

df2 = df2.groupby('weekday')['Weekly_Sales'].sum().reset_index()
plt.figure(figsize=(10,8))
plt.pie(df2['Weekly_Sales'],labels= df2['weekday'],autopct='%1.2f%%', normalize=True)

plt.figure(figsize=(10,10))
df5 = df.groupby('month')['Weekly_Sales'].sum().reset_index()
plt.pie(df5['Weekly_Sales'],labels=df5['month'],normalize=True,autopct='%1.2f%%')

df6 = df.groupby('Holiday_Flag')['Weekly_Sales'].sum().reset_index()
plt.pie(df6['Weekly_Sales'],labels= ['Non Special Holiday Week','Special Holiday Week'],normalize=True,autopct='%1.2f%%',startangle=90,explode=[0,0.3],shadow=True,colors=['hotpink','green'])

df.groupby('Store')['Weekly_Sales'].count().reset_index()

df.groupby('Store')['Weekly_Sales'].sum().max()

#Which store had the highest weekly sales in 2010?

df2010 = df[df.year==2010]
df2010[df2010.Weekly_Sales == df2010.Weekly_Sales.max()]

#Which store had the highest weekly sales in 2011?

df2011 = df[df.year==2011]
df2011[df2011.Weekly_Sales == df2011.Weekly_Sales.max()]

#Which store had the highest weekly sales in 2012?

df2012 = df[df.year==2012]
df2012[df2012.Weekly_Sales == df2012.Weekly_Sales.max()]

#Which store had the lowest weekly sales in 2010?

df2010[df2010.Weekly_Sales == df2010.Weekly_Sales.min()]

#Which store had the lowest weekly sales in 2011?

df2011[df2011.Weekly_Sales == df2011.Weekly_Sales.min()]

#Which store had the lowest weekly sales in 2012?

df2012[df2012.Weekly_Sales == df2012.Weekly_Sales.min()]

#Weekly Sales by Year

df2010.groupby('month')["Weekly_Sales"].mean().plot(linewidth=2,style='--o').set_title('Weekly Sales by 2010')

df2011.groupby('month')['Weekly_Sales'].mean().plot(linewidth=2,style='--o').set_title('Weekly Sales by 2011')

df2012.groupby('month')['Weekly_Sales'].mean().plot(linewidth=2,style='--o').set_title('Weekly Sales by 2012')

#Conclusion

The Temperature affects on the weekly sales where the customers are likly to go shopping more in cold weather specially in winter. The Unemployment rate took negative scale which affect badly on the weekly sales as there is no more employee to serve them. The fuel price affects by negative on the sales as by increasing the fuel price the products' price will increase which affect on the sales rate. The year 2011 was the richest year as the stores reached the maximum rate of weekly sales in it.

The year 2010 was the year with the highest revenues, followed by 2011. The stores with the highest revenues are number 14, 20 and 4. In general, holidays have the highest revenues. There are lower revenues when the temperature is low and higher revenues when the temperature is high. There are higher revenues when the CPI is low. The issue of rising fuel prices does not seem to affect store revenues.

# We can remove the date data now because it will not be giving us any important information about the prediction
df.drop(['Date'],axis=1,inplace=True)

df.head

df

df1=df.drop(['Temperature Category', 'FuelPriceCategory'], axis=1)

df1

#Splitting Data into Features(X) And Target(Y)

# Creating the target and the data separation

x = df1.drop(['Weekly_Sales'],axis=1)
y = df1['Weekly_Sales']

#Splitting into train test

# Creating the test-train split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)
Standardising the Data

#scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x

Model Training and Evaluation

Implementing Models

Model Used : Linear Regression

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lr = LinearRegression()

lr.fit(x_train,y_train)

y_predict = lr.predict(x_test)

print(y_predict)

# Calculating the mean squared error

mse = mean_squared_error(y_test,y_predict)

# Calculating root mean squared error
rmse = np.sqrt(mse)

print('MSE',mse)
print('RMSE',rmse)

Inference : Linear regression is not able to explain the variance of the data because the r2_score achieved is very less

Model Used : Decision Tree Regressor

# importing the model

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(max_depth=3)

tree_reg.fit(x_train, y_train)

# Predict the output for the test data
y_predict = tree_reg.predict(x_test)

# Evaluate the performance of the model using mean squared error
mse = mean_squared_error(y_test, y_predict)
print("Mean squared error: ", mse)

Inference 2 : Decision Tree regressor is better than the Linear Regression in terms of the fit as the r2_score for the DTR is better than LR

Model Used : Random Forest

# importing the model

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(x_train, y_train)

# Predict the output for the test data
y_pred = rf_reg.predict(x_test)

# Evaluate the performance of the model using mean squared error and R2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean squared error: ", mse)
print("R2 score: ", r2)

Inference : Random Forest Regressor is the best fitting model for the given dataset and it was able to nearly accurate about fitting the data.

** Lasso , Ridge Regression and elastic net**

from sklearn.linear_model import LassoCV , RidgeCV,ElasticNetCV
lcv = LassoCV(cv = 4)
lcv.fit(x_train,y_train)

Y_pred = lcv.predict(x_test)
r2_score(y_test,y_pred)
print(Y_pred)
print("R2 Score is ",r2_score(y_test,y_pred))

from sklearn.linear_model import LassoCV , RidgeCV,ElasticNetCV
rg = RidgeCV(cv = 4)
rg.fit(x_train,y_train)

from sklearn.linear_model import LassoCV , RidgeCV,ElasticNetCV
Y_pred = lcv.predict(x_test)
r2_score(y_test,y_pred)
print(Y_pred)
print("R2 Score is ",r2_score(y_test,y_pred))

from sklearn.linear_model import LassoCV , RidgeCV,ElasticNetCV
el = ElasticNetCV(cv = 4)
el.fit(x_train,y_train)

Y_pred = lcv.predict(x_test)
r2_score(y_test,y_pred)
print(Y_pred)
print("R2 Score is ",r2_score(y_test,y_pred))

Implementing KNeighbours and SVR

from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.metrics import r2_score , mean_squared_error

kn = KNeighborsRegressor()
kn.fit(x_train,y_train)

Y_pred = kn.predict(x_test)
print("R2 Score is ",r2_score(y_test,y_pred))
print("mean Squred error is",mean_squared_error(y_test,y_pred))
print(Y_pred)

svr = SVR()
svr.fit(x_train,y_train)

Y_pred = svr.predict(x_test)
print("R2 Score is ",r2_score(y_test,Y_pred))
print("mean Squred error is",mean_squared_error(y_test,Y_pred))
print(Y_pred)

df1 = pd.DataFrame(columns=["Model", "Accuracy for train","MSE for train","MAE for train","Accuracy for test","MSE for test","MAE for test"])

def pred_model(model,x_train,y_train,x_test,y_test):
    c = model()
    c.fit(x_train,y_train)
    x_pred = c.predict(x_train)
    y_pred = c.predict(x_test)

    print(c)

    print("For Training Data \n --------------------------------")
    print("MAE: ",mean_absolute_error(y_train, x_pred))
    print("MSE: ",mean_squared_error(y_train, x_pred))
    print("r2: ",r2_score(y_train, x_pred))
    print("RMSE: ",np.sqrt(mean_squared_error(y_train, x_pred)))

    print("")
    print("For Test Data \n --------------------------------")
    print("MAE: ",mean_absolute_error(y_test, y_pred))
    print("MSE: ",mean_squared_error(y_test, y_pred))
    print("r2: ",r2_score(y_test, y_pred))
    print("RMSE: ",np.sqrt(mean_squared_error(y_test,y_pred)))

   # print(f'MSE: {mean_squared_error(y_test,y_pred)}')
    #print(f'MAE: {mean_absolute_error(y_test,y_pred)}')
    #print(f'R2 : {r2_score(y_test,y_pred)}')


    print("Residual Analysis:")
    plt.figure(figsize = (20,5))
    plt.scatter(y_train,(y_train-x_pred),color = "red",label = 'Training Predictions')
    plt.scatter(y_test,(y_test-y_pred),color = "green",label = 'Testing Predictions')
    plt.legend()
    plt.show()

    re = {}
    re["Model"] = c
    re["Accuracy for train"] = 100*(r2_score(y_train, x_pred))
    re["MSE for train"] = mean_squared_error(y_test, y_pred)
    re["MAE for train"] = mean_absolute_error(y_test, y_pred)
    re["Accuracy for test"] = 100*(r2_score(y_test, y_pred))
    re["MSE for test"] = mean_squared_error(y_test,y_pred)
    re["MAE for test"] = mean_absolute_error(y_test,y_pred)

    return re

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
l = (LinearRegression,RandomForestRegressor,Lasso,Ridge,ElasticNet,DecisionTreeRegressor,)

for i in l:
    re = pred_model(i, x_train,y_train,x_test,y_test)

Training with 70:30 Ratio

df2 = pd.DataFrame(columns=["Model", "Accuracy for train","MSE for train","MAE for train","Accuracy for test","MSE for test","MAE for test"])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.3, random_state = 42)

def pred_model1(model,x_train,y_train,x_test,y_test):
    c = model()
    c.fit(x_train,y_train)
    x_pred = c.predict(x_train)
    y_pred = c.predict(x_test)

    print(c)

    print("For Training Data \n --------------------------------")
    print("MAE: ",mean_absolute_error(y_train, x_pred))
    print("MSE: ",mean_squared_error(y_train, x_pred))
    print("r2: ",r2_score(y_train, x_pred))
    print("RMSE: ",np.sqrt(mean_squared_error(y_train, x_pred)))

    print("")
    print("For Test Data \n --------------------------------")
    print("MAE: ",mean_absolute_error(y_test, y_pred))
    print("MSE: ",mean_squared_error(y_test, y_pred))
    print("r2: ",r2_score(y_test, y_pred))
    print("RMSE: ",np.sqrt(mean_squared_error(y_test,y_pred)))

    print("Residual Analysis:")
    plt.figure(figsize = (20,5))
    plt.scatter(y_train,(y_train-x_pred),color = "red",label = 'Training Predictions')
    plt.scatter(y_test,(y_test-y_pred),color = "green",label = 'Testing Predictions')
    plt.legend()
    plt.show()

    re1 = {}
    re1["Model"] = c
    re1["Accuracy for train"] = 100*(r2_score(y_train, x_pred))
    re1["MSE for train"] = mean_squared_error(y_test, y_pred)
    re1["MAE for train"] = mean_absolute_error(y_test, y_pred)
    re1["Accuracy for test"] = 100*(r2_score(y_test, y_pred))
    re1["MSE for test"] = mean_squared_error(y_test,y_pred)
    re1["MAE for test"] = mean_absolute_error(y_test,y_pred)

    return re1

l = (LinearRegression,RandomForestRegressor,Lasso,Ridge,ElasticNet,DecisionTreeRegressor)

for i in l:
    re1 = pred_model1(i, x_train,y_train,x_test,y_test)

Training with 60:40 Ratio

df3 = pd.DataFrame(columns=["Model", "Accuracy for train","MSE for train","MAE for train","Accuracy for test","MSE for test","MAE for test"])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.4, random_state = 42)

def pred_model2(model,x_train,y_train,x_test,y_test):
    c = model()
    c.fit(x_train,y_train)
    x_pred = c.predict(x_train)
    y_pred = c.predict(x_test)

    print(c)

    print("For Training Data \n --------------------------------")
    print("MAE: ",mean_absolute_error(y_train, x_pred))
    print("MSE: ",mean_squared_error(y_train, x_pred))
    print("r2: ",r2_score(y_train, x_pred))
    print("RMSE: ",np.sqrt(mean_squared_error(y_train, x_pred)))

    print("")
    print("For Test Data \n --------------------------------")
    print("MAE: ",mean_absolute_error(y_test, y_pred))
    print("MSE: ",mean_squared_error(y_test, y_pred))
    print("r2: ",r2_score(y_test, y_pred))
    print("RMSE: ",np.sqrt(mean_squared_error(y_test,y_pred)))

    print("Residual Analysis:")
    plt.figure(figsize = (20,5))
    plt.scatter(y_train,(y_train-x_pred),color = "red",label = 'Training Predictions')
    plt.scatter(y_test,(y_test-y_pred),color = "green",label = 'Testing Predictions')
    plt.legend()
    plt.show()

    re2 = {}
    re2["Model"] = c
    re2["Accuracy for train"] = 100*(r2_score(y_train, x_pred))
    re2["MSE for train"] = mean_squared_error(y_test, y_pred)
    re2["MAE for train"] = mean_absolute_error(y_test, y_pred)
    re2["Accuracy for test"] = 100*(r2_score(y_test, y_pred))
    re2["MSE for test"] = mean_squared_error(y_test,y_pred)
    re2["MAE for test"] = mean_absolute_error(y_test,y_pred)

    return re2

l = (LinearRegression,RandomForestRegressor,Lasso,Ridge,ElasticNet,DecisionTreeRegressor)

for i in l:
    re2 = pred_model2(i, x_train,y_train,x_test,y_test)

After comparing all the models with different test ratios, we get that 'RandomForest' with 70:30 ratio is giving us the best model with 93.12 accuracy. So now we will try to improve it's accuracy further by tuning it's parameters

Hyperparameter tuning

n_estimators = [5,20,50,100]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 120, num = 12)]
min_samples_split = [2, 6, 10]
min_samples_leaf = [1, 3, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,

'max_features': max_features,

'max_depth': max_depth,

'min_samples_split': min_samples_split,

'min_samples_leaf': min_samples_leaf,

'bootstrap': bootstrap}

rf = RandomForestRegressor()

from sklearn.model_selection import RandomizedSearchCV
rf_random = RandomizedSearchCV(estimator = rf,param_distributions = random_grid,
               n_iter = 100, cv = 5, verbose=2, random_state=35, n_jobs = -1)
rf_random.fit(x_train, y_train)

Getting the best parameters

print ('Random grid: ', random_grid, '\n')
print ('Best Parameters: ', rf_random.best_params_, ' \n')
Random grid:  {'n_estimators': [5, 20, 50, 100], 'max_features': ['auto', 'sqrt'], 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120], 'min_samples_split': [2, 6, 10], 'min_samples_leaf': [1, 3, 4], 'bootstrap': [True, False]} 

Best Parameters:  {'n_estimators': 100, 'min_samples_split': 6, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 40, 'bootstrap': True}  

Using the best parameters obtained in our model and getting the accuracy

randmf = RandomForestRegressor(n_estimators = 100, min_samples_split = 2, min_samples_leaf= 1, max_features = 'sqrt', max_depth= 120, bootstrap=True)
randmf.fit( x_train, y_train)

x_pred = randmf.predict(x_train)
y_pred = randmf.predict(x_test)

print("For Training Data \n --------------------------------")
print("MAE: ",mean_absolute_error(y_train, x_pred))
print("MSE: ",mean_squared_error(y_train, x_pred))
print("r2: ",r2_score(y_train, x_pred))
print("RMSE: ",np.sqrt(mean_squared_error(y_train, x_pred)))

print("")
print("For Test Data \n --------------------------------")
print("MAE: ",mean_absolute_error(y_test, y_pred))
print("MSE: ",mean_squared_error(y_test, y_pred))
print("r2: ",r2_score(y_test, y_pred))
print("RMSE: ",np.sqrt(mean_squared_error(y_test,y_pred)))

print("Residual Analysis:")
plt.figure(figsize = (20,5))
plt.scatter(y_train,(y_train-x_pred),color = "red",label = 'Training Predictions')
plt.scatter(y_test,(y_test-y_pred),color = "green",label = 'Testing Predictions')
plt.legend()
plt.show()

So, our accuracy has improved slightly after hyperparameter tuning from 92.133 to 93.5678 Hence, we will take this as our final model for further prediction

Deployment

1m
tempe=input("Temparature : ")
fu = input("Fuel_Price : ")
cp = input("CPI: ")
un = input("Unemployment : ")
print("Is it a holiday :")
holi_0 = int(input("For No-1, Yes-0 : "))
holi_1 = int(input("For No-0, Yes-1 : "))
store=int(input("Enter the store no (1-45): "))
l1=['Store_0','Store_1', 'Store_2', 'Store_3', 'Store_4', 'Store_5', 'Store_6',
    'Store_7', 'Store_8', 'Store_9', 'Store_10', 'Store_11', 'Store_12',
    'Store_13', 'Store_14', 'Store_15', 'Store_16', 'Store_17', 'Store_18',
    'Store_19', 'Store_20', 'Store_21', 'Store_22', 'Store_23', 'Store_24', 'Store_25',
    'Store_26', 'Store_27', 'Store_28', 'Store_29', 'Store_30', 'Store_31', 'Store_32', 'Store_33',
    'Store_34', 'Store_35', 'Store_36', 'Store_37', 'Store_38', 'Store_39', 'Store_40', 'Store_41', 'Store_42',
    'Store_43', 'Store_44', 'Store_45']

value = l1[store]

df13 = pd.DataFrame(columns = ['Fuel_Price', 'CPI', 'Unemployment', 'Store_1', 'Store_2', 'Store_3', 'Store_4', 'Store_5', 'Store_6', 'Store_7', 'Store_8', 'Store_9', 'Store_10', 'Store_11', 'Store_12', 'Store_13', 'Store_14', 'Store_15', 'Store_16', 'Store_17', 'Store_18', 'Store_19', 'Store_20', 'Store_21', 'Store_22', 'Store_23', 'Store_24', 'Store_25', 'Store_26', 'Store_27', 'Store_28', 'Store_29', 'Store_30', 'Store_31', 'Store_32', 'Store_33', 'Store_34', 'Store_35', 'Store_36', 'Store_37', 'Store_38', 'Store_39', 'Store_40', 'Store_41', 'Store_42', 'Store_43', 'Store_44', 'Store_45', 'Holiday_Flag_0', 'Holiday_Flag_1'])
data = {value:1}
df13['Temparature']=tempe
df13['Fuel_Price']=fu
df13['CPI']=cp
df13['Unemployment']=un
df13['Holiday_Flag_0']=holi_0
df13['Holiday_Flag_1']=holi_1

df13 = df13.fillna(0)
df13

randmf.predict(df13)

print("So the weekly sales for store 1 on a regular day is  : ",randmf.predict(df13))

Conclusion

So our dataset was labelled and our problem statement was of prediction, hence we have used different supervised learning algorithms used for prediction.

All the algorithms used in this project are :

Linear Regression Linear Regression : Lasso Linear Regression : Ridge Linear Regression : ELasticNet Decision Tree Random Forest Also we have used three different train-test ratio for training our model. And the best model that we obtained was 'RandomForest' with 70:30 ratio and with 92.133 accuracy. Also after tuning our model our accuracy further improved from 92.133 to 92.97.

In the end we have created a user-input deployment and predicted the output by the same.
