#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import googletrans 
from googletrans import Translator
import matplotlib.dates as mdates
from datetime import datetime
import calendar
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[67]:


df = pd.read_csv('/home/shristi/Downloads/Presentation/Price.csv')
df.head()


# In[68]:


# Rename columns

df.columns = ['Vegetable', 'Unit', 'low_price', 'high_price', "average", "Date", "priceType"]
df.head()


# In[69]:


# To translate the price 

def convert(val):
    result = []
    for v in val:
        result.append(int(v))
    return result

# df['न्यूनतम'] = convert(df['न्यूनतम'].tolist())
# df['अधिकतम'] = convert(df['अधिकतम'].tolist())
# df['औसत'] = convert(df['औसत'].tolist())

df['low_price'] = convert(df['low_price'].tolist())
df['high_price'] = convert(df['high_price'].tolist())
df['average'] = convert(df['average'].tolist())

df.head()


# In[70]:


#use pandas datetime to convert Date to datetime format

df['Date'] = pd.to_datetime(df['Date'])
# df.head()
df.head()


# In[71]:


df['Unit'] = df['Unit'].str.replace('के.जी.', 'kg')
df['Unit'] = df['Unit'].str.replace('के जी', 'kg')
df['Unit'] = df['Unit'].str.replace('केजी', 'kg')
df['Unit'] = df['Unit'].str.replace('प्रति गोटा', 'psc')
df['Unit'] = df['Unit'].str.replace('दर्जन', 'doz')
df_new = df

df['Unit'].value_counts()


# In[72]:


df.head()


# In[8]:


df.to_csv('/home/shristi/Downloads/Presentation/Price_upd.csv')


# In[9]:


df = pd.read_csv('/home/shristi/Downloads/Presentation/Price_upd.csv')
df.head()


# In[73]:


# To translate the price 
# df = df.drop('month_name', axis=1)

df['month']=''
# df['month_name']=''
df['year']=''
def getMonth(val):
    result = []
    for v in val:
#         result.append(calendar.month_name[v.month])
        result.append(v.month)
    return result

df['month'] = getMonth(df['Date'].tolist())
df['year'] = pd.DatetimeIndex(df['Date']).year

df.head()


# In[74]:


# List of vegetables
Key_vegetables = df.groupby(['Vegetable']).groups.keys()
Key_vegetables


# In[75]:


df.describe()


# In[76]:


vege = df['Vegetable'].value_counts()
vege


# In[77]:


df['Unit'].value_counts()


# In[78]:


veg_price_month_year = df.groupby(by=['month','year'], as_index=False)[['Vegetable','average']].mean()
veg_price_month_year
fig = plt.figure(figsize = (20,6))
# ax = sns.barplot(x='month', y='average', data = veg_price_month_year)
ax = sns.catplot(x="month", y="average", hue="year", data=veg_price_month_year,
                height=8,kind="bar", palette="muted")
# ax.set_title('Bar diagram showing mean average price - month-year')



# ax.despine(left=True)
ax.set_ylabels("Average Price")
ax.set_xlabels("Price comparison Month-Year")


# In[79]:


# Which month  vegetable price is max and min
veg_grp_month = df.groupby(by=['month'], as_index=False)[['Vegetable','average']].mean()
veg_grp_month.sort_values(by='month', ascending=True)
veg_grp_month


# In[80]:


#Bar diagram showing vegetables > 300 price
avg_data = veg_grp_month
fig = plt.figure(figsize = (12,6))
ax = sns.barplot(x='month', y='average', data = avg_data)
ax.set_title('Bar diagram showing mean average price - month')
ax.set_xticklabels(ax.get_xticklabels(),rotation=60)
# for index, row in avg_data.iterrows():
#     ax.text(row.month, row.average, round(row.average), color='black')


# In[81]:


veg_price_yearly = df.groupby(by=['year'], as_index=False)[['Vegetable','average']].mean()
veg_price_yearly
fig = plt.figure(figsize = (12,6))
ax = sns.barplot(x='year', y='average', data = veg_price_yearly)
ax.set_title('Bar diagram showing mean average price - yearly')
ax.set_xticklabels(ax.get_xticklabels(),rotation=60)


# In[82]:


translator = Translator()

def convert_name(val):
    result = []
    for v in val:
        b = translator.translate(v).text
        result.append(b)

    return result


# In[83]:


veg_grp_name_month_all = df.groupby(by=['Vegetable'], as_index=False)[['low_price','high_price', 'average']].max()

veg_grp_name_month_avg = df.groupby(by=['Vegetable'], as_index=False)[['average']].mean().reset_index()
#splitting data to 3 half
total = len(veg_grp_name_month_all)
ratio = total/3
veg_grp_1 = veg_grp_name_month_avg[:40]

veg_grp_2 = veg_grp_name_month_avg[40:40+40]

veg_grp_3 = veg_grp_name_month_avg[40+40:]

# #translation
veg_grp_1['Vegetable_trans']=convert_name(veg_grp_1['Vegetable'].tolist())


# In[90]:


#Bar diagram showing vegetables > 300 price
avg_data = veg_grp_1.reset_index()
fig = plt.figure(figsize = (20,6))
ax = sns.barplot(x='Vegetable_trans', y='average', data = avg_data)
ax.set_title('Bar diagram showing vegetables first 40 ')
ax.set_xticklabels(ax.get_xticklabels(),rotation=60)


# In[87]:


veg_grp_2['Vegetable_trans']=convert_name(veg_grp_2['Vegetable'].tolist())


# In[88]:


# Bar diagram showing vegetables between 250 & 140 price
avg_data_below = veg_grp_2.reset_index()
fig = plt.figure(figsize = (20,6))

ax=sns.barplot(x='Vegetable_trans', y='average', data = avg_data_below)
ax.set_title('Bar diagram showing vegetables 40-80 ')
ax.set_xticklabels(ax.get_xticklabels(),rotation=60)


# In[99]:


veg_grp_3['Vegetable_trans']=convert_name(veg_grp_3['Vegetable'].tolist())


# In[100]:


#Bar diagram showing vegetables less than 140 price
avg_data_below = veg_grp_3.reset_index()
fig = plt.figure(figsize = (20,6))

ax=sns.barplot(x='Vegetable_trans', y='average', data = avg_data_below)
ax.set_title('Bar diagram showing vegetables 80-120')
ax.set_xticklabels(ax.get_xticklabels(),rotation=60)


# In[118]:


max_price_by_name = df.groupby(by=['Vegetable','month','year'], as_index=False)[['low_price','high_price', 'average']].max()

exp_veg = veg_grp_2[veg_grp_2['Vegetable_trans'] =='Mushroom (maid)']
exp_veg


# In[119]:



mushroom = max_price_by_name[max_price_by_name['Vegetable'] == 'च्याउ(कन्य)']
mushroom['Vegetable_trans']=convert_name(mushroom['Vegetable'].tolist())
mushroom


# In[120]:


fig = plt.figure(figsize = (20,7))  #figsize
ax = sns.catplot(x="month", y="average", hue="year", data=mushroom,
                height=8,kind="bar", palette="muted")
ax.set_ylabels("Average Price")
ax.set_xlabels("Price comparison Month-Year")


# In[121]:


apple_fuji = veg_grp_3[veg_grp_3['Vegetable_trans'] =='Apples (Fuji)']
apple_fuji


# In[122]:


apple = max_price_by_name[max_price_by_name['Vegetable'] == 'स्याउ(फूजी)']
apple['Vegetable_trans']=convert_name(apple['Vegetable'].tolist())
apple


# In[127]:


fig = plt.figure(figsize = (10,4))  #figsize
ax = sns.catplot(x="month", y="average", hue="year", data=apple,
                height=6,kind="bar", palette="muted")
ax.set_ylabels("Average Price")
ax.set_xlabels("Price comparison Month-Year")


# In[146]:


#Chk vegetables lemon  
lemon = df[df["Vegetable"] == 'कागती'][['Vegetable','Unit','average', 'month','year','priceType']]
lemon

lemon_w = lemon[lemon["priceType"] == 'W'][['Vegetable','average','month', 'year']]

lemon_R = lemon[lemon["priceType"] == 'R'][['Vegetable','average', 'month','year']]


# In[140]:


a=lemon_w.groupby('month')
fig = plt.figure(figsize = (20,7))  #figsize
x = lemon_w['month']
x_r = lemon_R['month']

y_w = lemon_w['average']
y_R = lemon_R['average']
plt.plot(x, y_w, color = 'b', label = 'Wolesale' )
plt.plot(x_r, y_R, color = 'r', label = 'Retail' )

plt.legend()  # add legend
plt.xticks(rotation='vertical')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('2018-2020 Wholesale-Retail Price for Lemon')
plt.show()


# In[141]:


fig = plt.figure(figsize = (20,7))  #figsize
ax = sns.catplot(x="month", y="average", hue="priceType", data=lemon,
                height=8,kind="bar", palette="muted")
ax.set_ylabels("Average Price")
ax.set_xlabels("Price comparison Month-PriceType")


# In[144]:


lemon_w

fig = plt.figure(figsize = (20,7))  #figsize
ax = sns.catplot(x="month", y="average", hue="year", data=lemon_w,
                height=8,kind="bar", palette="muted")
ax.set_ylabels("Average Price")
ax.set_xlabels("Lemon Wholesale Price comparison")


# In[145]:



lemon_R

fig = plt.figure(figsize = (20,7))  #figsize
ax = sns.catplot(x="month", y="average", hue="year", data=lemon_R,
                height=8,kind="bar", palette="muted")
ax.set_ylabels("Average Price")
ax.set_xlabels("Lemon Retail Price Comparison")


# In[190]:


vegetable = df
vegetable['season']=''
def get_season(val):
    result = []
    
    for m in val:
        if m == 2:
            season = 'Winter'
        elif m == 1:
            season = 'Winter'
        elif m== 12:
            season = 'Winter'

        elif m == 3:
            season = 'Spring'
        elif m == 4:
            season = 'Spring'
        elif m== 5:
            season = 'Spring'
        elif m == 6:
            season = 'Summer'
        elif m == 7:
            season = 'Summer'
        elif m== 8:
            season = 'Summer'
        elif m == 9:
            season = 'Autumn'
        elif m == 10:
            season = 'Autumn'
        elif m== 11:
            season = 'Autumn'
        result.append(season)
    return result

vegetable['season'] = get_season(vegetable['month'].tolist())
vegetable.head()


# In[192]:


vegetable[vegetable['season']=='Autumn']


# In[ ]:




