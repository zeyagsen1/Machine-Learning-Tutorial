import numpy as np
import pandas as pd
##if you say header=None it is not gonna determine the first row as header
##if not it will set first row of the data as headers(labels)
## a=pd.read_csv('countries of the world.csv',header=None)

#df=pd.read_csv('world_population.csv')

##if we say header=None then we can assign header saying names as below
#a=pd.read_csv('countries of the world.csv',header=None,names=["country","region"])

#pd.set_option("display.max.rows",235)
#a.info()
##print(a.shape) ##returns rows and columns number
##print(a['Country'])##returns spesific column
##print(a.loc[224])
##print(a.iloc[224])##integer location
pd.set_option('display.max.columns',100)
#print(df["Rank"]) this returns spesific column

##print(df[df["Rank"]<10]) but this returns all df where rank is less than 10
##countries=["Turkey","Brazil"]

##print(df[df['Country'].isin(countries)])
##print(df[df["Country"].str.contains("United")])
##both code above do the same thing

##df2=df.set_index("Country")##created df2 that orders the tabel according to country names. instead default indexing 0,1,2...
##df3=pd.read_csv('world_population.csv',index_col="Country")
##both code snippets above do the same thing

##df.reset_index(inplace=True) used to reset indexing things
##print(df2.filter(items=["Continent","CCA3"]))
##print(df.filter(items=["Continent","CCA3"]))
##print(df[["Continent","CCA3"]])
##three code lines above do the same thing
##but if you use filter you can also search row items
##not only columns
##print(df2.filter(items=["Zimbabwe"],axis=0))
#print(df2.loc['United States'])
#print(df2.iloc[3])
## loc and iloc search items on the rows
## iloc searchs according to index loc string
##print(df[df['Rank']<10].sort_values(by=['Continent','Country'],ascending=[True,True]))

##print(df.set_index(["Continent","Country"]).sort_index(ascending=[True,False]))

