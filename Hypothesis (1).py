#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as mlt
import scipy .stats as stats


# In[2]:


df=pd.read_csv("Cutlets.csv")


# In[3]:


df


# In[4]:


df.describe()


# In[5]:


import warnings
warnings .filterwarnings('ignore')


# In[6]:


pip install chart-studio


# In[7]:


import statsmodels.stats.descriptivestats as sd 


# In[8]:


import chart_studio as cs


# In[9]:


#pip install plotly --upgrade


# In[10]:


import chart_studio.plotly as py
import plotly.graph_objects as go


# In[11]:


df.mean()


# In[12]:


df.std()


# In[13]:


UnitA =pd.Series(df.iloc[:,0])
UnitA


# In[14]:


UnitB =pd.Series(df.iloc[:,1])
UnitB


# In[15]:


p_value=stats.ttest_ind(UnitA,UnitB)
p_value


# In[16]:


# 2. Minitab File: LabTAT.mtw
import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import f_oneway as FO
import statsmodels.api as sm 
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt 


# In[17]:


import warnings
warnings .filterwarnings('ignore')


# In[18]:


lab=pd.read_csv("LabTAT.csv")
lab


# In[19]:


lab.count()


# In[20]:


print('The total number of rows in the dataset:', lab.size) 


# In[21]:


print (lab)


# In[22]:


lab.describe()


# In[23]:


Laboratory1=pd.Series(lab.iloc[:,0])
Laboratory1


# In[24]:


Laboratory2=pd.Series(lab.iloc[:,1])
Laboratory2


# In[25]:


Laboratory3=pd.Series(lab.iloc[:,2])
Laboratory3


# In[26]:


Laboratory4=pd.Series(lab.iloc[:,3])
Laboratory4


# In[27]:


# let us consider initially determining the confidence level of 95%, which also implies that we will accept only an error rate of 5%.


# In[28]:


FO(Laboratory1, Laboratory2, Laboratory3, Laboratory4)  


# In[29]:


# 3.Buyer Ratio 


# In[30]:


import pandas as pd
import numpy as np
import scipy 
from scipy import stats
import statsmodels.stats.descriptivestats as sd
from statsmodels.formula.api import ols
import chart_studio as cs
import seaborn as sns


# In[31]:


cof=pd.read_csv("BuyerRatio.csv")
print(cof.head()) 


# In[32]:


print('The total number of rows in the dataset:', cof.size)  


# In[33]:


cof.rename(columns={'Observed Values': 'obv'}, inplace=True)


# In[34]:


cof.obv[cof['obv'] == 'Males'] = 1
cof.obv[cof['obv'] == 'Females'] = 2
cof


# In[35]:


Chisquares_results=scipy.stats.chi2_contingency(cof)
Chisquares_results


# In[36]:


Chi_square=[['','Test Statistic','p-value'],['Sample Data',Chisquares_results[0],Chisquares_results[1]]]
Chi_square


# In[37]:


chisample_results=FF.create_table(Chi_square,index=True)
chisample_results 


# In[80]:


# Customer order form.
import pandas as pd
import numpy as np
import scipy 
from scipy import stats
import statsmodels.stats.descriptivestats as sd
from statsmodels.formula.api import ols
import chart_studio as cs
import seaborn as sns


# In[81]:


cof=pd.read_csv("Costomer+OrderForm.csv")
cof


# In[82]:


cof.count()


# In[83]:


cof.describe()


# In[84]:


cof2 = cof.groupby(['Phillippines']).size().reset_index(name='counts')
print(cof2)


# In[85]:


cof3=cof.groupby(['Indonesia']).size().reset_index(name='counts')
print(cof3)


# In[86]:


cof4=cof.groupby(['Malta']).size().reset_index(name='counts')
print(cof4)


# In[87]:


cof5=cof.groupby(['India']).size().reset_index(name='counts')
print(cof5)


# In[90]:


Error= ['Defective', 'Error free',]
Phillippines=['271','29']
Indonesia=['267','33']
Malta=['269','31']
India=['280','20']
list_of_tuples = list(zip(Error,Phillippines,Indonesia,Malta,India))
list_of_tuples


# In[91]:


df = pd.DataFrame(list_of_tuples, columns = ['Error','Phillippines','Indonesia','Malta','India'])
df


# In[96]:


print (df.dtypes)


# In[98]:


df.columns="Error","Phillippines","Indonesia","Malta","India"
df.columns


# In[99]:





# In[ ]:




