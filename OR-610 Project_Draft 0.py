#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil
import pandas as pd


train = os.path.join(os.path.expanduser('~'), 'Documents', 'Datasets','aptos2019-blindness-detection','train')
#test = os.path.join(os.path.expanduser('~'), 'Documents', 'Datasets','aptos2019-blindness-detection','test')


metaTrain = pd.read_csv(os.path.join(os.path.expanduser('~'), 'Documents', 'Datasets','aptos2019-blindness-detection','train.csv'))
#metaTest = pd.read_csv(os.path.join(os.path.expanduser('~'), 'Documents', 'Datasets','aptos2019-blindness-detection','test.csv'))

def mover(path, meta, className):
    meta = meta.id_code[meta.diagnosis==className]
    for i in meta:
        if not os.path.exists(os.path.join(path,str(className))):
            os.makedirs(os.path.join(path,str(className)))
        shutil.move(os.path.join(path,str(i)+'.png'),
                    os.path.join(path,str(className),str(i)+'.png')
                    )
   
for i in range(5):
    mover(train,metaTrain,i)
    #mover(test,metaTest,i)


# In[ ]:




