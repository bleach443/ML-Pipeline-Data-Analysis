#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def save_txt_file(results):
    myfile = open('results.txt','w')
    myfile.write(results)
    myfile.close()
    return myfile

