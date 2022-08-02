#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sqlite3
import sys
import pandas as pd
from pandas import DataFrame

#function to write the database file to a pandas dataframe
def sql_to_pd(path):
    connect = sqlite3.connect(str(path))
    data = pd.DataFrame(pd.read_sql("SELECT * FROM survive", connect))
    return data
