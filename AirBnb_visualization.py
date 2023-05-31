


# -*- coding: utf-8 -*-
"""

"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib inline
import seaborn as sns
import os


from io import BytesIO   
from PIL import Image


os.chdir("C:/Users/..../Downloads")

airbnb=pd.read_csv("AB_NYC_2019.csv")


sub_6=airbnb[airbnb.price < 500]
#viz_4=sub_6.plot(kind='scatter', x='longitude', y='latitude', label='availability_365', c='price',
#                  cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, figsize=(10,8))
#iz_4.legend()


import urllib
#initializing the figure size
plt.figure(figsize=(10,8))
#loading the png NYC image found on Google and saving to my local folder along with the project
i=urllib.request.urlopen('https://upload.wikimedia.org/wikipedia/commons/e/ec/Neighbourhoods_New_York_City_Map.PNG')


###加入這條(避免類似彈跳視窗的東西
b=BytesIO(i.read())

#NY_IMG=Image.open(b)

nyc_img=plt.imread(b)
#scaling the image based on the latitude and longitude max and mins for proper output
plt.imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])
ax=plt.gca()
#using scatterplot again
# =============================================================================
sub_6.plot(kind='scatter', x='longitude', y='latitude', label='availability_365', c='price', ax=ax, 
            cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, zorder=5)

plt.legend()
plt.show()
# =============================================================================
