from sentiment import *
from train import model
from scrap import ScrapComment
import os
import pandas as pd
import matplotlib.pyplot as plt

######################################################################################################
######## here the url given by user saved in csv is loaded to scrap############################

url = pd.read_csv('url.csv')

#url = 'https://www.youtube.com/watch?v=fmBuYaEB3pU'
df =   ScrapComment(url)

#################################################################################################
################# the scrapped comments are saved###############################################

file_path = 'data/scrap_comments.csv'
print('***********data scrapping completed***********')
df = datatranformation(file_path)


#################################################################################################
labeldata =  labeling(df)
print(len(labeldata))
proc_data = processing(labeldata)
Xdata,Ydata = sampling(proc_data)
cm=[[0,0,0],[0,0,0],[0,0,0]]
cm = np.array(cm)
print("*",cm)
####training
cm,accuracy= model(Xdata,Ydata)
print(cm)

#print(accuracy*100, '%')
#print('Accuracy:',accuracy*100,'%')
if cm[0][0] >0:
        positive_score= cm[0][0]
else:
        positive_score= 0

if cm[1][1]>0:                       
        neutral_score=cm[1][1]
else:
        neutral_score=0        

if cm[2][2]:        
        negative_score=cm[2][2]
else:
        negative_score=0

print("***",cm)

save_path = 'result'
filename = 'saved_image.jpg'

left = [1, 2, 3]
height = [positive_score,neutral_score,negative_score]
tick_label = ['Positive', 'Neutral', 'Negative']
plt.bar(left, height, tick_label = tick_label,
        width = 0.8, color = [ 'green','yellow','red'])
plt.ylabel('Scores (percentage)')
plt.title('Video Comments Analysis')
plt.savefig(os.path.join(save_path, filename))

# print('done')