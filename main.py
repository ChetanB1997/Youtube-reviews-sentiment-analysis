from sentiment_a import *
from train import model
from scrap import ScrapComment

import matplotlib.pyplot as plt

########### enter url ########
# url="https://www.youtube.com/watch?v=mBcBoGhFndY"
# df =   ScrapComment(url)
# f.to_csv('comment.csv')d
print('***********data scrapping completed***********')

###for csv file##############
file_path='comment.csv'
df=datatranformation(file_path)

#############################
labeldata =  labeling(df)

proc_data = processing(labeldata)
Xdata,Ydata = sampling(proc_data)

####training
cm,accuracy= model(Xdata,Ydata)
print(cm)
#print(accuracy*100, '%')
#print('Accuracy:',accuracy*100,'%')
positive_score= cm[0][0]
neutral_score=cm[1][1]
negative_score=cm[2][2]

left = [1, 2, 3]
height = [positive_score,neutral_score,negative_score]
tick_label = ['Positive', 'Neutral', 'Negative']
plt.bar(left, height, tick_label = tick_label,
        width = 0.8, color = [ 'green','yellow','red'])
plt.ylabel('Scores (percentage)')
plt.title('Video Comments Analysis')
plt.show()

print('done')