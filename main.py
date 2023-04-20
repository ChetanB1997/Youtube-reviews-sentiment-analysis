from sentiment_a import *
from train import model
from scrap import ScrapComment

########### enter url ########
url="https://www.youtube.com/watch?v=mBcBoGhFndY"
df =   ScrapComment(url)
#df.to_csv('comment.csv')
print('***********data scrapping completed***********')

###for csv file##############
#file_path='comment.csv'
#df=datatranformation(file_path)

#############################
labeldata =  labeling(df)

proc_data = processing(labeldata)
Xdata,Ydata = sampling(proc_data)

####training
cm,accuracy= model(Xdata,Ydata)
print(cm)
#print(accuracy*100, '%')
print('Accuracy:',accuracy*100,'%')

print('done')