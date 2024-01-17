!pip install simpletransformers
!pip install pandas
import numpy as np
import pandas as pd
import os
import shutil
os.getcwd()

from simpletransformers.classification import ClassificationModel


# Create a TransformerModel
model = ClassificationModel('bert', 'bert-base-uncased', num_labels=3, args={'reprocess_input_data': True, 
                                                                           'overwrite_output_dir': True},use_cuda=False)

df=pd.read_csv("Tweets.csv")
df.head()

df.shape
df.info()
 
df.isnull().sum()
df.airline_sentiment.unique()
df['airline_sentiment'].value_counts()

df1 = df[['text', 'airline_sentiment']].copy()
df1.head()

from sklearn.model_selection import train_test_split
train,test = train_test_split(df1, test_size = 0.2)

def making_label(st):
    if(st=='positive'):
        return 2
    elif(st=='neutral'):
        return 1
    else:
        return 0
    
train['label'] = train['airline_sentiment'].apply(making_label)
test['label'] = test['airline_sentiment'].apply(making_label)

print(train.shape)

train_df = pd.DataFrame({
    'text': train['text'][:3000].replace(r'\n', ' ', regex=True),
    'label': train['label'][:3000]
})
train_df['label'].value_counts()

test_df = pd.DataFrame({
    'text': test['text'][-400:].replace(r'\n', ' ', regex=True),
    'label': test['label'][-400:]
})

model.train_model(train_df)

result, model_outputs, wrong_predictions = model.eval_model(test_df)

result

model_outputs

lst = []
for arr in model_outputs:
    lst.append(np.argmax(arr))

true = test_df['label'].tolist()
predicted = lst   

import sklearn
mat = sklearn.metrics.confusion_matrix(true , predicted)
mat

sklearn.metrics.classification_report(true,predicted,target_names=['positive','neutral','negative'])
sklearn.metrics.accuracy_score(true,predicted)

def get_result(statement):
    result = model.predict([statement])
    pos = np.where(result[1][0] == np.amax(result[1][0]))
    pos = int(pos[0])
    sentiment_dict = {2:'positive',0:'negative',1:'neutral'}
    print(sentiment_dict[pos])
    return sentiment_dict[pos]

#Prediction
get_result("Seat was not comfortable")
get_result("food was excellent")
get_result("where is the flight rn?")

df_new=pd.read_csv("Tweets.csv"")
df_new.head()
                   
df_new1 = df_new[['text']].copy()
df_new1.head()

df_new2 = df_new1[:1000]
df_new2.head()

df_new2['sentiment'] = df_new2['text'].apply(get_result)
df_new2.head(10)

df_new2.sentiment.value_counts()
