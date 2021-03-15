import os
import zipfile
import pandas as pd

# 1. type <kaggle competitions download -c fake-news-pair-classification-challenge> in cmd

# 2. extract data
file_path = 'fake-news-pair-classification-challenge.zip'
zf = zipfile.ZipFile(file_path,mode='r')
zf.extractall(path='./input')
zf.close()

# 3.create trainset
df_train = pd.read_csv('input/train.csv')
empty_title = ((df_train['title2_zh'].isnull()) \
               | (df_train['title1_zh'].isnull()) \
               | (df_train['title2_zh'] == '') \
               | (df_train['title2_zh'] == '0'))
df_train = df_train[~empty_title]
# 剔除過長的樣本以避免 BERT 無法將整個輸入序列放入記憶體不多的 GPU
MAX_LENGTH = 30
df_train = df_train[~(df_train.title1_zh.apply(lambda x : len(x)) > MAX_LENGTH)]
df_train = df_train[~(df_train.title2_zh.apply(lambda x : len(x)) > MAX_LENGTH)]
# 去除不必要的欄位並重新命名兩標題的欄位名
df_train = df_train.reset_index()
df_train = df_train.loc[:, ['title1_zh', 'title2_zh', 'label']]
df_train.columns = ['text_a', 'text_b', 'label']
# idempotence, 將處理結果另存成 tsv 供 PyTorch 使用
df_train.to_csv("input/train.tsv", sep="\t", index=False)
print(df_train.shape)

# 4. create testset
df_test = pd.read_csv('input/test.csv')
df_test = df_test.loc[:, ["title1_zh", "title2_zh", "id"]]
df_test.columns = ["text_a", "text_b", "Id"]
df_test.to_csv("input/test.tsv", sep="\t", index=False)
print(df_test.shape)
