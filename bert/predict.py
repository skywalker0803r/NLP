import joblib
from utils import FakeNewsDataset,get_predictions,create_mini_batch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import pandas as pd

# load trained model
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = joblib.load('checkpoint/bert_model.pkl')
model.eval()

# load testset
testset = FakeNewsDataset("test", tokenizer=tokenizer)
testloader = DataLoader(testset, batch_size=256, collate_fn=create_mini_batch)

# get predictions
predictions , confidence = get_predictions(model, testloader)

# 用來將預測的 label id 轉回 label 文字
index_map = {v: k for k, v in testset.label_map.items()}

# 生成 Kaggle 繳交檔案
df = pd.DataFrame({"Category": predictions.tolist()})
df['Category'] = df.Category.apply(lambda x: index_map[x])
df['Confidence'] = confidence.detach().cpu().numpy()
df_pred = pd.concat([testset.df.loc[:, ["Id"]], df.loc[:, ['Category','Confidence']]], axis=1)
print(df_pred.head())

# 上傳到 Kaggle 網站 不需要 Confidence
df_pred = df_pred.drop('Confidence',axis=1)
df_pred.to_csv('output/bert_1_prec_training_samples.csv', index=False)
print('save done!')