from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd


df = pd.read_csv("forestfiresc.csv")
df = df.drop(columns=['day','month','year'])
# Map 'fire' to 1 and 'not fire' to 0
df['Classes'] = df['Classes'].map({'fire': 1, 'not fire': 0})

# Remove trailing spaces from column names
df.columns = df.columns.str.strip()

target_col = "Classes"
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
features = list(X_train.columns)

cb_model = CatBoostClassifier(iterations=230,random_state=2023,eval_metric="F1")

cb_model.fit(X_train, y_train, cat_features=["Temperature","RH","Ws"], plot= True, eval_set=(X_test,y_test))
y_pred = cb_model.predict(X_test)

f1 = f1_score(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)

print(f1)
print(accuracy)