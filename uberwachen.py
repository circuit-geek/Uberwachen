import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("datasheet_final.csv")

df = df.dropna(axis = 1, how = 'all')
new_df=df.drop(['weight','height','heart_pulse','muscle_mass','hydration','bone_mass','pulse_wave_velocity','systolic','diastolic','TEMP','SPO2'],axis=1)

df=new_df.values

X= df[:,0:4]
Y= df[:,4]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)
score = log_reg.score(X_test, y_test)
print("Accuracy is:", score)


pickle.dump(log_reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


