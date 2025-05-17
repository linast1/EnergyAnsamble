import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from xgboost import XGBClassifier
import datetime
import warnings
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, balanced_accuracy_score
)
warnings.filterwarnings('ignore')

# Matplotlibo nustatymai lietuvių kalbai
plt.rcParams['font.family'] = 'DejaVu Sans'

train_cols = ['building_id', 'timestamp', 'meter_reading', 'anomaly', 'air_temperature']
train = pd.read_csv('train_features.csv', usecols=train_cols)

test_cols = ['row_id', 'building_id', 'timestamp', 'meter_reading', 'air_temperature']
test = pd.read_csv('test_features.csv', usecols=test_cols)

print("\nTikrinamos trūkstamos reikšmės mokymo rinkinyje:")
for col in train.columns:
    k = train[col].isnull().sum()
    if k > 0:
        print(f'{col} : {k}')

print("\nTrūkstamos reikšmės anomalijų įrašuose:")
print(train[train['anomaly']==1].isnull().sum())

# Tikrinama, ar matuoklio rodmenys 1.0 dažnai yra anomalijos
print("\nAnomalijų pasiskirstymas, kai meter_reading=1.0:")
print(train[train['meter_reading']==1.0]['anomaly'].value_counts())


# Konvertuojama data į stulpelį iš laiko žymos
train['date'] = pd.to_datetime(train['timestamp']).dt.date
test['date'] = pd.to_datetime(test['timestamp']).dt.date

# Apskaičiuojamas dienos matuoklio rodmenų standartinis nuokrypis pagal pastatą
train['meterReadings_daily_std'] = train.groupby(['building_id','date'])['meter_reading'].transform('std')
test['meterReadings_daily_std'] = test.groupby(['building_id','date'])['meter_reading'].transform('std')

# Funkcija trūkstamų reikšmių užpildymui, remiantis pastato vidurkiu
def impute_nulls(data):
    # Užpildomos NaN reikšmės su vidutiniu meter_reading pagal pastato id
    mean_reading = data.groupby('building_id')['meter_reading'].mean()
    building_id = mean_reading.index
    values = mean_reading.values
    
    for i, idx in tqdm(enumerate(building_id), desc="Užpildomos trūkstamos reikšmės"):
        data.loc[data['building_id']==idx, 'meter_reading'] = data.loc[data['building_id']==idx, 'meter_reading'].fillna(values[i]) 
    
    return data

print("\nUžpildomos trūkstamos reikšmės mokymo duomenyse...")
train = impute_nulls(train)
print("\nUžpildomos trūkstamos reikšmės testavimo duomenyse...")
test = impute_nulls(test)

print("\nKuriami poslinkio požymiai...")
for shift_hours in tqdm([-1, 1, -24, 24]):
    # Mokymo duomenims
    meter_reading_shift = train[['building_id', 'timestamp', 'meter_reading']].copy()
    meter_reading_shift['timestamp'] = pd.to_datetime(meter_reading_shift['timestamp']) + datetime.timedelta(hours=shift_hours)
    meter_reading_shift['timestamp'] = meter_reading_shift['timestamp'].astype(str)
    meter_reading_shift = meter_reading_shift.rename(columns={'meter_reading': f'meter_reading_lag_{shift_hours}'})
    train = train.merge(meter_reading_shift, on=['building_id', 'timestamp'], how='left')
    
    # Apskaičiuojamas skirtumas
    train[f'diff_lag_{shift_hours}'] = train[f'meter_reading_lag_{shift_hours}'] - train['meter_reading']
    
    # Testavimo duomenims
    meter_reading_shift = test[['building_id', 'timestamp', 'meter_reading']].copy()
    meter_reading_shift['timestamp'] = pd.to_datetime(meter_reading_shift['timestamp']) + datetime.timedelta(hours=shift_hours)
    meter_reading_shift['timestamp'] = meter_reading_shift['timestamp'].astype(str)
    meter_reading_shift = meter_reading_shift.rename(columns={'meter_reading': f'meter_reading_lag_{shift_hours}'})
    test = test.merge(meter_reading_shift, on=['building_id', 'timestamp'], how='left')
    
    # Apskaičiuojamas skirtumas
    test[f'diff_lag_{shift_hours}'] = test[f'meter_reading_lag_{shift_hours}'] - test['meter_reading']

# Pridedamos valandos, savaitės dienų požymiai
train['hour'] = pd.to_datetime(train['timestamp']).dt.hour
train['day_of_week'] = pd.to_datetime(train['timestamp']).dt.dayofweek
test['hour'] = pd.to_datetime(test['timestamp']).dt.hour
test['day_of_week'] = pd.to_datetime(test['timestamp']).dt.dayofweek

# Išsaugomi row_id vėlesniam naudojimui
drop_cols = ['date', 'meterReadings_daily_std', 'timestamp'] + [f'meter_reading_lag_{shift}' for shift in [-1, 1, -24, 24]]
train = train.drop(drop_cols, axis=1, errors='ignore')
test = test.drop(drop_cols, axis=1, errors='ignore')

# Užpildomos likusios NaN reikšmės
row_ids = test['row_id'].copy()
test = test.drop('row_id', axis=1)

# Užpildome likusias NaN reikšmes
train = train.fillna(0)
test = test.fillna(0)

# Tvarkomas klasių disbalansas
neg = train[train['anomaly'] == 0]
pos = train[train['anomaly'] == 1]

print(f"\nKlasių pasiskirstymas - Normalūs: {neg.shape[0]}, Anomalijos: {pos.shape[0]}")

# Sukuriamas subalansuotas duomenų rinkinys
pos_count = pos.shape[0]
negs1 = neg.sample(n=pos_count, random_state=10)
negs2 = neg.sample(n=pos_count, random_state=20)
df_balanced = pd.concat([negs1, pos, negs2, pos], axis=0)
print(f"Subalansuoto rinkinio forma: {df_balanced.shape}")

# Padalijami duomenys mokymui ir testavimui
features = df_balanced.drop(['anomaly'], axis=1)
target = df_balanced['anomaly']

# Padalijame duomenis mokymui ir testavimui
X_train, X_val, y_train, y_val = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target
)

# Požymių normalizavimas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_scaled = scaler.transform(test)

print("\nApmokomas XGBoost modelis...")
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=200)
model.fit(X_train_scaled, y_train)

# Modelio vertinimas
train_pred = model.predict_proba(X_train_scaled)[:,1]
val_pred = model.predict_proba(X_val_scaled)[:,1]

train_auc = metrics.roc_auc_score(y_train, train_pred)
val_auc = metrics.roc_auc_score(y_val, val_pred)

# 1. PAGRINDINĖS KLASIFIKACIJOS METRIKOS
print("\n" + "="*50)
print("PAGRINDINĖS KLASIFIKACIJOS METRIKOS")
print("="*50)

# Konvertuojame tikimybes į dvejetaines prognozes
threshold = 0.5
train_pred_binary = (train_pred > threshold).astype(int)
val_pred_binary = (val_pred > threshold).astype(int)

# Accuracy (tikslumas)
train_accuracy = accuracy_score(y_train, train_pred_binary)
val_accuracy = accuracy_score(y_val, val_pred_binary)
print(f'Mokymo tikslumas: {train_accuracy:.4f}')
print(f'Testavimo tikslumas: {val_accuracy:.4f}')

# Balanced Accuracy (subalansuotas tikslumas)
train_balanced_acc = balanced_accuracy_score(y_train, train_pred_binary)
val_balanced_acc = balanced_accuracy_score(y_val, val_pred_binary)
print(f'Mokymo subalansuotas tikslumas: {train_balanced_acc:.4f}')
print(f'Testavimo subalansuotas tikslumas: {val_balanced_acc:.4f}')

# Precision (tikslumas pozityviems)
train_precision = precision_score(y_train, train_pred_binary)
val_precision = precision_score(y_val, val_pred_binary)
print(f'Mokymo tikslumas tikrai pozytivioms reikšmėms (true positives): {train_precision:.4f}')
print(f'Testavimo tikslumas tikrai pozytivioms reikšmėms (true positives): {val_precision:.4f}')

# Recall (jautrumas)
train_recall = recall_score(y_train, train_pred_binary)
val_recall = recall_score(y_val, val_pred_binary)
print(f'Mokymo jautrumas: {train_recall:.4f}')
print(f'Testavimo jautrumas: {val_recall:.4f}')

# F1-score
train_f1 = f1_score(y_train, train_pred_binary)
val_f1 = f1_score(y_val, val_pred_binary)
print(f'Mokymo F1-score: {train_f1:.4f}')
print(f'Testavimo F1-score: {val_f1:.4f}')

# Average Precision Score (PR AUC)
train_ap = average_precision_score(y_train, train_pred)
val_ap = average_precision_score(y_val, val_pred)
print(f'Mokymo PR AUC: {train_ap:.4f}')
print(f'Testavimo PR AUC: {val_ap:.4f}')

# Spausdinama AUC vertė
print(f'XGB Mokymo AUC: {train_auc:.4f}')
print(f'XGB Testavimo AUC: {val_auc:.4f}')

print("\nGeneruojamos prognozės...")
pred_probs = model.predict_proba(test_scaled)[:,1]

# Konvertuojamos tikimybės į dvejetaines prognozes (0 arba 1)
threshold = 0.5
predictions = (pred_probs > threshold).astype(int)

# Tikrinamos prognozės, kai meter_reading == 1.0
meter_reading_1 = test['meter_reading'] == 1.0
if meter_reading_1.sum() > 0:
    print(f"Testavimo pavyzdžių, kur meter_reading=1.0, skaičius: {meter_reading_1.sum()}")
    print("Nustatoma anomaly=1 šiems atvejams...")
    predictions[meter_reading_1.values] = 1

# Sukuriame pateikimo failą
submission = pd.DataFrame({
    'row_id': row_ids,
    'anomaly': predictions
})

# Išsaugomas pateikimo failas
submission.to_csv('submission_xgb.csv', index=False)
print("\nPateikimo failas sėkmingai sukurtas!")
print("\nPirmos 10 pateikimo failo eilučių:")
print(submission.head(10))

# Braižomas požymių svarbos grafikas
plt.figure(figsize=(12, 8))
feature_imp = pd.DataFrame({
    'požymis': X_train.columns,
    'svarba': model.feature_importances_
})
feature_imp = feature_imp.sort_values('svarba', ascending=False)
sns.barplot(x='svarba', y='požymis', data=feature_imp.head(15))
plt.title('XGBoost požymių svarba (Top 15)')
plt.xlabel('Svarba')
plt.ylabel('Požymis')
plt.tight_layout()
plt.savefig('xgb_požymių_svarba.png')
print("\nPožymių svarbos grafikas išsaugotas kaip 'xgb_požymių_svarba.png'")