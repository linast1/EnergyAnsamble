import pandas as pd
import numpy as np
import os
import json
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

def add_uk_holidays(df):
    # Importuoti JK švenčių biblioteką
    from holidays import UK
    
    # Sukurti JK švenčių objektą
    uk_holidays = UK()
    
    # Sukurti švenčių stulpelį (1 jei šventė, 0 jei ne)
    df['is_holiday'] = df['timestamp'].dt.date.apply(lambda x: 1 if x in uk_holidays else 0)
    
    return df

def timestamp_transform(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['weekday'] = df['timestamp'].dt.dayofweek
    df['day_night'] = np.where(((df['hour'] >= 23) | (df['hour'] < 7)), 0, 1)
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    
    # Pridėti JK šventes
    df = add_uk_holidays(df)
    
    return df

def add_missing_readings(df, is_train):
    # Kopijuoti originalią duomenų lentelę, kad išvengtume modifikacijų
    df = df.copy()
    
    # Inicializuoti trūkstamų duomenų kaukę ir įrašus
    missing_mask = df['meter_reading'].isna()
    new_records = []
    
    # Ieškome trūkstamų reikšmių
    medians = df.groupby(['building_id', 'weekday', 'hour'])['meter_reading'].transform('median')
    df.loc[missing_mask, 'meter_reading'] = medians[missing_mask]
    new_value_mask = missing_mask & df['meter_reading'].notna()
    
    # Įrašyti papildymus
    for idx in df[new_value_mask].index:
        new_records.append({
            'building_id': df.loc[idx, 'building_id'],
            'timestamp': df.loc[idx, 'timestamp'],
            'original_value': None,
            'new_value': df.loc[idx, 'meter_reading']
        })
    
    return df

# Vykdomas kumuliatyvios sumos apskaičiavimas
def detect_cusum(data, drift=0.03):
    # Konvertuojame duomenis į masyvą
    data = np.array(data)

    # Sukuriami masyvai teigiamiems ir neigiamiems
    # komuliatyvios sumos pokyčiams
    pos_cusum = np.zeros(len(data))
    neg_cusum = np.zeros(len(data))
    
    # Skaičiuojamos teigiamos ir neigiamos kumuliatyvios
    # sumos vertės
    for i in range(1, len(data)):
        pos_cusum[i] = max(0, pos_cusum[i-1] + data[i] - drift)
        neg_cusum[i] = max(0, neg_cusum[i-1] - data[i] - drift)

    # Sujungiame teigiamas ir neigiamas reikšmes į vieną metriką    
    combined_cusum = (1.3 * pos_cusum + 0.7 * neg_cusum)

    # Normalizuojame rezultatus 
    if combined_cusum.max() > 0:
        combined_cusum = combined_cusum / combined_cusum.max()
    
    return combined_cusum


# Taikomas atsitiktinio miško anomalijų aptikimo metodas
def random_forest_detector(X_train, y_train, X_test):

    # Sukuriamas atsitiktinio miško klasifikatorius
    rf = RandomForestClassifier(
        n_estimators=150,          # Nustatomas medžių kiekis
        class_weight= 'balanced',  # Nustatomas klasių svoris duomenų pasiskirstymui
        random_state=96
    )
    
    # Atliekamas apmokymas
    rf.fit(X_train, y_train)
    
    # Atliekamas prognozavimas
    probabilities = rf.predict_proba(X_test)[:, 1]
    return probabilities, rf

# Ansamblio algoritmo anomalijų vertinimo metodas
def combine_ensemble(cusum_scores, rf_scores, threshold=1):
    # Patikrinti skirtumų sąlygą |B₁ - B₂| > C
    score_difference = np.abs(cusum_scores - rf_scores)
    difference_anomalies = score_difference > threshold
    
    # Patikrinti sumos sąlygą B₁ + B₂ > C
    score_sum = cusum_scores + rf_scores
    sum_anomalies = score_sum > threshold
    
    # Sujungti abi sąlygas - jei bet kuri sąlyga tenkinama, pažymėti kaip anomaliją
    final_anomalies = np.logical_or(difference_anomalies, sum_anomalies).astype(int)
    
    return final_anomalies

# Ištrauktiami kontekstiniai požymiai
def extract_contextual_features(df):

    # Grupuoti pagal pastatą ir laiko požymius ieškant modelių
    result = df.copy()
    
    # Pridėti slankiojančius statistinius duomenis neįprastoms sekoms užfiksuoti
    result['rolling_mean_3h'] = df.groupby('building_id')['meter_reading'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    result['rolling_std_3h'] = df.groupby('building_id')['meter_reading'].transform(lambda x: x.rolling(3, min_periods=1).std())
    result['z_score_3h'] = (result['meter_reading'] - result['rolling_mean_3h']) / (result['rolling_std_3h'] + 0.001)
    
    # Pridėti naudojimo pasikeitimo greitį ir pagreitį
    result['usage_change'] = result.groupby('building_id')['meter_reading'].diff()
    result['usage_acceleration'] = result.groupby('building_id')['usage_change'].diff()
    
    # Apskaičiuoti konkrečiam pastatui būdingus modelių nuokrypius
    building_hour_means = df.groupby(['building_id', 'hour'])['meter_reading'].mean().reset_index()
    building_hour_means.columns = ['building_id', 'hour', 'hour_typical_usage']
    result = pd.merge(result, building_hour_means, on=['building_id', 'hour'], how='left')
    
    # Apskaičiuoti nuokrypį nuo tipinio modelio
    result['pattern_deviation'] = np.abs(result['meter_reading'] - result['hour_typical_usage'])
    result['pattern_deviation_ratio'] = result['pattern_deviation'] / (result['hour_typical_usage'] + 0.001)
    result = result.fillna(0)
    
    return result

# Anomalijų klasivikavimas 
#    1 - Sistemos/kibernetinės anomalijos 
#    0 - Žmogaus elgsenos variacijos
def classify_anomaly_types(anomaly_data, model, feature_columns):

    # Ištraukiama atsitiktinio miško algoritmo duomenų požymių svarba
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nDuomenų požymių svarba:")
        for f in range(min(10, len(feature_columns))):
            idx = indices[f]
            if idx < len(feature_columns):
                print(f"{f+1}. {feature_columns[idx]} ({importances[idx]:.4f})")
    
    available_features = anomaly_data.columns.tolist()
    
    # Paskirstomos duomenų svarbos  
    human_behavior_features = [
        'day_night', 'hour', 'weekday', 'is_holiday',   # Laiko kontekstas
        'pattern_deviation_ratio',                      # Pasikeitimai laiko modelyje
        'rolling_mean_3h', 'z_score_3h'                 # Trumpalaikis kontekstas
    ]
    
    system_anomaly_features = [
        'usage_acceleration', 'usage_change',          # Greiti pokyčiai
        'pattern_deviation', 'z_score_3h',             # Absoliutūs nuokrypiai
        'rolling_std_3h'                               # Nestabilumas
    ]
    
    human_behavior_features = [f for f in human_behavior_features if f in available_features]
    system_anomaly_features = [f for f in system_anomaly_features if f in available_features]
    
    anomaly_type = np.zeros(len(anomaly_data))
    
    # Izoliavimo miško algoritmas su 2% užteršimo lygiu
    iso = IsolationForest(contamination=0.02, random_state=42)
    
    # DBSCAN žmogaus elgesio anomalijų klasterizavimui
    scaler = StandardScaler()
    
    if len(human_behavior_features) >= 2 and len(system_anomaly_features) >= 2:
        X_human = scaler.fit_transform(anomaly_data[human_behavior_features])
        X_system = scaler.fit_transform(anomaly_data[system_anomaly_features])
        
        # Pritaikyti DBSCAN žmogaus elgsenos modeliams
        human_clusters = DBSCAN(eps=0.5, min_samples=5).fit(X_human)
        # Pritaikyti Isolation Forest sistemos anomalijų aptikimui
        system_scores = iso.fit_predict(X_system)
        
        human_behavior = (human_clusters.labels_ != -1).astype(int)
        system_anomaly = (system_scores == -1).astype(int)
        
        anomaly_type = np.where(
            # Stiprus signalas iš Isolation Forest, rodantis sistemos anomaliją
            (system_anomaly == 1),
            1,
            anomaly_type
        )
        
        # Taisyklėmis pagrįstas patikrinimas
        if 'usage_acceleration' in available_features and 'z_score_3h' in available_features:
            # Labai ekstremalūs pasikeitimai greičiausiai yra sistemos anomalijos
            extreme_threshold = anomaly_data['z_score_3h'].abs().quantile(0.98)  # Tik ekstremaliausi 2%
            extreme_accel_threshold = anomaly_data['usage_acceleration'].abs().quantile(0.98)
            
            anomaly_type = np.where(
                # Nepaprastai didelis pagreitis IR ekstremalus z-įvertis
                (anomaly_data['usage_acceleration'].abs() > extreme_accel_threshold) & 
                (anomaly_data['z_score_3h'].abs() > extreme_threshold),
                1,
                anomaly_type
            )
        
        # Patikrinti modelius, kurie aiškiai rodo žmogaus elgseną
        if 'day_night' in available_features and 'is_weekend' in available_features and 'is_holiday' in available_features and 'z_score_3h' in available_features:
            anomaly_type = np.where(
                (anomaly_type == 1) & (system_anomaly == 1),
                1,
                # Žmogaus elgsena, jei tenkinoma betkuri iš šių sąlygų:
                np.where(
                    (human_behavior == 1) |  # Priklauso žmogaus elgsenos klasteriui
                    # Įprastomis aktyvumo valandomis su vidutiniu z-įverčiu
                    ((anomaly_data['day_night'] == 1) & (anomaly_data['z_score_3h'].abs() < 5)) |
                    # Savaitgalis su vidutiniu z-įverčiu
                    ((anomaly_data['is_weekend'] == 1) & (anomaly_data['z_score_3h'].abs() < 7)) |
                    # Šventė su vidutiniu z-įverčiu (leidžiamas didesnis nuokrypis švenčių dienomis)
                    ((anomaly_data['is_holiday'] == 1) & (anomaly_data['z_score_3h'].abs() < 8)),
                    0,
                    anomaly_type
                )
            )
        elif 'day_night' in available_features and 'z_score_3h' in available_features:
            anomaly_type = np.where(
                (anomaly_type == 1) & (system_anomaly == 1),
                1,
                # Žmogaus elgsenos aptikimas be savaitgalio informacijos
                np.where(
                    (human_behavior == 1) |  # Priklauso žmogaus elgsenos klasteriui
                    # Įprastomis aktyvumo valandomis su vidutiniu z-įverčiu
                    ((anomaly_data['day_night'] == 1) & (anomaly_data['z_score_3h'].abs() < 5)),
                    0,
                    anomaly_type
                )
            )
        
        # Papildomas patikrinimas: jei vartojimas atitinka modelį, panašų į buvusį elgesį
        if 'pattern_deviation_ratio' in available_features:
            anomaly_type = np.where(
                (anomaly_type == 1) & (system_anomaly == 1),
                1,
                np.where(
                    # Mažas santykinis nuokrypis nuo tipinio modelio rodo žmogaus elgseną
                    (anomaly_data['pattern_deviation_ratio'] < 1.5),  # Mažiau nei 150% nuokrypis nuo tipinio
                    0,
                    anomaly_type
                )
            )
    else:
        # Naudoti paprastą taisyklę, pagrįstą z-įverčiu
        if 'z_score_3h' in available_features:
            anomaly_type = np.where(
                np.abs(anomaly_data['z_score_3h']) > 3.0,  # Labai didelis nuokrypis
                1,
                0
            )
        else:
            anomaly_type = np.ones(len(anomaly_data))
    
    return anomaly_type



def main():
    # Skaityti duomenis iš CSV failų su konkrečiais stulpeliais
    train_data = pd.read_csv('train_features.csv', usecols=['building_id', 'timestamp', 'meter_reading', 'anomaly', 'air_temperature'])
    test_data = pd.read_csv('test_features.csv', usecols=['row_id', 'building_id', 'timestamp', 'meter_reading', 'air_temperature'])
    
    train_data = timestamp_transform(train_data)
    test_data = timestamp_transform(test_data)
    
    train_data = add_missing_readings(train_data, is_train=True)
    test_data = add_missing_readings(test_data, is_train=False)
    
    feature_columns = [
        'meter_reading',
        'hour', 'day', 'month', 'weekday', 'day_night',
        'building_id', 'air_temperature',
        'is_weekend', 'is_holiday'
    ]
    
    # VALIDACIJOS DALIS
    scaler_val = StandardScaler()
    X_train_full = scaler_val.fit_transform(train_data[feature_columns])
    y_train_full = train_data['anomaly']
    
    # Atskirti validacijos duomenis SU INDEKSAIS
    train_idx, val_idx = train_test_split(
        range(len(train_data)), test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    X_train_split = X_train_full[train_idx]
    X_val = X_train_full[val_idx]
    y_train_split = y_train_full.iloc[train_idx]
    y_val = y_train_full.iloc[val_idx]
    
    # CUSUM skaičiavimas tik validacijos duomenims
    val_meter_readings = train_data.iloc[val_idx]['meter_reading'].values
    cusum_scores_val = detect_cusum(val_meter_readings)
    
    # Apmokyti modelį su train split duomenimis
    rf_scores_val, rf_model_val = random_forest_detector(X_train_split, y_train_split, X_val)
    
    # Užtikrinti, kad dydžiai sutampa
    min_len = min(len(cusum_scores_val), len(rf_scores_val), len(y_val))
    cusum_scores_val = cusum_scores_val[:min_len]
    rf_scores_val = rf_scores_val[:min_len]
    y_val = y_val.iloc[:min_len]
    
    final_predictions_val = combine_ensemble(cusum_scores_val, rf_scores_val)
    
    # VALIDACIJOS METRIKOS
    auc_score = roc_auc_score(y_val, rf_scores_val)
    accuracy = accuracy_score(y_val, final_predictions_val)
    precision = precision_score(y_val, final_predictions_val)
    recall = recall_score(y_val, final_predictions_val)
    f1 = f1_score(y_val, final_predictions_val)
    
    print(f"\nKLASIFIKACIJOS METRIKOS:")
    print(f"AUC: {auc_score:.4f}")
    print(f"Tikslumas (Accuracy): {accuracy:.4f}")
    print(f"Mokymo tikslumas tikrai pozytivioms reikšmėms (true positives) {precision:.4f}")
    print(f"Jautrumas (Recall): {recall:.4f}")
    print(f"F1 balas: {f1:.4f}")

    # GALUTINIS MODELIS SU VISAIS TRAIN DUOMENIMIS
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data[feature_columns])
    X_test = scaler.transform(test_data[feature_columns])
    
    y_train = train_data['anomaly']
    
    # Naudojamas komuliatyvios sumos apskaičiavimas
    cusum_scores = detect_cusum(test_data['meter_reading'])
    
    # Naudojamas atsitiktinio miško algoritmas
    rf_scores, rf_model = random_forest_detector(X_train, y_train, X_test)
    
    # Bendrų anomalijų aptikimo rezultatas
    final_predictions = combine_ensemble(cusum_scores, rf_scores)

    # Išsaugoti bendrų anomalijų aptikimo rezultatus
    original_results = pd.DataFrame({
        'row_id': test_data['row_id'],
        'anomaly': final_predictions
    })
    original_results.to_csv('results_ensemble_original.csv', index=False)

    print(f"\nRastų bendrų anomalijų skaičius: {int(original_results['anomaly'].sum())}")
    
    # Apdoroti tik eilutes, pažymėtas kaip anomalijas pagal originalų algoritmą
    anomaly_indices = original_results[original_results['anomaly'] == 1].index
    anomaly_data = test_data.loc[anomaly_indices].copy()
    
    # Ištraukti kontekstinius požymius, padedančius klasifikuoti
    anomaly_data = extract_contextual_features(anomaly_data)
    
    # Inicializuojamas vizualizacijos objektas
    visualization_data = pd.DataFrame({
        'row_id': test_data['row_id'],
        'timestamp': test_data['timestamp'],
        'building_id': test_data['building_id'],
        'meter_reading': test_data['meter_reading'],
        'anomaly': 0,
        'anomaly_type': 'normal'
    })
    
    if len(anomaly_data) > 0:
        # Klasifikuoti anomalijas kaip sistemos/kibernetines (1) arba žmogaus elgseną (0)
        anomaly_types = classify_anomaly_types(anomaly_data, rf_model, feature_columns)
        
        # Sukuriama bendrųjų anomalijų rezultatų kopija
        enhanced_results = original_results.copy()
        
        anomaly_type_map = {}
        
        # Pažymėti žmogaus elgsenos anomalijas kaip 0 (ne anomalijos) ir sistemos anomalijas kaip 1
        # Indeksams, kurie yra anomaly_indices
        for i, idx in enumerate(anomaly_indices):
            row_id = test_data.loc[idx, 'row_id']
            if anomaly_types[i] == 0:  # Jei klasifikuojama kaip žmogaus elgsena
                enhanced_results.loc[enhanced_results['row_id'] == row_id, 'anomaly'] = 0
                anomaly_type_map[row_id] = 'human_behavior'
            else:  # Sistemos anomalija
                anomaly_type_map[row_id] = 'system/cyber'
        
        # Sukurti galutinius rezultatų DataFrame
        final_results = pd.DataFrame({
            'row_id': test_data['row_id'],
            'anomaly': enhanced_results['anomaly']
        })
        
        # Pridėti anomalijos tipo stulpelį
        final_results['anomaly_type'] = 'normal'
        
        # Atnaujinti anomalijų tipus pagal klasifikaciją
        for row_id, anomaly_type in anomaly_type_map.items():
            final_results.loc[final_results['row_id'] == row_id, 'anomaly_type'] = anomaly_type
        
        # Išsaugoti klasifikuotus rezultatus
        final_results.to_csv('results_ensemble_enhanced.csv', index=False)
        
        # Spausdinti statistiką
        system_anomalies = int((final_results['anomaly_type'] == 'system/cyber').sum())
        human_behavior_anomalies = int((final_results['anomaly_type'] == 'human_behavior').sum())
        
        print(f"\nPATOBULINTŲ REZULTATŲ SUVESTINĖ:")
        print(f"Originalios aptiktos anomalijos: {int(original_results['anomaly'].sum())}")
        print(f"Anomalijos, klasifikuotos kaip sistemos/kibernetinės: {system_anomalies}")
        print(f"Anomalijos, klasifikuotos kaip žmogaus elgsena: {human_behavior_anomalies}")
        print(f"Normalūs rodmenys: {int((final_results['anomaly_type'] == 'normal').sum())}")
        
        # Atnaujinti vizualizacijos duomenis su galutinėmis anomalijų klasifikacijomis
        visualization_data['anomaly'] = final_results['anomaly']
        visualization_data['anomaly_type'] = final_results['anomaly_type']
        
        os.makedirs('dashboard/public/data', exist_ok=True)
        
        visualization_data.to_csv('dashboard/public/data/visualization_data.csv', index=False)
        print("\nVizualizacijos duomenys išsaugoti public/data/visualization_data.csv")
        
        # Sukurti ir išsaugoti suvestinę statistiką valdymo sąsajai
        summary_stats = {
            'total_anomalies': int(original_results['anomaly'].sum()),
            'anomaly_percentage': round(100 * original_results['anomaly'].sum() / len(original_results), 2),
            'total_readings': len(original_results),
            'system_anomalies': system_anomalies,
            'human_behavior_anomalies': human_behavior_anomalies
        }
        
        # Išsaugoti statistiką atvaizdavimui
        with open('dashboard/public/data/summary_stats.json', 'w') as f:
            json.dump(summary_stats, f)
        print("Suvestinė statistika išsaugota dashboard/public/data/summary_stats.json")
    else:
        print("\nNerasta anomalijų klasifikavimui.")

if __name__ == "__main__":
    main()