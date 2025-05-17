import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import gc


# Nustatomas atsitiktinių "sėklos" skaičių generavimas
np.random.seed(42)
tf.random.set_seed(42)

# Funkcija duomenų įkėlimui ir apdorojimui
def load_data(file_path, is_train=True):

    if is_train:

        data = pd.read_csv(file_path)
        data = data[['building_id', 'timestamp', 'meter_reading', 'anomaly', 'air_temperature']]
        print(f"Anomalijų pasiskirstymas mokymo duomenyse: {data['anomaly'].value_counts()}")
    else:
        data = pd.read_csv(file_path)
        data = data[['row_id', 'building_id', 'timestamp', 'meter_reading', 'air_temperature']]
    
    # Konvertuojama laiko žyma į datą
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    data['hour'] = data['timestamp'].dt.hour
    data['day'] = data['timestamp'].dt.day
    data['month'] = data['timestamp'].dt.month
    data['weekday'] = data['timestamp'].dt.weekday
    data['is_weekend'] = (data['weekday'] >= 5).astype(int)
    data['day_of_year'] = data['timestamp'].dt.dayofyear
    data['week_of_year'] = data['timestamp'].dt.isocalendar().week
    data['time_of_day'] = pd.cut(data['hour'], 
                                bins=[0, 6, 12, 18, 24], 
                                labels=[0, 1, 2, 3],  # naktis, rytas, diena, vakaras
                                include_lowest=True).astype(int)
    
    # Fiksuojamas darbo laikas ir ne darbo laikas
    data['is_business_hours'] = ((data['hour'] >= 8) & (data['hour'] <= 18) & 
                                (data['weekday'] < 5)).astype(int)
    
    print(f"NaN reikšmės prieš valymą: {data.isna().sum().sum()}")
    
    for col in data.columns:
        if col != 'timestamp' and data[col].dtype in ['float64', 'int64']:
            data[col] = data[col].replace([np.inf, -np.inf], np.nan)
            
            # Skaitiniams stulpeliams, užpildoma su stulpelio mediana
            if data[col].isna().any():
                # Grupuotiems duomenims, pvz., skaitiklio rodmenims, užpildoma grupėse
                if col in ['meter_reading', 'air_temperature']:
                    # Užpildoma pagal stebimo pastato mediana
                    data[col] = data.groupby('building_id')[col].transform(
                        lambda x: x.fillna(x.median() if not pd.isna(x.median()) else 0)
                    )
                else:
                    data[col].fillna(data[col].median(), inplace=True)
    
    if 'meter_reading' in data.columns:
        # Apkarpomi skaitiklio rodmenys, kad būtų pašalinami ekstremumai
        lower = data['meter_reading'].quantile(0.001)
        upper = data['meter_reading'].quantile(0.999)
        data['meter_reading'] = data['meter_reading'].clip(lower, upper)
        
        # Kuriami slenkantys požymiai
        for window in [3, 6, 12, 24]:
            try:
                data[f'rolling_mean_{window}h'] = data.groupby('building_id')['meter_reading'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                data[f'rolling_std_{window}h'] = data.groupby('building_id')['meter_reading'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
                )
                data[f'rolling_max_{window}h'] = data.groupby('building_id')['meter_reading'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
                data[f'rolling_min_{window}h'] = data.groupby('building_id')['meter_reading'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).min()
                )
                
                # Apskaičiuojamas nukrypimas nuo slenkančio vidurkio (svarbus anomalijos indikatorius)
                data[f'rolling_mean_diff_{window}h'] = data['meter_reading'] - data[f'rolling_mean_{window}h']
                
                # Apskaičiuojamas procentinis skirtumas
                data[f'rolling_mean_pct_{window}h'] = data[f'rolling_mean_diff_{window}h'] / (data[f'rolling_mean_{window}h'] + 1e-5)
                
                for col_name in [f'rolling_mean_{window}h', f'rolling_std_{window}h', 
                               f'rolling_max_{window}h', f'rolling_min_{window}h',
                               f'rolling_mean_diff_{window}h', f'rolling_mean_pct_{window}h']:
                    data[col_name].replace([np.inf, -np.inf], np.nan, inplace=True)
                    
                    # Užpildomos NaN su tinkamomis reikšmėmis
                    if 'std' in col_name:
                        data[col_name].fillna(0, inplace=True)
                    elif 'pct' in col_name or 'diff' in col_name:
                        data[col_name].fillna(0, inplace=True)
                    else:
                        data[col_name].fillna(data[col_name].median(), inplace=True)
                        
            except Exception as e:
                print(f"Klaida kuriant slenkančio lango {window}h požymius: {e}")
                data[f'rolling_mean_{window}h'] = data['meter_reading']
                data[f'rolling_std_{window}h'] = 0
                data[f'rolling_max_{window}h'] = data['meter_reading']
                data[f'rolling_min_{window}h'] = data['meter_reading']
                data[f'rolling_mean_diff_{window}h'] = 0
                data[f'rolling_mean_pct_{window}h'] = 0
        
        try:
            data['meter_reading_diff'] = data.groupby('building_id')['meter_reading'].diff().fillna(0)
            # Pridedami vėlinimo požymiai (svarbūs tendencijų pokyčių aptikimui)
            for lag in [1, 2, 3]:
                data[f'meter_reading_lag_{lag}'] = data.groupby('building_id')['meter_reading'].shift(lag).fillna(0)
                data[f'meter_reading_diff_lag_{lag}'] = data[f'meter_reading'] - data[f'meter_reading_lag_{lag}']
            
            # Apkarpomi skirtumai
            diff_cols = [col for col in data.columns if 'diff' in col]
            for col in diff_cols:
                diff_lower = data[col].quantile(0.001)
                diff_upper = data[col].quantile(0.999)
                data[col] = data[col].clip(diff_lower, diff_upper)
        except Exception as e:
            print(f"Klaida kuriant diff/lag požymius: {e}")
            data['meter_reading_diff'] = 0
            for lag in [1, 2, 3]:
                data[f'meter_reading_lag_{lag}'] = data['meter_reading']
                data[f'meter_reading_diff_lag_{lag}'] = 0
        
        try:
            # Naudojamos skirtingi grupavimai z-rezultatams (svarbu kontekstinėms anomalijoms)
            for group_cols in [['building_id'], 
                              ['building_id', 'hour'], 
                              ['building_id', 'time_of_day'],
                              ['building_id', 'is_weekend']]:
                
                # Sukuriamas grupavimo pavadinimas
                group_name = '_'.join([col.replace('_id', '') for col in group_cols])
                
                # Apskaičiuojamas vidurkis ir standartinis nuokrypis kiekvienai grupei
                group_mean = data.groupby(group_cols)['meter_reading'].transform('mean')
                group_std = data.groupby(group_cols)['meter_reading'].transform('std')
                # Užtikrinama, kad standartinis nuokrypis nėra nulis
                group_std = np.maximum(group_std, 1e-5)
                
                # Apskaičiuojamas z-rezultatas ir nukrypimo metrikos
                data[f'zscore_{group_name}'] = (data['meter_reading'] - group_mean) / group_std
                data[f'{group_name}_deviation'] = data['meter_reading'] - group_mean
                data[f'{group_name}_deviation_abs'] = np.abs(data[f'{group_name}_deviation'])
                
                data[f'zscore_{group_name}'].replace([np.inf, -np.inf], np.nan, inplace=True)
                data[f'zscore_{group_name}'].fillna(0, inplace=True)
            
            # Papildomi z-rezultatų požymiai ekstremalių pokyčių nustatymui
            data['zscore_extreme'] = (np.abs(data['zscore_building']) > 3).astype(int)
            data['zscore_extreme_high'] = (data['zscore_building'] > 3).astype(int)
            data['zscore_extreme_low'] = (data['zscore_building'] < -3).astype(int)
            
        except Exception as e:
            print(f"Klaida kuriant z-rezultatų požymius: {e}")
            data['zscore_building'] = 0
            data['building_deviation'] = 0
            data['building_deviation_abs'] = 0
        
        data['meter_reading_log'] = np.log1p(np.maximum(data['meter_reading'], 0))
        data['meter_reading_sqrt'] = np.sqrt(np.maximum(data['meter_reading'], 0))
        
        # Sąveika su temperatūra
        if 'air_temperature' in data.columns:
            temp_lower = data['air_temperature'].quantile(0.001)
            temp_upper = data['air_temperature'].quantile(0.999)
            data['air_temperature'] = data['air_temperature'].clip(temp_lower, temp_upper)
            
            # Įsitikinama, kad nedalijama iš nulio ar labai mažų reikšmių
            denominator = np.maximum(data['air_temperature'] + 273.15, 1)  # Užtikrinamas minimumas 1K
            data['temp_reading_ratio'] = data['meter_reading'] / denominator
            
            # Pridedama daugiau temperatūros sąveikų
            data['temp_squared'] = data['air_temperature'] ** 2
            data['temp_reading_product'] = data['meter_reading'] * data['air_temperature']
            
            # Apkarpomas santykis
            ratio_lower = data['temp_reading_ratio'].quantile(0.001)
            ratio_upper = data['temp_reading_ratio'].quantile(0.999)
            data['temp_reading_ratio'] = data['temp_reading_ratio'].clip(ratio_lower, ratio_upper)
    
    return data

# Sukuriamos LSTM modelio sekos
def create_sequences(data, seq_length=8, stride=1, feature_cols=None):
    if feature_cols is None:
        exclude_cols = ['timestamp', 'row_id', 'anomaly', 'building_id']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    X, y, building_ids, timestamps = [], [], [], []
    
    # Grupuojama pagal pastatą, kad būtų sukurtos atskiros sekos kiekvienam pastatui
    building_groups = data.groupby('building_id')
    print(f"Apdorojama {len(building_groups)} pastatų")
    
    for building_id, group in building_groups:
        # Rūšiuojama pagal laiko žymą
        group = group.sort_values('timestamp')
        
        # Gaunami požymiai
        features = group[feature_cols].values
        
        has_anomaly = 'anomaly' in group.columns
        if has_anomaly:
            target = group['anomaly'].values
        else:
            # Testavimo duomenims, užpildoma vėliau
            target = np.zeros(len(group))
        
        # Saugomi row_ids timestamps, vėliau naudojami testavimo duomenims
        if 'row_id' in group.columns:
            row_ids = group['row_id'].values
        else:
            row_ids = np.zeros(len(group))
        
        timestamps_arr = group['timestamp'].values
        
        # Kuriamos sekos su žingsniu
        for i in range(0, len(features) - seq_length + 1, stride):
            X.append(features[i:i+seq_length])
            y.append(target[i+seq_length-1])
            
            building_ids.append(building_id)
            timestamps.append(timestamps_arr[i+seq_length-1])
        
        # Valoma atmintis
        del features, target
        
    # Konvertuojama į numpy masyvus
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"Sukurta {len(X)} sekų su forma {X.shape}")
    return X, y, np.array(building_ids), np.array(timestamps)


# LSTM modelio kūrimas
def build_model(input_shape):
    model = Sequential()
    
    # Conv1D sluoksnis duomenų modelių sukūrimui
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', 
                    input_shape=input_shape, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Antras Conv1D sluoksnis
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',
                    kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Dvikryptis LSTM
    model.add(Bidirectional(LSTM(128, activation='tanh', return_sequences=True, 
                               kernel_regularizer=l2(1e-4))))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Antras dvikryptis LSTM
    model.add(Bidirectional(LSTM(64, activation='tanh', return_sequences=False,
                               kernel_regularizer=l2(1e-4))))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Pora išvesties sluoksnių
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Išvesties sluoksnis (dvejetainė klasifikacija)
    model.add(Dense(1, activation='sigmoid'))
    
    # Naudojamas mažesnis mokymosi greitis tikslesniam klasifikavimui
    optimizer = Adam(learning_rate=0.0005)
    
    # Kompiliuojamas modelis
    model.compile(optimizer=optimizer, loss='binary_crossentropy', 
                 metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    model.summary()
    return model

# Požymių svarbos analizė
def analyze_feature_importance(X, y, feature_names, max_features=30):
    # Apskaičiuojama kiekvieno požymio koreliacija su tiksline reikšme
    importances = []
    feature_list = []
    
    sample_size = min(100000, len(X))
    if len(X) > sample_size:
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[sample_idx]
        y_sample = y[sample_idx]
    else:
        X_sample = X
        y_sample = y
    
    # Kiekvienam laiko žingsniui ir požymiui, apskaičiuojama koreliacija su stebimu įrašu
    for t in range(X_sample.shape[1]):
        for f in range(X_sample.shape[2]):
            feature_name = f"t-{X_sample.shape[1]-t}_{feature_names[f]}"
            feature_list.append(feature_name)
            
            # Apskaičiuojama absoliutinė koreliacija
            corr = np.abs(np.corrcoef(X_sample[:, t, f], y_sample)[0, 1])
            if np.isnan(corr):
                corr = 0  # Tvarkomi NaN koreliacijos
            importances.append(corr)
    
    # Sukuriamas dataframe ir rūšiuojama svarba
    importance_df = pd.DataFrame({
        'Požymis': feature_list,
        'Svarba': importances
    }).sort_values('Svarba', ascending=False)
    
    # Grąžinami N svarbiausių požymių
    return importance_df.head(max_features)

def main():


    print("Įkeliami ir apdorojami duomenys...")
    train_data = load_data('train_features.csv', is_train=True)
    test_data = load_data('test_features.csv', is_train=False)
    
    print("Atliekamas papildomas duomenų valymas ekstremalių reikšmių apdorojimui...")
    
    # Funkcija duomenų valymui ir ekstremalių reikšmių apkarpymui
    def clean_and_clip_data(df):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Kiekvienam skaitiniam stulpeliui, apkarpomos ekstremalios reikšmės pagal kraštinius duomenis
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            if col not in ['building_id', 'row_id']:
                q_low = df[col].quantile(0.001)
                q_high = df[col].quantile(0.999)
                df[col] = df[col].clip(q_low, q_high)
                
        # Užpildomos likusios NaN su stulpelių medianomis
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            if col not in ['building_id', 'row_id']:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        # Galutinis patikrinimas
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in df.columns:
            if df[col].isna().any():
                print(f"Įspėjimas: Stulpelis {col} vis dar turi {df[col].isna().sum()} NaN reikšmių po valymo")
                if df[col].dtype in ['float64', 'int64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    # Taikomas valymas abiem duomenų rinkiniams
    train_data = clean_and_clip_data(train_data)
    test_data = clean_and_clip_data(test_data)
    
    # Apibrėžiami požymiai, įtraukiami visi skaitiniai požymiai išskyrus kategorinius (is_holiday)
    categorical_cols = ['hour', 'day', 'month', 'weekday', 'is_weekend', 'time_of_day', 'is_business_hours',
                      'zscore_extreme', 'zscore_extreme_high', 'zscore_extreme_low']
    exclude_from_scaling = ['building_id', 'anomaly', 'row_id'] + categorical_cols
    features_to_scale = [col for col in train_data.columns if col not in exclude_from_scaling 
                         and train_data[col].dtype in ['float64', 'int64']]
    
    print(f"Keičiami skalavimai (scalling) {len(features_to_scale)} požymiams")
    
    scaler = RobustScaler()
    
    # Pritaikoma ir transformuojami visi požymiai kartu geresniam skalavimui (scalling)
    if features_to_scale:
        print(f"Keičiami skalavimai (scalling) {len(features_to_scale)} skaitmeniniams požymiams")
        # Dar kartą patikrinamos likusios problemos
        for col in features_to_scale:
            if not np.isfinite(train_data[col]).all():
                print(f"Įspėjimas: Stulpelis {col} vis dar turi ne baigtines reikšmes")
                train_data[col] = train_data[col].replace([np.inf, -np.inf], np.nan)
                train_data[col].fillna(train_data[col].median(), inplace=True)
        
        train_data[features_to_scale] = scaler.fit_transform(train_data[features_to_scale])
        test_data[features_to_scale] = scaler.transform(test_data[features_to_scale])
    
    print("Kuriamos sekos...")
    seq_length = 8  # Naudojamas tas pats ilgis kaip originale
    stride = 1      # Naudojamas stride=1 maksimaliam duomenų padengimui
    
    # Apibrėžiami požymių stulpeliai - naudojami išsamesni požymių rinkiniai tikslumui
    exclude_cols = ['timestamp', 'row_id', 'anomaly', 'building_id']
    
    # Išlaikomas platus požymių rinkinys
    feature_cols = [col for col in train_data.columns if col not in exclude_cols]
    
    print(f"Iš viso naudojama {len(feature_cols)} požymių")
    
    X_train, y_train, train_buildings, train_timestamps = create_sequences(
        train_data, seq_length, stride, feature_cols
    )
    
    print(f"X_train forma: {X_train.shape}, y_train forma: {y_train.shape}")
    print(f"Klasių pasiskirstymas mokymo duomenyse: {np.bincount(y_train.astype(int))}")
    
    print("Analizuojama požymių svarba...")
    importance_df = analyze_feature_importance(X_train, y_train, feature_cols)
    print("20 svarbiausių požymių:")
    print(importance_df.head(20))
    
    # Apskaičiuojami klasių svoriai duomenų disbalansui spręsti
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )

    print("Dalijami duomenys į mokymo ir validavimo rinkinius...")
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    # Atlaisvinama atmintis
    del X_train, y_train
    gc.collect()
    
    print("Kuriamas ir apmokomas modelis...")
    input_shape = (X_train_final.shape[1], X_train_final.shape[2])
    model = build_model(input_shape)
    
    model_path = 'best_model.h5'
    checkpoint = ModelCheckpoint(
        model_path, 
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=0.00001,
        verbose=1
    )
    
    history = model.fit(
        X_train_final, y_train_final,
        epochs=30,  
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )

    # Įvertinamas modelis apmokymo rinkinyje
    val_metrics = model.evaluate(X_val, y_val)
    val_loss, val_acc = val_metrics[0], val_metrics[1]
    val_auc, val_precision, val_recall = val_metrics[2], val_metrics[3], val_metrics[4]
    
    print(f"Apmokymo nuostoliai: {val_loss:.4f}, Tikslumas: {val_acc:.4f}")
    print(f"AUC: {val_auc:.4f}, Tikslumas: {val_precision:.4f}, Pilnumas: {val_recall:.4f}")
    
    # Randama optimali riba naudojant tikslumo-pilnumo kreivę
    y_val_pred = model.predict(X_val, batch_size=128).flatten()
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_pred)
    
    # Randama riba su geriausiu F1 įverčiu
    f1_scores = 2 * recalls * precisions / (recalls + precisions + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    y_val_pred_binary = (y_val_pred > best_threshold).astype(int)
    val_f1 = f1_score(y_val, y_val_pred_binary)
    
    print(f"Optimali riba: {best_threshold:.4f} su F1 įverčiu: {val_f1:.4f}")
    
    # Apdorojami testavimo duomenys ir kuriamos sekos
    print("Apdorojami testavimo duomenys...")
    X_test, _, test_buildings, test_timestamps = create_sequences(
        test_data, seq_length, stride, feature_cols
    )
    
    # Ištraukiami row_ids, kurie atitinka prognozavimo taškus
    test_indices = []
    for i, (bid, ts) in enumerate(zip(test_buildings, test_timestamps)):
        # Randama atitinkanti eilutė test_data
        idx = test_data[(test_data['building_id'] == bid) & 
                         (test_data['timestamp'] == ts)].index
        if len(idx) > 0:
            test_indices.append(idx[0])
    
    print(f"Rasta {len(test_indices)} atitinkančių indeksų testavimo duomenyse")
    row_ids = test_data.loc[test_indices, 'row_id'].values
    
    print("Atliekamos prognozės...")
    batch_size = 512  # Mažesnis paketo dydis tikslesnėms prognozėms
    y_pred_proba = []
    
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size]
        batch_pred = model.predict(batch, verbose=0)
        y_pred_proba.extend(batch_pred)
    
    y_pred_proba = np.array(y_pred_proba).flatten()
    
    # Taikoma optimali riba, rasta naudojant apmokymo duomenis
    final_preds = (y_pred_proba > best_threshold).astype(int)
    
    print(f"Prognozių pasiskirstymas: {np.bincount(final_preds)}")
    
    # Sukuriamas galutinis rezultatų dataframe
    result_df = pd.DataFrame({
        'row_id': row_ids,
        'anomaly': final_preds
    })
    
    # Tvarkomi trūkstami row_ids
    all_row_ids = set(test_data['row_id'].values)
    missing_row_ids = all_row_ids - set(result_df['row_id'].values)
    
    if missing_row_ids:
        print(f"Pridedami {len(missing_row_ids)} trūkstami row_ids su numatytąja prognoze 0")
        missing_df = pd.DataFrame({
            'row_id': list(missing_row_ids),
            'anomaly': [0] * len(missing_row_ids)
        })
        result_df = pd.concat([result_df, missing_df])
    
    # Rūšiuojama pagal row_id nuosekliam pateikimui
    result_df = result_df.sort_values('row_id').reset_index(drop=True)
    
    # Išsaugomi rezultatai
    result_df.to_csv('anomaly_predictions_LSTM.csv', index=False)
    print("Prognozės išsaugotos 'anomaly_predictions_LSTM.csv' faile")
    
    print(f"Iš viso aptikta anomalijų: {result_df['anomaly'].sum()} iš {len(result_df)}")

if __name__ == "__main__":
    main()