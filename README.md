# ENERGIJOS NAUDOJIMO ANOMALIJŲ APTIKIMO METODAI IŠMANIŲJŲ NAMŲ IOT SISTEMOSE

## Magistro darbo programinis kodas

### Autorius: Linas Astrauskas

### Vadovas: prof. habil. dr. Antanas Čenys

Naudota OS: Windows 11 <br/>
Kodo redaktorius: Visual Studio Code<br/>
Apmokymo ir testavimo duomenų rinkinius atsisiųsti:<br/>
https://www.kaggle.com/competitions/energy-anomaly-detection/data<br/>

Kodo naudojimo instrukcija:

Ansamblio algoritmas:

Generuojami rezultatai:
1. py.exe ./Ensemble.py

Paleidžiama vartotojo sąsaja
1. Išskleidžiamas "dashboard.zip" archyvas
2. Einama į dashboard katalogą
3. npm run build
4. npm start
5. Einama į http://localhost:3000

Likusių algoritmų rezultatų generavimas:

LightGBM:
1. py.exe ./LightGBM.py

XGBoost:
1. py.exe ./XGBoost.py

LSTM:
1. py.exe ./LSTM.py