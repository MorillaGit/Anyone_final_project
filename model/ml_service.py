import json
import time
import settings
import pandas as pd
import numpy as np
import xgboost as xgb
import ligthgbm as lgb
import redis
import joblib

db = redis.Redis(
    host=settings.REDIS_IP ,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB_ID
)

model = xgb.XGBClassifier()
model.load_model('jobs/model.lgb')
transformer = joblib.load('jobs/transformer3.pkl')
scaler = joblib.load('jobs/scaler3.pkl')


def predict(data):
    
    df = pd.DataFrame.from_dict(data, orient='index', columns=['Value'])
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.transpose().sort_index(axis=1)
    
    df = df[['paagey', 'paarthre', 'pabathehlp', 'pacancre', 'pachair', 'pacholst',
       'paclims', 'padadage', 'padiabe', 'padrinkb', 'paeat', 'pafallinj',
       'pagender', 'paglasses', 'pagrossaa', 'pahearaid', 'paheight',
       'pahibpe', 'pahipe_m', 'palunge_m', 'pameds', 'pamomage', 'paosleep',
       'papaina', 'parafaany', 'parjudg', 'pasmokev', 'pastroke', 'paswell',
       'paweight', 'pawheeze']]
    
    df['paheight']=df['paheight'] / 100
    df = transformer.transform(df)
    
    np_arr = df
    # np_arr = df.to_numpy()
    
    np_arr = scaler.transform(np_arr)
    
    pred_probability = model.predict_proba(np_arr)[:, 1][0]
    pred = np.round(pred_probability)
    return float(pred), float(pred_probability)


def classify_process():

    while True:
        job = db.brpop(settings.REDIS_QUEUE)[1]
        
        job = json.loads(job.decode("utf-8"))
        class_name, pred_probability = predict(job["data_input"])
        
        pred = {
            "prediction": class_name,
            "score": float(pred_probability),
        }
        
        job_id = job["id"]
        db.set(job_id, json.dumps(pred))
        
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    print("Launching ML service...")
    classify_process()