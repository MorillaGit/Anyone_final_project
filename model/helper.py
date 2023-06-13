




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