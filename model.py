import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from pydantic import BaseModel
import pickle
import catboost as cb

from sklearn.preprocessing import QuantileTransformer


def map_2_cols(df, col1, col2, group, gby):
    return df.loc[:, [col1, col2]].astype(str).sum(axis=1).map(group[gby])


def group_2_cols(df,col1,col2,gby,func='mean'):
    group = df[[col1, col2,gby]].groupby([col1, col2],as_index=False).agg('mean')
    group['comp'] = group[[col1,col2]].astype(str).sum(axis=1)
    group.set_index('comp',inplace = True)
    group.drop([col1, col2],axis=1,inplace = True)
    return group


def cyclical(df, column, max_value):
    """
    The function is encoding time series cyclical features with sin and cos.
    Input: 
    ---------
    df - pandas DataFrame
    column - column name
    max_value - column max value
    Output: 
    -----------
    -same dataframe with _sin and _cos columns added
    """
    print('>>>>>>>>>>>>>>>>>>>>>', column, "\tmax: ", max_value, '\tdf: ',df[column].values)
    print(df.shape)
    df[column + '_sin'] = np.sin(2 * np.pi * df[column] / max_value)
    df[column + '_cos'] = np.cos(2 * np.pi * df[column] / max_value)
    df.fillna(0, inplace=True)
    return df


def shift_col(df, col_name, shift_num=1):
    df[f"{col_name}_shifted{shift_num}"]= df[col_name] - df[col_name].shift(shift_num)
    df[f"{col_name}_shifted{shift_num}"].fillna(method='bfill',inplace=True)
    df[f"{col_name}_shifted{shift_num}**"]= df[f"{col_name}_shifted{shift_num}"] * df[f"{col_name}_shifted{shift_num}"]
    df[f"{col_name}_shifted{shift_num}**"].fillna(method='bfill',inplace=True)
    df[f"{col_name}_shifted{shift_num}**"].fillna(method='ffill',inplace=True)
    df[f"{col_name}_shifted{shift_num}**"].fillna(method='ffill',inplace=True)
    df.fillna(0, inplace=True)
    return df


def preprocess(df, seasons_Hour_3cut=None, seaons_mean=None, test_set=False):
    df['Holiday'].replace({"Holiday": 0, "No Holiday": 1}, inplace=True)
    df['Functioning Day'].replace({"Yes": 0, "No": 1}, inplace=True)
    df['Seasons'].replace({"Autumn": 2, "Spring": 3, "Summer": 1, "Winter": 4}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].apply(lambda x:1 if x.year == 2018 else 0)
    df['Hour_3cut'] = pd.cut(df['Hour'],
                       bins=[-np.inf,7,18,np.inf],
                       labels=[1,2,3]).astype('int')
    dummies = pd.get_dummies(df, columns=['Hour', 'Seasons'], prefix=['col1', 'col2'])
#     print(dummies)
  
    # split the fractions
    df['snow_frac'] = df['Snowfall (cm)'].apply(lambda x: x - int(x))
    df['rain_frac'] = df['Rainfall(mm)'].apply(lambda x: x - int(x))
    df['solar_frac'] = df['Solar Radiation (MJ/m2)'].apply(lambda x: x - int(x))
    df['wind_frac'] = df['Wind speed (m/s)'].apply(lambda x: x - int(x))
#     df['wind_log'] = df['Wind speed (m/s)'].apply(lambda x: np.log(x+1))
    
    # create dates columns
    df['month'] = df['Date'].apply(lambda x:x.month)
    df['Week Days'] = df['Date'].apply(lambda x:x.dayofweek+1)

    # shift columns -1
    df = shift_col(df, 'Temperature(�C)', -1)
    df = shift_col(df, 'Rainfall(mm)', -1)
    df = shift_col(df, 'Humidity(%)', -1)
    df = shift_col(df, 'Wind speed (m/s)', -1)
    
    # shift columns
    df = shift_col(df, 'Temperature(�C)')
    df = shift_col(df, 'Rainfall(mm)')
    df = shift_col(df, 'Humidity(%)')
    df = shift_col(df, 'Wind speed (m/s)')

    df = cyclical(df, "month", 12)
    df = cyclical(df, "Hour", 23)
#     df = pd.concat([df, dummies], axis=1)
#     print(type(df))
    
    return df.drop(["ID", 'Date', 'Temperature(�C)_shifted-1', 'Temperature(�C)_shifted1', 'Functioning Day'], axis=1)


def pipeline(df, model, qt):
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True).dt.date
    test_X = preprocess(df, test_set=True)
    test_X = pd.DataFrame(qt.transform(test_X))
    y_cbm = model.predict(test_X)
    y_hat = np.exp(y_cbm) - 1
    return y_hat


def predict_file(test_path: str):
    df = pd.read_csv(test_path)
    columns = np.load('bin/columns.npy')
    df.columns = columns
    print('################################', df.info())
    model = cb.CatBoostRegressor()
    model.load_model('bin/cb_model', 'cbm')
    qt = pickle.load(open('bin/qt.pkl', 'rb'))
    predictions = pipeline(df, model, qt)
    ret = pd.DataFrame({'ID': df['ID'].to_numpy(), 'y': predictions}).to_numpy()
    return ret


def predict_single(p_id, date, hour, temperature, humidity, wind_speed, visibility, dew_point, solar_rad, rain_fall,
                   snow_fall, season, holiday, functioning_day):
    columns = np.load('bin/columns.npy')
    df = pd.DataFrame(columns=columns)

    df = df.append({columns[0]: p_id, columns[1]: date, columns[2]: hour, columns[3]: temperature, columns[4]: humidity,
                   columns[5]: wind_speed, columns[6]: visibility, columns[7]: dew_point, columns[8]: solar_rad,
                   columns[9]: rain_fall, columns[10]: snow_fall, columns[11]: season, columns[12]: holiday,
                   columns[13]: functioning_day}, ignore_index=True)
    df['Hour'] = df['Hour'].astype('int')
    model = cb.CatBoostRegressor()
    model.load_model('bin/cb_model', 'cbm')
    qt = pickle.load(open('bin/qt.pkl', 'rb'))
    predictions = pipeline(df, model, qt)
    ret = pd.DataFrame({'ID': df['ID'].to_numpy(), 'y': predictions}).to_numpy()
    return ret


class PredictionItem(BaseModel):
    p_id: str
    date: str
    hour: int
    temperature: float
    humidity: float
    wind_speed: float
    visibility: int
    dew_point: float
    solar_rad: float
    rain_fall: float
    snow_fall: float
    season: str
    holiday: str
    functioning_day: str

