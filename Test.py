import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from datetime import datetime

data_day_style = pd.read_csv(r'D:\Risk_Model\Style_Specific_risk.csv')  # std
factor_risk = pd.read_csv(r'D:\Risk_Model\factor_risk_record.csv')  # var


def yn_select_stocks(data):
    if data[0] == '3':
        return 1
    else:
        return 0


data_day_style['Second_board'] = data_day_style['WIND_CODE'].map(lambda x: yn_select_stocks(x))
data_day_style = data_day_style.loc[data_day_style.Second_board == 1]
data_day_style.TRADE_DT = pd.to_datetime(data_day_style.TRADE_DT)
data_day_style = data_day_style.loc[data_day_style.TRADE_DT >= datetime(2012, 1, 1)]
data_day_style.set_index(['TRADE_DT', 'WIND_CODE'], inplace=True)
data_day_style.sort_values(by=['TRADE_DT', 'WIND_CODE'], inplace=True)
data_day_style.drop('Unnamed: 0', axis=1, inplace=True)
#
factor_risk.TRADE_DT = pd.to_datetime(factor_risk.TRADE_DT)
factor_risk = factor_risk.loc[factor_risk.TRADE_DT >= datetime(2012, 1, 1)]
factor_risk.drop('Unnamed: 0', axis=1, inplace=True)
factor_risk.set_index(['TRADE_DT', 'FACTOR'], inplace=True)

# Index_return
Second_board = pd.DataFrame(columns=['NEXT_RETURN'], index=data_day_style.index.get_level_values(0).unique())
for i_num, i_date in enumerate(data_day_style.index.get_level_values(0).unique()):
    data_day_style_t = data_day_style.loc[i_date]
    index_return_t = np.dot(data_day_style_t.NEXT_RETURN, data_day_style_t.WEIGHT/data_day_style_t.WEIGHT.sum())
    Second_board.at[i_date, 'NEXT_RETURN'] = index_return_t

Second_board['Real_risk'] = Second_board['NEXT_RETURN'].rolling(21).apply(lambda x: np.std(x), 'raw=True')  # 默认ddof=0
Second_board['Real_risk'] = Second_board['Real_risk'] * np.sqrt(21)
Second_board['Predict_risk'] = np.nan

style = ['Value', 'LSIZE', 'MID_CAP', 'Momentum', 'Quality', 'Dividend_yield', 'Volatility', 'Growth', 'Liquidity', 'Sentiment']
date_index = list(data_day_style.index.get_level_values(0).unique())
for i_num, i_date in enumerate(data_day_style.index.get_level_values(0).unique()):
    data_day_style_t = data_day_style.loc[i_date].copy()
    # specific risk
    data_day_style_t['weight'] = data_day_style_t['WEIGHT'] / data_day_style_t['WEIGHT'].sum()
    specific_var_t = np.dot(data_day_style_t['specific_risk_VRA_Predict'] ** 2, data_day_style_t['weight'] ** 2)
    # factor_risk
    data_day_dummies = pd.get_dummies(data_day_style_t.INDUSTRY)
    data_day_style_t = pd.merge(data_day_style_t, data_day_dummies, how='left', left_index=True, right_index=True, sort=False)
    industry_t = list(data_day_style_t['INDUSTRY'].unique())
    columns_t = industry_t + style
    weights_matrix = np.mat(data_day_style_t['weight'].values.reshape(-1, 1))
    factor_matrix = np.mat(data_day_style_t[columns_t].values)
    factor_exposure = weights_matrix.T * factor_matrix  # matrix 行向量
    factor_exposure = factor_exposure.T
    factor_risk_t = factor_risk.loc[i_date].copy()
    factor_risk_matrix = np.mat(factor_risk_t.loc[columns_t, columns_t].values)
    style_var_t = float(factor_exposure.T * factor_risk_matrix * factor_exposure)
    #
    risk_all = np.sqrt(specific_var_t + style_var_t)
    Second_board.at[i_date, 'Predict_risk'] = risk_all















