from pytrends.request import TrendReq
from datetime import datetime
import wbdata
import pandas as pd



def load_gdp_data(start_year=2019, end_year=2022):
    indicators = {'NY.GDP.MKTP.CD': 'GDP_Current_USD'}
    gdp_data = wbdata.get_dataframe(indicators, country=countries)
    gdp_data.reset_index(inplace=True)
    gdp_data['date'] = pd.to_datetime(gdp_data['date'])
    gdp_data = gdp_data[gdp_data['date'].dt.year.between(start_year, end_year)]
    return gdp_data

def fetch_google_trends(keywords, start_date='2019-01-01', end_date=datetime.today().strftime('%Y-%m-%d')):
    pytrend = TrendReq()
    trends_data_combined = pd.DataFrame()
    for country in pycountries:
        for keyword in keywords:
            pytrend.build_payload(kw_list=[keyword], timeframe=f'{start_date} {end_date}', geo=country)
            data = pytrend.interest_over_time()
            if not data.empty:
                data.reset_index(inplace=True)
                data = data[['date', keyword]]
                if trends_data_combined.empty:
                    trends_data_combined = data
                else:
                    trends_data_combined = trends_data_combined.merge(data, on='date', how='left')
    return trends_data_combined
