#This example uses Python 2.7 and the python-request library.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json

url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {
  'start':'1',
  'limit':'5000',
  'convert':'USD'
}
headers = {
  'Accepts': 'application/json',
  'X-CMC_PRO_API_KEY': 'ac7ede71-0f45-4e36-87c4-b9b5943188ca',
}

session = Session()
session.headers.update(headers)

try:
  response = session.get(url, params=parameters)
  data = json.loads(response.text)
  normalizedData = pd.json_normalize(data)
  
  df = pd.DataFrame.from_dict(normalizedData)
  
  with open('cmcjson.txt', 'w') as outfile:
      json.dump(response.text, outfile)
      outfile.close();
  df.to_csv('cmc.csv', index = False)
  
  print(df)
except (ConnectionError, Timeout, TooManyRedirects) as e:
  print(e)
