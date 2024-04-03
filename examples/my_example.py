from pandasai import SmartDataframe
from pandasai.llm import BambooLLM
import pandas as pd
from pandas import DataFrame 
import os

pandasai_apikey = "$2a$10$qcCh/mubPCDz3pwwctq8t.bPCsoHJL99zdnNbYvGKnfotP6073QNC"

from ndeq.ttlsap.adapter.edw_gp import Edw

edwq = Edw()
q_sql = '''select table_catalog, table_schema, table_name, column_name, ordinal_position from information_schema.columns;'''
data = edwq.get_edw_data(q_sql)

data.to_csv('data\\edw_schema.csv')

df = pd.read_csv('data\\edw_schema.csv')

llm = BambooLLM(api_key=pandasai_apikey)

sdf = SmartDataframe(df, config={"llm": llm})

# sdf method :   'last_code_executed', 'last_code_generated', 'last_prompt'
# ask the count of the unique table_name

assert df.table_name.unique().size == sdf.chat("please count the unique table_name")

table_name = 'ie_be_dim_eq_res_tech_attr_hist'
subset_by_table = df[df.table_name == table_name]


def filter_df(df:DataFrame, type:int=1,**kwargs):
    """
        pandas DataFrame filter function : subset the dataframe rows or columns according the specified index lables
        type 1 : by columns filter with full name: items=['col1'..,'colk']
        type 2 : by columns filter with reg : regex='col$',axis=1
        type 3 : by rows filter with reg : regex='row$', axis=0
        kwargs required : 
            type 1 :{"items":value}
            type 2 & 3: {"regexp":exression}
    """
    if type==1 and 'items' in kwargs.keys(): result = df.filter(items=kwargs['items'])
    elif type==1 and 'items' not in kwargs.keys(): return f"require items list for columns"
    elif type==2 and 'regexp' in kwargs.keys(): result = df.filter(regex=kwargs['regexp'], axis=1)
    elif type==2 and 'regexp' not in kwargs.keys(): return f"require regexp for columns"
    elif type==3 and 'regexp' in kwargs.keys(): result = df.filter(regex=kwargs['regexp'], axis=0)
    elif type==3 and 'regexp' not in kwargs.keys(): return f"require regexp for rows"
    return result
    
def test_filter_df():
    r1 = filter_df(df, type=1, **{'items':['table_name']})
    assert r1.columns[0] == 'table_name' and r1.columns.size == 1
    r2 = filter_df(df, type=2, **{'regexp':'ta$'}) 