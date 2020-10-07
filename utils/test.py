from operator import index
import pandas as pd

df = pd.read_csv(r"C:\\Users\\nbhas\Desktop\Shishir\\utils\\csv_dump\\twerche.csv", index_col=None)

def get_data_lst(df):
    df['date'] = pd.to_datetime(df['date'])\

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # print(df)
    df_1 = df[df.groupby('year')['vol'].transform('max') == df['vol']]
    # print (df_1)
    final_df_1 = df_1[['date', 'vol']]
    final_df_1['date'] = final_df_1['date'].dt.strftime('%Y-%m-%d')

    # print(final_df_1)
    df_2 = df[df.groupby('year')['close'].transform('max') == df['max']]
    final_df_2 = df_2[['date', 'close']]
    final_df_2['date'] = final_df_2['date'].dt.strftime('%Y-%m-%d')
    # print(final_df_2)
    print([final_df_1, final_df_2])
    return[final_df_1, final_df_2]

get_data_lst(df)