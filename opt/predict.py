import os
import streamlit as st
import pandas as pd
import datetime
import streamlit.components.v1 as stc
from pycaret.regression import *


def app():
    st.markdown("# 予測フェーズ")
    st.markdown("# 2.モデルを読み込みます")
    pkl_lt = [''] + [f[:-4] for f in os.listdir(os.getcwd()) if f[-4:]=='.pkl']
    model_name = st.selectbox(label='ドロップダウンリストからモデルを選択してください',options=pkl_lt,key='model')

    try:
        if model_name != '':
            dt_saved = load_model(model_name)

            st.markdown("# 3.予測したいデータをアップロードします")
            uploaded_file = st.file_uploader("CSVファイルをドラッグ&ドロップ、またはブラウザから選択してください", type='csv', key='test')

            if uploaded_file is not None:
                df_new = pd.read_csv(uploaded_file,thousands=',')
                predictions = predict_model(dt_saved, data=df_new)

                predictions.to_csv(model_name+'_predict_'+datetime.date.today().strftime('%Y%m%d')+'.csv')
                st.dataframe(predictions)
                st.write("学習した変数が適用されます。")
                plot_model(predictions)
    except ValueError as error:
        print("データの形が間違っています")
        print(error)
