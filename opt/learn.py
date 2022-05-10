from matplotlib.style import use
import streamlit as st
import pandas as pd
from pycaret.regression import * 
import datetime
import os
import streamlit.components.v1 as stc
import time
from pycaret.regression import *


def app():
    st.markdown("# 学習フェーズ")
    st.markdown("## 1.学習させるデータをアップロードしてください")
    uploaded_file = st.file_uploader("CSVファイルをドラッグ&ドロップ", type='csv', key='train')

    #学習用データをアップロードされた後の処理
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("学習用データのアップロードが完了しました")
        
        #無理矢理型を変換する(ここは直さないとエラーが起こる可能性がある。エラー処理の場所として検討する)
        df = df.set_index('店舗名')
        df = df.replace(',','', regex=True)
        df['年商']= df['年商'].astype(int)
        df['レジ客数'] = df['レジ客数'].astype(int)

        st.dataframe(df, 800,300)
        
        st.markdown("## 2.予測したいターゲットの選択")
        #オプション　value = df.column[0]
        target = st.text_input(label='ターゲット名を正しく入力してください（例：年商）')
        
        if target != "":
            st.markdown("## 3.機械学習を始めます。")

            if st.button('学習を開始'):                
                st.markdown("学習中です…しばらくお待ち下さい…")
                
                #streamlitの前処理を表示できなくする。jupyter環境とは違うため。
                ml = setup(data=df,target=target, html=False,silent=True,)

                best = compare_models()
                best_model_results = pull() # 比較結果の取得
                st.write(best_model_results) # 比較結果の表示
                select_model = best_model_results.index[0]
                model = create_model(select_model)
                    #ここでエラーが出る。
                final = finalize_model(model)
                save_model(final, select_model+'_saved_'+datetime.date.today().strftime('%Y%m%d'))
                #特徴量寄与度
                plot_model(model, plot="feature", display_format="streamlit")
                #残差
                plot_model(model, plot="error", display_format="streamlit")
                st.markdown("モデル構築が完了しました")
                
                st.markdown("自分のパソコンに拡張子がpklのファイルがあることを確認して、予測フェーズへと進んでください")
        

 






#学習中にこの表示を行う
# latest_iteration = st.empty()
        
# bar = st.progress(0)
#         for i in range(100):
#             latest_iteration.text(f'Iteration {i + 1}')
#             bar.progress(i + 1)
#             time.sleep(0.1)
