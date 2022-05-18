from matplotlib.style import use
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib
import streamlit as st
import pandas as pd
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
        #オプションのthousandsでカンマを文字列として扱わなくなる。
        df = pd.read_csv(uploaded_file,thousands=',')
        st.write("学習用データのアップロードが完了しました")
        
        df = df.set_index('店舗名')
        st.dataframe(df, 800,300)
        
        #変数間の相関行列を出すとだいぶ重くなる。
        # fig, ax = plt.subplots(figsize=(10,10))
        # sns.heatmap(df.corr(), annot=True, ax=ax)
        # st.pyplot(fig)
        #削除したい説明変数を選択

        st.markdown("## 削除したい説明変数はありますか?")
        @st.cache()
        def delete_feature(deletes):
            df = df.drop(deletes, axis=1)
            return df
        deletes = st.multiselect("セレクトボックスから削除したい説明変数を選択してください",list(df.columns))
        df = df.drop(deletes, axis=1)
        st.dataframe(df, 800,300)
        st.markdown("## 2.予測したいターゲットの選択")
        #オプション　value = df.column[0]
        #あとで検討、一旦radioボタン
        target = st.radio("目的変数",("年商","レジ客数","客単価"))
        #target = st.text_input(label='ターゲット名を正しく入力してください（例：年商）')
        
        if target != "":
            st.markdown("## 3.機械学習を始めます。")

            if st.button('学習を開始'):                
                st.markdown("学習中です…しばらくお待ち下さい…")
                
                #streamlitの前処理を表示できなくする。jupyter環境とは違うため。
                #使えそうなオプション
                #normarize_method => いくつかの方法で正規化等ができる。
                #transformation =>　Trueにすることで乗数変換を用いて、データがガウス分布に従うようにする
                #remove_outliers =>Trueにすると、特異値分解を利用したPCA線形時限削減を用いて、トレーニングデータから外れ値が削除される。
                #log_data => Trueにすることで、訓練データとテストデータがcsvとして保存される。
                ml = setup(data=df,target=target, html=False,silent=True, train_size=0.8, data_split_shuffle=False)

                best = compare_models()
                best_model_results = pull() # 結果をデータフレームとして得る。
                
                #日本語に変換する
                best_model_results_columns = best_model_results.rename(columns={'Model': '機械学習モデル','MAE':'平均絶対誤差','MSE':'平均二乗誤差','RMSE':'二乗平均平方根誤差','R2':'決定係数','RMSLE':'対数平均二乗誤差','MAPE':'平均絶対%誤差'})
                best_model_results_japan = best_model_results_columns.replace({'Linear Regression': '線形回帰', 'Bayesian Ridge': 'ベイジアン回帰','Lasso Regression':'ラッソ回帰','Ridge Regression':'リッジ回帰','Elastic Net':'エラスティックネット','Lasso Least Angle Regression':'LARS', 'Huber Regressor':'ロバスト回帰','Random Forest Regressor':'ランダムフォレスト','K Neighbors Regressor':'k-近傍回帰','Gradient Boosting Regre':'勾配ブースティング','AdaBoost Regressor':'AdaBoost回帰','Light Gradient Boosting Machine':'Light勾配ブースティング','Decision Tree Regressor':'回帰木','Dummy Regressor':'ダミー回帰','Orthogonal Matching Pursuit':'マッチング追跡'})
                st.write(best_model_results_japan) # 比較結果の表示
                #今回は、何が高いのか調べる。ものを予測モデルとして選んでいるが、自分で学習モデルを選べるようにすることも考える。
                
                select_model = best_model_results.index[0]
                model = create_model(select_model)
                #ここでエラーが出る。
                final = finalize_model(model)
                save_model(final, select_model+target+'_saved_'+datetime.date.today().strftime('%Y%m%d'))
                #     #残差
                # plot_model(model, plot="error", display_format="streamlit")
                
                dashboard(model)
            
               
                st.markdown("モデル構築が完了しました")
                
                st.markdown("自分のパソコンに拡張子がpklのファイルがあることを確認して、予測フェーズへと進んでください")



 






#学習中にこの表示を行う
# latest_iteration = st.empty()
        
# bar = st.progress(0)
#         for i in range(100):
#             latest_iteration.text(f'Iteration {i + 1}')
#             bar.progress(i + 1)
#             time.sleep(0.1)

#初期化
    # def clear_cache():
    #     if st.button("initialize"):
    #         st.experimental_memo.clea