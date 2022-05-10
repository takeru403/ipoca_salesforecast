#全体のアプリのフロントエンドとなる画面。
#基本的にはプログラムに型アノテーションをつけていく。
import learn
import predict
import streamlit as st
PAGES = {
    "学習": learn,
    "予測": predict
}
st.sidebar.title('学習か予測かを選んでください')
selection = st.sidebar.radio("選択", list(PAGES.keys()))
page = PAGES[selection]
page.app()

#エクスパンだー
expander1 = st.beta_expander("モデルの精度が上がりません")
expander1.write("データに欠損値がないのか、確認してください")
expander2 = st.beta_expander("データが読み込みません")
expander2.write("拡張子がcsvになっているのか確認してください")
expander3 = st.beta_expander("その他")
expander3.write("こちらまで連絡ください")
