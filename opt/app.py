#全体のアプリのフロントエンドとなる画面。
#基本的にはプログラムに型アノテーションをつけていく。
import learn
import predict
import explanation
import streamlit as st
PAGES = {
    "学習": learn,
    "予測": predict,
    "説明": explanation
}
st.sidebar.title('学習か予測かを選んでください')
selection = st.sidebar.radio("選択", list(PAGES.keys()))
page = PAGES[selection]
page.app()
