#全体のアプリのフロントエンドとなる画面。
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