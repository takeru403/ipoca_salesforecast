mkdri -p ~/.streamlit/

echo "\
[general]\n\
email = \"tata.5593@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
poort = $PORT\n\
" > ~/.streamlit/config.toml