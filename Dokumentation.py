import streamlit as st

def main():
    st.set_page_config(
        page_title="Documentation",
        layout='wide',
        initial_sidebar_state = "collapsed"
    )

st.title("Dokumentation unserer App VoiceChoice")
st.write("Die VoiceChoice-App ist eine Anwendung, die entwickelt wurde, um anhand einer Audiodatei das Geschlecht der Stimme zu erkennen. Die App verwendet eine Kombination von Algorithmen und Machine Learning-Techniken, um eine Aussage darüber zu treffen, ob die aufgenommene Stimme weiblich oder männlich ist. Diese Dokumentation gibt einen Überblick über die Funktionen der App.")


if __name__ == "__main__":
    main()
