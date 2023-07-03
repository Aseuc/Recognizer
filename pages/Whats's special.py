import streamlit as st
import ballons_red as br
import ballons_blue as bb
import VoiceChoice as vc

st.set_page_config(
    page_title="Converter",
    page_icon="favicon.ico",
    layout='wide',
    initial_sidebar_state="auto"
)

vc.add_logo_sidebar()

st.header("What's special")

st.write("Zu guter Letzt zu den Dingen die speziell sind auf unserer Streamlit-App. Dabei haben wir eigene Komponenten "
         "entwickelt die es uns erlauben (siehe unten) Ballons für die jeweilige Klassifizierung aufsteigen zu "
         "lassen! Dabei werden immer wieder neue Personen angezeigt! Und schon gemerkt? Die Personen sind AI-generiert!")

br.ballons_red()
bb.ballons_blue()
