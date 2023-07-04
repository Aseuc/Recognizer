import streamlit as st
import ballons_red as br
import ballons_blue as bb
import VoiceChoice as vc
import randomFacts
st.set_page_config(
    page_title="What's special",
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

st.write(f"Ein weiteres Feature was wir uns überlegt haben ist auch das Ausgeben von Funfacts über die männliche oder "
         f"weibliche Stimme!")

st.write(f"Zum Beispiel: {randomFacts.random_fact_women()}")