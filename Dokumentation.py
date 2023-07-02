import streamlit as st

def main():
    st.set_page_config(
        page_title="Documentation",
        layout='wide',
        initial_sidebar_state = "collapsed"
    )

st.title("Wie unsere App funktioniert")

st.write("Die VoiceChoice-App ist eine Anwendung, die entwickelt wurde, um anhand einer Audiodatei das Geschlecht der Stimme zu erkennen. Die App verwendet eine Kombination von Algorithmen und Machine Learning-Techniken, um eine Aussage darüber zu treffen, ob die aufgenommene Stimme weiblich oder männlich ist. Diese Dokumentation gibt einen kurzen Überblick über die Funktionen der App.")

st.header("Funktionen")
st.write("Die VoiceChoice-App bietet folgende Funktionen:")
st.write("- Converter: Benutzer können eine Audiodatei im Format MP3, OPUS oder MP4 bzw. MPEG4 hochladen und diese im Anschluss als .wav wieder herunterladen.")
st.write("- Do it yourself: xy")
st.write("sowie VoiceChoice mit den verschiedenen Machine Learning Algorithmen:")
st.write("- Neuronales Netzwerk")
st.write("- Neuronales Netzwerk und Echtzeitaufnahme")
st.write("- Random Forest Classifier")
st.write("- Support Vectore Machine")


st.header("Funktionsweise der App")
st.write("")

st.header("Hinweis")
st.write("Die VoiceChoice-App bietet eine gute Genauigkeit bei der Geschlechtererkennung, jedoch können individuelle Ergebnisse variieren und die App sollte nicht als endgültiges Mittel zur Geschlechtsbestimmung verwendet werden.")

st.header("Systemanforderungen")
st.write("Die VoiceChoice-App hat folgende Systemanforderungen:")
st.write("- Ein aktueller Webbrowser (z.B. Google Chrome, Mozilla Firefox, Safari)")
st.write("- ...")
st.write("")
st.write("Es sind keine zusätzlichen Softwareinstallationen erforderlich. Die VoiceChoice-App ist vollständig webbasiert und kann von Benutzern direkt über den bereitgestellten Link aufgerufen werden.")


if __name__ == "__main__":
    main()
