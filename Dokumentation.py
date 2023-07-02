import streamlit as st

def main():
    st.set_page_config(
        page_title="Documentation",
        layout='wide',
        initial_sidebar_state = "collapsed"
    )

st.title("Wie unsere App funktioniert")

st.write("Die VoiceChoice-App ist eine Anwendung, die entwickelt wurde, um anhand einer Audiodatei das Geschlecht der Stimme zu erkennen. Die App verwendet eine Kombination von Algorithmen und Machine Learning-Techniken, um eine Aussage darüber zu treffen, ob die aufgenommene Stimme weiblich oder männlich ist. Diese Dokumentation gibt einen kurzen Überblick über die Funktionen der App.")


st.header("Funktionen", text_align='left')
st.write("Die VoiceChoice-App bietet folgende Funktionen:", text_align='left')
st.write("- Hochladen einer Audiodatei: Benutzer können eine Audiodatei im Format MP3, WAV oder FLAC hochladen.", text_align='left')
st.write("- Verarbeitung der Audiodatei: Die App verarbeitet die hochgeladene Audiodatei und extrahiert relevante Merkmale.", text_align='left')
st.write("- Geschlechtererkennung: Mithilfe von Algorithmen und Machine Learning-Techniken erkennt die App das Geschlecht der Stimme.", text_align='left')
st.write("- Anzeige der Ergebnisse: Die App zeigt das erkannte Geschlecht der Stimme zusammen mit weiteren Informationen an.", text_align='left')

st.header("Anwendung")
st.write("Die VoiceChoice-App kann in verschiedenen Szenarien nützlich sein, wie zum Beispiel:")
st.write("- Sprachanalyse: Die App kann bei der Analyse von Sprachaufnahmen unterstützen, indem sie automatisch das Geschlecht der Stimme identifiziert.")
st.write("- Benutzererfahrung: Unternehmen können die VoiceChoice-App verwenden, um die Benutzererfahrung ihrer sprachbasierten Anwendungen zu verbessern, indem sie personalisierte Inhalte basierend auf dem Geschlecht der Benutzerstimme bereitstellen.")
st.write("- Forschung und Studien: Die App kann in wissenschaftlichen Studien oder Forschungsprojekten eingesetzt werden, um Daten über das Geschlecht der Teilnehmerstimmen zu sammeln.")

st.header("Hinweis")
st.write("Die VoiceChoice-App bietet eine gute Genauigkeit bei der Geschlechtererkennung, jedoch können individuelle Ergebnisse variieren und die App sollte nicht als endgültiges Mittel zur Geschlechtsbestimmung verwendet werden.")




if __name__ == "__main__":
    main()
