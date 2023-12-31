import streamlit as st
from streamlit_lottie import st_lottie
import requests
import VoiceChoice as vc

# Animationen:
lottie_url = "https://assets2.lottiefiles.com/packages/lf20_bXsnEx.json"
additional_lottie_url = "https://assets5.lottiefiles.com/packages/lf20_ilaks9mg.json"
another_lottie_url = "https://assets4.lottiefiles.com/packages/lf20_GFK3CDFCrx.json"


def load_lottie_animation(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None


def main():
    st.set_page_config(
        page_title="Motivation",
        page_icon="favicon.ico",
        layout='wide',
        initial_sidebar_state="collapsed",
    )
    vc.add_logo_sidebar()
    co1, co2 = st.columns([1, 3])

    with co1:
        animation = load_lottie_animation(lottie_url)
        if animation:
            st_lottie(animation, width=200, height=200, key="animation")

    with co2:
        st.title("Moin und Herzlich Willkommen zu VoiceChoice!")
        st.subheader(
            "Bei VoiceChoice handelt es sich um eine App, die in der Lage ist, männliche und weibliche Stimmen zu "
            "erkennen.")

    # spalten erstellen
    col1, col2 = st.columns([3, 1])

    with col2:
        additional_animation = load_lottie_animation(additional_lottie_url)
        if additional_animation:
            st_lottie(additional_animation, width=400, height=400, key="additional_animation")

    # Buttons und zusätzliche Animation in der rechten Spalte anzeigen
    with col1:
        st.write("Doch bevor wir zu dieser App gekommen sind, haben wir intensiv über andere Ideen nachgedacht.")
        st.subheader("Welche Ideen hatten wir?")

        # CSS-Stil für die Buttons
        button_style = """
            <style>
            .custom-button.button1 {
                background-color: #0000FF; /* Blau */
                color: #FFFFFF;
                border-color: #0000FF;
                border-radius: 5px;
                padding: 0.5rem 1rem;
            }

            .custom-button.button2 {
                background-color: #FF00FF; /* Pink */
                color: #FFFFFF;
                border-color: #FF00FF;
                border-radius: 5px;
                padding: 0.5rem 1rem;
            }

            .custom-button.button3 {
                background-color: #FFA500; /* Orange */
                color: #FFFFFF;
                border-color: #FFA500;
                border-radius: 5px;
                padding: 0.5rem 1rem;
            }
             .custom-button.button4 {
            background-color: #00FF00; /* Grün */
            color: #FFFFFF;
            border-color: #00FF00;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        </style>
        """

        # CSS-Stil in Streamlit einfügen
        st.markdown(button_style, unsafe_allow_html=True)

        # Button 1 - Analyse von Essen
        if st.button("1. Analyse von Essen", key="button1", help="Eine Überlegung war die Analyse von Essen."):
            st.write(
                "Eine Überlegung war die Analyse von Essen. Die App sollte anhand von Fotos von Mahlzeiten "
                "Informationen über Nährwerte, Kaloriengehalt und ähnliche Daten liefern kann. Diese Funktion könnte "
                "beispielsweise in Restaurants oder der Lebensmittelindustrie von großem Nutzen sein, "
                "um Kunden detaillierte Informationen über ihre Mahlzeiten zu bieten.")
            st.write("<i>Wieso haben wir das nicht genommen?</i>", unsafe_allow_html=True)
            st.write(
                "Es wäre schwierig, genaue Informationen über Nährwerte und Kaloriengehalt ohne tiefgreifendes Wissen "
                "im Bereich der Nahrungsmittelanalyse zu liefern.")

        # Button 2 - Erkennung von Tiergeräuschen
        if st.button("2. Erkennung von Tiergeräuschen", key="button2",
                     help="Eine weitere Idee war die Fähigkeit zur Erkennung von Tiergeräuschen."):
            st.write(
                "Eine weitere Idee war die Fähigkeit zur Erkennung von Tiergeräuschen. Dies hätte insbesondere in "
                "landwirtschaftlichen oder zoologischen Anwendungen von großem Vorteil sein können, um Tierarten "
                "anhand ihrer charakteristischen Geräusche zu identifizieren und Überwachungssysteme zu verbessern.")
            st.write("<i>Wieso haben wir das nicht genommen?</i>", unsafe_allow_html=True)
            st.write(
                "Die Vielfalt der Tierstimmen und die Komplexität der Umgebungsgeräusche hätten es schwierig gemacht, "
                "eine genaue Klassifizierung zu erreichen.")

        # Button 3 - Musikgenre-Erkennung
        if st.button("3. Musikgenre-Erkennung", key="button3",
                     help="Diese Funktion könnte Musikliebhabern dabei helfen, neue Künstler und Songs zu entdecken."):
            st.write(
                "Diese Funktion könnte Musikliebhabern dabei helfen, neue Künstler und Songs zu entdecken, die ihrem individuellen Geschmack entsprechen.")
            st.write("<i>Wieso haben wir das nicht genommen?</i>", unsafe_allow_html=True)
            st.write(
                "Die Vielfalt der musikalischen Stile und die individuellen Unterschiede in den Stimmen machten eine präzise Klassifizierung schwierig.")

        # Button 4 - Nutzung von Schall zur Bestimmung von Entfernungen
        if st.button("4. Nutzung von Schall zur Bestimmung von Entfernungen", key="button4",
                     help="Dies wäre eine App zur Abschätzung von Entfernungen."):
            st.write(
                "Dies wäre eine App mit der man abschätzen könnte, wie weit entfernt eine Person oder ein Gegenstand ist. Dies könnte in Situationen, in denen die visuelle Wahrnehmung eingeschränkt ist, äußerst nützlich sein.")

    st.header("Aber jetzt zu VoiceChoice")
    st.write(
        "Inspiriert von Filmen wie James Bond, in denen der Held mit seiner Stimme Türen und Safes öffnet, haben wir uns entschlossen, eine Stimmerkennungs-App zu entwickeln, die nicht nur beeindruckend ist, sondern auch viele weitere Vorteile bietet.")
    st.write(
        "Stellt euch vor, eure Oma kann sich keine Passwörter merken, aber will ihr Tablet verwenden. Wie cool wäre es, wenn sie einfach nur hineinspricht und das iPad sie erkennt und entsperrt? ")
    st.write("Das ist doch absolut fantastisch!")
    st.write(
        "Aber das ist noch nicht alles! Stellt euch vor, ihr müsstet vor dem Betreten eines Gebäudes in einen Lautsprecher sprechen und die App entscheidet anhand eurer Stimme, ob ihr wütend oder ruhig seid und ob euch Einlass gewährt wird.")
    st.write("Das ist doch der Wahnsinn, oder?")

    st.subheader("Blick in die Zukunft:")
    st.write(
        "Unser großes Ziel ist die Erkennung von einzelnen Personen und Emotionen. Um den Alltag zu erleichern und Sicherheit anzustreben. ")
    st.write(
        "Überlegt haben wir uns Verfahren wie Sprachbiometrie zur Personenerkennung und Algorithmen des maschinellen Lernens zur Emotionserkennung zu nutzen. Durch kontinuierliches Training mit mehr und umfangreicheren Daten und Validierung in realen Anwendungen wollen wir die Genauigkeit verbessern und eine personalisierte und reaktionsschnelle Benutzererfahrung anstreben.")

    co3, co4, co5 = st.columns([1, 1, 1])

    with co4:
        another_lottie = load_lottie_animation(another_lottie_url)
        if another_lottie:
            st_lottie(another_lottie, width=400, height=400, key="additional_animation_2")


if __name__ == "__main__":
    main()
