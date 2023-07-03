import streamlit as st
from streamlit_lottie import st_lottie
import requests
from datetime import datetime
import json
import VoiceChoice as vc

st.set_page_config(
    page_title="Converter",
    page_icon="favicon.ico",
    layout='wide',
    initial_sidebar_state="auto"
)

vc.add_logo_sidebar()

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


text = """Nach unseren Recherchen dachten wir zunächst, dass es einfach wäre, die Varianzen, den Median, die Quantile 
etc. aus den Dezibel-Werte zu verwenden. Aber das war ein Irrtum! Diese Features führten nicht zum Ziel und waren 
nicht signifikant genug, um unser Modell zu trainieren. Also suchten wir weiter nach relevanten Features für 
Audiosequenzen. Dabei entdeckten wir, dass die MFCC und der Spektralkontrast zu den signifikantesten Merkmalen 
gehörten. Da uns das aber noch zu wenig erschien, suchten wir nach weiteren Features und fanden schließlich den MFCC, 
Spektralkontrast, Tonstärke, Bandbreite und die Zero Crossing Rate."""

st.header("Data Preparation")
st.write("Die Datenaufbereitung war für uns eine der längsten Aufgaben. Zunächst mussten wir die Features für die "
         "Daten festlegen. Dies war schwierig, da wir uns fragten: Welche Features sind essentiell? Wie sollen sie "
         "auf Sequenzen angewendet werden? Gibt es Fehlerquellen in unseren gesammelten Daten, die wir nicht sehen "
         "oder hören?")
lottie_url = "https://assets10.lottiefiles.com/packages/lf20_rp8vki3f.json"
lottie_url2 = "https://assets2.lottiefiles.com/packages/lf20_YBa32sJx1i.json"
lottie_url3 = "https://assets8.lottiefiles.com/packages/lf20_7Cyo9b.json"

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    lottie_animation = load_lottie_url(lottie_url)
    st_lottie(lottie_animation, speed=1, width=100, height=100, key="clock")

with col2:
    lottie_animation3 = load_lottie_url(lottie_url3)
    st_lottie(lottie_animation3, speed=1, width=100, height=100, key="dif")

with col3:
    lottie_animation2 = load_lottie_url(lottie_url2)
    st_lottie(lottie_animation2, speed=1, width=100, height=100, key="man")

st.write(text)
lottie_url4 = "https://assets3.lottiefiles.com/packages/lf20_9xRnlw.json"
lottie_animation4 = load_lottie_url(lottie_url4)
col2_lottie = st_lottie(lottie_animation4, speed=1, width=100, height=100, key="again")

col1, col2 = st.columns([10, 1])

with col2:
    lottie_url5 = "https://assets9.lottiefiles.com/packages/lf20_yuisinzc.json"
    lottie_animation5 = load_lottie_url(lottie_url5)
    st_lottie(lottie_animation5, speed=1, width=100, height=100, key="opt")

col1.write("Jedoch war das Festlegen der Features, gerade mal der Anfang. Jetzt mussten die Daten in eine geeignet "
           "Form gebracht werden. Da kamen zunächst weitere Komplikationen auf uns zu. Wir mussten uns überlegen: "
           "Verwenden wir Datensätze die aus Gruppen bestehen oder singulare Datensätze? Die Problematik die dahinter "
           "steht ist, dass MFCC-Werte und Spektral Kontrast-Werte immer mit 5 Zeilen extrahiert wurden und nicht nur "
           "eine einzelne! Wir entschlossen uns aber dazu beide Ansätze zu verwenden und hatten auf beiden Ansätzen "
           "gleich gute Ergebnisse.")
col1.write("  ")

col1.write("Jetzt folgten jedoch weitere Probleme die es zu lösen gab, wir mussten beispielsweise die verschiedenen "
           "Datentypen"
           "festlegen. Werte die vorher als Strings extrahiert wurden mussten umgewandelt werden in Int-/Float-Werte. Trainingsdatensätze "
           "und zur vorhersagende Datensätze mussten in Einklang gebracht werden.")

st.header("Modeling")

st.subheader("Auswahl der Modellierungstechnik")

st.write(
    "Zunächst starteten wir mit der Auswahl unserer Modellierungstechnik. Hierbei wählten wir den Ansatz des UML-Diagramms. Wir überlegten uns zunächst welche Machine Learning Modelle zu unsere Daten passen. "
    "Und definierten schon Funktionen, Klassen und deren Beziehungen zu einander. Uns wurde jedoch während der Implementierung schnell klar, dass wir doch lieber den agilen Prozess verwenden wollten und somit war das UML-Modell nach kurzer Zeit "
    "hinfällig.")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    lottie_url5 = "https://assets8.lottiefiles.com/packages/lf20_rG8vsHpaeB.json"
    lottie_animation5 = load_lottie_url(lottie_url5)
    st_lottie(lottie_animation5, speed=1, width=400, height=400, key="model")

st.subheader("Testmodell erstellen")

col1, col2 = st.columns([10, 1])
with col2:
    lottie_url6 = "https://assets6.lottiefiles.com/packages/lf20_q77jpumk.json"
    lottie_animation6 = load_lottie_url(lottie_url6)
    st_lottie(lottie_animation6, speed=1, width=150, height=150, key="optimize")

col1.write(
    "Nach dem Erstellen unseres UML-Diagramms haben wir verschiedene Test-Modelle gebaut. Dabei haben wir den K-Nearest-Neighbour, "
    "Random-Forest und Support Vector Machines ausprobiert, da diese auf kleinen Datensätzen laut unseren Recherchen immer noch recht gut laufen"
    "hätten sollen. Jedoch überzeugten uns die Ergebnisse dieser Modelle nicht und wir sind zurück zur "
    "Datensammlung gegangen, um für ein"
    "neuronale Netz entsprechend weitere Daten zu sammeln, um große Datenmengen zu erhalten.")

st.subheader("Bewertung des Modells")

st.write("Nachdem wir das Gefühl hatten, wir haben genug Datensätze, erstellten wir nochmals mehrere ML-Modelle. "
         "Dabei waren wir sehr verwundert, denn fast alle unsere Modelle performten zunächst herausragend. Genau das "
         "machte uns stutzig und “Ta Da!” wir hatten wieder mehrere Probleme, die es zu lösen gab. Beispielsweise "
         "hatten wir"
         "das Problem des Overfittings, dass das Modell zu gut auf den Trainingsdatensatz reagierte und schlecht auf "
         "nicht bekannte Datensätze! Somit löschten wir Features, Spalten, Zeilen, alles, wo wir dachten, "
         "es könnte unsere Modelle verbessern! Jedoch vergebens!")

st.write(" ")
col1, col2 = st.columns([2, 1])
with col2:
    with open('97832-m5-progress-bar.json', 'r') as f:
        lottie_animation = json.load(f)
        st_lottie(lottie_animation, speed=1, width=200, height=200, key="progress")

col1.write("Wir gaben aber nicht auf und optimierten zunächst unser neuronales Netz mit verschiedenen Layern und "
           "Neuronen und kamen immer mehr in die richtige Richtung. Somit entschlossen wir uns dazu, "
           "mehrere neuronale Netze mit verschiedenen Layern und Schichten zu kombinieren um die "
           "Validierungsgenauigkeit zu erhöhen!")

st.header("Evaluation")

col1, col2, col3 = st.columns([10, 2, 10])
with col2:
    with open('146225-evaluation.json', 'r') as f:
        lottie_animation = json.load(f)
        st_lottie(lottie_animation, speed=1, width=200, height=200, key="evaluation")

col1.write(
    "In diesem Teil haben wir immer wieder unsere Modelle getestet und sie bewertet. Wir gingen davon aus, dass "
    "unsere Modelle recht gut waren aber noch viel Platz nach oben war. Beispielsweise konnten die Modelle, "
    "was irgendwie auch normal erscheint nicht zwischen männlichen oder weiblichen Stimmen unterscheiden wenn "
    "beide Personen sehr laut gesprochen haben.")
with col3:
    with open('19169-user-testing.json', 'r') as f:
        lottie_animation2 = json.load(f)
        st_lottie(lottie_animation2, speed=1, width=300, height=300, key="test")

st.header("Deployment")

col1, col2, col3 = st.columns([10, 2, 1])
with col2:
    with open('32056-user-experience.json', 'r') as f:
        lottie_animation = json.load(f)
        st_lottie(lottie_animation, speed=1, width=250, height=250, key="ue")

col1.write("Im letzten Schritt haben wir uns viele Gedanken gemacht! Wie kann man dem User eine nutzerfreundliche App "
           "bieten? Was wirkt ansprechend auf den User? Welche Features würde man gerne selbst oder der User in der App"
           "haben wollen? Und vieles mehr... Wir haben uns letztendlich für interaktive Elemente entschieden aber auch "
           "zum Beispiel die Möglichkeit selbst mal ein neuronale Netz trainieren zu können und das mit geringen "
           "Aufwand. Oder "
           "in Echtzeit Aufnahmen zu tätigen und direkt das Neuronale Netz zu verwenden. Oder das Feature bequem aus "
           "Whats App Sprachnotizen oder Videos eine WAV-Datei zu erstellen. Dies alles um die Userexperience so hoch "
           "wie möglich zuhalten.")


