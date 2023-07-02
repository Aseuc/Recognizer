import streamlit as st
def main():
    st.header("Moin und auch hier nochmal Herzlich Willkommen zu VoiceChoice!")

    st.write("Es handelt es sich um eine App, die in der Lage ist, männliche und weibliche Stimmen zu erkennen.")
    st.write("Doch bevor wir zu dieser App gekommen sind, haben wir intensiv über andere Ideen nachgedacht.")
    
             
    st.subheader("Welche Ideen hatten wir?")
    
    st.subheader("1. Analyse von Essen")
    st.write("Eine Überlegung war die Analyse von Essen. Die App sollte anhand von Fotos von Mahlzeiten Informationen über Nährwerte, Kaloriengehalt und ähnliche Daten liefern kann. Diese Funktion könnte beispielsweise in Restaurants oder der Lebensmittelindustrie von großem Nutzen sein, um Kunden detaillierte Informationen über ihre Mahlzeiten zu bieten.")
    st.write("Wieso haben wir das nicht genommen?")
    st.write("Es wäre schwierig, genaue Informationen über Nährwerte und Kaloriengehalt ohne tiefgreifendes Wissen im Bereich der Nahrungsmittelanalyse zu liefern.")
  
    st.subheader("2. Erkennung von Tiergeräuschen")
    st.write("Eine weitere Idee war die Fähigkeit zur Erkennung von Tiergeräuschen. Dies hätte insbesondere in landwirtschaftlichen oder zoologischen Anwendungen von großem Vorteil sein können, um Tierarten anhand ihrer charakteristischen Geräusche zu identifizieren und Überwachungssysteme zu verbessern.")
    st.write("Wieso haben wir das nicht genommen?")
    st.write("Die Vielfalt der Tierstimmen und die Komplexität der Umgebungsgeräusche hätten es schwierig gemacht, eine genaue Klassifizierung zu erreichen.")
    
    st.subheader("3. Musikgenre-Erkennung")
    st.write("Diese Funktion könnte Musikliebhabern dabei helfen, neue Künstler und Songs zu entdecken, die ihrem individuellen Geschmack entsprechen.")
    st.write("Wieso haben wir das nicht genommen?")
    st.write("Die Vielfalt der musikalischen Stile und die individuellen Unterschiede in den Stimmen machten eine präzise Klassifizierung schwierig.")
   
    st.subheader("4. Nutzung von Schall zur Bestimmung von Entfernungen.")
    st.write("Dies wäre eine App mit der man abschätzen könnte, wie weit entfernt eine Person oder ein Gegenstand ist. Dies könnte in Situationen, in denen die visuelle Wahrnehmung eingeschränkt ist, äußerst nützlich sein.")

    st.header("Aber jetzt zu VoiceChoice")
    st.write("Inspiriert von Filmen wie James Bond, in denen der Held mit seiner Stimme Türen und Safes öffnet, haben wir uns entschlossen, eine Stimmerkennungs-App zu entwickeln, die nicht nur beeindruckend ist, sondern auch viele weitere Vorteile bietet.")
    st.write("Stellt euch vor, eure Oma kann sich keine Passwörter merken, aber will ihr iPad verwenden. Wie cool wäre es, wenn sie einfach nur hineinspricht und das iPad sie erkennt und entsperrt? ")
    st.write("Das ist doch absolut fantastisch!")
    st.write("Aber das ist noch nicht alles! Stellt euch vor, ihr müsstet vor dem Betreten eines Gebäudes in einen Lautsprecher sprechen und die App entscheidet anhand eurer Stimme, ob ihr wütend oder ruhig seid und ob euch Einlass gewährt wird.")
    st.write("Das ist doch der Wahnsinn, oder?")

    st.subheader("Und nun zur Datensammlung:")
    
    st.write("Für die Datensammlung zur Stimmerkennung haben wir einen sorgfältigen Ansatz gewählt, um eine vielfältige und repräsentative Datenbasis zu erhalten. Wir haben Freunde, Familie und Kollegen gebeten, an der Datenerhebung teilzunehmen. Sie haben wiederum ihre eigenen Freunde und Bekannten eingeladen, sich zu beteiligen.")
    st.write("Auf diese Weise konnten wir eine breite Palette von Stimmen unterschiedlicher Altersgruppen, Geschlechter und Dialekte einschließen.")
    st.write("Um sicherzustellen, dass die Datenerhebung von hoher Qualität ist, haben wir sowohl standardisierte Aufgaben als auch kreative Ansätze verwendet. Bei den standardisierten Aufgaben haben die Teilnehmer zum Beispiel das Alphabet aufgesagt oder Witze erzählt.")
    st.write("Dadurch konnten wir strukturierte Daten sammeln und eine Vergleichbarkeit zwischen den Aufnahmen sicherstellen.")
    st.write("Darüber hinaus haben wir uns für einen kreativen Ansatz entschieden. Gemeinsam haben wir das erste Kapitel von "Stolz und Vorurteil" vorgelesen und die Aufnahmen in kurze 3-Sekunden-Segmente aufgeteilt. Dadurch konnten wir natürliche Sprachmuster und Variationen erfassen und die Vielfalt der Stimmen besser abbilden.")
    st.write("Nach der Datensammlung haben wir umfangreiche Vorverarbeitungsschritte durchgeführt, um die Daten für die Modellentwicklung vorzubereiten. Dabei haben wir verschiedene Merkmale angewendet, wie zum Beispiel Mel-frequency Cepstral Coefficients (MFCC) zur Darstellung des Spektrums von Audiosignalen,")
    st.write("spektrale Kontraste zur Identifizierung herausragender Merkmale und Muster im Frequenzbereich, Lautstärke zur Erfassung der Klangintensität, Zero Crossing Rate zur Unterscheidung zwischen stimmhaften und stimmlosen Klängen sowie Bandbreite zur Erfassung von Klangfarbe und Verteilung der Frequenzkomponenten.")
    st.write("Diese umfangreiche Vorverarbeitung hilft uns, relevante Informationen aus den Audiodaten zu extrahieren und eine geeignete Darstellungsform für die Modellentwicklung zu erzeugen. Dadurch können wir Aufgaben wie Spracherkennung oder Klassifikation von Klangereignissen effizient durchführen.")
    st.write("Mit all diesen Vorarbeiten konnten wir schließlich zur eigentlichen Entwicklung der App übergehen. Unser Ziel war es, eine benutzerfreundliche und zuverlässige Streamlit-App zu schaffen, die in der Lage ist, die Stimmen von Männern und Frauen zuverlässig zu erkennen. Die App bietet eine intuitive Benutzeroberfläche,")
    st.write("auf der die Benutzer ihre Aufnahmen hochladen und die Ergebnisse in Echtzeit anzeigen können.")

    st.subheader("Blick in die Zukunft:")
    st.write("Unser großes Ziel ist die Erkennung von einzelnen Personen und Emotionen. Um den Alltag zu erleichern und Sicherheit anzustrebe. ")
    st.write(" Überlegt haben wir uns Verfahren wie Sprachbiometrie zur Personenerkennung und Algorithmen des maschinellen Lernens zur Emotionserkennung zu nutzen. Durch kontinuierliches Training mit mehr und umfangreicheren Daten und Validierung in realen Anwendungen wollen wir die Genauigkeit verbessern und eine personalisierte und reaktionsschnelle Benutzererfahrung anstreben.")
  
#st.subheader("vv")
#st.markdown()
#st.title()
