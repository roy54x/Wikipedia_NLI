# This Python file uses the following encoding: utf-8

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import wikipedia
import numpy as np
import matplotlib.pyplot as plt
model_name = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name )

def compare_sentenses(s1, s2):

       input = tokenizer(s1, s2, truncation=True, return_tensors="pt")
       output = model(input["input_ids"])

       prediction = torch.softmax(output["logits"][0], -1).tolist()
       diff = prediction[2]-prediction[0]
       label_names = ["entailment", "neutral", "contradiction"]
       prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
       return diff

wikipedia.set_lang("ar")
ar_list  = wikipedia.page('Al-Ahli Arab Hospital explosion').content.split('\n')

wikipedia.set_lang("he")
he_list  = wikipedia.page('Al-Ahli Arab Hospital explosion').content.split('\n')

l_1 = len(ar_list)

l_2 = len(he_list)
print(l_1, l_2) 

heatmap = np.zeros((l_1, l_2))
for i, s1 in enumerate(ar_list):
       for j, s2 in enumerate(he_list):
              heatmap[i, j] = compare_sentenses(s1, s2)
              print(i, j)
fig, ax = plt.subplots()
im = ax.imshow(heatmap)

# Add the color bar
cbar = ax.figure.colorbar(im, ax = ax)
cbar.ax.set_ylabel("Color bar", rotation = -90, va = "bottom")

plt.show()    
#ar_1=ar_1.decode('utf-8','ignore').encode("utf-8")
t1 = "the explosion was caused by an israeli air strike"
t2 = "the explosion was caused by a missle shot by hammas"
en_1 = "On 17 October 2023, an explosion took place in the parking lot of the courtyard of " \
          "al-Ahli Arab Hospital in Gaza City during the 2023 Israel–Hamas war, resulting in " \
          "a large number of fatalities and injuries among displaced Palestinians seeking shelter there."
en_2 = "Reports of the number of fatalities vary widely. The Gaza Health Ministry reported 342 injured and " \
            "471 killed. The Anglican diocese that manages the hospital reported 200 people killed. US intelligence " \
            "agencies assessed a death toll between 100 and 300. A report by Human Rights Watch also questioned " \
            "the Health Ministry's casualty figures. The cause of the explosion is contested. Israel, " \
            "the United States, France, the United Kingdom, and Canada said that their intelligence sources " \
            "indicate the cause of the explosion was a failed rocket launch from within Gaza by the Palestinian " \
            "Islamic Jihad (PIJ). Hamas and PIJ stated the explosion was caused by an Israeli airstrike."

es_1 = "La masacre del Hospital Bautista Al-Ahli fue la matanza de un gran grupo de personas, " \
               "producida el 17 de octubre de 2023 en el patio del Hospital Bautista Al-Ahli Arabi, situado " \
               "en el barrio de Al Zeitún en el centro de Gaza, en el que varios miles de palestinos desplazados " \
               "buscaban refugio, lo que posiblemente causó un número de muertes mayor que cualquier " \
               "otro evento en Gaza desde 2008."
es_2 = "Las bajas reportadas varían dependiendo de las fuentes, en un primer momento el Ministerio de " \
       "Sanidad de Gaza habló de más de 500 muertos. Posteriormente, la Defensa Civil " \
       "de Gaza situó el número de fallecidos en 300 y el día 18, el viceministro de Sanidad de la " \
       "Franja, Youssef Abu Al Rish, reportó una cifra de 400 muertos. Ese mismo día, el " \
       "Ministerio de Salud de Gaza, cifró el número de víctimas mortales en al menos 471 y 28 " \
       "heridos en estado crítico, además de 314 personas con heridas de diversa consideración." \
       "Fuentes in situ elevan el número de víctimas a más de 800, pudiendo llegar a " \
       " Según fuentes de inteligencia de los Estados Unidos hubo entre 100 y 300 muertos." \
       "Fuentes occidentales consideran que no es posible determinar el número de fallecidos de " \
       "forma concluyente por falta de evidencia, pero las imágenes del hospital verificadas " \
       "por The New York Times y los relatos de los testigos dejan claro que la cifra es alta."
es_3 = "a causa de la explosión es discutida. El Ministerio de Salud de Gaza desde el primer momento dijo " \
       "que fue un ataque aéreo israelí; mientras que las Fuerzas de Defensa de Israel (FDI) " \
       "aseguran que fue provocada por el lanzamiento fallido de un cohete por parte de la Yihad Islámica " \
       "Palestina. La Yihad Islámica Palestina, la Autoridad Nacional Palestina y los países árabes y " \
       "musulmanes acusaron a Israel de ser el perpetrador. Las autoridades Israelíes han negado la " \
       "acusación con el apoyo de Estados Unidos (su principal aliado)."

