from spaczz.matcher import FuzzyMatcher
import spacy
from spacy.tokens import Span
from spacy.util import filter_spans
import os
from transformers import pipeline
import networkx as nx
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import pytesseract
import cv2
import time
from time import perf_counter
import json
from pdf2image.pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
from PyPDF2 import PdfFileReader
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import glob
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider
from PyQt5.QtCore import Qt
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#1. Ordner anlegen mit Namen "image" und "images/many_images" im gleichen Verzeichnis wie die .py Datei
#2. Ordner anlegen mit Namen "txt" und "txt/many_txts" im gleichen Verzeichnis wie die .py Datei
#3. Ordner anlegen mit Namen "pdf" im gleichen Verzeichnis wie die .py Datei und die PDFs dort ablegen
#4. (Absoluter) Pfad für poppler und tesseract überprüfen und notfalls anpassen  -> https://github.com/Belval/pdf2image/issues/101
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
absoluter_pfad_image = "/Users/abinashselvarajah/Desktop/Bachelorarbeit/Code/images"
absoluter_pfad_txt = "/Users/abinashselvarajah/Desktop/Bachelorarbeit/Code/txts"
def txt_preprocessing(txt):
    txt = txt.replace('$', '§')
    lower_txt = txt.lower()
    index = lower_txt.find("textliche festsetzungen")
    if index != -1:
        return txt[index:]
    else:
        index = lower_txt.find("planungsrechtliche festsetzungen")
        if index != -1:
            return txt[index:]
        else:
            index = lower_txt.find("art der baulichen nutzung")
            if index != -1:
                return txt[index:]
            else:
                index = lower_txt.find("maß der baulichen nutzung")
                if index != -1:
                    return txt[index:]
                else:
                    index = lower_txt.find("baulichen Nutzung")
                    if index != -1:
                        return txt[index:]
    return 0

#Eingabe 1.) relativer Pfad zum Ordner mit den PDFs von .py Datei
def dir_pdf_txt_conversion(input_path,output_path):
    # Laufvariable für die Bildbenennung
    i = 0
    # Pfad zu PDF angeben
    for file in os.listdir(input_path):
        # Bool-Wert, ob viele Seiten in PDF vorliegen oder nicht
        txt_name = file.split("/")[-1].split(".")[0]
        many_images = False
        if file.endswith(".pdf"):
            tmp_path = os.path.join(input_path, file)
            # PDF in jpg konvertieren                                                   BITTE ÜBERPRÜFEN
            try:
                images = convert_from_path(tmp_path,300, poppler_path='/opt/homebrew/Cellar/poppler/23.06.0/bin')
            except:
                images = convert_from_path(tmp_path,200, poppler_path='/opt/homebrew/Cellar/poppler/21.08.0/bin')
            # pdf zu jpg konvertieren
            if(len(images) == 1):
                images[0].save(absoluter_pfad_image + '/page'+ str(i) +'.jpg', 'JPEG')
            elif(len(images) >= 2):
                # viele Seiten liegen vor (aeltere Dokumente)
                many_images = True
                for j in range(0,len(images)):
                    # Bilder werden in einem gesonderten Ordner "many_images" gespeichert
                    images[j].save('/Users/abinashselvarajah/Desktop/Bachelorarbeit/Code/images/many_images/page'+ str(j) +'.jpg', 'JPEG')
            else:
                raise Exception("Keine Seiten")

            if(many_images == False):
                # Pfad zu jpg-Bild angeben
                image = cv2.imread('images/page'+ str(i) +'.jpg')
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Pfad zu Tesseract.exe angeben (MacOS)                                 BITTE ÜBERPRÜFEN
                pytesseract.pytesseract.tesseract_cmd = r'/Users/abinashselvarajah/anaconda3/envs/bachelorarbeit/bin/tesseract'
                # Texterkennung aus jpg-Bild, Sprache: Deutsch
                result = pytesseract.image_to_string(gray, lang='deu', config='--psm 1')
                with open(output_path+'/'+txt_name+'.txt', 'w') as f:
                    #result = txt_preprocessing(result)
                    f.write(result)
                    f.close()
            elif(many_images == True):
                for k in range(0,len(images)):
                    # Pfad zu jpg-Bild angeben
                    image = cv2.imread('images/many_images/page'+ str(k) +'.jpg')
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # Pfad zu Tesseract.exe angeben                                     BITTE ÜBERPRÜFEN
                    pytesseract.pytesseract.tesseract_cmd = r'/Users/abinashselvarajah/anaconda3/envs/bachelorarbeit/bin/tesseract'
                    # Texterkennung aus jpg-Bild, Sprache: Deutsch
                    result = pytesseract.image_to_string(gray, lang='deu', config='--psm 1')
                    with open(output_path+'/many_txts/page_'+ str(k)+ '.txt', 'w') as f:
                        #result = txt_preprocessing(result)
                        f.write(result)
                        f.close()


                # Löschen der jpg-Bilder
                for m in range(0,len(images)):
                    os.remove(absoluter_pfad_image+'/many_images/page'+ str(m) +'.jpg')

                # Fetch filenames with pattern 'page_*.txt' from directory 'many_txts'
                filenames = glob.glob(os.path.join(output_path, 'many_txts', 'page_*.txt'))

                # sort filenames by number that comes after 'page_' in their names
                filenames.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

                # Merge the text files
                with open(output_path+'/'+txt_name+'.txt', 'w') as outfile:
                    for filename in filenames:
                        with open(filename, 'r') as infile:
                            outfile.write(infile.read())

                # Löschen der txt-Dateien
                for m in range(0,len(images)):
                    os.remove(output_path+'/many_txts/page_'+ str(m)+ '.txt')
                i += 1

def file_pdf_txt_conversion(input_path,output_path):
    # Bool-Wert, ob viele Seiten in PDF vorliegen oder nicht
    many_images = False
    i = 0
    txt_name = input_path.split("/")[-1].split(".")[0]
    if input_path.endswith(".pdf"):
        # PDF in jpg konvertieren                                                   BITTE ÜBERPRÜFEN
        try:
            images = convert_from_path(input_path,300,poppler_path='/opt/homebrew/Cellar/poppler/23.06.0/bin')
        except:
            images = convert_from_path(input_path,200,poppler_path='/opt/homebrew/Cellar/poppler/21.08.0/bin')
        # pdf zu jpg konvertieren
        if(len(images) == 1):
            images[0].save('/Users/abinashselvarajah/Desktop/Bachelorarbeit/Code/images/page'+ str(i) +'.jpg', 'JPEG')
            print("Eine Seite")
        elif(len(images) >= 2):
            # viele Seiten liegen vor (aeltere Dokumente)
            many_images = True
            for j in range(0,len(images)):
                # Bilder werden in einem gesonderten Ordner "many_images" gespeichert
                images[j].save('/Users/abinashselvarajah/Desktop/Bachelorarbeit/Code/images/many_images/page'+ str(j) +'.jpg', 'JPEG')
            print("Mehr als 2 Seiten")
        else:
            raise Exception("Keine Seiten")

        if(many_images == False):
            # Pfad zu jpg-Bild angeben
            image = cv2.imread('images/page'+ str(i) +'.jpg')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Pfad zu Tesseract.exe angeben (MacOS)                                 BITTE ÜBERPRÜFEN
            pytesseract.pytesseract.tesseract_cmd = r'/Users/abinashselvarajah/anaconda3/envs/bachelorarbeit/bin/tesseract'
            # Texterkennung aus jpg-Bild, Sprache: Deutsch
            result = pytesseract.image_to_string(gray, lang='deu', config='--psm 1')
            with open(output_path+'/'+txt_name+'.txt', 'w') as f:
                #result = txt_preprocessing(result)
                f.write(result)
                f.close()
        elif(many_images == True):
            for k in range(0,len(images)):
                # Pfad zu jpg-Bild angeben
                image = cv2.imread('images/many_images/page'+ str(k) +'.jpg')
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Pfad zu Tesseract.exe angeben                                     BITTE ÜBERPRÜFEN
                pytesseract.pytesseract.tesseract_cmd = r'/Users/abinashselvarajah/anaconda3/envs/bachelorarbeit/bin/tesseract'
                # Texterkennung aus jpg-Bild, Sprache: Deutsch
                result = pytesseract.image_to_string(gray, lang='deu', config='--psm 1')
                with open(output_path+'/many_txts/page_'+ str(k)+ '.txt', 'w') as f:
                    #result = txt_preprocessing(result)
                    f.write(result)
                    f.close()

            # Löschen der jpg-Bilder
            for m in range(0,len(images)):
                os.remove('images/many_images/page'+ str(m) +'.jpg')

            # Fetch filenames with pattern 'page_*.txt' from directory 'many_txts'
            filenames = glob.glob(os.path.join(output_path, 'many_txts', 'page_*.txt'))

            # sort filenames by number that comes after 'page_' in their names
            filenames.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

            # Merge the text files
            with open(output_path+'/'+txt_name+'.txt', 'w') as outfile:
                for filename in filenames:
                    with open(filename, 'r') as infile:
                        outfile.write(infile.read())

            # Löschen der txt-Dateien
            for m in range(1,len(images)):
                os.remove(output_path+'/many_txts/page_'+ str(m-1)+ '.txt')
        return output_path+'/'+txt_name+'.txt'

#mDeBERTa-Modell für QA
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="deutsche-telekom/electra-base-de-squad2",
    tokenizer="deutsche-telekom/electra-base-de-squad2"
)


#Hol dir Antworten
def get_answers(question, context):
    # Führe das QA-Model mit der Frage aus
    result = qa_pipeline({
        'question': question,
        'context': context
    })
    #print(f"Question: {question}")
    #print(f"Answer: {result['answer']}")
    #print(f"Score: {result['score']}")
    return result['answer'], result['score']

#Vergleiche die Antworten über alle Gebiete hinweg und hol dir das Ergebnis mit dem größten Confidence-Score
def get_highest_score_answer(scores, answers):
    
    # Check both scores and answers have same length
    assert len(scores) == len(answers), "scores and answers must have the same length"
    
    # Find the index of the highest score
    max_score_index = scores.index(max(scores)) 
    
    # Return the corresponding answer
    return answers[max_score_index], scores[max_score_index]

def get_results(text, threshold):
    text = text.replace("\n", " ")
    nlp_ner = spacy.load("model-best")
    many_baugebiete = 0
    betreten = 0
    ents = nlp_ner(text)
    baugebiete = []
    alle_baugebiete = []
    final = []
    for i in range(len(ents.ents)):
        if("BAUGEBIETE" in ents.ents[i].label_):
            alle_baugebiete.append(ents.ents[i].text)
            if(i != len(ents.ents)-1):
                if("BAUGEBIETE" not in ents.ents[i+1].label_ and betreten == 1):
                    baugebiete.append(ents.ents[i].text)
                    betreten = 0
                    baugebiete = list(dict.fromkeys(baugebiete))
                if("BAUGEBIETE" in ents.ents[i+1].label_ and i-1 <= len(ents.ents)):
                    baugebiete.append(ents.ents[i].text)
                    many_baugebiete = 1
                    betreten = 1
                for j in range(i+1, len(ents.ents)):
                    if("ART_DER_BAULICHEN_NUTZUNG" in ents.ents[j].label_):
                        if many_baugebiete == 1:
                            for baugebiet in baugebiete:
                                answer, score = get_answers("Sind " + ents.ents[j].text + " im " + baugebiet + " zulässig, nicht zulässig, unzulässig, ausnahmsweise zulässig oder nicht Bestandteil des Bebauungsplanes ? ", text)
                                if(score >= threshold):
                                    final.append([score, answer, ents.ents[j].text, baugebiet])
                        elif many_baugebiete == 0:
                            answer, score = get_answers("Sind " + ents.ents[j].text + " im " + ents.ents[i].text + " zulässig, nicht zulässig, unzulässig, ausnahmsweise zulässig oder nicht Bestandteil des Bebauungsplanes ? ", text)
                            if(score >= threshold):
                                final.append([score, answer, ents.ents[j].text, ents.ents[i].text])
                        many_baugebiete = 0
                        baugebiete = []
                    if("MASS_DER_BAULICHEN_NUTZUNG" in ents.ents[j].label_):
                        if many_baugebiete == 1:
                            for baugebiet in baugebiete:
                                answer, score = get_answers("Welche " + ents.ents[j].text + " ist in dem " + baugebiet + " festgelegt ? ", text)
                                if(score >= threshold):
                                    final.append([score, answer, ents.ents[j].text, baugebiet])
                        elif many_baugebiete == 0:
                            answer, score = get_answers("Welche " + ents.ents[j].text + " ist in dem " + ents.ents[i].text + " festgelegt ? ", text)
                            if(score >= threshold):
                                final.append([score, answer, ents.ents[j].text, ents.ents[i].text])
                        many_baugebiete = 0
                        baugebiete = []
                    if("BAUWEISE" in ents.ents[j].label_):
                        if many_baugebiete == 1:
                            for baugebiet in baugebiete:
                                answer, score = get_answers("Ist die Bauweise offen, geschlossen oder abweichend im " + baugebiet + " ? ", text)
                                if(score >= threshold):
                                    final.append([score, answer, ents.ents[j].text, baugebiet])
                        elif many_baugebiete == 0:
                            answer, score = get_answers("Ist die Bauweise offen, geschlossen oder abweichend im " + ents.ents[i].text + " ? ", text)
                            if(score >= threshold):
                                final.append([score, answer, ents.ents[j].text, ents.ents[i].text])
                        many_baugebiete = 0
                        baugebiete = []
                    if("BAUGEBIETE" in ents.ents[j].label_):
                        break
    alle_baugebiete = list(dict.fromkeys(alle_baugebiete))
    return final, alle_baugebiete, len(alle_baugebiete), len(ents.ents)-len(alle_baugebiete)
def create_networkx_graph(naming_string, input_list):
    # Create a directed graph
    G = nx.DiGraph()

    # Add the main node
    G.add_node(naming_string)

    # Iterate over the input list and add the nodes and edges to the graph
    for item in input_list:
        weight, description, name, category = item
        
        # Build the edge attribute string
        edge_info = str(weight) + ", " + description.strip()

        # Add the subnode (category) to the graph if it's not already there
        if not G.has_node(category):
            G.add_node(category)
        
        # Each subnode has a unique subsubnode which is the 'name' in current context
        subsubnode_name = f"{name}"

        # Add the edge from category to the subsubnode (name) along with its information
        G.add_edge(category, subsubnode_name, description=edge_info)

        # Connect main_node to category
        G.add_edge(naming_string, category)

        # we'll define the node lists as per the hierarchy in your graph with naming_string being the main node, 
    # category acting as sub-nodes and subsubnodes as the last hierarchy in your graph.

    main_node = [n for n in G if G.in_degree(n)==0]  # The main node has a in-degree of 0
    sub_nodes = [n for n in G if G.out_degree(n)!=0 and G.in_degree(n)!=0]  # sub nodes have both in-degrees and out-degrees
    subsub_nodes = [n for n in G if G.out_degree(n)==0]  # subsub nodes have a out-degree of 0

    # Prepare for drawing
    plt.figure(figsize=(15, 10))
    pos = nx.shell_layout(G, [main_node, sub_nodes, subsub_nodes])  # Notice how we're providing the node lists here   

    # Draw the graph using nx.draw_networkx instead of nx.draw
    nx.draw_networkx(G, pos, with_labels=True)

    # Edge Descriptions
    edge_labels = nx.get_edge_attributes(G, 'description')  # Get edge descriptions
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)  # Draw edge descriptions

    # Show the plot
    plt.show()
    return G

class FileDragAndDrop(QLabel):
    def __init__(self):
        super().__init__()

        self.setAcceptDrops(True)
        self.setText('Drag & Drop your file here')
        self.slider_value = 0

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()
        self.setText('File path: ' + file_path)
        self.placeholder_function(file_path, self.slider_value)

    def placeholder_function(self, file_path, slider_value):
        filename = file_pdf_txt_conversion(file_path, "bsp_txt_src")
        with open(filename) as f:
            file = f.read()
            filename = filename.split("/")[-1]
            file = file.replace('\n', ' ')
            result, len_bau, len_ents = (get_results(file, slider_value))
            # Step 1: Create a dictionary with the highest confidence scores for each entity and construction area and standardize the entity
            dict_scores = {}
            for entry in result:
                try:
                    entry[2] = normalize_entity(entry[2]).ents[0].label_
                except:
                    pass
                if "Stellplätze" in entry[2] or "Garagen" in entry[2]:
                    entry[2] = entry[2] + " -> Voraussetzungen für den Bau von Stellplätzen/Gebäuden sind zu prüfen"
                if "einzelhandel" in entry[2]:
                    entry[2] = entry[2] + " -> Sortiment und Verkaufsfläche sind zu prüfen"
                score = entry[0]
                elements = (entry[2], entry[3])
                if elements in dict_scores:
                    if score > dict_scores[elements]:
                        dict_scores[elements] = score
                else:
                    dict_scores[elements] = score

            # Step 2: Sort the confidence scores in descending order
            sorted_scores = sorted(dict_scores.values(), reverse=True)

            # Step 3: Create a new list with the entries corresponding to the highest confidence scores
            summarized = []
            for score in sorted_scores:
                for entry in result:
                    if entry[0] == score and (entry[2], entry[3]) not in [x[2:4] for x in summarized]:
                        summarized.append(entry)
                        break

            # Step 4: Create a json file with the summarized results
            json_data = {}
            for item in result:
                if item[3] not in json_data:
                    json_data[item[3]] = []
                if("zahl" in item[2] or "höhe" in item[2] or "BMZ" in item[2] or "GFZ" in item[2] or "GRZ" in item[2]):
                    json_data[item[3]].append({"confidence_score": item[0], "Zulaessigkeit": item[1], "Maß der baulichen Nutzung": item[2]})
                if("bauweise" in item[2]):
                    json_data[item[3]].append({"confidence_score": item[0], "zulaessigkeit": item[1], "Bauweise": item[2]})
                else:
                    json_data[item[3]].append({"confidence_score": item[0], "zulaessigkeit": item[1], "Art der baulichen Nutzung": item[2]})
            with open(filename+'.json', 'w', encoding='utf-8') as outfile:
                json.dump(json_data, outfile, ensure_ascii=False)
                
            create_networkx_graph(filename, result)
    def set_slider_value(self, value):
        self.slider_value = value

class AppDemo(QWidget):
    def __init__(self):
        super().__init__()

        self.resize(400, 300)
        self.setWindowTitle('ML_Informationsextraktion')
        self.setStyleSheet("background-color: #333; color: #ddd;")  # Set background and text color

        mainLayout = QVBoxLayout()

        self.label = FileDragAndDrop()
        self.label.setStyleSheet("font-size: 20px;")  # Increase font size
        mainLayout.addWidget(self.label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setStyleSheet("background-color: #555;")  # Change slider color
        self.slider.valueChanged.connect(self.slider_changed)
        mainLayout.addWidget(self.slider)

        self.slider_label = QLabel('Threshold: 0')
        self.slider_label.setStyleSheet("font-size: 16px;")  # Increase font size
        mainLayout.addWidget(self.slider_label)

        self.setLayout(mainLayout)

    def slider_changed(self, value):
        self.label.set_slider_value(value / 100)
        self.slider_label.setText('Threshold: ' + str(value / 100))


app = QApplication(sys.argv)
demo = AppDemo()
demo.show()
sys.exit(app.exec_())
