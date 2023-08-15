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
    model="timpal0l/mdeberta-v3-base-squad2",
    tokenizer="timpal0l/mdeberta-v3-base-squad2"
)

#Erkenne Baugebiete und Objekte, die auf den Baugebieten gebaut werden sollen
def normalize_entity(doc):
    nlp = spacy.blank("de")
    ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
    patterns_baugebiete = [
        {"label":"KLEINSIEDLUNGSGEBIETE","pattern":[{"TEXT":{"FUZZY":"kleinsiedlungsgebiet"}}]},
        {"label":"KLEINSIEDLUNGSGEBIETE","pattern":[{"TEXT":{"FUZZY":"kleinsiedlungsgebiete"}}]},
        {"label":"KLEINSIEDLUNGSGEBIETE","pattern":[{'IS_UPPER': True, 'LENGTH': 2, 'LOWER': 'ws'}]},
        {"label": "KLEINSIEDLUNGSGEBIETE","pattern": [{"TEXT": {"REGEX": "^WS($|[^0-9])"}}]},
        {"label":"KLEINSIEDLUNGSGEBIETE_1","pattern":[{'LOWER': 'ws', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '1'}]},
        {"label":"KLEINSIEDLUNGSGEBIETE_2","pattern":[{'LOWER': 'ws', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '2'}]},
        {"label":"KLEINSIEDLUNGSGEBIETE_3","pattern":[{'LOWER': 'ws', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '3'}]},
        {"label":"KLEINSIEDLUNGSGEBIETE_4","pattern":[{'LOWER': 'ws', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '4'}]},
        {"label":"KLEINSIEDLUNGSGEBIETE_5","pattern":[{'LOWER': 'ws', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '5'}]},
        {"label":"KLEINSIEDLUNGSGEBIETE_6","pattern":[{'LOWER': 'ws', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '6'}]},
        {"label":"KLEINSIEDLUNGSGEBIETE_7","pattern":[{'LOWER': 'ws', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '7'}]},
        {"label":"KLEINSIEDLUNGSGEBIETE_1","pattern":[{'LOWER': 'ws1', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"KLEINSIEDLUNGSGEBIETE_2","pattern":[{'LOWER': 'ws2', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"KLEINSIEDLUNGSGEBIETE_3","pattern":[{'LOWER': 'ws3', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"KLEINSIEDLUNGSGEBIETE_4","pattern":[{'LOWER': 'ws4', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"KLEINSIEDLUNGSGEBIETE_5","pattern":[{'LOWER': 'ws5', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"KLEINSIEDLUNGSGEBIETE_6","pattern":[{'LOWER': 'ws6', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"KLEINSIEDLUNGSGEBIETE_7","pattern":[{'LOWER': 'ws7', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"REINE_WOHNGEBIETE","pattern":[{"TEXT":{"FUZZY":"reines"}},{"TEXT":{"FUZZY":"wohngebiet"}}]},
        {"label":"REINE_WOHNGEBIETE","pattern":[{"TEXT":{"FUZZY":"reine"}},{"TEXT":{"FUZZY":"wohngebiete"}}]},
        {"label":"REINE_WOHNGEBIETE","pattern":[{'IS_UPPER': True, 'LENGTH': 2, 'LOWER': 'wr'}]},
        {"label": "KLEINSIEDLUNGSGEBIETE","pattern": [{"TEXT": {"REGEX": "^WR($|[^0-9])"}}]},
        {"label":"REINE_WOHNGEBIETE_1","pattern":[{'LOWER': 'wr', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '1'}]},
        {"label":"REINE_WOHNGEBIETE_2","pattern":[{'LOWER': 'wr', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '2'}]},
        {"label":"REINE_WOHNGEBIETE_3","pattern":[{'LOWER': 'wr', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '3'}]},
        {"label":"REINE_WOHNGEBIETE_4","pattern":[{'LOWER': 'wr', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '4'}]},
        {"label":"REINE_WOHNGEBIETE_5","pattern":[{'LOWER': 'wr', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '5'}]},
        {"label":"REINE_WOHNGEBIETE_6","pattern":[{'LOWER': 'wr', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '6'}]},
        {"label":"REINE_WOHNGEBIETE_7","pattern":[{'LOWER': 'wr', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '7'}]},
        {"label":"REINE_WOHNGEBIETE_1","pattern":[{'LOWER': 'wr1', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"REINE_WOHNGEBIETE_2","pattern":[{'LOWER': 'wr2', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"REINE_WOHNGEBIETE_3","pattern":[{'LOWER': 'wr3', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"REINE_WOHNGEBIETE_4","pattern":[{'LOWER': 'wr4', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"REINE_WOHNGEBIETE_5","pattern":[{'LOWER': 'wr5', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"REINE_WOHNGEBIETE_6","pattern":[{'LOWER': 'wr6', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"REINE_WOHNGEBIETE_7","pattern":[{'LOWER': 'wr7', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"ALLGEMEINE_WOHNGEBIETE","pattern":[{"TEXT":{"FUZZY": "allgemeines"}},{"TEXT":{"FUZZY":"wohngebiet"}}]},
        {"label":"ALLGEMEINE_WOHNGEBIETE","pattern":[{"TEXT":{"FUZZY": "allgemeine"}},{"TEXT":{"FUZZY":"wohngebiete"}}]},
        {"label":"ALLGEMEINE_WOHNGEBIETE","pattern":[{'IS_UPPER': True, 'LENGTH': 2, 'LOWER': 'wa'}]},
        {"label":"ALLGEMEINE_WOHNGEBIETE","pattern": [{"TEXT": {"REGEX": "^WA($|[^0-9])"}}]},
        {"label":"ALLGEMEINE_WOHNGEBIETE_1","pattern":[{'LOWER': 'wa', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '1'}]},
        {"label":"ALLGEMEINE_WOHNGEBIETE_2","pattern":[{'LOWER': 'wa', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '2'}]},
        {"label":"ALLGEMEINE_WOHNGEBIETE_3","pattern":[{'LOWER': 'wa', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '3'}]},
        {"label":"ALLGEMEINE_WOHNGEBIETE_4","pattern":[{'LOWER': 'wa', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '4'}]},
        {"label":"ALLGEMEINE_WOHNGEBIETE_5","pattern":[{'LOWER': 'wa', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '5'}]},
        {"label":"ALLGEMEINE_WOHNGEBIETE_6","pattern":[{'LOWER': 'wa', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '6'}]},
        {"label":"ALLGEMEINE_WOHNGEBIETE_7","pattern":[{'LOWER': 'wa', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '7'}]},
        {"label":"ALLGEMEINE_WOHNGEBIETE_1","pattern":[{'LOWER': 'wa1', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"ALLGEMEINE_WOHNGEBIETE_2","pattern":[{'LOWER': 'wa2', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"ALLGEMEINE_WOHNGEBIETE_3","pattern":[{'LOWER': 'wa3', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"ALLGEMEINE_WOHNGEBIETE_4","pattern":[{'LOWER': 'wa4', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"ALLGEMEINE_WOHNGEBIETE_5","pattern":[{'LOWER': 'wa5', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"ALLGEMEINE_WOHNGEBIETE_6","pattern":[{'LOWER': 'wa6', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"ALLGEMEINE_WOHNGEBIETE_7","pattern":[{'LOWER': 'wa7', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"BESONDERE_WOHNGEBIETE","pattern":[{"TEXT":{"FUZZY": "besonderes"}},{"TEXT":{"FUZZY":"wohngebiet"}}]},
        {"label":"BESONDERE_WOHNGEBIETE","pattern":[{"TEXT":{"FUZZY": "besondere"}},{"TEXT":{"FUZZY":"wohngebiete"}}]},
        {"label":"BESONDERE_WOHNGEBIETE","pattern":[{'IS_UPPER': True, 'LENGTH': 2, 'LOWER': 'wb'}]},
        {"label":"BESONDERE_WOHNGEBIETE","pattern": [{"TEXT": {"REGEX": "^WB($|[^0-9])"}}]},
        {"label":"BESONDERE_WOHNGEBIETE_1","pattern":[{'LOWER': 'wb', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '1'}]},
        {"label":"BESONDERE_WOHNGEBIETE_2","pattern":[{'LOWER': 'wb', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '2'}]},
        {"label":"BESONDERE_WOHNGEBIETE_3","pattern":[{'LOWER': 'wb', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '3'}]},
        {"label":"BESONDERE_WOHNGEBIETE_4","pattern":[{'LOWER': 'wb', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '4'}]},
        {"label":"BESONDERE_WOHNGEBIETE_5","pattern":[{'LOWER': 'wb', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '5'}]},
        {"label":"BESONDERE_WOHNGEBIETE_6","pattern":[{'LOWER': 'wb', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '6'}]},
        {"label":"BESONDERE_WOHNGEBIETE_7","pattern":[{'LOWER': 'wb', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '7'}]},
        {"label":"BESONDERE_WOHNGEBIETE_1","pattern":[{'LOWER': 'wb1', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"BESONDERE_WOHNGEBIETE_2","pattern":[{'LOWER': 'wb2', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"BESONDERE_WOHNGEBIETE_3","pattern":[{'LOWER': 'wb3', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"BESONDERE_WOHNGEBIETE_4","pattern":[{'LOWER': 'wb4', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"BESONDERE_WOHNGEBIETE_5","pattern":[{'LOWER': 'wb5', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"BESONDERE_WOHNGEBIETE_6","pattern":[{'LOWER': 'wb6', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"BESONDERE_WOHNGEBIETE_7","pattern":[{'LOWER': 'wb7', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"DORFGEBIETE","pattern":[{"TEXT":{"FUZZY":"dorfgebiete"}}]},
        {"label":"DORFGEBIETE","pattern":[{"TEXT":{"FUZZY":"dorfgebiet"}}]},
        {"label":"DORFGEBIETE","pattern":[{'IS_UPPER': True, 'LENGTH': 2, 'LOWER': 'md'}]},
        {"label":"DORFGEBIETE","pattern": [{"TEXT": {"REGEX": "^MD($|[^0-9])"}}]},
        {"label":"DORFGEBIETE_1","pattern":[{'LOWER': 'md', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '1'}]},
        {"label":"DORFGEBIETE_2","pattern":[{'LOWER': 'md', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '2'}]},
        {"label":"DORFGEBIETE_3","pattern":[{'LOWER': 'md', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '3'}]},
        {"label":"DORFGEBIETE_4","pattern":[{'LOWER': 'md', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '4'}]},
        {"label":"DORFGEBIETE_5","pattern":[{'LOWER': 'md', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '5'}]},
        {"label":"DORFGEBIETE_6","pattern":[{'LOWER': 'md', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '6'}]},
        {"label":"DORFGEBIETE_7","pattern":[{'LOWER': 'md', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '7'}]},
        {"label":"DORFGEBIETE_1","pattern":[{'LOWER': 'md1', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"DORFGEBIETE_2","pattern":[{'LOWER': 'md2', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"DORFGEBIETE_3","pattern":[{'LOWER': 'md3', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"DORFGEBIETE_4","pattern":[{'LOWER': 'md4', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"DORFGEBIETE_5","pattern":[{'LOWER': 'md5', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"DORFGEBIETE_6","pattern":[{'LOWER': 'md6', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"DORFGEBIETE_7","pattern":[{'LOWER': 'md7', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"DOERFLICHE_WOHNGEBIETE","pattern":[{"TEXT":{"FUZZY": "dörfliches"}},{"TEXT":{"FUZZY":"wohngebiet"}}]},
        {"label":"DOERFLICHE_WOHNGEBIETE","pattern":[{"TEXT":{"FUZZY": "dörfliche"}},{"TEXT":{"FUZZY":"wohngebiete"}}]},
        {"label":"DOERFLICHE_WOHNGEBIETE","pattern":[{'IS_UPPER': True, 'LENGTH': 3, 'LOWER': 'mdw'}]},
        {"label":"DOERFLICHE_WOHNGEBIETE","pattern": [{"TEXT": {"REGEX": "^MDW($|[^0-9])"}}]},
        {"label":"DOERFLICHE_WOHNGEBIETE_1","pattern":[{'LOWER': 'mdw', 'IS_UPPER': True, 'LENGTH': 3},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '1'}]},
        {"label":"DOERFLICHE_WOHNGEBIETE_2","pattern":[{'LOWER': 'mdw', 'IS_UPPER': True, 'LENGTH': 3},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '2'}]},
        {"label":"DOERFLICHE_WOHNGEBIETE_3","pattern":[{'LOWER': 'mdw', 'IS_UPPER': True, 'LENGTH': 3},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '3'}]},
        {"label":"DOERFLICHE_WOHNGEBIETE_4","pattern":[{'LOWER': 'mdw', 'IS_UPPER': True, 'LENGTH': 3},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '4'}]},
        {"label":"DOERFLICHE_WOHNGEBIETE_5","pattern":[{'LOWER': 'mdw', 'IS_UPPER': True, 'LENGTH': 3},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '5'}]},
        {"label":"DOERFLICHE_WOHNGEBIETE_6","pattern":[{'LOWER': 'mdw', 'IS_UPPER': True, 'LENGTH': 3},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '6'}]},
        {"label":"DOERFLICHE_WOHNGEBIETE_7","pattern":[{'LOWER': 'mdw', 'IS_UPPER': True, 'LENGTH': 3},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '7'}]},
        {"label":"DOERFLICHE_WOHNGEBIETE_1","pattern":[{'LOWER': 'mdw1', 'IS_UPPER': True, 'LENGTH': 4}]},
        {"label":"DOERFLICHE_WOHNGEBIETE_2","pattern":[{'LOWER': 'mdw2', 'IS_UPPER': True, 'LENGTH': 4}]},
        {"label":"DOERFLICHE_WOHNGEBIETE_3","pattern":[{'LOWER': 'mdw3', 'IS_UPPER': True, 'LENGTH': 4}]},
        {"label":"DOERFLICHE_WOHNGEBIETE_4","pattern":[{'LOWER': 'mdw4', 'IS_UPPER': True, 'LENGTH': 4}]},
        {"label":"DOERFLICHE_WOHNGEBIETE_5","pattern":[{'LOWER': 'mdw5', 'IS_UPPER': True, 'LENGTH': 4}]},
        {"label":"DOERFLICHE_WOHNGEBIETE_6","pattern":[{'LOWER': 'mdw6', 'IS_UPPER': True, 'LENGTH': 4}]},
        {"label":"DOERFLICHE_WOHNGEBIETE_7","pattern":[{'LOWER': 'mdw7', 'IS_UPPER': True, 'LENGTH': 4}]},
        {"label":"MISCHGEBIETE","pattern":[{"TEXT":{"FUZZY":"mischgebiete"}}]},
        {"label":"MISCHGEBIETE","pattern":[{"TEXT":{"FUZZY":"mischgebiet"}}]},
        {"label":"MISCHGEBIETE","pattern":[{'IS_UPPER': True, 'LENGTH': 2, 'LOWER': 'mi'}]},
        {"label":"MISCHGEBIETE","pattern": [{"TEXT": {"REGEX": "^MI($|[^0-9])"}}]},
        {"label":"MISCHGEBIETE_1","pattern":[{'LOWER': 'mi', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '1'}]},
        {"label":"MISCHGEBIETE_2","pattern":[{'LOWER': 'mi', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '2'}]},
        {"label":"MISCHGEBIETE_3","pattern":[{'LOWER': 'mi', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '3'}]},
        {"label":"MISCHGEBIETE_4","pattern":[{'LOWER': 'mi', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '4'}]},
        {"label":"MISCHGEBIETE_5","pattern":[{'LOWER': 'mi', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '5'}]},
        {"label":"MISCHGEBIETE_6","pattern":[{'LOWER': 'mi', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '6'}]},
        {"label":"MISCHGEBIETE_7","pattern":[{'LOWER': 'mi', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '7'}]},
        {"label":"MISCHGEBIETE_1","pattern":[{'LOWER': 'mi1', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"MISCHGEBIETE_2","pattern":[{'LOWER': 'mi2', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"MISCHGEBIETE_3","pattern":[{'LOWER': 'mi3', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"MISCHGEBIETE_4","pattern":[{'LOWER': 'mi4', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"MISCHGEBIETE_5","pattern":[{'LOWER': 'mi5', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"MISCHGEBIETE_6","pattern":[{'LOWER': 'mi6', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"MISCHGEBIETE_7","pattern":[{'LOWER': 'mi7', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"KERNGEBIETE","pattern":[{"TEXT":{"FUZZY":"kerngebiete"}}]},
        {"label":"KERNGEBIETE","pattern":[{"TEXT":{"FUZZY":"kerngebiet"}}]},
        {"label":"KERNGEBIETE","pattern":[{'IS_UPPER': True, 'LENGTH': 2, 'LOWER': 'mk'}]},
        {"label": "KERNGEBIETE","pattern": [{"TEXT": {"REGEX": "^MK($|[^0-9])"}}]},
        {"label":"KERNGEBIETE_1","pattern":[{'LOWER': 'mk', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '1'}]},
        {"label":"KERNGEBIETE_2","pattern":[{'LOWER': 'mk', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '2'}]},
        {"label":"KERNGEBIETE_3","pattern":[{'LOWER': 'mk', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '3'}]},
        {"label":"KERNGEBIETE_4","pattern":[{'LOWER': 'mk', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '4'}]},
        {"label":"KERNGEBIETE_5","pattern":[{'LOWER': 'mk', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '5'}]},
        {"label":"KERNGEBIETE_6","pattern":[{'LOWER': 'mk', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '6'}]},
        {"label":"KERNGEBIETE_7","pattern":[{'LOWER': 'mk', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '7'}]},
        {"label":"KERNGEBIETE_1","pattern":[{'LOWER': 'mk1', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"KERNGEBIETE_2","pattern":[{'LOWER': 'mk2', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"KERNGEBIETE_3","pattern":[{'LOWER': 'mk3', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"KERNGEBIETE_4","pattern":[{'LOWER': 'mk4', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"KERNGEBIETE_5","pattern":[{'LOWER': 'mk5', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"KERNGEBIETE_6","pattern":[{'LOWER': 'mk6', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"KERNGEBIETE_7","pattern":[{'LOWER': 'mk7', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"URBANE_GEBIETE","pattern":[{"TEXT":{"FUZZY":"urbane"}},{"TEXT":{"FUZZY":"gebiete"}}]},
        {"label":"URBANE_GEBIETE","pattern":[{"TEXT":{"FUZZY":"urbanes"}},{"TEXT":{"FUZZY":"gebiet"}}]},
        {"label":"URBANE_GEBIETE","pattern":[{'IS_UPPER': True, 'LENGTH': 2, 'LOWER': 'mu'}]},
        {"label":"URBANE_GEBIETE","pattern": [{"TEXT": {"REGEX": "^MU($|[^0-9])"}}]},
        {"label":"URBANE_GEBIETE_1","pattern":[{'LOWER': 'mu', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '1'}]},
        {"label":"URBANE_GEBIETE_2","pattern":[{'LOWER': 'mu', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '2'}]},
        {"label":"URBANE_GEBIETE_3","pattern":[{'LOWER': 'mu', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '3'}]},
        {"label":"URBANE_GEBIETE_4","pattern":[{'LOWER': 'mu', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '4'}]},
        {"label":"URBANE_GEBIETE_5","pattern":[{'LOWER': 'mu', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '5'}]},
        {"label":"URBANE_GEBIETE_6","pattern":[{'LOWER': 'mu', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '6'}]},
        {"label":"URBANE_GEBIETE_7","pattern":[{'LOWER': 'mu', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '7'}]},
        {"label":"URBANE_GEBIETE_1","pattern":[{'LOWER': 'mu1', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"URBANE_GEBIETE_2","pattern":[{'LOWER': 'mu2', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"URBANE_GEBIETE_3","pattern":[{'LOWER': 'mu3', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"URBANE_GEBIETE_4","pattern":[{'LOWER': 'mu4', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"URBANE_GEBIETE_5","pattern":[{'LOWER': 'mu5', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"URBANE_GEBIETE_6","pattern":[{'LOWER': 'mu6', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"URBANE_GEBIETE_7","pattern":[{'LOWER': 'mu7', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"EINGESCHRÄNKTES_GEWERBEGEBIET","pattern":[{"TEXT":{"FUZZY":"eingeschränktes"}},{"TEXT":{"FUZZY":"gewerbegebiet"}}]},
        {"label":"EINGESCHRÄNKTES_GEWERBEGEBIET","pattern":[{'IS_UPPER': True, 'LENGTH': 3, 'LOWER': 'gee'}]},
        {"label":"GEWERBEGEBIETE","pattern":[{"TEXT":{"FUZZY":"gewerbegebiete"}}]},
        {"label":"GEWERBEGEBIETE","pattern":[{"TEXT":{"FUZZY":"gewerbegebiet"}}]},
        {"label":"GEWERBEGEBIETE","pattern":[{'IS_UPPER': True, 'LENGTH': 2, 'LOWER': 'ge'}]},
        {"label":"GEWERBEGEBIETE","pattern": [{"TEXT": {"REGEX": "^GE($|[^0-9])"}}]},
        {"label":"GEWERBEGEBIETE_1","pattern":[{'LOWER': 'ge', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '1'}]},
        {"label":"GEWERBEGEBIETE_2","pattern":[{'LOWER': 'ge', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '2'}]},
        {"label":"GEWERBEGEBIETE_3","pattern":[{'LOWER': 'ge', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '3'}]},
        {"label":"GEWERBEGEBIETE_4","pattern":[{'LOWER': 'ge', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '4'}]},
        {"label":"GEWERBEGEBIETE_5","pattern":[{'LOWER': 'ge', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '5'}]},
        {"label":"GEWERBEGEBIETE_6","pattern":[{'LOWER': 'ge', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '6'}]},
        {"label":"GEWERBEGEBIETE_7","pattern":[{'LOWER': 'ge', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '7'}]},
        {"label":"GEWERBEGEBIETE_1","pattern":[{'LOWER': 'ge1', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"GEWERBEGEBIETE_2","pattern":[{'LOWER': 'ge2', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"GEWERBEGEBIETE_3","pattern":[{'LOWER': 'ge3', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"GEWERBEGEBIETE_4","pattern":[{'LOWER': 'ge4', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"GEWERBEGEBIETE_5","pattern":[{'LOWER': 'ge5', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"GEWERBEGEBIETE_6","pattern":[{'LOWER': 'ge6', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"GEWERBEGEBIETE_7","pattern":[{'LOWER': 'ge7', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"INDUSTRIEGEBIETE","pattern":[{"TEXT":{"FUZZY":"industriegebiete"}}]},
        {"label":"INDUSTRIEGEBIETE","pattern":[{"TEXT":{"FUZZY":"industriegebiet"}}]},
        {"label":"INDUSTRIEGEBIETE","pattern":[{'IS_UPPER': True, 'LENGTH': 2, 'LOWER': 'gi'}]},
        {"label":"INDUSTRIEGEBIETE","pattern": [{"TEXT": {"REGEX": "^GI($|[^0-9])"}}]},
        {"label":"INDUSTRIEGEBIETE_1","pattern":[{'LOWER': 'gi', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '1'}]},
        {"label":"INDUSTRIEGEBIETE_2","pattern":[{'LOWER': 'gi', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '2'}]},
        {"label":"INDUSTRIEGEBIETE_3","pattern":[{'LOWER': 'gi', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '3'}]},
        {"label":"INDUSTRIEGEBIETE_4","pattern":[{'LOWER': 'gi', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '4'}]},
        {"label":"INDUSTRIEGEBIETE_5","pattern":[{'LOWER': 'gi', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '5'}]},
        {"label":"INDUSTRIEGEBIETE_6","pattern":[{'LOWER': 'gi', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '6'}]},
        {"label":"INDUSTRIEGEBIETE_7","pattern":[{'LOWER': 'gi', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '7'}]},
        {"label":"INDUSTRIEGEBIETE_1","pattern":[{'LOWER': 'gi1', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"INDUSTRIEGEBIETE_2","pattern":[{'LOWER': 'gi2', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"INDUSTRIEGEBIETE_3","pattern":[{'LOWER': 'gi3', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"INDUSTRIEGEBIETE_4","pattern":[{'LOWER': 'gi4', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"INDUSTRIEGEBIETE_5","pattern":[{'LOWER': 'gi5', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"INDUSTRIEGEBIETE_6","pattern":[{'LOWER': 'gi6', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"INDUSTRIEGEBIETE_7","pattern":[{'LOWER': 'gi7', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"SONDERGEBIETE","pattern":[{"TEXT":{"FUZZY":"sondergebiete"}}]},
        {"label":"SONDERGEBIETE","pattern":[{"TEXT":{"FUZZY":"sondergebiet"}}]},
        {"label":"SONDERGEBIETE","pattern":[{'IS_UPPER': True, 'LENGTH': 2, 'LOWER': 'so'}]},
        {"label":"SONDERGEBIETE","pattern": [{"TEXT": {"REGEX": "^SO($|[^0-9])"}}]},
        {"label":"SONDERGEBIETE_1","pattern":[{'LOWER': 'so', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '1'}]},
        {"label":"SONDERGEBIETE_2","pattern":[{'LOWER': 'so', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '2'}]},
        {"label":"SONDERGEBIETE_3","pattern":[{'LOWER': 'so', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '3'}]},
        {"label":"SONDERGEBIETE_4","pattern":[{'LOWER': 'so', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '4'}]},
        {"label":"SONDERGEBIETE_5","pattern":[{'LOWER': 'so', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '5'}]},
        {"label":"SONDERGEBIETE_6","pattern":[{'LOWER': 'so', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '6'}]},
        {"label":"SONDERGEBIETE_7","pattern":[{'LOWER': 'so', 'IS_UPPER': True, 'LENGTH': 2},{'IS_DIGIT': True, 'LENGTH': 1, 'LOWER': '7'}]},
        {"label":"SONDERGEBIETE_1","pattern":[{'LOWER': 'so1', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"SONDERGEBIETE_2","pattern":[{'LOWER': 'so2', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"SONDERGEBIETE_3","pattern":[{'LOWER': 'so3', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"SONDERGEBIETE_4","pattern":[{'LOWER': 'so4', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"SONDERGEBIETE_5","pattern":[{'LOWER': 'so5', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"SONDERGEBIETE_6","pattern":[{'LOWER': 'so6', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"SONDERGEBIETE_7","pattern":[{'LOWER': 'so7', 'IS_UPPER': True, 'LENGTH': 3}]},
        {"label":"wochenendhausgebiete","pattern":[{"TEXT":{"FUZZY":"wochenendhausgebiete"}}]},
        {"label":"BAUHOEHE","pattern":[{"TEXT":{"FUZZY":"bauhöhe"}}]},
        {"label":"BAUHOEHE","pattern":[{"TEXT":{"FUZZY":"gebäudehöhe"}}]},
        {"label":"BAUHOEHE","pattern":[{"TEXT":{"FUZZY":"höhe"}},{"TEXT":{"FUZZY":"der"}},{"TEXT":{"FUZZY":"baulichen"}},{"TEXT":{"FUZZY":"anlage"}}]},
        {"label":"GRUNDFLAECHENZAHL","pattern":[{"TEXT":{"FUZZY":"grundflächenzahl"}}]},
        {"label":"GRUNDFLAECHENZAHL","pattern":[{'LOWER': 'grz'}]},
        {"label":"GESCHOSSFLAECHENZAHL","pattern":[{'LOWER': 'gfz'}]},
        {"label":"GESCHOSSFLAECHENZAHL","pattern":[{"TEXT":{"FUZZY":"geschossflächenzahl"}}]},
        {"label":"OFFENE_BAUWEISE","pattern":[{"TEXT":{"FUZZY":"offene"}},{"TEXT":{"FUZZY":"bauweise"}}]},
        {"label":"GESCHLOSSENE_BAUWEISE","pattern":[{"TEXT":{"FUZZY":"geschlossene"}},{"TEXT":{"FUZZY":"bauweise"}}]},
        {"label":"Betriebe_des_Beherbergungsgewerbes","pattern":[{"TEXT":{"FUZZY":"betriebe"}},{"TEXT":{"FUZZY":"des"}},{"TEXT":{"FUZZY":"beherbergungsgewerbes"}}]},
        {"label":"sonstige_nicht_störende_Gewerbebetriebe","pattern":[{"TEXT":{"FUZZY":"sonstige"}},{"TEXT":{"FUZZY":"nicht"}},{"TEXT":{"FUZZY":"störende"}},{"TEXT":{"FUZZY":"gewerbebetriebe"}}]},
        {"label":"Anlagen_für_Verwaltungen","pattern":[{"TEXT":{"FUZZY":"anlagen"}},{"TEXT":{"FUZZY":"für"}},{"TEXT":{"FUZZY":"verwaltungen"}}]}, 
        {"label":"der_Versorgung_des_Gebiets_dienende_Läden","pattern":[{"TEXT":{"FUZZY":"der"}},{"TEXT":{"FUZZY":"versorgung"}},{"TEXT":{"FUZZY":"des"}},{"TEXT":{"FUZZY":"gebiets"}},{"TEXT":{"FUZZY":"dienende"}},{"TEXT":{"FUZZY":"läden"}}]},    
        {"label":"Anlagen_für_sportliche_Zwecke","pattern":[{"TEXT":{"FUZZY":"anlagen"}}, {"TEXT":{"FUZZY":"für"}}, {"TEXT":{"FUZZY":"sportliche"}}, {"TEXT":{"FUZZY":"zwecke"}} ]},   
        {"label":"Betriebe,_die_Waren_oder_Dienstleistungen_zur_Erregung_sexueller_Bedürfnisse_oder_deren_Befriedigung_anbieten","pattern":[{"TEXT":{"FUZZY":"betriebe,"}},{"TEXT":{"FUZZY":"die"}},{"TEXT":{"FUZZY":"waren"}},{"TEXT":{"FUZZY":"oder"}},{"TEXT":{"FUZZY":"Dienstleistungen"}},{"TEXT":{"FUZZY":"zur"}},{"TEXT":{"FUZZY":"Erregung"}},{"TEXT":{"FUZZY":"sexueller"}},{"TEXT":{"FUZZY":"bedürfnisse"}},{"TEXT":{"FUZZY":"oder"}},{"TEXT":{"FUZZY":"deren"}},{"TEXT":{"FUZZY":"befriedigung"}},{"TEXT":{"FUZZY":"anbieten"}}]},
        {"label":"Wohnungen_in_den_Erdgeschossen","pattern":[{"TEXT":{"FUZZY":"wohnungen"}},{"TEXT":{"FUZZY":"in"}},{"TEXT":{"FUZZY":"den"}},{"TEXT":{"FUZZY":"erdgeschossen"}}]},
        {"label":"Wohnungen_im_Keller","pattern":[{"TEXT":{"FUZZY":"wohnungen"}},{"TEXT":{"FUZZY":"im"}},{"TEXT":{"FUZZY":"keller"}}]},
        {"label":"Wohnungen_in_den_Obergeschoss","pattern":[{"TEXT":{"FUZZY":"wohnungen"}},{"TEXT":{"FUZZY":"in"}},{"TEXT":{"FUZZY":"den"}},{"TEXT":{"FUZZY":"obergeschoss"}}]},  
        {"label":"Wohnungen_auf_dem_Dach","pattern":[{"TEXT":{"FUZZY":"wohnungen"}},{"TEXT":{"FUZZY":"auf"}},{"TEXT":{"FUZZY":"dem"}},{"TEXT":{"FUZZY":"dach"}}]},
        {"label":"sonstige_Gewerbebetriebe","pattern":[{"TEXT":{"FUZZY":"sonstige"}}, {"TEXT":{"FUZZY":"gewerbebetriebe"}}]},
        {"label":"Betriebe_mit_Lärmemissionen","pattern":[{"TEXT":{"FUZZY":"betriebe"}},{"TEXT":{"FUZZY":"mit"}},{"TEXT":{"FUZZY":"lärmemissionen"}}]}, 
        {"label":"Verkaufsflächen_für_Schwerpunktsortimente","pattern":[{"TEXT":{"FUZZY":"verkaufsflächen"}},{"TEXT":{"FUZZY":"für"}},{"TEXT":{"FUZZY":"schwerpunktsortimente"}}]},  
        {"label":"Lärmemissionen_durch_Freizeitaktivitäten","pattern":[{"TEXT":{"FUZZY":"lärmemissionen"}},{"TEXT":{"FUZZY":"durch"}},{"TEXT":{"FUZZY":"freizeitaktivitäten"}}]}, 
        {"label":"Vermietung_von_Geräten_oder_Fahrzeugen","pattern":[{"TEXT":{"FUZZY":"vermietung"}}, {"TEXT":{"FUZZY":"von"}}, {"TEXT":{"FUZZY":"geräten"}}, {"TEXT":{"FUZZY":"fahrzeugen"}}]},    
        {"label":"Lagerung_von_gefährlichen_Stoffen","pattern":[{"TEXT":{"FUZZY":"lagerung"}},{"TEXT":{"FUZZY":"von"}},{"TEXT":{"FUZZY":"gefährlichen"}},{"TEXT":{"FUZZY":"stoffen"}}]},  
        {"label":"Vermietung_von_Wohnraum","pattern":[{"TEXT":{"FUZZY":"vermietung"}},{"TEXT":{"FUZZY":"von"}},{"TEXT":{"FUZZY":"wohnraum"}}]}, 
        {"label":"Vermietung_von_Büroräumen","pattern":[{"TEXT":{"FUZZY":"vermietung"}},{"TEXT":{"FUZZY":"von"}},{"TEXT":{"FUZZY":"büroräumen"}}]}, 
        {"label":"Räume_für_freie_Berufe","pattern":[{"TEXT":{"FUZZY":"räume"}},{"TEXT":{"FUZZY":"für"}},{"TEXT":{"FUZZY":"freie"}},{"TEXT":{"FUZZY":"berufe"}}]},
        {"label":"Vermietung_von_Lagerflächen","pattern":[{"TEXT":{"FUZZY":"vermietung"}},{"TEXT":{"FUZZY":"von"}},{"TEXT":{"FUZZY":"lagerflächen"}}]},  
        {"label":"Werbung_oder_Plakatierung","pattern":[{"TEXT":{"FUZZY":"werbung"}},{"TEXT":{"FUZZY":"oder"}},{"TEXT":{"FUZZY":"plakatierung"}}]}, 
        {"label":"Vermietung_von_Werbeflächen","pattern":[{"TEXT":{"FUZZY":"vermietung"}}, {"TEXT":{"FUZZY":"von"}}, {"TEXT":{"FUZZY":"werbeflächen"}}]},
        {"label":"Stellplätze_für_LKWs_oder_Schwertransporter","pattern":[{"TEXT":{"FUZZY":"stellplätze"}},{"TEXT":{"FUZZY":"für"}},{"TEXT":{"FUZZY":"lkws"}},{"TEXT":{"FUZZY":"oder"}},{"TEXT":{"FUZZY":"schwertransporter"}}]},
        {"label":"Stellplätze_für_LKWs_oder_Schwertransporter","pattern":[{"TEXT":{"FUZZY":"stellplätze"}},{"TEXT":{"FUZZY":"für"}},{"TEXT":{"FUZZY":"wohnwagen"}},{"TEXT":{"FUZZY":"oder"}},{"TEXT":{"FUZZY":"mobile"}},{"TEXT":{"FUZZY":"heime"}}]},
        {"label":"Stellplätze_für_Wohnmobile","pattern":[{"TEXT":{"FUZZY":"stellplätze"}},{"TEXT":{"FUZZY":"für"}},{"TEXT":{"FUZZY":"wohnmobile"}}]},
        {"label":"Einrichtungen_zur_medizinischen_Versorgung","pattern":[{"TEXT":{"FUZZY":"einrichtungen"}},{"TEXT":{"FUZZY":"zur"}},{"TEXT":{"FUZZY":"medizinischen"}},{"TEXT":{"FUZZY":"versorgung"}}]},
        {"label":"militärische_Einrichtungen","pattern":[{"TEXT":{"FUZZY":"militärische"}},{"TEXT":{"FUZZY":"einrichtungen"}}]},
        {"label":"tankstellen","pattern":[{'LOWER':"tankstellen"}]},
        {"label":"gartenbaubetriebe","pattern":[{'LOWER':"gartenbaubetriebe"}]},
        {"label":"einzelhandelsbetriebe","pattern":[{'LOWER':"einzelhandelsbetriebe"}]},
        {"label":"industrieanlagen","pattern":[{'LOWER':"industrieanlagen"}]},
        {"label":"veranstaltungsräume","pattern":[{'LOWER':"veranstaltungsräume"}]},
        {"label":"öffentliche Betriebe","pattern":[{'LOWER':"öffentliche"}, {'LOWER':"betriebe"}]},
        {"label":"spielhallen","pattern":[{'LOWER':"spielhallen"}]},
        {"label":"baumärkte","pattern":[{'LOWER':"baumärkte"}]},
        {"label":"gartenzentren","pattern":[{'LOWER':"gartenzentren"}]},
        {"label":"sex-shops","pattern":[{'LOWER':"sex-shops"}]},
        {"label":"sex-kinos","pattern":[{'LOWER':"sex-kinos"}]},
        {"label":"peep-shows","pattern":[{'LOWER':"peep-shows"}]},
        {"label":"bordelle","pattern":[{'LOWER':"bordelle"}]},
        {"label":"bordellbetriebe","pattern":[{'LOWER':"bordellbetriebe"}]},
        {"label":"laufhäuser","pattern":[{'LOWER':"laufhäuser"}]},
        {"label":"freudenhäuser","pattern":[{'LOWER':"freudenhäuser"}]},
        {"label":"gartenlauben","pattern":[{'LOWER':"gartenlauben"}]},
        {"label":"baustoffhandel","pattern":[{'LOWER':"baustoffhandel"}]},
        {"label":"kleingartenanlagen","pattern":[{'LOWER':"kleingartenanlagen"}]},
        {"label":"parkhäuser","pattern":[{'LOWER':"parkhäuser"}]},
        {"label":"sportplaetze","pattern":[{'LOWER':"sportplaetze"}]},
        {"label":"sportanlagen","pattern":[{'LOWER':"sportanlagen"}]},
        {"label":"fitnessstudios","pattern":[{'LOWER':"fitnessstudios"}]},
        {"label":"sporthallen","pattern":[{'LOWER':"sporthallen"}]},
        {"label":"sportclubs","pattern":[{'LOWER':"sportclubs"}]},
        {"label":"Parkanlagen","pattern":[{'LOWER':"Parkanlagen"}]},
        {"label":"Spielplätze","pattern":[{'LOWER':"Spielplätze"}]},
        {"label":"Lagerflächen","pattern":[{'LOWER':"Lagerflächen"}]},
        {"label":"Kindergärten","pattern":[{'LOWER':"kindergärten"}]},
        {"label":"krippen","pattern":[{'LOWER':"krippen"}]},
        {"label":"sektkellereien","pattern":[{'LOWER':"sektkellereien"}]},
        {"label":"kinderkrippen","pattern":[{'LOWER':"kinderkrippen"}]},
        {"label":"schulen","pattern":[{'LOWER':"schulen"}]},
        {"label":"krankenhäuser","pattern":[{'LOWER':"krankenhäuser"}]},
        {"label":"kinos","pattern":[{'LOWER':"kinos"}]},
        {"label":"theater","pattern":[{'LOWER':"theater"}]},
        {"label":"konzerthallen","pattern":[{'LOWER':"konzerthallen"}]},
        {"label":"museen","pattern":[{'LOWER':"museen"}]},
        {"label":"galerien","pattern":[{'LOWER':"galerien"}]},
        {"label":"kulturzentren","pattern":[{'LOWER':"kulturzentren"}]},
        {"label":"kulturhaus","pattern":[{'LOWER':"kulturhaus"}]},
        {"label":"bibliotheken","pattern":[{'LOWER':"bibliotheken"}]},
        {"label":"buchhandlungen","pattern":[{'LOWER':"buchhandlungen"}]},
        {"label":"lagerhäuser","pattern":[{'LOWER':"lagerhäuser"}]},
        {"label":"lagerplätze","pattern":[{'LOWER':"lagerplätze"}]},
        {"label":"schwimmbäder","pattern":[{'LOWER':"schwimmbäder"}]},
        {"label":"gartenbau","pattern":[{'LOWER':"gartenbau"}]},
        {"label":"blumenläden","pattern":[{'LOWER':"blumenläden"}]},
        {"label":"bäckereien","pattern":[{'LOWER':"bäckereien"}]},
        {"label":"konditoreien","pattern":[{'LOWER':"konditoreien"}]},
        {"label":"Metzgereien","pattern":[{'LOWER':"Metzgereien"}]},
        {"label":"schlachter","pattern":[{'LOWER':"schlachter"}]},
        {"label":"schlachthof","pattern":[{'LOWER':"schlachthof"}]},
        {"label":"gemüseläden","pattern":[{'LOWER':"gemüseläden"}]},
        {"label":"fleischerei","pattern":[{'LOWER':"fleischerei"}]},
        {"label":"fischläden","pattern":[{'LOWER':"fischläden"}]},
        {"label":"käseladen","pattern":[{'LOWER':"käseladen"}]},
        {"label":"lebensmittelmärkte","pattern":[{'LOWER':"lebensmittelmärkte"}]},
        {"label":"lebensmittelgeschäfte","pattern":[{'LOWER':"lebensmittelgeschäfte"}]},
        {"label":"lebensmittelhandel","pattern":[{'LOWER':"lebensmittelhandel"}]},
        {"label":"supermarkt","pattern":[{'LOWER':"supermarkt"}]},
        {"label":"discounter","pattern":[{'LOWER':"discounter"}]},
        {"label":"einkaufszentrum","pattern":[{'LOWER':"einkaufszentrum"}]},
        {"label":"einkaufszentren","pattern":[{'LOWER':"einkaufszentren"}]},
        {"label":"einkaufscenter","pattern":[{'LOWER':"einkaufscenter"}]},
        {"label":"discotheken", "pattern":[{'LOWER':"discotheken"}]},
        {"label":"Wohnnutzung","pattern":[{'LOWER':"wohnnutzung"}]},
        {"label":"Getränkemärkte","pattern":[{'LOWER':"getränkemärkte"}]},
        {"label":"Getränkehandel","pattern":[{'LOWER':"getränkehandel"}]},
        {"label":"Getränkefachmärkte","pattern":[{'LOWER':"getränkefachmärkte"}]},
        {"label":"Spirituosenhandel","pattern":[{'LOWER':"spirituosenhandel"}]},
        {"label":"Spirituosenmärkte","pattern":[{'LOWER':"spirituosenmärkte"}]},
        {"label":"Spirituosenfachmärkte","pattern":[{'LOWER':"spirituosenfachmärkte"}]},
        {"label":"weinkeller","pattern":[{'LOWER':"weinkeller"}]},
        {"label":"weinhandel","pattern":[{'LOWER':"weinhandel"}]},
        {"label":"brauerei","pattern":[{'LOWER':"brauerei"}]},
        {"label":"eisdielen","pattern":[{'LOWER':"eisdielen"}]},
        {"label":"gastronomie","pattern":[{'LOWER':"gastronomie"}]},
        {"label":"snacksverkauf","pattern":[{'LOWER':"snacksverkauf"}]},
        {"label":"kleidungsgeschäft","pattern":[{'LOWER':"kleidungsgeschäft"}]},
        {"label":"modegeschäft","pattern":[{'LOWER':"modegeschäft"}]},
        {"label":"schuhgeschäft","pattern":[{'LOWER':"schuhgeschäft"}]},
        {"label":"accessoiresgeschäft","pattern":[{'LOWER':"accessoiresgeschäft"}]},
        {"label":"schmuckgeschäft","pattern":[{'LOWER':"schmuckgeschäft"}]},
        {"label":"spielzeuggeschäft","pattern":[{'LOWER':"spielzeuggeschäft"}]},
        {"label":"sportgeschäft","pattern":[{'LOWER':"sportgeschäft"}]},
        {"label":"parfümerien","pattern":[{'LOWER':"parfümerien"}]},
        {"label":"kosmetikgeschäft","pattern":[{'LOWER':"kosmetikgeschäft"}]},
        {"label":"haushaltswarengeschäft","pattern":[{'LOWER':"haushaltswarengeschäft"}]},
        {"label":"baumarktgeschäft","pattern":[{'LOWER':"baumarktgeschäft"}]},
        {"label":"blumencenter","pattern":[{'LOWER':"blumencenter"}]},
        {"label":"parkplätze","pattern":[{'LOWER':"parkplätze"}]},
        {"label":"gartencenter","pattern":[{'LOWER':"gartencenter"}]},
        {"label":"gewerbebetriebe","pattern":[{'LOWER':"gewerbebetriebe"}]},
        {"label":"buchhandlungen","pattern":[{'LOWER':"buchhandlungen"}]},
        {"label":"zeitschriftengeschäft","pattern":[{'LOWER':"zeitschriftengeschäft"}]},
        {"label":"Anlagen_für_soziale,_gesundheitliche_und_sportliche_Zwecke", "pattern": [{'TEXT': {'FUZZY': 'Anlagen'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'soziale,'}}, {'TEXT': {'FUZZY': 'gesundheitliche'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'sportliche'}}, {'TEXT': {'FUZZY': 'Zwecke'}}]},
        {"label":"Anlagen_von_Gewerbebetrieben", "pattern": [{'TEXT': {'FUZZY': 'Anlagen'}}, {'TEXT': {'FUZZY': 'von'}}, {'TEXT': {'FUZZY': 'Gewerbebetrieben'}}]},
        {"label":"Büro-_und_Verwaltungsgebäude", "pattern": [{'TEXT': {'FUZZY': 'Büro-'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'Verwaltungsgebäude'}}]},
        {"label":"Anlagen_für_soziale,_gesundheitliche_und_sportliche_Zwecke", "pattern": [{'TEXT': {'FUZZY': 'Anlagen'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'soziale,'}}, {'TEXT': {'FUZZY': 'gesundheitliche'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'sportliche'}}, {'TEXT': {'FUZZY': 'Zwecke'}}]},
        {'label':'Schank-_und_Speisewirtschaften', 'pattern':[{'TEXT': {'FUZZY': 'Schank-'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'Speisewirtschaften'}}]},
        {"label":"Büro-,_Dienstleistungs-_und_Verwaltungsgebäuden", "pattern": [{'TEXT': {'FUZZY': 'Büro-'}}, {'TEXT': {'FUZZY': 'Dienstleistungs-'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'Verwaltungsgebäuden'}}]},
        {"label":"Anlagen_des_Post-_und_Fernmeldewesens", "pattern": [{'TEXT': {'FUZZY': 'Anlagen'}}, {'TEXT': {'FUZZY': 'des'}}, {'TEXT': {'FUZZY': 'Post-'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'Fernmeldewesens'}}]},
        {'label':'Kleinsiedlungen_einschließlich_Wohngebäude_mit_entsprechenden_Nutzgärten,_landwirtschaftliche_Nebenerwerbsstellen_und_Gartenbaubetriebe', 'pattern': [{'TEXT': {'FUZZY': 'Kleinsiedlungen'}}, {'TEXT': {'FUZZY': 'einschließlich'}}, {'TEXT': {'FUZZY': 'Wohngebäude'}}, {'TEXT': {'FUZZY': 'mit'}}, {'TEXT': {'FUZZY': 'entsprechenden'}}, {'TEXT': {'FUZZY': 'Nutzgärten,'}}, {'TEXT': {'FUZZY': 'landwirtschaftliche'}}, {'TEXT': {'FUZZY': 'Nebenerwerbsstellen'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'Gartenbaubetriebe'}}]},
        {'label':'die_der_Versorgung_des_Gebiets_dienenden_Läden,_Schank-_und_Speisewirtschaften_sowie_nicht_störende_Handwerksbetriebe', 'pattern': [{'TEXT': {'FUZZY': 'die'}}, {'TEXT': {'FUZZY': 'der'}}, {'TEXT': {'FUZZY': 'Versorgung'}}, {'TEXT': {'FUZZY': 'des'}}, {'TEXT': {'FUZZY': 'Gebiets'}}, {'TEXT': {'FUZZY': 'dienenden'}}, {'TEXT': {'FUZZY': 'Läden,'}}, {'TEXT': {'FUZZY': 'Schank-'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'Speisewirtschaften'}}, {'TEXT': {'FUZZY': 'sowie'}}, {'TEXT': {'FUZZY': 'nicht'}}, {'TEXT': {'FUZZY': 'störenden'}}, {'TEXT': {'FUZZY': 'Handwerksbetriebe'}} ]},
        {'label': 'nicht_störende_Handwerksbetriebe', 'pattern': [{'TEXT': {'FUZZY': 'nicht'}}, {'TEXT': {'FUZZY': 'störende'}}, {'TEXT': {'FUZZY': 'Handwerksbetriebe'}}]},
        {'label': 'sonstige_Wohngebäude_mit_nicht_mehr_als_zwei_Wohnungen', 'pattern': [{'TEXT': {'FUZZY': 'sonstige'}}, {'TEXT': {'FUZZY': 'Wohngebäude'}}, {'TEXT': {'FUZZY': 'mit'}}, {'TEXT': {'FUZZY': 'nicht'}}, {'TEXT': {'FUZZY': 'mehr'}}, {'TEXT': {'FUZZY': 'als'}}, {'TEXT': {'FUZZY': 'zwei'}}, {'TEXT': {'FUZZY': 'Wohnungen'}}]},
        {'label': 'Anlagen_für_kirchliche,_kulturelle,_soziale,_gesundheitliche_und_sportliche_Zwecke', 'pattern': [{'TEXT': {'FUZZY': 'Anlagen'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'kirchliche,'}}, {'TEXT': {'FUZZY': 'kulturelle,'}}, {'TEXT': {'FUZZY': 'soziale,'}}, {'TEXT': {'FUZZY': 'gesundheitliche'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'sportliche'}}, {'TEXT': {'FUZZY': 'Zwecke'}}]},
        {'label': 'Tankstellen', 'pattern': [{'LOWER': 'tankstellen'}]},
        {'label': 'nicht_störende_Gewerbebetriebe', 'pattern': [{'TEXT': {'FUZZY': 'nicht'}}, {'TEXT': {'FUZZY': 'störende'}}, {'TEXT': {'FUZZY': 'Gewerbebetriebe'}}]},
        {'label': 'Anlagen_zur_Kinderbetreuung,_die_den_Bedürfnissen_der_Bewohner_des_Gebiets_dienen', 'pattern': [{'TEXT': {'FUZZY': 'Anlagen'}}, {'TEXT': {'FUZZY': 'zur'}}, {'TEXT': {'FUZZY': 'Kinderbetreuung,'}}, {'TEXT': {'FUZZY': 'die'}}, {'TEXT': {'FUZZY': 'den'}}, {'TEXT': {'FUZZY': 'Bedürfnissen'}}, {'TEXT': {'FUZZY': 'der'}}, {'TEXT': {'FUZZY': 'Bewohner'}}, {'TEXT': {'FUZZY': 'des'}}, {'TEXT': {'FUZZY': 'Gebiets'}}, {'TEXT': {'FUZZY': 'dienen'}}]},
        {'label': 'Läden_und_nicht_störende_Handwerksbetriebe,_die_zur_Deckung_des_täglichen_Bedarfs_für_die_Bewohner_des_Gebiets_dienen,_sowie_kleine_Betriebe_des_Beherbergungsgewerbes', 'pattern': [{'TEXT': {'FUZZY': 'Läden'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'nicht'}}, {'TEXT': {'FUZZY': 'störende'}}, {'TEXT': {'FUZZY': 'Handwerksbetriebe,'}}, {'TEXT': {'FUZZY': 'die'}}, {'TEXT': {'FUZZY': 'zur'}}, {'TEXT': {'FUZZY': 'Deckung'}}, {'TEXT': {'FUZZY': 'des'}}, {'TEXT': {'FUZZY': 'täglichen'}}, {'TEXT': {'FUZZY': 'Bedarfs'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'die'}}, {'TEXT': {'FUZZY': 'Bewohner'}}, {'TEXT': {'FUZZY': 'des'}}, {'TEXT': {'FUZZY': 'Gebiets'}}, {'TEXT': {'FUZZY': 'dienen,'}}, {'TEXT': {'FUZZY': 'sowie'}}, {'TEXT': {'FUZZY': 'kleine'}}, {'TEXT': {'FUZZY': 'Betriebe'}}, {'TEXT': {'FUZZY': 'des'}}, {'TEXT': {'FUZZY': 'Beherbergungsgewerbes'}}]},
        {'label': 'sonstige_Anlagen_für_soziale_Zwecke_sowie_den_Bedürfnissen_der_Bewohner_des_Gebiets_dienende_Anlagen_für_kirchliche,_kulturelle,_gesundheitliche_und_sportliche_Zwecke', 'pattern': [{'TEXT': {'FUZZY': 'sonstige'}}, {'TEXT': {'FUZZY': 'Anlagen'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'soziale'}}, {'TEXT': {'FUZZY': 'Zwecke'}}, {'TEXT': {'FUZZY': 'sowie'}}, {'TEXT': {'FUZZY': 'den'}}, {'TEXT': {'FUZZY': 'Bedürfnissen'}}, {'TEXT': {'FUZZY': 'der'}}, {'TEXT': {'FUZZY': 'Bewohner'}}, {'TEXT': {'FUZZY': 'des'}}, {'TEXT': {'FUZZY': 'Gebiets'}}, {'TEXT': {'FUZZY': 'dienende'}}, {'TEXT': {'FUZZY': 'Anlagen'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'kirchliche,'}}, {'TEXT': {'FUZZY': 'kulturelle,'}}, {'TEXT': {'FUZZY': 'gesundheitliche'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'sportliche'}}, {'TEXT': {'FUZZY': 'Zwecke'}}]},
        {'label': 'Betriebe_des_Beherbergungsgewerbes', 'pattern': [{'TEXT': {'FUZZY': 'Betriebe'}}, {'TEXT': {'FUZZY': 'des'}}, {'TEXT': {'FUZZY': 'Beherbergungsgewerbes'}}]},
        {'label': 'sonstige_nicht_störende_Gewerbebetriebe', 'pattern': [{'TEXT': {'FUZZY': 'sonstige'}}, {'TEXT': {'FUZZY': 'nicht'}}, {'TEXT': {'FUZZY': 'störende'}}, {'TEXT': {'FUZZY': 'Gewerbebetriebe'}}]}, 
        {'label': 'Zentrale_Einrichtungen_für_Verwaltungen', 'pattern': [{'TEXT': {'FUZZY': 'zentrale'}}, {'TEXT': {'FUZZY': 'einrichtungen'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'öffentlicher'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'privater'}}, {'TEXT': {'FUZZY': 'werwaltungen'}}]},
        {'label': 'Gartenbaubetriebe', 'pattern':[{'LOWER': 'gartenbaubetriebe'}]},    
        {'label': 'Wohngebäude', 'pattern':[{'text': 'wohngebäude'}]},
        {'label': 'Läden', 'pattern':[{'LOWER': 'läden'}]},
        {'label': 'Ställe_für_Kleintierhaltung_als_Zubehör_zu_Kleinsiedlungen_und_landwirtschaftlichen_Nebenerwerbsstellen', 'pattern': [{'TEXT': {'FUZZY': 'Ställe'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'Kleintierhaltung'}}, {'TEXT': {'FUZZY': 'als'}}, {'TEXT': {'FUZZY': 'Zubehör'}}, {'TEXT': {'FUZZY': 'zu'}}, {'TEXT': {'FUZZY': 'Kleinsiedlungen'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'landwirtschaftlichen'}}, {'TEXT': {'FUZZY': 'Nebenerwerbsstellen'}}]},
        {'label': 'Einzelhandelsbetriebe', 'pattern': [{'TEXT': {'FUZZY': 'einzelhandelsbetriebe'}}]},
        {'label': 'sonstige_nicht_störende_Gewerbebetriebe', 'pattern': [{'TEXT': {'FUZZY': 'sonstige'}}, {'TEXT': {'FUZZY': 'nicht'}}, {'TEXT': {'FUZZY': 'störende'}}, {'TEXT': {'FUZZY': 'Gewerbebetriebe'}}]},
        {'label': 'nicht_gewerbliche_Einrichtungen_und_Anlagen_für_die_Tierhaltung', 'pattern': [{'TEXT': {'FUZZY': 'nicht'}}, {'TEXT': {'FUZZY': 'gewerbliche'}}, {'TEXT': {'FUZZY': 'Einrichtungen'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'Anlagen'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'die'}}, {'TEXT': {'FUZZY': 'Tierhaltung'}}]},
        {'label': 'Läden,_Betriebe_des_Beherbergungsgewerbes,_Schank-_und_Speisewirtschaften', 'pattern': [{'TEXT': {'FUZZY': 'Läden,'}}, {'TEXT': {'FUZZY': 'Betriebe'}}, {'TEXT': {'FUZZY': 'des'}}, {'TEXT': {'FUZZY': 'Beherbergungsgewerbes,'}}, {'TEXT': {'FUZZY': 'Schank-'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'Speisewirtschaften'}}]},
        {'label': 'sonstige_Gewerbebetriebe', 'pattern': [{'TEXT': {'FUZZY': 'sonstige'}}, {'TEXT': {'FUZZY': 'Gewerbebetriebe'}}]},
        {'label': 'Geschäfts-_und_Bürogebäude', 'pattern': [{'TEXT': {'FUZZY': 'Geschäfts-'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'Bürogebäude'}}]},
        {'label': 'Anlagen_für_zentrale_Einrichtungen_der_Verwaltung', 'pattern': [{'TEXT': {'FUZZY': 'Anlagen'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'zentrale'}}, {'TEXT': {'FUZZY': 'Einrichtungen'}}, {'TEXT': {'FUZZY': 'der'}}, {'TEXT': {'FUZZY': 'Verwaltung'}}]},
        {'label': 'Vergnügungsstätten', 'pattern': [{'TEXT': {'FUZZY': 'vergnügungsstätten'}}]},
        {'label': 'oberhalb_eines_im_Bebauungsplan_bestimmten_Geschosses_nur_Wohnungen_zulässig_sind', 'pattern': [{'TEXT': {'FUZZY': 'oberhalb'}}, {'TEXT': {'FUZZY': 'eines'}}, {'TEXT': {'FUZZY': 'im'}}, {'TEXT': {'FUZZY': 'Bebauungsplan'}}, {'TEXT': {'FUZZY': 'bestimmten'}}, {'TEXT': {'FUZZY': 'Geschosses'}}, {'TEXT': {'FUZZY': 'nur'}}, {'TEXT': {'FUZZY': 'Wohnungen'}}, {'TEXT': {'FUZZY': 'zulässig'}}, {'TEXT': {'FUZZY': 'sind'}}]},
        {'label': 'in_Gebäuden_ein_im_Bebauungsplan_bestimmter_Anteil_der_zulässigen_Geschossfläche_oder_eine_bestimmte_Größe_der_Geschossfläche_für_Wohnungen_zu_verwenden_ist', 'pattern': [{'TEXT': {'FUZZY': 'in'}}, {'TEXT': {'FUZZY': 'Gebäuden'}}, {'TEXT': {'FUZZY': 'ein'}}, {'TEXT': {'FUZZY': 'im'}}, {'TEXT': {'FUZZY': 'Bebauungsplan'}}, {'TEXT': {'FUZZY': 'bestimmter'}}, {'TEXT': {'FUZZY': 'Anteil'}}, {'TEXT': {'FUZZY': 'der'}}, {'TEXT': {'FUZZY': 'zulässigen'}}, {'TEXT': {'FUZZY': 'Geschossfläche'}}, {'TEXT': {'FUZZY': 'oder'}}, {'TEXT': {'FUZZY': 'eine'}}, {'TEXT': {'FUZZY': 'bestimmte'}}, {'TEXT': {'FUZZY': 'Größe'}}, {'TEXT': {'FUZZY': 'der'}}, {'TEXT': {'FUZZY': 'Geschossfläche'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'Wohnungen'}}, {'TEXT': {'FUZZY': 'zu'}}, {'TEXT': {'FUZZY': 'verwenden'}}, {'TEXT': {'FUZZY': 'ist'}}]},
        {'label': 'Wirtschaftsstellen_land-_und_forstwirtschaftlicher_Betriebe_und_die_dazugehörigen_Wohnungen_und_Wohngebäude', 'pattern': [{'TEXT': {'FUZZY': 'Wirtschaftsstellen'}}, {'TEXT': {'FUZZY': 'land-'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'forstwirtschaftlicher'}}, {'TEXT': {'FUZZY': 'Betriebe'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'die'}}, {'TEXT': {'FUZZY': 'dazugehörigen'}}, {'TEXT': {'FUZZY': 'Wohnungen'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'Wohngebäude'}}]},
        {'label': 'Kleinsiedlungen_einschließlich_Wohngebäude_mit_entsprechenden_Nutzgärten_und_landwirtschaftliche_Nebenerwerbsstellen', 'pattern': [{'TEXT': {'FUZZY': 'Kleinsiedlungen'}}, {'TEXT': {'FUZZY': 'einschließlich'}}, {'TEXT': {'FUZZY': 'Wohngebäude'}}, {'TEXT': {'FUZZY': 'mit'}}, {'TEXT': {'FUZZY': 'entsprechenden'}}, {'TEXT': {'FUZZY': 'Nutzgärten'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'landwirtschaftliche'}}, {'TEXT': {'FUZZY': 'Nebenerwerbsstellen'}}]},
        {'label': 'sonstige_Wohngebäude', 'pattern': [{'TEXT': {'FUZZY': 'sonstige'}}, {'TEXT': {'FUZZY': 'wohngebäude'}}]},
        {'label': 'Läden_für_den_täglichen_Bedarf', 'pattern': [{'TEXT': {'FUZZY': 'läden'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'den'}}, {'TEXT': {'FUZZY': 'täglichen'}}, {'TEXT': {'FUZZY': 'bedarf'}}]},
        {'label': 'Betriebe_zur_Be-_und_Verarbeitung_und_Sammlung_land-_und_forstwirtschaftlicher_Erzeugnisse', 'pattern': [{'TEXT': {'FUZZY': 'Betriebe'}}, {'TEXT': {'FUZZY': 'zur'}}, {'TEXT': {'FUZZY': 'Be-'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'Verarbeitung'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'Sammlung'}}, {'TEXT': {'FUZZY': 'land-'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'forstwirtschaftlicher'}}, {'TEXT': {'FUZZY': 'Erzeugnisse'}}]},
        {'label': 'Einzelhandelsbetriebe,_Schank-_und_Speisewirtschaften_sowie_Betriebe_des_Beherbergungsgewerbes', 'pattern': [{'TEXT': {'FUZZY': 'Einzelhandelsbetriebe,'}}, {'TEXT': {'FUZZY': 'Schank-'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'Speisewirtschaften'}}, {'TEXT': {'FUZZY': 'sowie'}}, {'TEXT': {'FUZZY': 'Betriebe'}}, {'TEXT': {'FUZZY': 'des'}}, {'TEXT': {'FUZZY': 'Beherbergungsgewerbes'}}]},
        {'label': 'Anlagen_für_örtliche_Verwaltungen_sowie_für_kirchliche,_kulturelle,_soziale,_gesundheitliche_und_sportliche_Zwecke', 'pattern': [{'TEXT': {'FUZZY': 'Anlagen'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'örtliche'}}, {'TEXT': {'FUZZY': 'Verwaltungen'}}, {'TEXT': {'FUZZY': 'sowie'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'kirchliche,'}}, {'TEXT': {'FUZZY': 'kulturelle,'}}, {'TEXT': {'FUZZY': 'soziale,'}}, {'TEXT': {'FUZZY': 'gesundheitliche'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'sportliche'}}, {'TEXT': {'FUZZY': 'Zwecke'}}]},
        {'label': 'Anlagen_für_Verwaltungen_sowie_für_kirchliche,_kulturelle,_soziale,_gesundheitliche_und_sportliche_Zwecke', 'pattern': [{'TEXT': {'FUZZY': 'Anlagen'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'Verwaltungen'}}, {'TEXT': {'FUZZY': 'sowie'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'kirchliche,'}}, {'TEXT': {'FUZZY': 'kulturelle,'}}, {'TEXT': {'FUZZY': 'soziale,'}}, {'TEXT': {'FUZZY': 'gesundheitliche'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'sportliche'}}, {'TEXT': {'FUZZY': 'Zwecke'}}]},
        {'label': 'im_Erdgeschoss_an_der_Straßenseite_eine_Wohnnutzung_nicht_oder_nur_ausnahmsweise_zulässig_ist', 'pattern': [{'TEXT': {'FUZZY': 'im'}}, {'TEXT': {'FUZZY': 'Erdgeschoss'}}, {'TEXT': {'FUZZY': 'an'}}, {'TEXT': {'FUZZY': 'der'}}, {'TEXT': {'FUZZY': 'Straßenseite'}}, {'TEXT': {'FUZZY': 'eine'}}, {'TEXT': {'FUZZY': 'Wohnnutzung'}}, {'TEXT': {'FUZZY': 'nicht'}}, {'TEXT': {'FUZZY': 'oder'}}, {'TEXT': {'FUZZY': 'nur'}}, {'TEXT': {'FUZZY': 'ausnahmsweise'}}, {'TEXT': {'FUZZY': 'zulässig'}}, {'TEXT': {'FUZZY': 'ist'}}]},
        {'label': 'oberhalb_eines_im_Bebauungsplan_bestimmten_Geschosses_nur_Wohnungen_zulässig_sind', 'pattern': [{'TEXT': {'FUZZY': 'oberhalb'}}, {'TEXT': {'FUZZY': 'eines'}}, {'TEXT': {'FUZZY': 'im'}}, {'TEXT': {'FUZZY': 'Bebauungsplan'}}, {'TEXT': {'FUZZY': 'bestimmten'}}, {'TEXT': {'FUZZY': 'Geschosses'}}, {'TEXT': {'FUZZY': 'nur'}}, {'TEXT': {'FUZZY': 'Wohnungen'}}, {'TEXT': {'FUZZY': 'zulässig'}}, {'TEXT': {'FUZZY': 'sind'}}]},
        {'label': 'ein_im_Bebauungsplan_bestimmter_Anteil_der_zulässigen_Geschossfläche_oder_eine_im_Bebauungsplan_bestimmte_Größe_der_Geschossfläche_für_Wohnungen_zu_verwenden_ist,_oder', 'pattern': [{'TEXT': {'FUZZY': 'ein'}}, {'TEXT': {'FUZZY': 'im'}}, {'TEXT': {'FUZZY': 'Bebauungsplan'}}, {'TEXT': {'FUZZY': 'bestimmter'}}, {'TEXT': {'FUZZY': 'Anteil'}}, {'TEXT': {'FUZZY': 'der'}}, {'TEXT': {'FUZZY': 'zulässigen'}}, {'TEXT': {'FUZZY': 'Geschossfläche'}}, {'TEXT': {'FUZZY': 'oder'}}, {'TEXT': {'FUZZY': 'eine'}}, {'TEXT': {'FUZZY': 'im'}}, {'TEXT': {'FUZZY': 'Bebauungsplan'}}, {'TEXT': {'FUZZY': 'bestimmte'}}, {'TEXT': {'FUZZY': 'Größe'}}, {'TEXT': {'FUZZY': 'der'}}, {'TEXT': {'FUZZY': 'Geschossfläche'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'Wohnungen'}}, {'TEXT': {'FUZZY': 'zu'}}, {'TEXT': {'FUZZY': 'verwenden'}}, {'TEXT': {'FUZZY': 'ist,'}}, {'TEXT': {'FUZZY': 'oder'}}]},
        {'label': 'ein_im_Bebauungsplan_bestimmter_Anteil_der_zulässigen_Geschossfläche_oder_eine_im_Bebauungsplan_bestimmte_Größe_der_Geschossfläche_für_gewerbliche_Nutzungen_zu_verwenden_ist', 'pattern': [{'TEXT': {'FUZZY': 'ein'}}, {'TEXT': {'FUZZY': 'im'}}, {'TEXT': {'FUZZY': 'Bebauungsplan'}}, {'TEXT': {'FUZZY': 'bestimmter'}}, {'TEXT': {'FUZZY': 'Anteil'}}, {'TEXT': {'FUZZY': 'der'}}, {'TEXT': {'FUZZY': 'zulässigen'}}, {'TEXT': {'FUZZY': 'Geschossfläche'}}, {'TEXT': {'FUZZY': 'oder'}}, {'TEXT': {'FUZZY': 'eine'}}, {'TEXT': {'FUZZY': 'im'}}, {'TEXT': {'FUZZY': 'Bebauungsplan'}}, {'TEXT': {'FUZZY': 'bestimmte'}}, {'TEXT': {'FUZZY': 'Größe'}}, {'TEXT': {'FUZZY': 'der'}}, {'TEXT': {'FUZZY': 'Geschossfläche'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'gewerbliche'}}, {'TEXT': {'FUZZY': 'Nutzungen'}}, {'TEXT': {'FUZZY': 'zu'}}, {'TEXT': {'FUZZY': 'verwenden'}}, {'TEXT': {'FUZZY': 'ist'}}]},
        {'label': 'Geschäfts-_,_Büro-_und_Verwaltungsgebäude', 'pattern': [{'TEXT': {'FUZZY': 'Geschäfts-'}}, {'TEXT': {'FUZZY': ','}}, {'TEXT': {'FUZZY': 'Büro-'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'Verwaltungsgebäude'}}]},
        {'label': 'Büro-,_Geschäfts-_und_Verwaltungsgebäude', 'pattern': [{'TEXT': {'FUZZY': 'Büro-'}}, {'TEXT': {'FUZZY': ','}}, {'TEXT': {'FUZZY': 'Geschäfts-'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'Verwaltungsgebäude'}}]},
        {'label': 'Einzelhandelsbetriebe,_Schank-_und_Speisewirtschaften,_Betriebe_des_Beherbergungsgewerbes_und_Vergnügungsstätten', 'pattern': [{'TEXT': {'FUZZY': 'Einzelhandelsbetriebe,'}}, {'TEXT': {'FUZZY': 'Schank-'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'Speisewirtschaften,'}}, {'TEXT': {'FUZZY': 'Betriebe'}}, {'TEXT': {'FUZZY': 'des'}}, {'TEXT': {'FUZZY': 'Beherbergungsgewerbes'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'Vergnügungsstätten'}}]},
        {'label': 'sonstige_nicht_wesentlich_störende_Gewerbebetriebe', 'pattern': [{'TEXT': {'FUZZY': 'sonstige'}}, {'TEXT': {'FUZZY': 'nicht'}}, {'TEXT': {'FUZZY': 'wesentlich'}}, {'TEXT': {'FUZZY': 'störende'}}, {'TEXT': {'FUZZY': 'Gewerbebetriebe'}}]},
        {'label': 'Tankstellen_im_Zusammenhang_mit_Parkhäusern_und_Großgaragen', 'pattern': [{'TEXT': {'FUZZY': 'Tankstellen'}}, {'TEXT': {'FUZZY': 'im'}}, {'TEXT': {'FUZZY': 'Zusammenhang'}}, {'TEXT': {'FUZZY': 'mit'}}, {'TEXT': {'FUZZY': 'Parkhäusern'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'Großgaragen'}}]},
        {'label': 'Wohnungen_für_Aufsichts-_und_Bereitschaftspersonen_sowie_für_Betriebsinhaber_und_Betriebsleiter', 'pattern': [{'TEXT': {'FUZZY': 'Wohnungen'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'Aufsichts-'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'Bereitschaftspersonen'}}, {'TEXT': {'FUZZY': 'sowie'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'Betriebsinhaber'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'Betriebsleiter'}}]},
        {'label': 'Wohnungen_für_Aufsichts-_und_Bereitschaftspersonen', 'pattern': [{'TEXT': {'FUZZY': 'Wohnungen'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'Aufsichts-'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'Bereitschaftspersonen'}}]},
        {'label': 'Betriebe_des_produzierenden_Gewerbes', 'pattern': [{'TEXT': {'FUZZY': 'Betriebe'}}, {'TEXT': {'FUZZY': 'des'}}, {'TEXT': {'FUZZY': 'produzierenden'}}, {'TEXT': {'FUZZY': 'Gewerbes'}}]},
        {'label': 'sonstige_Wohnungen_nach_Maßgabe_von_Festsetzungen_des_Bebauungsplans', 'pattern': [{'TEXT': {'FUZZY': 'sonstige'}}, {'TEXT': {'FUZZY': 'Wohnungen'}}, {'TEXT': {'FUZZY': 'nach'}}, {'TEXT': {'FUZZY': 'Maßgabe'}}, {'TEXT': {'FUZZY': 'von'}}, {'TEXT': {'FUZZY': 'Festsetzungen'}}, {'TEXT': {'FUZZY': 'des'}}, {'TEXT': {'FUZZY': 'Bebauungsplans'}}]},
        {'label': 'Tankstellen,_die_nicht_unter_Absatz_2_Nummer_5_fallen', 'pattern': [{'TEXT': {'FUZZY': 'Tankstellen,'}}, {'TEXT': {'FUZZY': 'die'}}, {'TEXT': {'FUZZY': 'nicht'}}, {'TEXT': {'FUZZY': 'unter'}}, {'TEXT': {'FUZZY': 'Absatz'}}, {'TEXT': {'FUZZY': '2'}}, {'TEXT': {'FUZZY': 'Nummer'}}, {'TEXT': {'FUZZY': '5'}}, {'TEXT': {'FUZZY': 'fallen'}}]},
        {'label': 'Wohnungen,_die_nicht_unter_Absatz_2_Nummer_6_und_7_fallen', 'pattern': [{'TEXT': {'FUZZY': 'Wohnungen,'}}, {'TEXT': {'FUZZY': 'die'}}, {'TEXT': {'FUZZY': 'nicht'}}, {'TEXT': {'FUZZY': 'unter'}}, {'TEXT': {'FUZZY': 'Absatz'}}, {'TEXT': {'FUZZY': '2'}}, {'TEXT': {'FUZZY': 'Nummer'}}, {'TEXT': {'FUZZY': '6'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': '7'}}, {'TEXT': {'FUZZY': 'fallen'}}]},
        {'label': 'oberhalb_eines_im_Bebauungsplan_bestimmten_Geschosses_nur_Wohnungen_zulässig_sind_oder', 'pattern': [{'TEXT': {'FUZZY': 'oberhalb'}}, {'TEXT': {'FUZZY': 'eines'}}, {'TEXT': {'FUZZY': 'im'}}, {'TEXT': {'FUZZY': 'Bebauungsplan'}}, {'TEXT': {'FUZZY': 'bestimmten'}}, {'TEXT': {'FUZZY': 'Geschosses'}}, {'TEXT': {'FUZZY': 'nur'}}, {'TEXT': {'FUZZY': 'Wohnungen'}}, {'TEXT': {'FUZZY': 'zulässig'}}, {'TEXT': {'FUZZY': 'sind'}}, {'TEXT': {'FUZZY': 'oder'}}]},
        {'label': 'in_Gebäuden_ein_im_Bebauungsplan_bestimmter_Anteil_der_zulässigen_Geschossfläche_oder_eine_bestimmte_Größe_der_Geschossfläche_für_Wohnungen_zu_verwenden_ist', 'pattern': [{'TEXT': {'FUZZY': 'in'}}, {'TEXT': {'FUZZY': 'Gebäuden'}}, {'TEXT': {'FUZZY': 'ein'}}, {'TEXT': {'FUZZY': 'im'}}, {'TEXT': {'FUZZY': 'Bebauungsplan'}}, {'TEXT': {'FUZZY': 'bestimmter'}}, {'TEXT': {'FUZZY': 'Anteil'}}, {'TEXT': {'FUZZY': 'der'}}, {'TEXT': {'FUZZY': 'zulässigen'}}, {'TEXT': {'FUZZY': 'Geschossfläche'}}, {'TEXT': {'FUZZY': 'oder'}}, {'TEXT': {'FUZZY': 'eine'}}, {'TEXT': {'FUZZY': 'bestimmte'}}, {'TEXT': {'FUZZY': 'Größe'}}, {'TEXT': {'FUZZY': 'der'}}, {'TEXT': {'FUZZY': 'Geschossfläche'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'Wohnungen'}}, {'TEXT': {'FUZZY': 'zu'}}, {'TEXT': {'FUZZY': 'verwenden'}}, {'TEXT': {'FUZZY': 'ist'}}]},
        {'label': 'Gewerbebetriebe_aller_Art_einschließlich_Anlagen_zur_Erzeugung_von_Strom_oder_Wärme_aus_solarer_Strahlungsenergie_oder_Windenergie,_Lagerhäuser,_Lagerplätze_und_öffentliche_Betriebe', 'pattern': [{'TEXT': {'FUZZY': 'Gewerbebetriebe'}}, {'TEXT': {'FUZZY': 'aller'}}, {'TEXT': {'FUZZY': 'Art'}}, {'TEXT': {'FUZZY': 'einschließlich'}}, {'TEXT': {'FUZZY': 'Anlagen'}}, {'TEXT': {'FUZZY': 'zur'}}, {'TEXT': {'FUZZY': 'Erzeugung'}}, {'TEXT': {'FUZZY': 'von'}}, {'TEXT': {'FUZZY': 'Strom'}}, {'TEXT': {'FUZZY': 'oder'}}, {'TEXT': {'FUZZY': 'Wärme'}}, {'TEXT': {'FUZZY': 'aus'}}, {'TEXT': {'FUZZY': 'solarer'}}, {'TEXT': {'FUZZY': 'Strahlungsenergie'}}, {'TEXT': {'FUZZY': 'oder'}}, {'TEXT': {'FUZZY': 'Windenergie,'}}, {'TEXT': {'FUZZY': 'Lagerhäuser,'}}, {'TEXT': {'FUZZY': 'Lagerplätze'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'öffentliche'}}, {'TEXT': {'FUZZY': 'Betriebe'}}]},
        {'label': 'Anlagen_für_sportliche_Zwecke', 'pattern': [{'TEXT': {'FUZZY': 'Anlagen'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'sportliche'}}, {'TEXT': {'FUZZY': 'Zwecke'}}]},
        {'label': 'Geschäfts-_und_Büronutzungen', 'pattern': [{'TEXT': {'FUZZY': 'Geschäfts-'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'Büronutzungen'}}, {'TEXT': {'FUZZY': 'Zwecke'}}]},
        {'label': 'großflächige_Einzelhandelsbetriebe', 'pattern': [{'TEXT': {'FUZZY': 'großflächige'}}, {'TEXT': {'FUZZY': 'Einzelhandelsbetriebe'}}]},
        {'label': 'Wohnungen_für_Aufsichts-_und_Bereitschaftspersonen_sowie_für_Betriebsinhaber_und_Betriebsleiter', 'pattern': [{'TEXT': {'FUZZY': 'Wohnungen'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'Aufsichts-'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'Bereitschaftspersonen'}}, {'TEXT': {'FUZZY': 'sowie'}}, {'TEXT': {'FUZZY': 'für'}}, {'TEXT': {'FUZZY': 'Betriebsinhaber'}}, {'TEXT': {'FUZZY': 'und'}}, {'TEXT': {'FUZZY': 'Betriebsleiter'}}]},
        {'label': 'Läden_mit_nahversorgungsrelevanten_Sortimenten', 'pattern': [{'TEXT': {'FUZZY': 'Läden'}}, {'TEXT': {'FUZZY': 'mit'}}, {'TEXT': {'FUZZY': 'nahversorgungsrelevanten'}}, {'TEXT': {'FUZZY': 'Sortimenten'}}]}
    ]
    ruler.add_patterns(patterns_baugebiete)
    doc = nlp(doc)
    return doc



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
    ents = nlp_ner(text)
    baugebiete = []
    scores = []
    answers = []
    final = []
    for i in range(len(ents.ents)):
        if("GEBIETE" in ents.ents[i].label_):
            try:
                if(normalize_entity(ents.ents[i].text).ents[0].label_ not in baugebiete):
                    baugebiete.append(normalize_entity(ents.ents[i].text).ents[0].label_)
            except:
                if(ents.ents[i].text not in baugebiete):
                    baugebiete.append(ents.ents[i].text)
    for i in range(len(ents.ents)):
        if("GEBIETE" not in ents.ents[i].label_ and "MASS" not in ents.ents[i].label_ and "BAUWEISE" not in ents.ents[i].label_):
            for baugebiet in baugebiete:
                answer, score = get_answers("Sind " + ents.ents[i].text + " im " + baugebiet + " zulässig, nicht zulässig, unzulässig, ausnahmsweise zulässig oder nicht Bestandteil des Bebauungsplanes ? ", text)
                scores.append(score)
                answers.append([answer, baugebiet])
            if scores:
                final_answer, final_score = (get_highest_score_answer(scores, answers))
                # Convert final list of lists to list of tuples for hashability
                final = [tuple(i) for i in final]

                # Create a set for faster lookups
                final_set = set(final)

                while True:
                    final_tuple = (final_score, final_answer[0], ents.ents[i].text, final_answer[1])
                    if final_tuple not in final_set:
                        break

                    # If execution reaches this point it means final_tuple was in final_set
                    max_score_index = scores.index(max(scores)) if scores else None
                    if max_score_index is not None:
                        del scores[max_score_index]
                        del answers[max_score_index]

                    # Recalculate final_answer and final_score if there are still scores left
                    if scores:
                        final_answer, final_score = get_highest_score_answer(scores, answers)
                    else:
                        break
                # Convert final list of tuples back to list of lists
                final = [list(i) for i in final]

                scores = []
                answers = []
                    
                if(final_score >= threshold):
                    final.append([final_score, final_answer[0], ents.ents[i].text, final_answer[1]])
            scores = []
            answers = []
        if("MASS" in ents.ents[i].label_):
            for baugebiet in baugebiete:
                answer, score = get_answers("Welche " + ents.ents[i].text + " ist in dem " + baugebiet + " festgelegt ? ", text)
                scores.append(score)
                answers.append([answer, baugebiet])
            if scores:
                final_answer, final_score = (get_highest_score_answer(scores, answers))
                # Convert final list of lists to list of tuples for hashability
                final = [tuple(i) for i in final]

                # Create a set for faster lookups
                final_set = set(final)

                while True:
                    final_tuple = (final_score, final_answer[0], ents.ents[i].text, final_answer[1])
                    if final_tuple not in final_set:
                        break

                    # If execution reaches this point it means final_tuple was in final_set
                    max_score_index = scores.index(max(scores)) if scores else None
                    if max_score_index is not None:
                        del scores[max_score_index]
                        del answers[max_score_index]

                    # Recalculate final_answer and final_score if there are still scores left
                    if scores:
                        final_answer, final_score = get_highest_score_answer(scores, answers)
                    else:
                        break
                # Convert final list of tuples back to list of lists
                final = [list(i) for i in final]

                scores = []
                answers = []
                if(final_score >= threshold):
                    final.append([final_score, final_answer[0], ents.ents[i].text, final_answer[1]])
            scores = []
            answers = []
        if("BAUWEISE" in ents.ents[i].label_):
            for baugebiet in baugebiete:
                answer, score = get_answers("Ist die Bauweise offen, geschlossen oder abweichend im " + baugebiet + " ? ", text)
                scores.append(score)
                answers.append([answer, baugebiet])
            if scores:
                final_answer, final_score = (get_highest_score_answer(scores, answers))
                # Convert final list of lists to list of tuples for hashability
                final = [tuple(i) for i in final]

                # Create a set for faster lookups
                final_set = set(final)

                while True:
                    final_tuple = (final_score, final_answer[0], ents.ents[i].text, final_answer[1])
                    if final_tuple not in final_set:
                        break

                    # If execution reaches this point it means final_tuple was in final_set
                    max_score_index = scores.index(max(scores)) if scores else None
                    if max_score_index is not None:
                        del scores[max_score_index]
                        del answers[max_score_index]

                    # Recalculate final_answer and final_score if there are still scores left
                    if scores:
                        final_answer, final_score = get_highest_score_answer(scores, answers)
                    else:
                        break
                # Convert final list of tuples back to list of lists
                final = [list(i) for i in final]

                scores = []
                answers = []
                if(final_score >= threshold):
                    final.append([final_score, final_answer[0], ents.ents[i].text, final_answer[1]])
            scores = []
            answers = []
    return final, len(baugebiete), len(ents.ents)-len(baugebiete)

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
