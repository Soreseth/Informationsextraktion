import pytesseract
import cv2
from pdf2image.pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
import os
from PyPDF2 import PdfFileReader
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import xml.etree.ElementTree as ET
import glob



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
    return " "

#Eingabe 1.) relativer Pfad zum Ordner mit den PDFs von .py Datei
def pdf_txt_conversion(input_path,output_path):
    # Laufvariable für die Bildbenennung
    i = 0
    # Pfad zu PDF angeben
    for file in os.listdir(input_path):
        # Bool-Wert, ob viele Seiten in PDF vorliegen oder nicht
        many_images = False
        if file.endswith(".pdf"):
            tmp_path = os.path.join(input_path, file)
            # PDF in jpg konvertieren
            images = convert_from_path(tmp_path,200, poppler_path='/opt/homebrew/Cellar/poppler/23.06.0/bin')
            # pdf zu jpg konvertieren
            if(len(images) == 2):
                images[1].save('/Users/abinashselvarajah/Desktop/Bachelorarbeit/Code/images/page'+ str(i) +'.jpg', 'JPEG')
            elif(len(images) == 1):
                images[0].save('/Users/abinashselvarajah/Desktop/Bachelorarbeit/Code/images/page'+ str(i) +'.jpg', 'JPEG')
            elif(len(images) > 2):
                # viele Seiten liegen vor (aeltere Dokumente)
                many_images = True
                for j in range(1,len(images)):
                    # Bilder werden in einem gesonderten Ordner "many_images" gespeichert
                    images[j].save('/Users/abinashselvarajah/Desktop/Bachelorarbeit/Code/images/many_images/page'+ str(j-1) +'.jpg', 'JPEG')
                print("Mehr als 2 Seiten")
            else:
                print("Keine Seiten")
                continue

            if(many_images == False):
                # Pfad zu jpg-Bild angeben
                image = cv2.imread('images/page'+ str(i) +'.jpg')

                # Pfad zu Tesseract.exe angeben
                pytesseract.pytesseract.tesseract_cmd = r'/Users/abinashselvarajah/anaconda3/envs/bachelorarbeit/bin/tesseract'
                # Texterkennung aus jpg-Bild, Sprache: Deutsch
                result = pytesseract.image_to_string(image, lang='deu')
                with open(output_path+'/page_'+ str(i)+ '.txt', 'w') as f:
                    #result = txt_preprocessing(result)
                    f.write(result)
                    f.close()
                #Inkrementation für Dateinamen
                i += 1
            elif(many_images == True):
                for k in range(1,len(images)):
                    # Pfad zu jpg-Bild angeben
                    image = cv2.imread('images/many_images/page'+ str(k-1) +'.jpg')

                    # Pfad zu Tesseract.exe angeben
                    pytesseract.pytesseract.tesseract_cmd = r'/Users/abinashselvarajah/anaconda3/envs/bachelorarbeit/bin/tesseract'
                    # Texterkennung aus jpg-Bild, Sprache: Deutsch
                    result = pytesseract.image_to_string(image, lang='deu')
                    with open(output_path+'/many_txts/page_'+ str(k-1)+ '.txt', 'w') as f:
                        result = txt_preprocessing(result)
                        f.write(result)
                        f.close()

                # Löschen der jpg-Bilder
                for m in range(1,len(images)):
                    os.remove('images/many_images/page'+ str(m-1) +'.jpg')

                # Zusammenfügen der einzelnen Textdateien zu einer Textdatei
                read_files = glob.glob(output_path+'/many_txts/*.txt')
                with open(output_path+'/page_'+ str(i)+ '.txt', "wb") as outfile:
                    # l ist Laufvariable, damit beim Zusammenfügen die richtige Reihenfolge der Texte gewährleistet wird
                    l = 0
                    for f in read_files:
                        with open(f, "rb") as infile:
                            if(f == output_path+'/many_txts/page_' + str(l) + '.txt'):
                                outfile.write(infile.read())
                                l += 1

                # Löschen der txt-Dateien
                for m in range(1,len(images)):
                    os.remove(output_path+'/many_txts/page_'+ str(m-1)+ '.txt')
                #Inkrementation für Dateinamen
                i += 1

    # Löschen der jpg-Bilder
    for k in range(i):
        os.remove('images/page'+ str(k) +'.jpg')

def pdf_xml_conversion(input_path,output_path):
    # Laufvariable für die Bildbenennung
    i = 0
    # Pfad zu PDF angeben
    for file in os.listdir(input_path):
        # Bool-Wert, ob viele Seiten in PDF vorliegen oder nicht
        many_images = False
        if file.endswith(".pdf"):
            tmp_path = os.path.join(input_path, file)
            images = convert_from_path(tmp_path,200)

            # pdf zu jpg konvertieren
            if(len(images) == 2):
                images[1].save('/Users/abinashselvarajah/Desktop/Bachelorarbeit/Code/images/page'+ str(i) +'.jpg', 'JPEG')
            elif(len(images) == 1):
                images[0].save('/Users/abinashselvarajah/Desktop/Bachelorarbeit/Code/images/page'+ str(i) +'.jpg', 'JPEG')
            elif(len(images) > 2):
                # viele Seiten liegen vor (aeltere Dokumente)
                many_images = True
                for j in range(1,len(images)):
                    # Bilder werden in einem gesonderten Ordner "many_images" gespeichert
                    images[j].save('/Users/abinashselvarajah/Desktop/Bachelorarbeit/Code/images/many_images/page'+ str(j-1) +'.jpg', 'JPEG')
                print("Mehr als 2 Seiten")
            else:
                print("Keine Seiten")
                continue

            if(many_images == False):
                # xml Root-Knoten erstellen
                xml_root = ET.Element('root')
                image = cv2.imread('images/page'+ str(i) +'.jpg')

                # Pfad zu Tesseract.exe angeben
                pytesseract.pytesseract.tesseract_cmd = r'/Users/abinashselvarajah/anaconda3/envs/bachelorarbeit/bin/tesseract'
                # Texterkennung aus jpg-Bild, Sprache: Deutsch
                result = pytesseract.image_to_string(image, lang='deu')
                # Text in xml-Datei schreiben
                page_element = ET.SubElement(xml_root, 'page')
                page_element.text = result

                # xml-Datei speichern
                tree = ET.ElementTree(xml_root)
                output_xml = 'output'+str(i)+'.xml'
                tree.write(output_xml, xml_declaration=True, encoding="utf-8")
                #Inkrementation für Dateinamen
                i += 1
#%%
#pdf_txt_conversion('bsp_pdf_src','bsp_txt_src')
#%%
