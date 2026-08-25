"""Modulo de la interfaz grafica (Tkinter) para la deteccion de neumonia."""

from tkinter import *
from tkinter import ttk, font, filedialog
from tkinter.messagebox import askokcancel, showinfo, WARNING

import csv
import pyautogui
from PIL import ImageTk, Image

from src.read_img import read_dicom_file, read_jpg_file
from src.integrator import predict


class App:
    """Ventana principal de la aplicacion de deteccion de neumonia.

    Contiene todos los botones y campos con los que interactua el
    usuario (cargar imagen, predecir, guardar resultados, generar PDF
    y borrar), y delega todo el trabajo de lectura de imagenes y
    prediccion en los modulos read_img.py e integrator.py.
    """

    def __init__(self):
        self.root = Tk()
        self.root.title("Herramienta para la deteccion rapida de neumonia")

        #   BOLD FONT
        fonti = font.Font(weight="bold")

        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        #   LABELS
        self.lab1 = ttk.Label(self.root, text="Imagen Radiografica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cedula Paciente:", font=fonti)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNOSTICO MEDICO DE NEUMONIA",
            font=fonti,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        #   TWO STRING VARIABLES TO CONTAIN ID AND RESULT
        self.ID = StringVar()
        self.result = StringVar()

        #   TWO INPUT BOXES
        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)

        #   GET ID
        self.ID_content = self.text1.get()

        #   TWO IMAGE INPUT BOXES
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        #   BUTTONS
        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )

        #   WIDGETS POSITIONS
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)
        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)
        self.text1.place(x=200, y=350)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)

        #   FOCUS ON PATIENT ID
        self.text1.focus_set()

        #  se reconoce como un elemento de la clase
        self.array = None

        #   NUMERO DE IDENTIFICACION PARA GENERAR PDF
        self.reportID = 0

        #   RUN LOOP
        self.root.mainloop()

    #   METHODS
    def load_img_file(self):
        """Abre el dialogo de seleccion de archivo y carga la imagen elegida.

        Detecta la extension del archivo seleccionado: si es .dcm usa
        read_dicom_file, en cualquier otro caso (.jpg, .jpeg, .png) usa
        read_jpg_file. La imagen cargada se guarda en self.array para
        que run_model pueda usarla despues, y se muestra en el panel
        izquierdo de la interfaz.
        """
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Select image",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("jpg files", "*.jpg"),
                ("png files", "*.png"),
            ),
        )
        if filepath:
            if filepath.lower().endswith(".dcm"):
                self.array, img2show = read_dicom_file(filepath)
            else:
                self.array, img2show = read_jpg_file(filepath)
            self.img1 = img2show.resize((250, 250), Image.LANCZOS)
            self.img1 = ImageTk.PhotoImage(self.img1)
            self.text_img1.image_create(END, image=self.img1)
            self.button1["state"] = "enabled"

    def run_model(self):
        """Ejecuta la prediccion sobre la imagen cargada y muestra el resultado.

        Limpia los campos de resultado y probabilidad antes de insertar
        los nuevos, para que no se acumulen con predicciones anteriores.
        Llama a integrator.predict, que retorna la clase, la
        probabilidad y el heatmap, y los muestra en la interfaz.
        """
        self.text2.delete(1.0, "end")
        self.text3.delete(1.0, "end")
        self.label, self.proba, self.heatmap = predict(self.array)
        self.img2 = Image.fromarray(self.heatmap)
        self.img2 = self.img2.resize((250, 250), Image.LANCZOS)
        self.img2 = ImageTk.PhotoImage(self.img2)
        self.text_img2.image_create(END, image=self.img2)
        self.text2.insert(END, self.label)
        self.text3.insert(END, "{:.2f}".format(self.proba) + "%")

    def save_results_csv(self):
        """Guarda la cedula, el resultado y la probabilidad en historial.csv."""
        with open("historial.csv", "a") as csvfile:
            w = csv.writer(csvfile, delimiter="-")
            w.writerow(
                [self.text1.get(), self.label, "{:.2f}".format(self.proba) + "%"]
            )
            showinfo(title="Guardar", message="Los datos se guardaron con exito.")

    def create_pdf(self):
        """Captura la ventana actual y genera un reporte en PDF.

        Toma una captura de pantalla de la region exacta donde esta la
        ventana (usando pyautogui) y la convierte a PDF. Cada reporte
        generado incrementa self.reportID para no sobreescribir los
        anteriores.
        """
        ID = "Reporte" + str(self.reportID) + ".jpg"
        x = self.root.winfo_rootx()
        y = self.root.winfo_rooty()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        screenshot.save(ID)
        img = Image.open(ID)
        img = img.convert("RGB")
        pdf_path = r"Reporte" + str(self.reportID) + ".pdf"
        img.save(pdf_path)
        self.reportID += 1
        showinfo(title="PDF", message="El PDF fue generado con exito.")

    def delete(self):
        """Borra los datos y campos de la interfaz, previa confirmacion."""
        answer = askokcancel(
            title="Confirmacion", message="Se borraran todos los datos.", icon=WARNING
        )
        if answer:
            self.text1.delete(0, "end")
            self.text2.delete(1.0, "end")
            self.text3.delete(1.0, "end")
            self.text_img1.delete(self.img1, "end")
            self.text_img2.delete(self.img2, "end")
            showinfo(title="Borrar", message="Los datos se borraron con exito")


def main():
    """Punto de entrada de la aplicacion."""
    App()
    return 0


if __name__ == "__main__":
    main()