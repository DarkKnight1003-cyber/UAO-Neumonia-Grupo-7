import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import detector_neumonia


class _Widget:
    def __init__(self, value=""):
        self.value = value
        self.deleted = []
        self.inserted = []
        self.images = []

    def get(self, *args):
        return self.value

    def delete(self, *args):
        self.deleted.append(args)
        self.value = ""

    def insert(self, *args):
        self.inserted.append(args)
        if len(args) > 1:
            self.value = args[1]

    def image_create(self, *args, **kwargs):
        self.images.append((args, kwargs))

    def __setitem__(self, key, value):
        setattr(self, key, value)


def _app_without_init():
    app: Any = detector_neumonia.App.__new__(detector_neumonia.App)
    app.text1 = _Widget("123")
    app.text2 = _Widget()
    app.text3 = _Widget()
    app.text_img1 = _Widget()
    app.text_img2 = _Widget()
    app.button1 = _Widget()
    app.root = _Widget()
    app.reportID = 0
    app.array = np.ones((2, 2, 3), dtype=np.uint8)
    return app


@pytest.mark.parametrize("extension", [".dcm", ".jpg", ".jpeg", ".png"])
def test_app_load_img_file_selects_reader_and_enables_prediction(monkeypatch, extension):
    app = _app_without_init()
    image = Image.new("RGB", (4, 4), "white")
    loaded = np.ones((4, 4, 3), dtype=np.uint8)
    calls = []
    monkeypatch.setattr(detector_neumonia.filedialog, "askopenfilename", lambda **_: "/tmp/image" + extension)
    monkeypatch.setattr(detector_neumonia, "read_dicom_file", lambda path: calls.append("dcm") or (loaded, image))
    monkeypatch.setattr(detector_neumonia, "read_jpg_file", lambda path: calls.append("jpg") or (loaded, image))
    monkeypatch.setattr(detector_neumonia.ImageTk, "PhotoImage", lambda value: value)
    app.load_img_file()
    assert app.array is loaded
    assert calls == (["dcm"] if extension.lower() == ".dcm" else ["jpg"])
    assert app.button1.state == "enabled"


@pytest.mark.parametrize("prediction", [("bacteriana", 12.345), ("normal", 50.0), ("viral", 99.999), ("normal", 0.0)])
def test_app_run_model_clears_and_displays_prediction(monkeypatch, prediction):
    app = _app_without_init()
    app.img1 = object()
    heatmap = np.zeros((4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(detector_neumonia, "predict", lambda _: (prediction[0], prediction[1], heatmap))
    monkeypatch.setattr(detector_neumonia.ImageTk, "PhotoImage", lambda value: value)
    app.run_model()
    assert app.text2.value == prediction[0]
    assert app.text3.value == f"{prediction[1]:.2f}%"
    assert app.text2.deleted and app.text3.deleted
    assert app.text_img2.images


@pytest.mark.parametrize("label,proba", [("bacteriana", 1.0), ("normal", 25.5), ("viral", 99.0), ("normal", 50.25)])
def test_app_save_results_csv_writes_delimited_history(monkeypatch, tmp_path, label, proba):
    app = _app_without_init()
    app.label = label
    app.proba = proba
    messages = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(detector_neumonia, "showinfo", lambda **kwargs: messages.append(kwargs))
    app.save_results_csv()
    with Path("historial.csv").open(newline="") as csvfile:
        rows = list(csv.reader(csvfile, delimiter="-"))
    assert rows == [["123", label, f"{proba:.2f}%"]]
    assert messages[0]["title"] == "Guardar"


@pytest.mark.parametrize("confirmed", [True, False, True, False])
def test_app_delete_respects_confirmation(monkeypatch, confirmed):
    app = _app_without_init()
    app.img1 = object()
    app.img2 = object()
    messages = []
    monkeypatch.setattr(detector_neumonia, "askokcancel", lambda **_: confirmed)
    monkeypatch.setattr(detector_neumonia, "showinfo", lambda **kwargs: messages.append(kwargs))
    app.delete()
    if confirmed:
        assert app.text1.deleted and app.text2.deleted and app.text3.deleted
        assert app.text_img1.deleted and app.text_img2.deleted
        assert messages[0]["title"] == "Borrar"
    else:
        assert not app.text1.deleted
        assert messages == []


@pytest.mark.parametrize("report_id", [0, 1, 7, 19])
def test_app_create_pdf_captures_unique_report(monkeypatch, tmp_path, report_id):
    app = _app_without_init()
    app.reportID = report_id
    app.root.winfo_rootx = lambda: 1
    app.root.winfo_rooty = lambda: 2
    app.root.winfo_width = lambda: 3
    app.root.winfo_height = lambda: 4
    saved = []

    class Screenshot:
        def save(self, path, *args, **kwargs):
            saved.append(path)

    screenshot = Screenshot()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(detector_neumonia.pyautogui, "screenshot", lambda region: screenshot)
    monkeypatch.setattr(detector_neumonia.Image, "open", lambda path: Image.new("RGB", (3, 4), "black"))
    monkeypatch.setattr(detector_neumonia, "showinfo", lambda **_: None)
    app.create_pdf()
    assert app.reportID == report_id + 1
    assert saved[0] == f"Reporte{report_id}.jpg"