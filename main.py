
import PySimpleGUI as sg
import nlp
import brain_cancer_cnn as model
from PIL import Image, ImageTk
layout = [
    [
        sg.Column(
            [
            [sg.Text("Hello from AI doctor")],
            [sg.Multiline(size=(80, 20), key="-CHAT-")],
            [sg.In(size=(25, 1), enable_events=True, key="-IN-"),sg.Button("SEND")],
            ]
        ),
        sg.VSeperator(),
        sg.Column(
            [
                [sg.In("Choose an image",key="-IMG_PATH-"), sg.FileBrowse(target='-IMG_PATH-'),sg.Button("OK")],
                [sg.Image(key="-IMAGE-")]
            ]
        )
    ]
]
# Create the window
window = sg.Window("Demo", layout)
state = "chat"
# Create an event loop
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    if event == "SEND":
        words = values["-IN-"]
        if len(words) == 0:
            continue
        window['-CHAT-'].print("YOU: " + words)
        command = nlp.process(words)
        if command == "brain_tumor":
            window['-CHAT-'].print("Doctor: Please provide a mri image of your brain")
            state = "brain_tumor"
        elif command == "help":
            window['-CHAT-'].print("Doctor: I can do brain tumor scan")
            state = "chat"
    elif event == "OK":
        path = values["-IMG_PATH-"]
        window['-CHAT-'].print("SYSTEM: you choose \""+ path + "\" as your input image.")
        if state == "brain_tumor":
            prediction_path = model.predict_image(path)
            image = Image.open(prediction_path)
            window['-IMAGE-'].update(data = ImageTk.PhotoImage(image))
            window['-CHAT-'].print("Doctor: the segmentation of this result is showed in the right")
            state = "chat"
        # show the prediction
window.close() 
