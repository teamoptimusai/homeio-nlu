from engine import NLUEngine
from tkinter import *

DEFAULT_WEIGHTS = 'scripts/epoch50_best_model_trace.pth'

nlu_engine = NLUEngine(DEFAULT_WEIGHTS)

root = Tk()
root.title("NLUEngine")
root.geometry("300x300")


label = Label(root, text="Enter your text here:", fg="blue")
label.pack()
text_input = Text(root, width=30, height=5)
text_input.pack()


def submit():
    global i, scenario_output, intent_output, entities_output, submit_button
    scenario_output = Label(root, text="Scenario", fg="green")
    intent_output = Label(root, text="Intent", fg="green")
    entities_output = Label(root, text="Entities", fg="green")
    text = text_input.get("1.0", END)
    predictions = nlu_engine.predict(sentence=text)
    text_input.delete("1.0", END)
    scenario_output.config(text=str(predictions["scenario"]))
    intent_output.config(text=str(predictions["intent"]))
    entities_output.config(text=str(predictions["entities"]))
    scenario_output.pack()
    intent_output.pack()
    entities_output.pack()
    submit_button.config(state=DISABLED)


def clear():
    scenario_output.pack_forget()
    intent_output.pack_forget()
    entities_output.pack_forget()
    submit_button.config(state=NORMAL)


submit_button = Button(root, text="Submit", fg="green",
                       command=submit)
clear_button = Button(root, text="Clear", fg="red",
                      command=clear)
submit_button.pack()
clear_button.pack()

root.mainloop()
