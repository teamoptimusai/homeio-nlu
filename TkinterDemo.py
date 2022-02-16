from typing import Any

from engine import NLUEngine
from tkinter import *

DEFAULT_WEIGHTS = 'scripts/epoch50_best_model_trace.pth'

nlu_engine = NLUEngine(DEFAULT_WEIGHTS)

root = Tk()
root.title("NLUEngine")
root.geometry("600x450")
root.resizable(width=False, height=False)
root.attributes("-alpha", 0.9)
# root.configure(background='white')


label = Label(root, text="Enter your text here:",
              fg="black", font=("Helvetica", 20))

label.pack()
text_input = Text(root, width=50, height=5, font=("Helvetica", 20))
text_input.pack(pady=10)


def submit():
    global i, scenario_output, intent_output, entities_output, submit_button

    scenario_output = Label(root, text="Scenario",
                            fg="purple", font=("Helvetica", 20))
    intent_output = Label(root, text="Intent", fg="green",
                          font=("Helvetica", 20))
    entities_output = Label(root, text="Entities",
                            fg="orange", font=("Helvetica", 20))

    text = text_input.get("1.0", END)
    text_input.delete("1.0", END)

    predictions: dict[str, Any] = nlu_engine.predict(sentence=text)
    scenario = predictions["scenario"]
    intent = predictions["intent"]
    entities = predictions["entities"]
    entity_str = ""
    for entity in entities:
        print(entity)
        entity_str += f"{entity['word']} : {entity['entity']} | Confidence: {round(entity['score'],2)} \n"

    scenario_output.config(
        text=f"Scenario: {scenario['class']} | Confidence: {round(scenario['score'],2)}")
    intent_output.config(
        text=f"Intent: {intent['class']} | Confidence: {round(intent['score'], 2)}")
    entities_output.config(text=entity_str)
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
                       command=submit, font=("Helvetica", 20))
clear_button = Button(root, text="Clear", fg="red",
                      command=clear, font=("Helvetica", 20))
clear_button.pack(side=BOTTOM, padx=10, pady=10)
submit_button.pack(side=BOTTOM, padx=10)

root.mainloop()
