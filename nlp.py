
import spacy
import classy_classification

# train data
data = {
    "help": ["What can you do",
               "How to use you",
               "Help"],
    "brain_tumor": ["scan my brain",
                "do brain tumor scan for me",
                "check my brain"]
}

# load the modal
nlp = spacy.load('en_core_web_md')
nlp.add_pipe("text_categorizer", 
    config={
        "data": data,
        "model": "spacy"
    }
)

# predict
def process(words):
    result = nlp(words)._.cats
    choose = "help"
    for i in result.keys():
        if result[i] > result[choose]:
            choose = i
    return choose