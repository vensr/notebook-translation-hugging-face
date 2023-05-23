## Using Pre-Trained Models

As of now if you check on [hugging face models for translation](https://huggingface.co/tasks/translation), there are around 2117 models.

Check out one of these models from Meta [M2M100 1.2B](https://huggingface.co/facebook/m2m100_1.2B) that can be used for translating to and from more than 200 languages.

## Translation Task

Before we start the translation task, ensure that the following python libraries are installed.

```bash
# packages required for creating the pipeline and loading pre-trained models
pip install transformers datasets -q

# packages required for tokenization
pip install  sentencepiece sacremoses -q
```

Import the required libraries

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
```

The translation process using hugging face pipeline is a 4 step process.

* Set the device
```python
device = torch.cuda.current_device() if torch.cuda.is_available() else -1
````
* Load the pre-trained model
```python
model_name = 'facebook/m2m100_1.2B'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
````

* Load the Tokenizer
```python
# source language is hindi and target language is english
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="hi", tgt_lang="en")
````

* Create a Hugging Face Pipeline
```python
translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang="hi", tgt_lang="en",device=device)
````

* Translate the text
```python
text = "यह वास्तव में आश्चर्यजनक है, क्योंकि यह महान अनुवाद करने में सक्षम है।"
target_seq = translator(text)
print(target_seq [0]['translation_text'])
````

You should see the output that is something like this
"This is really amazing, because it is able to translate great."

That is all, you can also expose this as an API service and create a great interface like google translate and use them in your projects.

## References

* [Hugging Face Translation Models](https://huggingface.co/models?pipeline_tag=translation&sort=downloads)

* [Hugging Face Translation Task](https://huggingface.co/tasks/translation)

* [M2M100 1.2B Model Details](https://huggingface.co/facebook/m2m100_1.2B)
