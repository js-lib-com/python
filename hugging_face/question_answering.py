import sys
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM, GenerationConfig

model_name = "google/flan-t5-xl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

config = GenerationConfig(max_new_tokens=200)
for line in sys.stdin:
    if line.startswith("quit"):
        print("bye bye")
        break
    tokens = tokenizer(line, return_tensors="tf")
    outputs = model.generate(**tokens, generation_config=config)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
