import openai
import pygame

openai.api_key = ""

# text = input("Enter text to generate speech: ")
text = 'my name is julien. this is a simple text to speech test.'

# response = openai.Completion.create(
#     engine="text-davinci-002",
#     prompt=f"Convert the following text to speech: {text}",
#     temperature=0.5,
#     max_tokens=100,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0
# )

response = openai.Completion.create(
    engine="davinci",
    prompt=f"Speak: {text}",
    max_tokens=1024,
    temperature=0.5,
)

speech = response.choices[0].text
speech = speech.strip()

pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(speech)
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    continue
