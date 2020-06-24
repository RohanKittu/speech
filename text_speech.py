#libraries for convert text to sppech
from gtts import gTTS
import os

import vlc

#chatter bot libraries
from chatterbot import ChatBot
import spacy
from chatterbot.trainers import ChatterBotCorpusTrainer
from pygame import mixer


#chatbot initializer
bot = ChatBot("Chatterbot",storage_adapter="chatterbot.storage.SQLStorageAdapter")
trainer = ChatterBotCorpusTrainer(bot)
trainer.train("chatterbot.corpus.english")
 
#taking the input from the user.
st = input("\ntalk to bot :- ")
#bot response
bot_response=bot.get_response(st)
output = gTTS(text = str(bot_response.text),lang='en',slow=False)
output.save("output.mp3")
os.system("start output.mp3")
#mixer.init()
#mixer.music.load('/Users/rohankittu/Desktop/speech/output.mp3')
#mixer.music.play()

p = vlc.MediaPlayer("/Users/rohankittu/Desktop/speech/output.mp3")
p.play()


'''
class Generic_reply():
    messaage = " "
    def __init__(self,text):
        self.message = text
    def funct(self):
        print("displaying through function 1 :- ",self.message)
    def func(self):
        print("displaying through function 2 :- ",self.message)

st = input("\nenter the string :- ")
x = Generic_reply(st)
x.func()
x.funct()
'''