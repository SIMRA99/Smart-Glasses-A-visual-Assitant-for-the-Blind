# importing library
import requests
from bs4 import BeautifulSoup

import pyttsx3
from datetime import date
from datetime import datetime

# enter city name
city = "Dubai"
LANGUAGE = "en-US,en;q=0.5"

# creating url and requests instance
url = "https://www.google.com/search?q="+"weather"+city
html = requests.get(url).content

# getting raw data
soup = BeautifulSoup(html, 'html.parser')
temp = soup.find('div', attrs={'class': 'BNeawe iBp4i AP7Wnd'}).text
str = soup.find('div', attrs={'class': 'BNeawe tAd8D AP7Wnd'}).text

# formatting data
data = str.split('\n')
time = data[0]
sky = data[1]

# getting all div tag
listdiv = soup.findAll('div', attrs={'class': 'BNeawe s3v9rd AP7Wnd'})
strd = listdiv[5].text

# getting other required data
pos = strd.find('Wind')
other_data = strd[pos:]

# printing all data
print("Temperature is", temp)
print("Time: ", time)
print("Sky Description: ", sky)
print(other_data)

today = date.today()
print("Today's date:", today)
d2 = today.strftime("%B %d, %Y")
print("d2 =", d2)
now = datetime.now()
dt_string = now.strftime("%H  %M o clock ")
 
print("now =", dt_string)

engine = pyttsx3.init() 

# an audio will be played which speaks the test if pyttsx3 recognizes it 
engine.say("the temperature is "+temp + "degree")
engine.say("Today's date:" + d2)
engine.say("The time is :" + dt_string)


engine.runAndWait() 
