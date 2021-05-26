# Chatbot-Project

In this project, I used Reddit comments from different scientific subreddits to make a generative chatbot by using the encoder-decoder (seq2seq) model. 
A generative chatbot will generate a response based on the data we use to train the model. 

## Gathering  the data

The first step is to scrape the comments from Reddit from the desired subreddits. 
I used the Reddit pushshift API to get the data. A sleep time of 1 second between each iteration was required to avoid the HTTP 429 Too Many Requests error. 
The collected data was in JSON format so Pandas was used to make a data frame that I  could work with.

##  Gathering  the data and matching the comments


The second step of this project was to clean the data and to match the comments with the parent comments (Answers to the comment and the comment). 

We need to clean the data to avoid having unnecessary words or characters that would make the training of the model harder. Then I matched the comments with the answer to those comments. If one comment had more than one answer I chose the one with the most scores. The highest score comments should be the higher quality comments.

## seq2seq model 
The seq2seq model uses Long short Term Memory (LSTM) artificial recurrent neural network (RNN) for text generation. 

## deployment 
To deploy the project I used the Flask API.

## Result

The chatbot works, but it needs improvement. One of the reasons why I think the bot does not perform well could be because I did not use lots of comments/response pairs. A project like this normally needs a lot of data to have a good result. Also, Reddit comments are not always conversations, so a response could be completely unrelated to the original comment. 

![img](https://github.com/KevinSeyranoglu/Chatbot-Project/blob/main/chatbotEx.JPG)




## Bonus Data cleaning
Because those comments are from people from the internet. some comments contain swears. So I had to censor the comments. So If you want to remove the sensor, just set censor=False in the chatbot_.py 

## Try it yourself! 

you will only need glove.6B.50d.txt and make sure to put that file in the right Folder (or change the path in the codes).

You will need to download glove.6B 
at  http://nlp.stanford.edu/data/glove.6B.zip 

or it can be done from the command lines with 

wget  http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.zip
rm glove.zip

run app.py and click on the link
