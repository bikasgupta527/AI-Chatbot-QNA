from flask import Flask, request, json, jsonify,make_response,render_template
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from flask_socketio import SocketIO, emit
import speech_recognition as sr
import time
import pandas as pd
import torch


tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

with open('cwc-2023.txt',encoding='utf8') as file:
  book_text = file.read()
  
def qna_ans(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    outputs = model(**inputs)

    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_confidence = torch.nn.functional.softmax(start_logits, dim=1)[0][start_idx].item()
    end_confidence = torch.nn.functional.softmax(end_logits, dim=1)[0][end_idx].item()

    confidence_threshold = 0.5

    if start_confidence > confidence_threshold or end_confidence > confidence_threshold:
        answer_tokens = inputs["input_ids"][0][start_idx: end_idx + 1]
        answer = tokenizer.decode(answer_tokens)
    else:
        answer = "Apology.. I don't have an answer for that, try asking a question related cricket world cup 2023"

    return answer, start_confidence, end_confidence


def process_long_text(question, text=book_text, max_chunk_len=512):
    chunks = [text[i:i + max_chunk_len] for i in range(0, len(text), max_chunk_len)]
    answers = []
    S,E = [],[]

    for chunk in chunks:
        answer,s,e = qna_ans(question, chunk)
        answers.append(answer)
        S.append(s)
        E.append(e)

    return answers,S,E


app = Flask(__name__)
socketio = SocketIO(app)


@socketio.on('voice_search')
def handle_voice_search():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Say something...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        try:
            audio = recognizer.listen(source, timeout=1)  # Set the timeout here
            try:
                text = recognizer.recognize_google(audio)
                words = text.split()
                
                for word in words:
                    time.sleep(0.5)
                    emit('voice_search_response', word)
                emit('voice_search_response', {'end_of_conversation': True})
            except sr.UnknownValueError:
                print("Could not understand audio")
                emit('voice_search_response', {'end_of_conversation2': True})
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                emit('voice_search_response', {'end_of_conversation2': True})
        except sr.WaitTimeoutError:
            print("Speech recognition timed out.")
            emit('voice_search_response', {'end_of_conversation2': True})

    
    # for i in range(5):
    #     result = "Hello from Python!" 
    #     emit('voice_search_response', result)
    #     time.sleep(1) 
    
@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/qna',methods=['POST'])
def qna():
    data = request.get_json()
    
    question = data['question']
    ans,start_confidence,end_confidence  = process_long_text(question)
    
    output = pd.DataFrame({
        "Answer":ans,
        "Confidence-Start": start_confidence,
        "Confidence-End": end_confidence,
    })
    output.sort_values(by=['Confidence-Start','Confidence-End'],ascending=False,inplace=True)
    
    if output.shape[0] > 0:
        return jsonify({'message':output.iloc[0]['Answer']})
    else:
        return jsonify({'message': "Apology.. I don't have an answer for that, try asking a question related cricket world cup 2023"})



if __name__ == "__main__":
    app.run(port=5000,debug=True)
    