import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
import csv
import json
import itertools
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import joblib
from flask import Flask, render_template, request, session, redirect, url_for, render_template_string 
from flask_ngrok import run_with_ngrok


app = Flask(__name__)
run_with_ngrok(app)
import random
import string
S = 10
ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k=S))
app.secret_key = str(ran)
#app.secret_key = "secret key"
nlp = spacy.load('en_core_web_sm')

data = {"users": []}
with open('history.json', 'w') as outfile:
    json.dump(data, outfile)


def write_json(new_data, filename='history.json'):
    with open(filename, 'r+') as file:
        file_data = json.load(file)

        file_data["users"].append(new_data)

        file.seek(0)

        json.dump(file_data, file, indent=4)


df_tr = pd.read_csv('Training.csv')
df_tt = pd.read_csv('Testing.csv')
videos = pd.read_csv('videos.csv')

symp = []
disease = []
for i in range(len(df_tr)):
    symp.append(df_tr.columns[df_tr.iloc[i] == 1].to_list())
    disease.append(df_tr.iloc[i, -1])


all_symp_col = list(df_tr.columns[:-1])


def clean_symp(sym):
    return sym.replace('_', ' ').replace('.1', '').replace('(typhos)', '').replace('yellowish', 'yellow').replace(
        'yellowing', 'yellow')


all_symp = [clean_symp(sym) for sym in (all_symp_col)]


def preprocess(doc):
    nlp_doc = nlp(doc)
    d = []
    for token in nlp_doc:
        if (not token.text.lower() in STOP_WORDS and token.text.isalpha()):
            d.append(token.lemma_.lower())
    return ' '.join(d)


all_symp_pr = [preprocess(sym) for sym in all_symp]

col_dict = dict(zip(all_symp_pr, all_symp_col))

def powerset(seq):
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]] + item
            yield item

def sort(a):
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if len(a[j]) > len(a[i]):
                a[i], a[j] = a[j], a[i]
    a.pop()
    return a

def permutations(s):
    permutations = list(itertools.permutations(s))
    return ([' '.join(permutation) for permutation in permutations])

def DoesExist(txt):
    txt = txt.split(' ')
    combinations = [x for x in powerset(txt)]
    sort(combinations)
    for comb in combinations:
        for sym in permutations(comb):
            if sym in all_symp_pr:
                return sym
    return False

def jaccard_set(str1, str2):
    list1 = str1.split(' ')
    list2 = str2.split(' ')
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def syntactic_similarity(symp_t, corpus):
    most_sim = []
    poss_sym = []
    for symp in corpus:
        d = jaccard_set(symp_t, symp)
        most_sim.append(d)
    order = np.argsort(most_sim)[::-1].tolist()
    for i in order:
        if DoesExist(symp_t):
            return 1, [corpus[i]]
        if corpus[i] not in poss_sym and most_sim[i] != 0:
            poss_sym.append(corpus[i])
    if len(poss_sym):
        return 1, poss_sym
    else:
        return 0, None

def check_pattern(inp, dis_list):
    import re
    pred_list = []
    ptr = 0
    patt = "^" + inp + "$"
    regexp = re.compile(inp)
    for item in dis_list:
        if regexp.search(item):
            pred_list.append(item)
    if (len(pred_list) > 0):
        return 1, pred_list
    else:
        return ptr, None

from nltk.wsd import lesk
from nltk.tokenize import word_tokenize


def WSD(word, context):
    sens = lesk(context, word)
    return sens

def semanticD(doc1, doc2):
    doc1_p = preprocess(doc1).split(' ')
    doc2_p = preprocess(doc2).split(' ')
    score = 0
    for tock1 in doc1_p:
        for tock2 in doc2_p:
            syn1 = WSD(tock1, doc1)
            syn2 = WSD(tock2, doc2)
            if syn1 is not None and syn2 is not None:
                x = syn1.wup_similarity(syn2)
                if x is not None and x > 0.25:
                    score += x
    return score / (len(doc1_p) * len(doc2_p))


def semantic_similarity(symp_t, corpus):
    max_sim = 0
    most_sim = None
    for symp in corpus:
        d = semanticD(symp_t, symp)
        if d > max_sim:
            most_sim = symp
            max_sim = d
    return max_sim, most_sim

def suggest_syn(sym):
    symp = []
    synonyms = wordnet.synsets(sym)
    lemmas = [word.lemma_names() for word in synonyms]
    lemmas = list(set(itertools.chain(*lemmas)))
    for e in lemmas:
        res, sym1 = semantic_similarity(e, all_symp_pr)
        if res != 0:
            symp.append(sym1)
    return list(set(symp))

def OHV(cl_sym, all_sym):
    l = np.zeros([1, len(all_sym)])
    for sym in cl_sym:
        l[0, all_sym.index(sym)] = 1
    return pd.DataFrame(l, columns=all_symp)


def contains(small, big):
    a = True
    for i in small:
        if i not in big:
            a = False
    return a

def possible_diseases(l):
    poss_dis = []
    for dis in set(disease):
        if contains(l, symVONdisease(df_tr, dis)):
            poss_dis.append(dis)
    return poss_dis

def symVONdisease(df, disease):
    ddf = df[df.prognosis == disease]
    m2 = (ddf == 1).any()
    return m2.index[m2].tolist()

knn_clf = joblib.load('knn.pkl')

severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()


def getDescription():
    global description_list
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open('symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)

getSeverityDict()
getprecautionDict()
getDescription()

def calc_condition(exp, days):
    sum = 0
    for item in exp:
        if item in severityDictionary.keys():
            sum = sum + severityDictionary[item]
    if ((sum * days) / (len(exp)) > 13):
        return 1
        print("You should take the consultation from doctor. ")
    else:
        return 0
        print("It might not be that bad but you should take precautions.")

def related_sym(psym1):
    s = "Please be more specific, <br>"
    i = len(s)
    for num, it in enumerate(psym1):
        s += str(num) + ") " + clean_symp(it) + "<br>"
    if num != 0:
        s += "Select the one you meant."
        return s
    else:
        return 0

def select_symptom(symptom_list, target):
    for symptom in symptom_list:
        if symptom == target:
            return symptom  # Return the selected symptom
    return None


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/get")
def get_bot_response():
    s = request.args.get('msg')
    if "step" in session:
        if session["step"] == "Q_C":
            name = session["name"]
            age = session["age"]
            gender = session["gender"]
            session.clear()
            if s.lower() == "exit":
                "Thank you for using our web site " + name
            else:
                session["step"] = "FS"
                session["name"] = name
                session["age"] = age
                session["gender"] = gender
    if any(word in s.lower().split() for word in ["start"]):
        return "What is your name ?"
    if 'name' not in session and 'step' not in session:
        session['name'] = s
        session['step'] = "age"
        return "How old are you? "
    if session["step"] == "age":
        session["age"] = s
        session["step"] = "gender"
        return "Can you specify your gender ?"
    if session["step"] == "gender":
        session["gender"] = s
        session["step"] = "Depart"
    if session['step'] == "Depart":
        session['step'] = "BFS"
        return "Well, Hello again " + session[
            "name"] + ", now I will be asking some few questions about your symptoms to see what you should do. Just use the word begin?"
    if session['step'] == "BFS":
        session['step'] = "FS"
        return "Can you precise your main symptom " + session["name"] + " ?"
    if session['step'] == "FS":
        sym1 = s
        sym1 = preprocess(sym1)
        sim1, psym1 = syntactic_similarity(sym1, all_symp_pr)
        temp = [sym1, sim1, psym1]
        session['FSY'] = temp
        session['step'] = "SS"
        if sim1 == 1:
            session['step'] = "RS1"
            s = related_sym(psym1)
            if s != 0:
                return s
        else:
            return "You are probably facing another symptom, if so, can you specify it?"
    if session['step'] == "RS1":
        temp = session['FSY']
        psym1 = temp[2]
        psym1 = psym1[int(s)]
        temp[2] = psym1
        session['FSY'] = temp
        session['step'] = 'SS'
        return "You are probably facing another symptom, if so, can you specify it?"
    if session['step'] == "SS":
        sym2 = s
        sym2 = preprocess(sym2)
        sim2 = 0
        psym2 = []
        if len(sym2) != 0:
            sim2, psym2 = syntactic_similarity(sym2, all_symp_pr)
        temp = [sym2, sim2, psym2]
        session['SSY'] = temp
        session['step'] = "semantic"
        if sim2 == 1:
            session['step'] = "RS2"
            s = related_sym(psym2)
            if s != 0:
                return s
    if session['step'] == "RS2":
        temp = session['SSY']
        psym2 = temp[2]
        psym2 = psym2[int(s)]
        temp[2] = psym2
        session['SSY'] = temp
        session['step'] = "semantic"
    if session['step'] == "semantic":
        temp = session["FSY"]
        sym1 = temp[0]
        sim1 = temp[1]
        temp = session["SSY"]
        sym2 = temp[0]
        sim2 = temp[1]
        if sim1 == 0 or sim2 == 0:
            session['step'] = "BFsim1=0"
        else:
            session['step'] = 'PD'
    if session['step'] == "BFsim1=0":
        if sim1 == 0 and len(sym1) != 0:
            sim1, psym1 = semantic_similarity(sym1, all_symp_pr)
            temp = []
            temp.append(sym1)
            temp.append(sim1)
            temp.append(psym1)
            session['FSY'] = temp
            session['step'] = "sim1=0"
        else:
            session['step'] = "BFsim2=0"
    if session['step'] == "sim1=0":
        temp = session["FSY"]
        sym1 = temp[0]
        sim1 = temp[1]
        if sim1 == 0:
            if "suggested" in session:
                sugg = session["suggested"]
                if any(word in s.lower().split() for word in ["yes", "yeah", 'yea']):
                    psym1 = sugg[0]
                    sim1 = 1
                    temp = session["FSY"]
                    temp[1] = sim1
                    temp[2] = psym1
                    session["FSY"] = temp
                    sugg = []
                else:
                    del sugg[0]
            if "suggested" not in session:
                session["suggested"] = suggest_syn(sym1)
                sugg = session["suggested"]
            if len(sugg) > 0:
                msg = "are you experiencing any  " + sugg[0] + "?"
                return msg
        if "suggested" in session:
            del session["suggested"]
        session['step'] = "BFsim2=0"
    if session['step'] == "BFsim2=0":
        temp = session["SSY"]
        sym2 = temp[0]
        sim2 = temp[1]
        if sim2 == 0 and len(sym2) != 0:
            sim2, psym2 = semantic_similarity(sym2, all_symp_pr)
            temp = []
            temp.append(sym2)
            temp.append(sim2)
            temp.append(psym2)
            session['SSY'] = temp
            session['step'] = "sim2=0"
        else:
            session['step'] = "TEST"
    if session['step'] == "sim2=0":
        temp = session["SSY"]
        sym2 = temp[0]
        sim2 = temp[1]
        if sim2 == 0:
            if "suggested_2" in session:
                sugg = session["suggested_2"]
                if any(word in s.lower().split() for word in ["yes", "yeah", 'yea']):
                    psym2 = sugg[0]
                    sim2 = 1
                    temp = session["SSY"]
                    temp[1] = sim2
                    temp[2] = psym2
                    session["SSY"] = temp
                    sugg = []
                else:
                    del sugg[0]
            if "suggested_2" not in session:
                session["suggested_2"] = suggest_syn(sym2)
                sugg = session["suggested_2"]
            if len(sugg) > 0:
                msg = "Are you experiencing " + sugg[0] + "?"
                session["suggested_2"] = sugg
                return msg
        if "suggested_2" in session:
            del session["suggested_2"]
        session['step'] = "TEST"
    if session['step'] == "TEST":
        temp = session["FSY"]
        sim1 = temp[1]
        psym1 = temp[2]
        temp = session["SSY"]
        sim2 = temp[1]
        psym2 = temp[2]
        if sim1 == 0 and sim2 == 0:
            result = None
            session['step'] = "END"
        else:
            if sim1 == 0:
                psym1 = psym2
                temp = session["FSY"]
                temp[2] = psym2
                session["FSY"] = temp
            if sim2 == 0:
                psym2 = psym1
                temp = session["SSY"]
                temp[2] = psym1
                session["SSY"] = temp
            session['step'] = 'PD'
    if session['step'] == 'PD':
        temp = session["FSY"]
        sim1 = temp[1]
        psym1 = temp[2]
        temp = session["SSY"]
        sim2 = temp[1]
        psym2 = temp[2]
        print("hey2")
        if "all" not in session:
            session["asked"] = []
            session["all"] = [col_dict[psym1], col_dict[psym2]]
            print(session["all"])
        session["diseases"] = possible_diseases(session["all"])
        print(session["diseases"])
        all_sym = session["all"]
        diseases = session["diseases"]
        dis = diseases[0]
        session["dis"] = dis
        session['step'] = "for_dis"
    if session['step'] == "DIS":
        if "symv" in session:
            if len(s) > 0 and len(session["symv"]) > 0:
                symts = session["symv"]
                all_sym = session["all"]
                if any(word in s.lower().split() for word in ["yes", "yeah", 'yea']):
                    all_sym.append(symts[0])
                    session["all"] = all_sym
                    print(possible_diseases(session["all"]))
                del symts[0]
                session["symv"] = symts
        if "symv" not in session:
            session["symv"] = symVONdisease(df_tr, session["dis"])
        if len(session["symv"]) > 0:
            if symts[0] not in session["all"] and symts[0] not in session["asked"]:
                asked = session["asked"]
                asked.append(symts[0])
                session["asked"] = asked
                symts = session["symv"]
                msg = "Are you experiencing " + clean_symp(symts[0]) + "?"
                return msg
            else:
                del symts[0]
                session["symv"] = symts
                s = ""
                print("HANAAA")
                return get_bot_response()
        else:
            PD = possible_diseases(session["all"])
            diseases = session["diseases"]
            if diseases[0] in PD:
                session["testpred"] = diseases[0]
                PD.remove(diseases[0])
            session["diseases"] = PD
            session['step'] = "for_dis"
    if session['step'] == "for_dis":
        diseases = session["diseases"]
        if len(diseases) <= 0:
            session['step'] = 'PREDICT'
        else:
            session["dis"] = diseases[0]
            session['step'] = "DIS"
            session["symv"] = symVONdisease(df_tr, session["dis"])
            return get_bot_response()
    if session['step'] == "PREDICT":
        result = knn_clf.predict(OHV(session["all"], all_symp_col))
        session['step'] = "END"
    if session['step'] == "END":
        if result is not None:
            if result[0] != session["testpred"]:
                session['step'] = "Q_C"
                return "As you provide me with few symptoms, I regret to inform you that I cannot predict your \
                condition at the moment. Could you please provide more details about what you're experiencing, \
                and when you're ready to end the conversation, just use the word exit."
            session['step'] = "Description"
            session["disease"] = result[0]
            return "Well " + session["name"] + ", you may have " + result[
                0] + ". Do you want to get a description of the disease?"
        else:
            session['step'] = "Q_C"
            return "Can you specify more what you feel or when you're ready to end the conversation, just use the word exit."
    if session['step'] == "Description":
        y = {"Name": session["name"], "Age": session["age"], "Gender": session["gender"], "Disease": session["disease"],
             "Sympts": session["all"]}
        write_json(y)
        session['step'] = "Severity"
        if session["disease"] in description_list.keys():
            return description_list[session["disease"]] + " \n <br>  How many days have you had symptoms?"
        else:
            if " " in session["disease"]:
                session["disease"] = session["disease"].replace(" ", "_")
            return "Pplease visit <a href='" + "https://en.wikipedia.org/wiki/" + session["disease"] + "'>  here  </a>"
    if session['step'] == "Severity":
        session['step'] = 'FINAL'
        if calc_condition(session["all"], int(s)) == 1:
            return "You should take the consultation from doctor <br> To end the conversation, just use the word exit.<br>To watch the first aid demonstration use the word video!"
        else:
            msg = 'Nothing to worry about, but you should take the following precautions :<br> '
            i = 1
            for e in precautionDictionary[session["disease"]]:
                msg += '\n ' + str(i) + ' - ' + e + '<br>'
                i += 1
            msg += ' To end the conversation, just use the word exit.<br>To watch the first aid demonstration use the word video!'
            return msg

    if session['step'] == "FINAL":
        session['step'] = "BYE"
        if any(word in s.lower().split() for word in ["video", "videos"]):
          result_disease = session["disease"]
          firstaid_video = videos.loc[videos['prognosis'] == result_disease, 'videos'].values
          firstaid_video = str(firstaid_video[0])
          html_template = """
                  <iframe src="{{ firstaid_video }}" width="400" height="250" frameborder="0" allowfullscreen></iframe>
                  <p>Your diagnosis was perfectly completed. Do you need another medical consultation (yes or no)?</p>
                  """
          rendered_html = render_template_string(html_template, firstaid_video=firstaid_video)
          return rendered_html
    if session['step'] == "BYE":
        name = session["name"]
        age = session["age"]
        gender = session["gender"]
        session.clear()
        if any(word in s.lower().split() for word in ["yes", "yeah", 'yea']):
            session["gender"] = gender
            session["name"] = name
            session["age"] = age
            session['step'] = "FS"
            return "HELLO again " + session["name"] + " Please tell me your main symptom. "
        else:
            return "THANKS " + name + " for using me for more information please contact <b> 0500-78601</b>"

'''app.route('/play_youtube_video')
def play_youtube_video():
  if 'disease' in session:
    result_disease = session["disease"]
    firstaid_video = videos.loc[videos['prognosis'] == result_disease, 'videos'].values
    return str(firstaid_video[0])'''




if __name__ == "__main__":
    app.run()
