import nltk
import pandas as pd
import regex as re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
nltk.download("stopwords")


def load_data(input_path):
    return pd.read_csv(input_path)


def save_data(df, output_path):
    df.to_csv(output_path , index=False)
    
# no need to handle contractions as stopword handles them this is just for eda
contractions = [
    "i’m",
    "you’re",
    "he’s",
    "she’s",
    "it’s",
    "we’re",
    "they’re",
    "i’ve",
    "you’ve",
    "we’ve",
    "they’ve",
    "i’d",
    "you’d",
    "he’d",
    "she’d",
    "we’d",
    "they’d",
    "i’ll",
    "you’ll",
    "he’ll",
    "she’ll",
    "we’ll",
    "they’ll",
    "isn’t",
    "aren’t",
    "wasn’t",
    "weren’t",
    "haven’t",
    "hasn’t",
    "hadn’t",
    "don’t",
    "doesn’t",
    "didn’t",
    "can’t",
    "couldn’t",
    "won’t",
    "wouldn’t",
    "shouldn’t",
    "mustn’t",
    "what’s",
    "where’s",
    "when’s",
    "who’s",
    "how’s",
    "that’s",
    "there’s",
    "y’all"
]

positive_words = [
    "ambitious", "brave", "compassionate", "dazzling", "empowering", "flourishing",
    "generous", "hopeful", "innovative", "joyful", "kindhearted", "luminous",
    "motivated", "nurturing", "optimistic", "passionate", "resilient", "serene",
    "thriving", "unwavering", "vibrant", "warmhearted", "exuberant", "youthful",
    "affable", "bountiful", "charismatic", "delightful", "eloquent", "faithful",
    "gracious",
    # Common YT positives
    "awesome", "amazing", "love", "liked", "favorite", "best", "wonderful", "great",
    "impressive", "epic", "brilliant", "fantastic", "beautiful", "excellent",
    "masterful", "helpful", "informative", "inspiring", "thank you", "thanks",
    "legend", "wholesome", "valid", "based", "respect"
]

negative_words = [
    "abrasive", "bleak", "clumsy", "dismal", "evasive", "frivolous", "gruesome",
    "harsh", "ignorant", "jaded", "lethargic", "malicious", "nefarious", "obnoxious",
    "pernicious", "querulous", "repugnant", "sinister", "toxic", "unsettling", "vile",
    "warped", "xenophobic", "accusatory", "belligerent", "coercive", "deceptive",
    "expendable", "flippant", "glib",
    # Common YT negatives
    "bad", "hate", "worst", "boring", "trash", "awful", "fail", "useless", "sucks",
    "cringe", "disappointing", "scam", "fake", "terrible", "mistake", "weak",
    "stupid", "annoying", "dumb", "flawed", "spam", "clickbait", "bot", "repetitive",
    "waste", "mid", "copium", "lame"
]

neutral_words = [
    # Mostly descriptive / factual / context-dependent
    "antique", "brisk", "candid", "durable", "eclectic", "formal", "generic",
    "hybrid", "implicit", "juxtaposed", "kinetic", "literal", "methodical",
    "objective", "pragmatic", "resolute", "sparse", "technical", "uniform",
    "analytical", "benchmark", "comprehensive", "debatable", "efficient",
    "fragmented", "groundbreaking", "average", "decent", "typical", "normal",
    "fair", "sure", "possible", "depends", "middle", "moderate", "factual",
    # Context-sensitive words (can be + or - depending on tone)
    "interesting", "okay", "ok", "fine", "cool", "alright", "neutral", "so-so",
    "maybe", "sometimes"
]

neutral_wrod_list=[ps.stem(word) for word in neutral_words]
positive_wrod_list=[ps.stem(word) for word in positive_words]
negative_wrod_list=[ps.stem(word) for word in negative_words]
def stemming(txt):
    new_txt = []
    for word in txt.split():
        if word not in stpWrd:
            new_txt.append(ps.stem(word))
    return " ".join(new_txt)




def countNegative(lst):
    count=0
    for i in lst.split():
        if i in negative_wrod_list:
            count+=1
    return count
    

def countPositive(lst):
    count=0
    for i in lst.split():
        if i in positive_wrod_list:
            count+=1
    return count


def countNeutral(lst):
    count=0
    for i in lst.split():
        if i in neutral_wrod_list:
            count+=1
    return count


def apos_count(txt):
    count=0
    for i in txt.split():
        if i in contractions:  
            count+=1
    return count

stpWrd = stopwords.words("english")
def remove_stopword(txt):
    new_txt=[]
    for word in txt.split():
        if word not in stpWrd:
            new_txt.append(word)
    return " ".join(new_txt)

def count_stopword(txt):
    count=0
    for word in txt.split():
        if word in stpWrd:
            count+=1
    return count


def preprocess_data(df,comment_len=True,word_count=True,char_per_words=True):
    df.fillna("")
    df['Comment']=df['Comment'].str.lower().str.strip()
    
    # punctuation mark removal
    df["Comment"]=df["Comment"].apply(lambda x:re.sub(r"[^\w\s]","",x))
    
    df.fillna("")
    # length of comment
    if comment_len:
        df['comment_len']=df['Comment'].apply(lambda x:len(x))

    if word_count:
        df['word_count']=df['Comment'].apply(lambda x:len(x.split()))
        
    if comment_len and word_count and char_per_words:
        df['char_per_words']=df['comment_len']/df['word_count']

    # apos count
    df['apos_count']=df['Comment'].apply(apos_count)

    # stopword removal
    # it also removes words like not which can be important
    df['stopword_count']=df['Comment'].apply(count_stopword)
    df['Comment']=df['Comment'].apply(remove_stopword)

    # stemming
    df['Comment']=df['Comment'].apply(stemming)
    df["PositiveWordCount"]=df["Comment"].apply(countPositive)
    df["NegativeWordCount"]=df["Comment"].apply(countNegative)
    df["NeutralWordCount"]=df["Comment"].apply(countNeutral)
    # df.dropna(inplace=True)
    
    return df