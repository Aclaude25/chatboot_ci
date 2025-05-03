import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st

# Téléchargement des ressources NLTK (une seule fois au démarrage)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download('averaged_perceptron_tagger')


@st.cache_data  # Mise en cache pour améliorer les performances
def load_and_preprocess_data(filepath):
    """Charge, prétraite et retourne le corpus."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = f.read().replace('\n', ' ')

    sentences = sent_tokenize(data, language='french')

    def preprocess(sentence):
        words = word_tokenize(sentence, language='french')  # Spécifiez la langue
        words = [word.lower() for word in words if
                 word.lower() not in stopwords.words('french') and word not in string.punctuation]
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        return words

    corpus = [preprocess(sentence) for sentence in sentences]
    return corpus, sentences  # Retourne également les phrases originales


# Charger et prétraiter les données
corpus, original_sentences = load_and_preprocess_data('Quatre_grands_groupes_ethniques_ci.txt')


@st.cache_data  # Mise en cache pour améliorer les performances
def get_most_relevant_sentence(query, corpus, original_sentences):
    """Trouve la phrase la plus pertinente dans le corpus."""
    def preprocess(sentence):  # Définition locale pour éviter de la dupliquer globalement
        words = word_tokenize(sentence, language='french')
        words = [word.lower() for word in words if
                 word.lower() not in stopwords.words('french') and word not in string.punctuation]
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        return words

    query = preprocess(query)
    max_similarity = 0
    most_relevant_index = -1  # Stocke l'index de la phrase la plus pertinente

    for i, sentence in enumerate(corpus):
        similarity = len(set(query).intersection(sentence)) / float(len(set(query).union(sentence)))
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_index = i

    if most_relevant_index != -1:
        return original_sentences[most_relevant_index]  # Retourne la phrase originale
    else:
        return "Je n'ai pas trouvé d'informations pertinentes pour cette question."


def chatbot(question, corpus, original_sentences):
    """Fonction principale du chatbot."""
    most_relevant_sentence = get_most_relevant_sentence(question, corpus, original_sentences)
    return most_relevant_sentence


# Interface Streamlit
st.title("Chatbot sur l'Histoire de la Côte d'Ivoire")
st.write("Bonjour ! Je suis un chatbot. Posez-moi des questions sur l'histoire de la Côte d'Ivoire.")

# Zone de saisie de la question
question = st.text_input("Votre question :")

# Bouton de soumission
if st.button("Envoyer"):
    # Appel du chatbot et affichage de la réponse
    with st.spinner("Recherche en cours..."):  # Indique que la recherche est en cours
        response = chatbot(question, corpus, original_sentences)
    st.write("Réponse :", response)  # Affichage de la réponse dans une phrase complète