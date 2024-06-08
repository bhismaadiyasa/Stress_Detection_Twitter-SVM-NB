from numpy.core.numeric import True_
from sklearn import metrics
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import altair as alt
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS

def main():
    st.title("Twitter Sentiment Analysis with SVM & Naive Bayes")
    st.sidebar.title("- Content to Show -")

if __name__ == '__main__':
    main()


@st.cache_data(persist= True)
def load_dataset():
    url = "https://raw.githubusercontent.com/bhismaadiyasa/Stress_Detection_Twitter-SVM-NB/main/tweet_emotions_preprocessed.csv"
    data = pd.read_csv(url)
    return data
df = load_dataset()

def load_eval():
    url = "https://raw.githubusercontent.com/bhismaadiyasa/Stress_Detection_Twitter-SVM-NB/main/df_results_svm_nb.csv"
    data = pd.read_csv(url)
    return data
df_eval = load_eval()

option = st.sidebar.radio("Choose action", ("Display Data + EDA", "Classification Result"))

if option == "Display Data + EDA":
    st.subheader("Show Twitter dataset")
    st.write(df)

    # Bar Chart untuk visualisasi Distribusi Jumlah Sentimen
    st.subheader("Distribusi Jumlah Sentimen")
    sentiment_counts = df['sentiment'].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
    plt.title('Distribusi Jumlah Sentimen')
    plt.xlabel('Sentimen')
    plt.ylabel('Jumlah')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    st.pyplot(plt)
    st.write("Dari data yang disediakan, terlihat bahwa distribusi sentimen dalam dataset tidak seimbang. Mayoritas tweet terkumpul dalam kategori 'neutral' dan 'worry', sementara frekuensi tweet dalam kategori lainnya sangat bervariasi. Sentimen positif seperti 'happiness' dan 'love' memiliki frekuensi yang lebih rendah, sementara beberapa sentimen negatif seperti 'anger' dan 'boredom' bahkan memiliki frekuensi yang sangat minim. Ketidakseimbangan ini dapat memengaruhi performa model klasifikasi atau analisis sentimen, di mana kelas dengan frekuensi tinggi cenderung memiliki dampak lebih besar. Untuk memperbaiki ini, mungkin diperlukan penanganan khusus seperti oversampling atau undersampling untuk menyeimbangkan dataset, sehingga analisis lebih akurat dan representatif.")

    # Menambahkan kolom fitur
    df['Polarity'] = df['content'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['word_count'] = df['content'].apply(lambda x: len(str(x).split()))
    df['review_len'] = df['content'].apply(lambda x: len(str(x)))

    # Plot distribusi Polarity
    st.subheader("Distribusi Polarity")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.histplot(df['Polarity'], bins=50, color='#9966ff', kde=True, ax=ax1)
    sns.kdeplot(df['Polarity'], color='#9966ff', linestyle='--', ax=ax1)
    ax1.set_title('Polarity Distribution', size=15)
    ax1.set_xlabel('Polarity')
    ax1.set_ylabel('Density')
    st.pyplot(fig1)
    st.write("Dari hasil analisis polaritas, mayoritas nilai terkonsentrasi di sekitar 0, yang menunjukkan bahwa mayoritas data memiliki polaritas netral. Rentang nilai polaritas dari -1 hingga 1 menunjukkan sejauh mana sentimen dalam data cenderung positif atau negatif, di mana nilai positif menunjukkan sentimen positif dan nilai negatif menunjukkan sentimen negatif. Dengan mayoritas nilai polaritas mendekati 0, hal ini menunjukkan bahwa data cenderung netral secara emosional, dengan sedikit kecenderungan positif atau negatif. Hal ini bisa mengindikasikan bahwa dalam kumpulan data tersebut, tweet-tweet yang dianalisis memiliki tingkat kesetimbangan antara sentimen positif dan negatif.")

    # Plot distribusi Review Length
    st.subheader("Distribusi Review Length")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.histplot(df['review_len'], bins=50, color='#3399ff', kde=True, ax=ax2)
    sns.kdeplot(df['review_len'], color='#3399ff', linestyle='--', ax=ax2)
    ax2.set_title('Review Length Distribution', size=15)
    ax2.set_xlabel('Review Length')
    ax2.set_ylabel('Density')
    st.pyplot(fig2)
    st.write("sebagian besar panjang tweet berkisar antara 0 hingga sekitar 175 karakter. Nilai terbanyak terlihat berada di sekitar 140 karakter, menunjukkan bahwa mayoritas tweet memiliki panjang yang relatif pendek. Hal ini mungkin mengindikasikan bahwa pengguna cenderung menyampaikan pendapat atau informasi dalam format yang singkat dan padat. Panjang tweet yang cenderung pendek ini juga dapat mempermudah pembacaan dan pemahaman, serta memungkinkan untuk ditampilkan secara penuh dalam platform media sosial yang memiliki batasan karakter tertentu.")

    # Plot distribusi Word Count
    st.subheader("Distribusi Word Count")
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.histplot(df['word_count'], bins=50, color='#00ff00', kde=True, ax=ax3)
    sns.kdeplot(df['word_count'], color='#00ff00', linestyle='--', ax=ax3)
    ax3.set_title('Word Count Distribution', size=15)
    ax3.set_xlabel('Word Count')
    ax3.set_ylabel('Density')
    st.pyplot(fig3)
    st.write("Dari grafik distribusi jumlah kata dalam tweet, terlihat bahwa sebagian besar tweet memiliki jumlah kata berkisar antara 0 hingga sekitar 35 kata. Nilai terbanyak terkonsentrasi di sekitar 5 hingga 10 kata, menunjukkan bahwa mayoritas tweet memiliki panjang yang relatif singkat.")
    
    # Visualisasi Wordcloud
    comment_words = ''
    stopwords = set(STOPWORDS)

    for val in df.content:
        val = str(val)
        tokens = val.split()
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        comment_words += " ".join(tokens) + " "

    # Membuat word cloud
    wordcloud = WordCloud(width=800, height=800,
                        background_color='white',
                        stopwords=stopwords,
                        min_font_size=10).generate(comment_words)

    st.subheader("Word Cloud dari Content Twitter")

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    st.pyplot(plt)
    st.write("Dari wordcloud yang dihasilkan setelah menghilangkan stop word, kata-kata yang sering muncul, ditandai dengan ukuran yang besar, adalah 'Iam', 'today', 'love', 'work', dan 'now'. Ini menunjukkan bahwa dalam dataset tersebut, kata-kata tersebut memiliki frekuensi kemunculan yang tinggi dan mungkin menjadi fokus utama dalam tweet-tweet yang dianalisis.")



    # --------------------------------------------------------------------------------------------------------
else:
    st.sidebar.markdown(" ")
    st.sidebar.title("Complete the configuration first to start classification")
    class_names=['anger', 'boredom', 'empty', 'enthusiasm', 'fun', 'happiness', 'hate', 'love', 'neutral', 'relief', 'sadness', 'surprise', 'worry']

    def plot_metrics(metrics_list):
        if show_conf_matrix:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)  # Compute confusion matrix
            
            fig, ax = plt.subplots(figsize=(10, 7))  # Increase figure size
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 14})
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title('Confusion Matrix')
            
            st.pyplot(fig)  # Display the plot in Streamlit

    classifier = st.sidebar.selectbox("Choose classifier", ("Support Vector Machine (SVM)", "Naive Bayes"))

    show_conf_matrix = st.sidebar.checkbox("Show Confusion Matrix")

    if st.sidebar.button("Classify", key="classify"):

        if classifier == "Support Vector Machine (SVM)":
            st.subheader("Support Vector Machine (SVM) results")         
        
            y_test = df_eval['y_test_svm']
            y_pred = df_eval['y_pred_svm']


        else:
            st.subheader("Naive Bayes (MultinomialNB) results")
            
            y_test = df_eval['y_test_nb']
            y_pred = df_eval['y_pred_nb']

        # Hitung dan print hasil evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        st.write("Accuracy: ", round(accuracy, 7))
        st.write("Precision: ", round(precision, 7))
        st.write("Recall: ", round(recall, 7))
        st.write("F1-Score: ", round(f1, 7))

        plot_metrics(metrics)


        





        st.write("-- End of Process --")
        
