import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Sauvegarde et chargement de modèle et fichier
import joblib
#import pickle #plus utile pas supporter par Heroku
# Analyse de l'importance des features
import shap
import plotly.express as px
import lightgbm
from sklearn.cluster import KMeans
#from zipfile import ZipFile #plus utile pas supporter par Heroku
# version du 13/02/2022
# Ne pouvant pas revoir la structure du programme entièrement après remarques mentor les structure sample et data sont identiques
#
def main():
    def load_data():
        #Version on-line
        data=pd.read_csv("app_test_domain_norm_idx.csv", sep=",")
        data = data.set_index('SK_ID_CURR')
        sample=pd.read_csv("app_test_domain_norm_idx.csv", sep=",")
        sample = sample.set_index('SK_ID_CURR')
        description = pd.read_csv("HomeCredit_columns_description.csv",
                          usecols=['Row', 'Description'], index_col=0, encoding='unicode_escape')
        return data, sample, description

    def identite_client(data, id):
        data_client = data[data.index == int(id)]
        return data_client

    @st.cache(allow_output_mutation=True)
    def load_knn(sample):
        knn = knn_training(sample)
        return knn

    # @st.cache
    def load_model():
        model = joblib.load("strealit_model.sav")
        return model

    def load_model_shap():
        model = joblib.load("strealit_model_shap.sav")
        return model

    # @st.cache
    def pretraitement(sample, id):
        sample = sample[id,:]
        return sample

    @st.cache
    def load_prediction(sample, id, clf):
        X=sample
        score_proba = clf.predict_proba(X[X.index == int(id)])
        score_pred= clf.predict(X[X.index == int(id)])
        return score_proba, score_pred

    @st.cache
    def load_kmeans(sample, id, knn):
        index = sample[sample.index == int(id)].index.values
        data_client = pd.DataFrame(sample.loc[sample.index, :])
        df_neighbors = pd.DataFrame(knn.fit_predict(data_client), index=data_client.index)
        df_neighbors = pd.concat([df_neighbors, data], axis=1)
        return df_neighbors.iloc[:,1:].sample(10)

    @st.cache
    def knn_training(sample):
        knn = KMeans(n_clusters=2).fit(sample)
        return knn

    @st.cache
    def load_age_population(data):
        data_age = round((data["DAYS_BIRTH"]/365), 2)
        return data_age

    @st.cache
    def load_income_population(sample):
        df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
        df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
        return df_income

    @st.cache
    def load_infos_gen(data):
        lst_infos = [data.shape[0],
                     round(data["AMT_INCOME_TOTAL"].mean(), 2),
                     round(data["AMT_CREDIT"].mean(), 2)]

        nb_credits = lst_infos[0]
        rev_moy = lst_infos[1]
        credits_moy = lst_infos[2]
        #targets = data.TARGET.value_counts()
        return nb_credits, rev_moy, credits_moy

    def undummify(df, prefix_sep="_"):
        cols2collapse = {
            item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
        }
        series_list = []
        for col, needs_to_collapse in cols2collapse.items():
            if needs_to_collapse:
                undummified = (
                    df.filter(like=col)
                        .idxmax(axis=1)
                        .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                        .rename(col)
                )
                series_list.append(undummified)
            else:
                series_list.append(df[col])
        undummified_df = pd.concat(series_list, axis=1)
        return undummified_df

    # Chargement des dataframe et du modèle

    # Commentaire: à faire
    # Loading data……
    data, sample, description = load_data()
    #print(data.index.value)
    id_client = sample.index
    numero_client = data.index
    # j'ai 2 variables client
    # 1 - id_client : pour les prédictions
    # 2 - numero_client : pour fournir l'accès à tous les clients
    feature = data.columns[1:]
    clf = load_model()

    # PARTIE WEB
    # PAGE CLIENT
    def page_client():
        # Affichage du jeu de données
        #######################################
        # SIDEBAR
        #######################################
        html_temp = """
        	    <div style="background-color: tomato; padding:10px; border-radius:10px">
        	    <h1 style="color: white; text-align:center">"CONSULTER VOS DONNEES"</h1>
        	    </div>
        	    <p style="font-size: 20px; font-weight: bold; text-align:center">Choisissez votre code client</p>
        	    """
        st.markdown(html_temp, unsafe_allow_html=True)

        st.sidebar.header("**JEU DE DONNEES TOUT CLIENT**")

        chk_id = st.sidebar.selectbox("Numero de client", numero_client)

        st.header("**CONSULTATION DE MES DONNEES **")
        if st.button('Consultation'):
            consult = data[data.index == chk_id]
            st.subheader("**Mes données:**")
            st.dataframe(consult.style.highlight_max(axis=0))

            infos_client = data[data.index == chk_id]
            if infos_client["CODE_GENDER_F"].values[0]==1:
                st.write("**Gender : **", "F")
            else:
                st.write("**Gender : **", "M")
            st.write("**Age : **{:.0f} ans".format(int(infos_client["DAYS_BIRTH"]/365)))
            if infos_client["NAME_FAMILY_STATUS_Civil marriage"].values[0]==1:
                st.write("**Family status : **", "Civil marriage")
            elif infos_client["NAME_FAMILY_STATUS_Married"].values[0]==1:
                st.write("**Family status : **", "Married")
            elif infos_client["NAME_FAMILY_STATUS_Separated"].values[0]==1:
                st.write("**Family status : **", "Separated")
            elif infos_client["NAME_FAMILY_STATUS_Single / not married"].values[0]==1:
                st.write("**Family status : **", "Single or not married")
            elif infos_client["NAME_FAMILY_STATUS_Widow"].values[0]==1:
                st.write("**Family status : **", "SWidow")
            st.write("**Number of children : **{:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))

            # Age distribution plot
            data_age = load_age_population(data)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(data_age, edgecolor='k', color="goldenrod", bins=20)
            ax.axvline(int(infos_client["DAYS_BIRTH"].values / 365), color="green", linestyle='--')
            ax.set(title='Customer age', xlabel='Age(Year)', ylabel='')
            st.pyplot(fig)

            st.subheader("*Income (USD)*")
            st.write("**Income total : **{:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
            st.write("**Credit amount : **{:.0f}".format(infos_client["AMT_CREDIT"].values[0]))
            st.write("**Credit annuities : **{:.0f}".format(infos_client["AMT_ANNUITY"].values[0]))
            st.write("**Amount of property for credit : **{:.0f}".format(infos_client["AMT_GOODS_PRICE"].values[0]))

            # Income distribution plot
            data_income = load_income_population(data)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor='k', color="goldenrod", bins=10)
            ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="green", linestyle='--')
            ax.set(title='Customer income', xlabel='Income (USD)', ylabel='')
            st.pyplot(fig)

            # Relationship Age / Income Total interactive plot
            data_sk = data.reset_index(drop=False)
            data_sk.DAYS_BIRTH = (data_sk['DAYS_BIRTH'] / 365).round(1)
            data_sk2=undummify(data_sk[['CODE_GENDER_F','CODE_GENDER_M']])
            data_sk3 = pd.concat([data_sk, data_sk2], axis=1)
            data_sk3 = data_sk3.drop(columns=['CODE_GENDER_F', 'CODE_GENDER_M'])
            data_sk3.rename(columns={'CODE': 'CODE_GENDER'}, inplace=True)
            fig, ax = plt.subplots(figsize=(10, 10))
            fig = px.scatter(data_sk3, x='DAYS_BIRTH', y="AMT_INCOME_TOTAL",
                             size="AMT_INCOME_TOTAL", color='CODE_GENDER',
                             hover_data=['NAME_FAMILY_STATUS_Civil marriage', 'NAME_FAMILY_STATUS_Married', 'NAME_FAMILY_STATUS_Separated', 'NAME_FAMILY_STATUS_Single / not married', 'NAME_FAMILY_STATUS_Widow', 'CNT_CHILDREN', 'NAME_CONTRACT_TYPE', 'SK_ID_CURR'])
            fig.update_layout({'plot_bgcolor':'#f0f0f0'},
                          title={'text':"Relationship Age / Income Total", 'x':0.5, 'xanchor': 'center'},
                          title_font=dict(size=20, family='Verdana'), legend=dict(y=1.1, orientation='h'))


            fig.update_traces(marker=dict(line=dict(width=0.5, color='#3a352a')), selector=dict(mode='markers'))
            fig.update_xaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                             title="Age", title_font=dict(size=18, family='Verdana'))
            fig.update_yaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                             title="Income Total", title_font=dict(size=18, family='Verdana'))

            st.plotly_chart(fig)
        else:
            st.markdown("<i>…</i>", unsafe_allow_html=True)

        ### FIN DE LA PAGE CLIENT

    #-----------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------
    # PARTIE WEB (suite)
    # PAGE INTERNE
    def aff_page_prediction():
        html_temp = """
        	    <div style="background-color: tomato; padding:10px; border-radius:10px">
        	    <h1 style="color: white; text-align:center">"Dashboard Scoring Credit"</h1>
        	    </div>
        	    <p style="font-size: 20px; font-weight: bold; text-align:center">Choisissez le code client</p>
        	    """
        st.markdown(html_temp, unsafe_allow_html=True)

        # Customer ID selection
        st.sidebar.header("**General Info**")
        # Loading selectbox
        chk_id = st.sidebar.selectbox("Client ID", id_client)

        # Loading general info
        nb_credits, rev_moy, credits_moy= load_infos_gen(data)

        ### Display of information in the sidebar ###
         # Number of loans in the sample
        st.sidebar.markdown("<u>Number of loans in the sample :</u>", unsafe_allow_html=True)
        st.sidebar.text(nb_credits)

        # Average income
        st.sidebar.markdown("<u>Average income (USD) :</u>", unsafe_allow_html=True)
        st.sidebar.text(rev_moy)

        # AMT CREDIT
        st.sidebar.markdown("<u>Average loan amount (USD) :</u>", unsafe_allow_html=True)
        st.sidebar.text(credits_moy)

        # PieChart
        # st.sidebar.markdown("<u>......</u>", unsafe_allow_html=True)
        #fig, ax = plt.subplots(figsize=(5, 5))
        #plt.pie(targets, explode=[0, 0.1], labels=['No default', 'Default'], autopct='%1.1f%%', startangle=90)
        #st.sidebar.pyplot(fig)

        chk_id_2 = st.sidebar.selectbox("Numero de client", id_client)

        # Customer solvability display
        st.header("**ANALYSE DES PREDICTIONS**")
        if st.button('Prediction'):
            prediction_proba, prediction = load_prediction(sample, chk_id_2, clf)
            st.write(prediction_proba)
            st.markdown("<u>Customer Data :</u>", unsafe_allow_html=True)
            st.write(identite_client(data, chk_id_2))

        # Feature importance / description
        if st.checkbox("Customer ID {:.0f} feature importance ?".format(chk_id_2)):
        #st.header("**Feature importance**")
            shap.initjs()
            #X = sample.iloc[:, :-1]
            X = sample
            X_features=list(X.columns)
            number = st.slider("Pick a number of features…", 0, 200, 5)
            fig, ax = plt.subplots(figsize=(10, 10))
            explainer = shap.TreeExplainer(load_model_shap(),X)
            shap_values = explainer.shap_values(X)
            shap.summary_plot(shap_values, X, plot_type="bar", max_display=number, color_bar=False, plot_size=(5, 5))
            st.pyplot(fig)

        else:
            st.markdown("<i>…</i>", unsafe_allow_html=True)
        if st.checkbox("Need help about feature description ?"):
            list_features = description.index.to_list()
            feature = st.selectbox('Feature checklist…', list_features)
            st.table(description.loc[description.index == feature][:1])

        # Similar customer files display

        chk_voisins = st.checkbox("Show similar customer files ?")
        if chk_voisins:
            knn = load_knn(sample)
            st.markdown("<u>List of the 10 files closest to this Customer :</u>", unsafe_allow_html=True)
            st.dataframe(load_kmeans(sample, chk_id_2, knn))
            #st.markdown("<i>Target 1 = Customer with default</i>", unsafe_allow_html=True)
        else:
            st.markdown("<i>…</i>", unsafe_allow_html=True)
    #------------------------------------------------------------------
    # FIN DE LA PAGE INTERNE
    #------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------
    # PARTIE WEB
    # Appel à la page principale

    #######################################
    # HOME PAGE - MAIN CONTENT
    #######################################

    # PARTIE CLIENT - on travaille sur le dataframe général car il faut que tous les clients
    # puissent accéder à leurs données personnelles
    html_temp = """
            <div style="background-color: tomato; padding:10px; border-radius:10px">
    	    <h1 style="color: white; text-align:center">"PRET A DEPENSER"</h1>
    	    </div>
    	    <p style="font-size: 20px; font-weight: bold; text-align:center">Bienvenue</p>
    	    """
    st.markdown(html_temp, unsafe_allow_html=True)
    # Display Customer ID from Sidebar
    # Title
    st.title("Suivi de votre demande")
    st.header("Veuillez, svp, préciser si vous êtes client on interne ")
    status = st.radio("Votre choix",
                      ('Client', 'Interne'))  # conditional statement to print Male if male is selected else print female
    # show the result using the success function

    if (status == 'Client'):

        page_client()
    else:
        aff_page_prediction()

if __name__ == '__main__':
    main()

