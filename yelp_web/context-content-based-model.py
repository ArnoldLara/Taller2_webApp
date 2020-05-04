def ccbm():
    import pandas as pd
    import numpy as np
    import scipy
    #import seaborn as sns
    #import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import CountVectorizer

    #from collections import defaultdict

    #from surprise import SVD
    #from surprise import Reader
    #from surprise import Dataset
    #from surprise import accuracy

    ### Lectura de Información

    business = pd.read_csv("/content/drive/My Drive/Sistemas de recomendacion/Taller 2/JSON to csv/data/T2 - business.csv", sep=",")
    svd = pd.read_csv("/content/drive/My Drive/Sistemas de recomendacion/Taller 2/JSON to csv/data/SVD_global_filtered.csv", sep=",", usecols=[1,2,3])
    reviews = pd.read_csv("/content/drive/My Drive/Sistemas de recomendacion/Taller 2/JSON to csv/data/T2 - reviews.csv", usecols = [0, 1, 10])

    ### Funciones para reutilizar trabajo

    ## Función para calcular % de missing values por columna

    def missing_values_table(df):
            mis_val = df.isnull().sum()
            mis_val_percent = 100 * df.isnull().sum()/len(df)
            mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
            mis_val_table_ren_columns = mis_val_table.rename(
            columns = {0 : 'Missing Values', 1 : '% of Total Values'})
            return mis_val_table_ren_columns

    ## Función para catalogar franja de horario más visitada

    def time_day(row):
        if row['hour'] > 5  and row["hour"] < 12:
            val = "Morning"
        elif row['hour'] >= 12  and row["hour"] <= 18:
            val = "Afternoon"
        elif row['hour'] > 18  and row["hour"] <= 23:
            val = "Night"
        elif row['hour'] >= 0  and row["hour"] <= 5:
            val = "Dawn"
        else:
            val = "Whenever"
        return val



    ### Código para obtener máximos solo 1000 business máximo por ciudad

    reviews_list = business.groupby('city', group_keys=False).apply(lambda x: x.sample(min(len(x), 1000)))

    reviews = reviews[reviews["business_id"].isin(reviews_list["business_id"].unique())]

    ### Análisis exploratorio y Wrangling de datos

    reviews["date"] = pd.to_datetime(reviews["date"])
    reviews["hour"] = reviews["date"].dt.hour
    reviews['hour_plan'] = reviews.apply(time_day, axis=1)

    business_hour = reviews[["business_id", "name", "hour_plan"]].groupby(["business_id", "hour_plan"]).count().reset_index()
    business_hour = business_hour.sort_values(by=["business_id", "name"], ascending=False)
    business_hour = business_hour.drop_duplicates(subset=["business_id"], keep='first')
    business_hour = business_hour[["business_id", "hour_plan"]]
    business_hour.columns = ["business_id", "hour_more_visited"]

    metadata = reviews.copy()[["business_id", "name"]].drop_duplicates(subset=["business_id"], keep="first")
    metadata = metadata.merge(business[["business_id", "address",  "city", "categories", "stars", "review_count"]].drop_duplicates(), how='left', on=["business_id"])
    metadata = metadata.merge(business_hour, how='left', on=["business_id"])
    metadata["hour_more_visited"] = metadata["hour_more_visited"].apply(lambda x: [x,x])
    metadata["city"] = metadata["city"].apply(lambda x: [x,x])
    metadata["categories"] = metadata["categories"].apply(lambda x: [x])

    metadata["city"] = metadata["city"].astype('str')
    metadata["categories"] = metadata["categories"].astype('str')
    metadata["hour_more_visited"] = metadata["hour_more_visited"].astype('str')

    metadata["city"] = metadata["city"].apply(lambda x: x.replace("[", "").replace("'", ""))
    metadata["city"] = metadata["city"].apply(lambda x: x.replace("]", ""))

    metadata["categories"] = metadata["categories"].apply(lambda x: x.replace("[", "").replace("'", ""))
    metadata["categories"] = metadata["categories"].apply(lambda x: x.replace("]", ""))

    metadata["hour_more_visited"] = metadata["hour_more_visited"].apply(lambda x: x.replace("[", "").replace("'", ""))
    metadata["hour_more_visited"] = metadata["hour_more_visited"].apply(lambda x: x.replace("]", ""))

    metadata["features"] = metadata["categories"] + ' ' + metadata["city"] + ' ' + metadata["hour_more_visited"]

    del reviews
    gc.collect()

    C = metadata["stars"].mean()
    print(C)

    m = metadata['review_count'].quantile(0.50)

    q_business = metadata.copy().loc[metadata['review_count'] >= m]

    def weighted_rating(x, m=m, C=C):
        v = x['review_count']
        R = x['stars']
        # Calculation based on the IMDB formula
        return (v/(v+m) * R) + (m/(m+v) * C)

    q_business['score'] = q_business.apply(weighted_rating, axis=1)

    q_business = q_business.sort_values('score', ascending=False)

    ### Construcción de modelo con CountVectorizer

    metadata['categories'] = metadata['categories'].fillna('')

    def clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(",", " ")) for i in x]
        else:
            #Check if director exists. If not, return empty string
            if isinstance(x, str):
                return str.lower(x.replace(",", " "))
            else:
                return ''

    features = ['features']

    for feature in features:
        metadata[feature] = metadata[feature].apply(clean_data)

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(metadata['features'])
    print(count_matrix.shape)

    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

    metadata = metadata.reset_index()
    indices = pd.Series(metadata.index, index=metadata['address'])

    ### Construcción de modelo con matriz tf-idf

    #tfidf = TfidfVectorizer(stop_words='english')

    #tfidf_matrix = tfidf.fit_transform(metadata['features'])

    #cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    def get_recommendations2(address, cosine_sim=cosine_sim2):
        # Get the index of the movie that matches the title
        idx = indices[address]

        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim2[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:6]

        # Get the movie indices
        business_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar movies
        return metadata['address'].iloc[business_indices]

    def get_recommendations(address, cosine_sim=cosine_sim):
        # Get the index of the movie that matches the title
        idx = indices[address]

        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:6]

        # Get the movie indices
        business_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar movies
        return metadata['address'].iloc[business_indices]
