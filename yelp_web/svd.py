def svd():
    import pandas as pd
    import numpy as np
    import scipy
    import seaborn as sns
    import matplotlib.pyplot as plt

    from collections import defaultdict

    from surprise import SVD
    from surprise import Reader
    from surprise import Dataset
    from surprise import accuracy

    # Carga de datos ********

    usuario = pd.read_csv("T2 - users_yelp.csv")
    reviews = pd.read_csv("T2 - reviews.csv")
    business = pd.read_csv("T2 - business.csv", sep=",")
    coffee = pd.read_csv('usuarios.csv')

    reviews_train = reviews[~reviews['user_id'].isin(coffee['user_id'])]
    reviews_test = reviews[reviews['user_id'].isin(coffee['user_id'])]
    reviews_train = reviews_train[["user_id", "business_id", "review_stars"]]
    reviews_test = reviews_test[["user_id", "business_id", "review_stars"]]

    # División Test, train, validate

    # Dataset a Surprise ********

    # Se establece el rango en el cual se aceptaran los ratings
    reader = Reader(rating_scale = ( 1, 5 ))

    # Transformación de los datasets, puede leer los datasets directamente desde el disco sin necesidad de
    # pasar por pandas
    train_data = Dataset.load_from_df( reviews_train[ [ 'user_id', 'business_id', 'review_stars' ] ], reader )
    train_d = Dataset.load_from_df( reviews_train[ [ 'user_id', 'business_id', 'review_stars' ] ], reader )
    test_data = Dataset.load_from_df( reviews_test[ [ 'user_id', 'business_id', 'review_stars' ] ], reader )

    # Surprise requiere que explicitamente los datasets sean transformados a datasets de entrenamiento y
    # prueba en cada caso
    # Si bien no se entrenará sobre los datasets de validación y prueba, surprise requiere que sean
    # tranformados a entrenamiento para posteriormente ser transformados a prueba
    train_data = train_data.build_full_trainset()
    test_data = test_data.build_full_trainset()

    # Finalmente, se convierten los 3 datasets a prueba ya que se medirá el error obtenido en los 3
    train_data_2 = train_data.build_testset()
    test_data = test_data.build_testset()

    # SVD ********

    mean = train_data.global_mean

    # Hiperparámetros ********

    from surprise.model_selection import GridSearchCV

    # param_grid = {'n_factors': [25, 30, 35, 40, 100],
    #               'n_epochs': [15, 20, 25],
    #               'lr_all': [0.001, 0.003, 0.005, 0.008],
    #               'reg_all': [0.08, 0.1, 0.15, 0.02]}

    # param_grid = {'n_factors': [25, 30],
    #               'n_epochs': [15, 20],
    #               'lr_all': [0.001, 0.003],
    #               'reg_all': [0.08, 0.1]}

    # param_grid = {'n_factors': [5, 25, 30],
    #               'n_epochs': [15, 20, 25, 30],
    #               'lr_all': [0.001, 0.003, 0.005, 0.008],
    #               'reg_all': [0.08, 0.1, 0.15, 0.02]}

    # gs = GridSearchCV(SVD, param_grid, measures=['RMSE', 'MAE'], cv=3)
    # gs.fit(train_d)
    # algo2 = gs.best_estimator['rmse']

    # print(gs.best_score['rmse'])
    # print(gs.best_params['rmse'])

    # #Assigning values
    # t = gs.best_params
    # factors = t['rmse']['n_factors']
    # epochs = t['rmse']['n_epochs']
    # lr_value = t['rmse']['lr_all']
    # reg_value = t['rmse']['reg_all']

    algo = SVD( n_factors = 5, n_epochs = 25, biased = True, lr_all = 0.008, reg_all = 0.08, init_mean = 0, init_std_dev = 0.01, verbose = True )

    # Se realiza el entrenamiento a partir del dataset debido
    algo.fit( train_data )

    # Evaluación de predicciones ********

    predictions_train = algo.test( train_data_2 )
    predictions_test = algo.test( test_data )

    accuracy.rmse( predictions_train, verbose = True )
    accuracy.rmse( predictions_test, verbose = True )

    # Recomendaciones por usuario ********

    usertest = 'qKpkRCPk4ycbllTfFcRbNw'

    df_Upredictions2 = []
    for x in business['business_id']:
      Upredictions2 = algo.predict(usertest,x)
      df_Upredictions2.append(Upredictions2)

    #Ordenamos de mayor a menor estimación de relevancia
    df_Upredictions2.sort(key=lambda x : x.est, reverse=True)

    # Se convierte a dataframe
    labels = ['user_id', 'business_id','rating']
    df_Upredictions2 = pd.DataFrame.from_records(list(map(lambda x: (x.uid, x.iid, x.est) , df_Upredictions2)), columns=labels)

    df_Upredictions2.to_csv('SVD_user_rec.csv')

    # Recomendaciones globales ********

    df_Gpredictions2 = []
    for y in coffee['user_id']:
      for x in business['business_id']:
        Gpredictions2 = algo.predict(y,x)
        df_Gpredictions2.append(Gpredictions2)

    # Se convierte a dataframe
    labels = ['user_id', 'business_id','rating']
    df_Gpredictions2 = pd.DataFrame.from_records(list(map(lambda x: (x.uid, x.iid, x.est) , df_Gpredictions2)), columns=labels)

    df_Gpredictions2.to_csv('SVD_global_rec.csv')
