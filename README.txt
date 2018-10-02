PARÂMENTROS
    Para o correto funcionamento dos códigos alguns para parametro devem ser configurados antes de sua execucao
    Em util.py
        Path de todos os dataset
        num_cores utilizando nos processamentos. Todas de longo tempo desenvolvidas nesta trabalho foram paralelizadas, e todas elas utilizarao o parametro num_cores.
    Em train.py
        MODEL_ACTION
            TEST_ALL_CLASSIFIERS - Testa todos os classificadores sobre os features selecionados
            TEST_BIG_CLASSIFIER  - Monta o big classificador apresentado e explicado no "artigo"
                                   gerado desse trabalho
                IMAGE_ROTULATION - Rotulas novas imagens com o big classificar, sendo a nova imagem
                                   adiciona ao dataset somente se todos os classificadores gerarem 
                                   a mesma classe
                    AMOS - Rotula dados do dataset AMOS

Os códigos possuiem documentações resumidas em todas as suas funções, que devem ser lidas para uma correta execução

ARQUIVOS E SUAS FUNCIONALIDADES
    deep_learning_train.py
        Realiza o treinamento das CNNs com fine tunning

    feature_extraction_CNN.py
        Realiza a extracao de features utilizando CNNs

    feature_extraction.py
        Realiza a extracao de features do histograma, co-matriz e LBP

    process_dataset_AMOS.py
        Funçoes de processamento de dados para o dataset AMOS

    process_dataset_nexet.py
        Funçoes de processamento de dados para o dataset nexet

    process_dataset_transient.py
        Funçoes de processamento de dados para o dataset transient

    train.py
        Realiza o treinamento de todos os classificadores (os features já devem ter sido extraidos), exceto os deep models

    util.py
        Diversas funções de "uso geral" que foram necessárias no decorrer deste trabalho

    vgg16_places_365.py
        Modelo pre treinado do vgg16_places_365