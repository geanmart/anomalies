def ajusta_gaussianas(X, n_components=5, n_artificial=1, D1_factor=2, growth_rate=1.5):
    """
    Ajusta gaussianas usando GMM e adiciona gaussianas artificiais seguindo regras específicas.
    As médias e demais parâmetros são ordenados em ordem crescente de médias.
    """
    # Ajuste inicial do GMM
    gmm = GaussianMixture(n_components=n_components, random_state=0).fit(X.reshape(-1, 1))
    
    # Parâmetros ajustados
    weights = gmm.weights_.tolist()
    means = gmm.means_.flatten().tolist()
    covariances = gmm.covariances_.flatten().tolist()
    
    # Configuração inicial para gaussianas artificiais
    max_mean = max(means)
    max_std = np.sqrt(max(covariances))
    D1 = D1_factor * max_std  # Diferença inicial baseada no maior desvio padrão

    # Adicionando gaussianas artificiais
    last_mean = max_mean
    for i in range(n_artificial):
        # Calcular D_i com crescimento exponencial
        Di = D1 * (growth_rate ** i)
        new_mean = last_mean + Di
        new_covariance = np.var(X)  # Variância baseada nos dados
        new_weight = 0.1  # Peso arbitrário para as gaussianas artificiais

        # Atualizar parâmetros
        weights.append(new_weight)
        means.append(new_mean)
        covariances.append(new_covariance)
        last_mean = new_mean

    # Normalizar os pesos
    weights = np.array(weights)
    weights /= weights.sum()

    # Ordenar parâmetros com base nas médias
    order = np.argsort(means)  # Índices em ordem crescente de médias
    weights = np.array(weights)[order]
    means = np.array(means)[order]
    covariances = np.array(covariances)[order]

    # Atualizar o modelo GMM com parâmetros ordenados
    new_gmm = GaussianMixture(n_components=n_components + n_artificial, random_state=0)
    new_gmm.weights_ = weights
    new_gmm.means_ = means.reshape(-1, 1)
    new_gmm.covariances_ = covariances.reshape(-1, 1, 1)
    new_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(new_gmm.covariances_))
    
    return new_gmm



def roda_tudo(df_daily_means,feature='feature_1'):
    train_period = "2023-04-30"
    test_period = "2023-05-31"
    deploy_period = "2023-06-01"

    df_means_feat_1 = df_daily_means.query("feature==@feature").drop(["overall_mean","abs_diff"],axis=1).reset_index(drop=True)
    df_means_feat_1.date = pd.to_datetime(df_means_feat_1.date)

    train_set = df_means_feat_1.query("date<=@train_period").reset_index(drop=True)
    test_set = df_means_feat_1.query("date<=@test_period and date>@train_period").reset_index(drop=True)
    deploy_set = df_means_feat_1.query("date>=@deploy_period").reset_index(drop=True)

    train_set['set'] = 'train'
    test_set['set'] = 'test'
    deploy_set['set'] = 'deploy'

    train_mean = train_set['value'].mean()
    test_mean = train_set['value'].mean()
    deploy_mean = train_set['value'].mean()

    train_set.eval("abs_diff=abs(value-@train_mean)",inplace=True)
    test_set.eval("abs_diff=abs(value-@test_mean)",inplace=True)
    deploy_set.eval("abs_diff=abs(value-@deploy_mean)",inplace=True)
    ##training the model

    ## adding the new gaussian
    new_gmm = ajusta_gaussianas(train_set['abs_diff'].values.reshape(-1,1),3,4)
    
    #evaluation
    X = train_mean

    Y = deploy_set['value'].values
    kde_y = gaussian_kde(Y)
    # Gerar valores de z para avaliar
    z_values = np.linspace(0, 0.1, 100)
    pdf_values = np.array([pdf_abs_diff(z) for z in z_values])

    
    ## definindo multipliers
    multipliers = [30,50,70,100]
    severity = ["INFO","WARNING","ERROR","CRITICAL"]

    limits = [abs(X) * alpha for alpha in multipliers]

    #medias das gaussianas
    limits_proposed = new_gmm.means_.ravel()
    severity_proposta = ["normal_0","normal_1","normal_2","anormal_1","anormal_2","anormal_3","anormal_4"]
    # Definindo cores personalizadas para o segundo for
    colors_proposed = [cm.Dark2(i / len(limits_proposed)) for i in range(len(limits_proposed))]  # Usando a paleta Set2

    # Configurando subplots lado a lado
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

    # Primeiro subplot
    axs[0].plot(z_values, pdf_values, label="Original", color='blue')
    for i, limit in enumerate(limits):
        axs[0].axvline(x=limit, color=f"C{i}", linestyle="--")
        axs[0].text(limit, 0.5, severity[i], color=f"C{i}", fontsize=10, 
                    verticalalignment='center', horizontalalignment='left', rotation=90)
    axs[0].set_title("Gráfico Original")
    axs[0].set_xlabel("Z")
    axs[0].set_ylabel("Densidade de Probabilidade")
    axs[0].grid(alpha=0.3)
    axs[0].legend()

    # Segundo subplot
    axs[1].plot(z_values, pdf_values, label="Proposed", color='blue')
    for i, limit in enumerate(limits_proposed):
        axs[1].axvline(x=limit, color=colors_proposed[i], linestyle=":")
        axs[1].text(limit, 0.5, severity_proposta[i], color=colors_proposed[i], fontsize=10, 
                    verticalalignment='center', horizontalalignment='left', rotation=90)
    axs[1].set_title("Gráfico Proposto")
    axs[1].set_xlabel("Z")
    axs[1].grid(alpha=0.3)
    axs[1].legend()

    # Ajustar layout e exibir
    plt.tight_layout()
    plt.show()

    info_deploy = pd.DataFrame({"obs":deploy_set['abs_diff'].values,"gaussiana":new_gmm.predict(deploy_set['abs_diff'].values.reshape(-1,1))})
    # Usando sns.histplot
    sns.histplot(data=info_deploy, x='obs', hue='gaussiana', kde=True, palette="tab10")
    return 1
