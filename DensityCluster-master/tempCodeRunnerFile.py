pca = PCA(n_components=2)
    Data=pca.fit_transform(points)
    print(Data)