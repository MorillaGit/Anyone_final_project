



    # split dataset and get stratified target feature
    X_train, X_test, y_train, y_test = utils_features.split(
        df, "TARGET_LABEL_BAD=1"
    )