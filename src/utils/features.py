def add_days_granularity(out, features_list):
    for f in features_list:
        days_15 = out[f].rolling(15, min_periods=14).mean().fillna(method="bfill")
        out["{}_15days".format(f)] = days_15

        days_10 = out[f].rolling(10, min_periods=9).mean().fillna(method="bfill")
        out["{}_10days".format(f)] = days_10

        days_5 = out[f].rolling(5, min_periods=4).mean().fillna(method="bfill")
        out["{}_5days".format(f)] = days_5

        days_30 = out[f].rolling(30, min_periods=29).mean().fillna(method="bfill")
        out["{}_30days".format(f)] = days_30

    return out
