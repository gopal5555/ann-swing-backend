import pandas as pd

def build_features(df):
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    out = []

    for sym, g in df.groupby("symbol"):
        g = g.copy()
        g["ret_5d"] = g["close"].pct_change(5)
        g["ma20"] = g["close"].rolling(20).mean()
        g["close_vs_ma20"] = (g["close"] - g["ma20"]) / g["ma20"]
        out.append(g)

    feat = pd.concat(out).dropna().reset_index(drop=True)
    return feat

def nlp_advice(score, advice):
    if advice == "STRONG":
        return (
            "Stock me strong positive momentum dikh raha hai. "
            "Market support bhi positive hai. Normal quantity ke "
            "sath trade liya ja sakta hai, stop-loss follow karna zaroori hai."
        )
    elif advice == "WEAK":
        return (
            "Momentum positive hai, lekin strength limited hai. "
            "Is trade ko skip karna ya half quantity ke sath "
            "tight stop-loss rakhna better rahega."
        )
    else:
        return (
            "Aaj model ko koi clear opportunity nahi mili. "
            "Capital protection ke liye trade avoid karna better hota hai."
        )