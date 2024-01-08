import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def read_csv(path):
    return pd.read_csv(path)




def merge_df(df, y_name, x_name, plot_x, plot_y):

    data = {
        plot_x:[],
        plot_y:[],
        "model" :[]
    }
    for i, row in df.iterrows():

        if row["mode"] == "favor":
            data["model"].append("RFK")

        elif row["mode"] == "linear":
            data["model"].append("linear")

        elif row["mode"] == "rand":
            p = row["prob"]
            data["model"].append(f"rand {p}")

        elif row["mode"] == "topk":
            p = round(1 - row["prob"], 1)
            data["model"].append(f"topk {p}")

        elif row["mode"] == "sparsemax":
            data["model"].append("sparsemax")

        elif row["mode"] == "softmax":
            data["model"].append("softmax")

        elif row["mode"] == "window":
            data["model"].append("window")
        else:
            data["model"].append(row["mode"])
        
        data[plot_x].append(row[x_name])
        data[plot_y].append(row[y_name])

        return data
    

def plot(data):

    df = pd.DataFrame(data)
    
