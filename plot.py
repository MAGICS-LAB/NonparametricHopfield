import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# df = pd.read_csv('./favor_0.5_0.csv')

# for name in ['softmax', 'sparsemax', 'linear', 'favor', 'topk', 'rand']:

#     if name in ['topk', 'rand']:
#         for p in [0.2, 0.5, 0.8]:
#             for i in range(5):
#                 df2 = pd.read_csv(f"{name}_{p}_{i}.csv")
#                 df = pd.concat([df, df2], keys=df.columns, ignore_index=True)
#     else:
#         for p in [0.5]:
#             for i in range(5):
#                 df2 = pd.read_csv(f"{name}_{p}_{i}.csv")
#                 df = pd.concat([df, df2], keys=df.columns,ignore_index=True)


def load_data(tgt_name, tgt_plot_name):


    data = {
        'epoch':[],
        tgt_name:[],
        'Model':[]
    }

    for i, row in df.iterrows():

        data['epoch'].append(row['epoch'])
        data[tgt_name].append(row[tgt_name])
        if row['model'] == 'sparsemax':
            data['Model'].append('Sparse Hopfield [Hu et al., 2023]')
        elif row['model'] == 'softmax':
            data['Model'].append('Dense Hopfield [Ramsauer et al. 2020]')
        elif row['model'] == 'favor':
            data['Model'].append('Random Feature Hopfield')
        elif row['model'] == 'linear':
            data['Model'].append('Linear Hopfield')
        elif row['model'] == 'topk':
            p = round(float(row['prob'])*100)
            data['Model'].append(f'Top {p}% Hopfield')
        elif row['model'] == 'rand':
            p = round(float(row['prob']), 1)
            data['Model'].append(f'Random Masked Hopfield {p}')
    return pd.DataFrame(data)

def generate_plot(tgt_name, tgt_plot_name):

    data = load_data(tgt_name, tgt_plot_name)

    fig = plt.figure(figsize=(12,8))
    plt.title(tgt_plot_name, fontsize=20)
    # sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    sns.despine(left=False,top=True, right=True, bottom=False)
    sns.lineplot(data=data, x="Memory Size", y=tgt_name, hue="Model", errorbar=None)

    plt.xlabel("Memory Size",fontsize=20)
    plt.ylabel(tgt_plot_name,fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(fontsize=20, prop = {"size": 20})
    fig.tight_layout()
    plt.savefig(f"{tgt_plot_name}.png")
    plt.clf()


# generate_plot("train acc", "Train Accuracy")
# generate_plot("test acc", "Validation Accuracy")
# generate_plot("train loss", "Train Loss")
# generate_plot("test loss", "Validation Loss")

def load_data_rand_only(tgt_name, tgt_plot_name):

    df = pd.read_csv('./mts_cost.csv')

    data = {
        'Memory Size':[],
        tgt_name:[],
        'Model':[]
    }

    for i, row in df.iterrows():

        if row['model'] == 'rand_fast':
            data['Memory Size'].append(row['Sequence Length'])
            data[tgt_name].append(row[tgt_name])
            p = round(float(row['Prob']), 1)
            data['Model'].append(f'Random Masked Hopfield {p}')

    return pd.DataFrame(data)

def load_data2(tgt_name, tgt_plot_name):

    df = pd.read_csv('./mts_cost.csv')

    data = {
        'Memory Size':[],
        tgt_name:[],
        'Model':[]
    }

    for i, row in df.iterrows():

        if row['model'] == 'sparsemax':
            data['Model'].append('Sparse Hopfield [Hu et al., 2023]')
            data['Memory Size'].append(row['Sequence Length'])
            data[tgt_name].append(row[tgt_name])
        elif row['model'] == 'softmax':
            data['Model'].append('Dense Hopfield [Ramsauer et al. 2020]')
            data['Memory Size'].append(row['Sequence Length'])
            data[tgt_name].append(row[tgt_name])
        elif row['model'] == 'favor':
            data['Model'].append('Random Feature Hopfield')
            data['Memory Size'].append(row['Sequence Length'])
            data[tgt_name].append(row[tgt_name])
        elif row['model'] == 'linear':
            data['Model'].append('Linear Hopfield')
            data['Memory Size'].append(row['Sequence Length'])
            data[tgt_name].append(row[tgt_name])

        elif row['model'] == 'topk':
            p = round(float(row['Prob'])*100)
            data['Model'].append(f'Top {p}% Hopfield')
            data['Memory Size'].append(row['Sequence Length'])
            data[tgt_name].append(row[tgt_name])

        elif row['model'] == 'window':
            data['Model'].append('Window Hopfield')
            data['Memory Size'].append(row['Sequence Length'])
            data[tgt_name].append(row[tgt_name])

        else:
            print()
        # elif row['model'] == 'rand_fast':
        #     p = round(float(row['Prob']), 1)
        #     data['Model'].append(f'Random Masked Hopfield {p}')


    return pd.DataFrame(data)

def generate_plot2(tgt_name, tgt_plot_name):

    data = load_data2(tgt_name, tgt_plot_name)

    fig = plt.figure(figsize=(12,8))
    plt.title(tgt_plot_name, fontsize=20)
    # sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    sns.despine(left=False,top=True, right=True, bottom=False)
    sns.lineplot(data=data, x="Memory Size", y=tgt_name, hue="Model", errorbar=None, alpha=0.9, style='Model', linewidth=3)

    plt.xlabel("Memory Size",fontsize=20)
    plt.ylabel(tgt_plot_name,fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(fontsize=20, prop = {"size": 20})
    fig.tight_layout()
    plt.savefig(f"{tgt_plot_name}.png")
    plt.clf()

# generate_plot2("duration", "duration")
# generate_plot2("Flops", "Flops")


def generate_plot_rand_only(tgt_name, tgt_plot_name):

    data = load_data_rand_only(tgt_name, tgt_plot_name)

    fig = plt.figure(figsize=(12,8))
    plt.title(tgt_plot_name, fontsize=20)
    # sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    sns.despine(left=False,top=True, right=True, bottom=False)
    sns.lineplot(data=data, x="Memory Size", y=tgt_name, hue="Model", errorbar=None, alpha=0.8, style='Model', linewidth=3)

    plt.xlabel("Memory Size",fontsize=20)
    plt.ylabel(tgt_plot_name,fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(fontsize=20, prop = {"size": 20})
    fig.tight_layout()
    plt.savefig(f"{tgt_plot_name}_rand.png")
    plt.clf()


# generate_plot_rand_only("duration", "duration")
# generate_plot_rand_only("Flops", "Flops")



def load_data_mil(tgt_name=None, tgt_plot_name=None):

    df = pd.read_csv('loss_results/mnist_mil.csv')

    tgt_name = 'best test acc'
    data = {
        'Bag Size':[],
        "Test Accuracy":[],
        'Model':[]
    }

    for i, row in df.iterrows():

        data['Bag Size'].append(int(row['bag_size']))
        data["Test Accuracy"].append(float(row[tgt_name]))
        if row['mode'] == 'sparsemax':
            data['Model'].append('Sparse Hopfield [Hu et al., 2023]')
        elif row['mode'] == 'softmax':
            data['Model'].append('Dense Hopfield [Ramsauer et al. 2020]')
        elif row['mode'] == 'favor':
            data['Model'].append('Random Feature Hopfield')
        elif row['mode'] == 'linear':
            data['Model'].append('Linear Hopfield')
        elif row['mode'] == 'topk':
            p = round(float(row['prob'])*100)
            data['Model'].append(f'Top {p}% Hopfield')
        elif row['mode'] == 'rand':
            p = round(float(row['prob']), 1)
            data['Model'].append(f'Random Masked Hopfield {p}')
    return pd.DataFrame(data).sort_values(by=['Model'])

def generate_plot_mil(tgt_name, tgt_plot_name):

    data = load_data_mil(tgt_name, tgt_plot_name)

    fig = plt.figure(figsize=(12,6))
    plt.title(tgt_plot_name, fontsize=20)
    # sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)  
    sns.despine(left=False,top=True, right=True, bottom=False)
    sns.lineplot(data=data, x="Bag Size", y="Test Accuracy", hue="Model", alpha=0.8,marker="o", errorbar=None,  markersize=5, linewidth=3)

    plt.xlabel("Memory Size",fontsize=20)
    plt.ylabel(tgt_plot_name,fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(fontsize=20, prop = {"size": 20})
    fig.tight_layout()
    plt.savefig(f"{tgt_plot_name}_mil.png")
    plt.clf()


# generate_plot_mil("", "")

# logs = {
#     'duration':[],
#     'Model':[],
#     'Horizon Length':[],
#     'Prob':[]
# }
# df = pd.DataFrame(logs)
# df.to_csv('./time_series_duration_etth1.csv', index=False)

def load_data_mts(tgt_name=None, tgt_plot_name=None):

    df = pd.read_csv('time_series_duration_etth1.csv')

    tgt_name = 'duration'
    data = {
        'Horizon Length':[],
        "Time (ms) per epoch":[],
        'Model':[]
    }

    for i, row in df.iterrows():

        # data['Horizon Length'].append(int(row['Horizon Length']))
        # data["Time (ms) per epoch"].append(float(row[tgt_name]))
        if row['Model'] == 'sparsemax':
            data['Horizon Length'].append(int(row['Horizon Length']))
            data["Time (ms) per epoch"].append(float(row[tgt_name]))
            data['Model'].append('Sparse Hopfield [Hu et al., 2023]')
        elif row['Model'] == 'window':
            data['Horizon Length'].append(int(row['Horizon Length']))
            data["Time (ms) per epoch"].append(float(row[tgt_name]))
            data['Model'].append('Window Hopfield')
        elif row['Model'] == 'softmax':
            data['Horizon Length'].append(int(row['Horizon Length']))
            data["Time (ms) per epoch"].append(float(row[tgt_name]))
            data['Model'].append('Dense Hopfield [Ramsauer et al. 2020]')
        elif row['Model'] == 'favor':
            data['Horizon Length'].append(int(row['Horizon Length']))
            data["Time (ms) per epoch"].append(float(row[tgt_name]))
            data['Model'].append('Random Feature Hopfield')
        elif row['Model'] == 'linear':
            data['Horizon Length'].append(int(row['Horizon Length']))
            data["Time (ms) per epoch"].append(float(row[tgt_name]))
            data['Model'].append('Linear Hopfield')
        elif row['Model'] == 'topk':
            data['Horizon Length'].append(int(row['Horizon Length']))
            data["Time (ms) per epoch"].append(float(row[tgt_name]))
            p = round(float(row['Prob'])*100)
            data['Model'].append(f'Top {p}% Hopfield')

    return pd.DataFrame(data).sort_values(by=['Model'])

def generate_plot_mts(tgt_name, tgt_plot_name):

    data = load_data_mts(tgt_name, tgt_plot_name)

    fig = plt.figure(figsize=(12,6))
    plt.title(tgt_plot_name, fontsize=20)
    # sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)  
    sns.despine(left=False,top=True, right=True, bottom=False)
    sns.lineplot(data=data, x="Horizon Length", y="Time (ms) per epoch", hue="Model", style="Model", alpha=0.8,marker="o", errorbar=None,  markersize=5, linewidth=3)

    plt.xlabel("Memory Size",fontsize=20)
    plt.ylabel(tgt_plot_name,fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(fontsize=20, prop = {"size": 20})
    fig.tight_layout()
    plt.savefig(f"{tgt_plot_name}_mts.png")
    plt.clf()

generate_plot_mts("", "")