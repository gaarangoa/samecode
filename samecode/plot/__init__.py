import seaborn as sns
import numpy as np 

def plot_arrows(data, **kwargs):
    x1 = kwargs.get('x1', None)
    y1 = kwargs.get('y1', None)
    x2 = kwargs.get('x2', None)
    y2 = kwargs.get('y2', None)
    ax = kwargs.get('ax', None)
    
    for ix, i in data.iterrows():
        ax.plot(i[x1], i[y1], 'ro', color='white', alpha=0)
        ax.plot(i[x2], i[y2], 'ro', color='white', alpha=0)

        ax.annotate('', xy=(i[x2], i[y2]), xytext=(i[x1], i[y1]), arrowprops= {'arrowstyle': '->'})



def volcano_plot(data, x, y, top=10, ax=[], label='gene', **kwargs):
    
    xpos = kwargs.get('xpos', -0.1)
    ypos_rate = kwargs.get('ypos_rate', 0)
    
    data['log_'] = -np.log10(data[y])
    sns.scatterplot(
        data=data, 
        x=x, 
        y='log_',
        ax=ax
    );


    sdf = data.sort_values(y).reset_index(drop=True)
    
    max_x = np.max(sdf[x])
    max_y = np.max(sdf.log_)
    min_x = np.min(sdf[x])
    min_y = np.min(sdf.log_)    
    
    offset=kwargs.get('offset', 1)

    for ix,i in sdf[:top].iterrows():
        annotation = ax.annotate(
            i[label], 
            ( i[x], i.log_), 
            xytext=(xpos, max_y-ix*ypos_rate), 
            # xytext=(i[x], i.log_), 
            # xycoords='axes fraction',
            bbox=dict(fc='white', lw=0, alpha=0.5),
            arrowprops=dict(facecolor='black', arrowstyle="-", connectionstyle="arc3,rad=0.0"),
        )

    ax.set_xlabel(kwargs.get('xlabel', 'Fold Change'))
    ax.set_ylabel(kwargs.get('ylabel', '-log10(P-val)'))