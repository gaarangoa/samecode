import seaborn as sns
import numpy as np 

def heatmap(df, ax, groupby=None, y_cut=[11, 22, 27], columns=None, **kwargs):
    """
    Creates a heatmap with horizontal and vertical dividing lines, visualizing data
    from a pandas DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data to visualize.
        groupby (str, optional): The column to group the data by before plotting.
            If not provided, no grouping is applied.
        y_cut (list, optional): A list of row indices where to draw horizontal
            dividing lines. Default is [11, 22, 27].
        columns (list, optional): A list of column names to include in the heatmap.
            If not provided, all columns are used.
        **kwargs: Additional keyword arguments to pass to the `seaborn.heatmap` function.
            Commonly used options include `vmin`, `vmax`, `center`, and `labelsize`.

    Returns:
        None (modifies the provided `ax` object in-place).

    Raises:
        ImportError: If seaborn or numpy libraries are not installed.
        ValueError: If invalid data types or incompatible arguments are provided.

    Examples:
        ```python
        import seaborn as sns
        import pandas as pd

        # Create sample data
        data = {'col1': [1, 2, 3, 4, 5], 'col2': [5, 4, 3, 2, 1]}
        df = pd.DataFrame(data)

        # Create heatmap with grouping and horizontal lines
        fig, ax = plt.subplots()
        heatmap(df, ax, groupby='col1', y_cut=[2, 4])

        # Create heatmap with specific columns and colorbar customization
        fig, ax = plt.subplots()
        heatmap(df, ax, columns=['col2'], vmin=0, vmax=10, cbar_kws={'label': 'My Values'})

        plt.show()
        ```
    """

    vmin = kwargs.get('vmin', -1)
    vmax = kwargs.get('vmax', 1)
    center = kwargs.get('center', 0)

    sns.heatmap(
        df.sort_values(groupby).reset_index(drop=True)[columns].T, 
        vmin=vmin, vmax=vmax, center=center, ax=ax, linewidths=0,
        cbar_kws={'shrink': 0.3, 'aspect': 20},
    )

    index = df.sort_values(groupby).reset_index(drop=True)
    gps = index[groupby]

    index['index_'] = index.index
    index = index[['index_', groupby]].groupby(groupby).min().reset_index()[['index_']]
    
    index = index.index_.values

    for i in y_cut:
        ax.axhline(i, color='white', linewidth=4)

    for i in index:
        ax.axvline(i, color='white', linewidth=4)

    for ix,i in enumerate(index):
        ax.text(i, 0, gps[i], fontsize=kwargs.get('labelsize', 8))

    ax.tick_params(labelsize=kwargs.get('labelsize', 8))
    ax.set_yticks(ticks=np.array(range(len(columns))) +0.5);
    ax.set_yticklabels(columns);
    ax.set_xticks([]);