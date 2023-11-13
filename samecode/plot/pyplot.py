import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# from judge.experimental.survival import PlotSurvival
# from judge.dataset.vendors.GuardantHealth.OMNI import format_OMNI_report
# from judge.experimental.survival import forestplot

from ..fonts import set_font
from scipy import stats

def subplots(**kwargs):
    '''
    axs = subplots(rows=1, cols=4, w=8, h=4)

    sns.boxplot(y=performance.loc[1].scalars, ax=axs[0])
    sns.boxplot(y=performance.loc[7].scalars, ax=axs[1])

    [ax.set_ylim([0.7, 1]) for ax in axs];
    '''
    
    rows = kwargs.get("rows", 1) 
    cols = kwargs.get("cols", 2) 
    w = kwargs.get("w", 8)
    h = kwargs.get("h", 4)

    return_f = kwargs.get("return_f", False)
    set_font(kwargs.get('font', 'arial'))
    plt.rcParams['figure.facecolor'] = kwargs.get('facecolor', 'white')

    f = plt.figure(constrained_layout=True,  figsize=(w, h))
    gs = f.add_gridspec(rows, cols)
    axs = []
    for i in range(rows):
        for j in range(cols):
            ax_ = f.add_subplot(gs[i,j])
            # ax_.set_facecolor((1, 1, 1, 0))
            axs.append( ax_ )

    if return_f:
        return f, axs
    else:
        return axs

def clear_plot(ax, **kwargs):
    ax.set_title('')
    ax.set_ylabel('')
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis = "y", which = "both", left = False, right = False)
    ax.tick_params(axis = "x", which = "both", left = False, right = False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

def skdeplot(df, x, y, ax, **kwargs):

    '''
    Usage:
    ---------------
    skdeplot(
        beta_dist, 
        x='β', 
        y='group', 
        ax=axs[0], 
        offset=0.9,
        order = ['Original Scores', "Perturbing: " + var2, "Perturbing: " + var1, 'Perturbing: Both'],
        alpha=1
    )
    '''
    offset2 = kwargs.get('offset', 0)
    points = kwargs.get('points', 100)
    interval = np.linspace(df[x].min() - offset2, df[x].max() + offset2, points, endpoint=True)
    offset = kwargs.get('offset', 0.01)
    colors = kwargs.get('colors', sns.color_palette('Paired', 100, desat=0.4))
    alpha = kwargs.get('alpha', 0.5)
    
    
    groups = kwargs.get('order', set(df[y]))
    textures = kwargs.get('textures', ['']*len(groups))
    edgecolors = kwargs.get('edgecolors', ['black']*len(groups))

    of_i = offset
    for ig, group in enumerate(groups):

        kernel = stats.gaussian_kde(df[df[y] == group][x])
        yval = kernel(interval)
        yval = (yval - np.min(yval)) / (np.max(yval) - np.min(yval))
        
        sns.lineplot(y=yval-offset, x=interval, ax=ax, color='black', zorder=ig)
        ax.fill_between(interval, np.min(yval)-offset, yval-offset, facecolor=colors[ig], zorder=ig, alpha=alpha, label=group, hatch=textures[ig], edgecolor=edgecolors[ig])
        
        
        
        offset += of_i
        
    ax.set_xlabel(x)
    ax.set_ylabel('')
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis = "y", which = "both", left = False, right = False)
    ax.set_yticklabels([])
    
    ax.legend(bbox_to_anchor=(1.0, 1.0))

def dibarplot(x, y, legend='', color='black', title='', ylim=[], figsize=(10, 4), x_label=''):
    '''
    Input is a dataframe with the required data.
    
    ----
    Example: 
    
    dibarplot(
        x=['orange', 'abd'], 
        y=[100, 3],
        legend='test',
        ylim=[0, 2, 29],
        figsize=(4, 4)
    )

    '''
    
    f = plt.figure(constrained_layout=True, figsize=figsize)
    gs = f.add_gridspec(2, 1)

    ax=f.add_subplot(gs[0, 0])
    ax2=f.add_subplot(gs[1, 0])

    sns.barplot(x=x, y=y, fill=True, ax=ax, label=legend, color=color, alpha=0.8)
    sns.barplot(x=x, y=y, fill=True, ax=ax2, label=legend, color=color, alpha=0.8)
    

    ylim_ = ax.get_ylim()
    ax.set_ylim(ylim[2], ylim_[1])  # outliers only
    ax2.set_ylim(ylim[0], ylim[1])  # most of the data

    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.axes.xaxis.set_ticks([])

    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    ax.legend()
    ax2.set_xlabel(x_label);
    ax.set_xlabel('');
    ax.set_ylabel('');
    ax2.set_ylabel('');

    d = .005  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    # ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    # ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs);

    ax.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False);
    
    ax.set_title(title, weight='bold')

def variable_change_percentil(data, percentil, ref_x, ref_y, event='event', OS='OS', iterations=10, axs=[], f=[], ref_x_lab='Target Population'):

    data.loc[:, 'point_diff'] = data[ref_x] - data[ref_y]
    
    min_cut_x = np.quantile(np.array(data[ref_x]), percentil)
    min_cut_y = np.quantile(np.array(data[ref_y]), percentil)
    
    
    data['point_diff_cut'] = (data[ref_x] >= min_cut_x) & (data[ref_y] <= min_cut_y)
    
    sns.stripplot(data=data[[
            ref_x, ref_y
        ]], linewidth=1, ax=axs[0], orient='v', size=6);
    
    sns.violinplot(data=data[[
            ref_x, ref_y
        ]], linewidth=1, ax=axs[0], orient='v', size=6);
    
    axs[0].plot( [-0.4, 0.2], [min_cut_x, min_cut_x], color='black');
    axs[0].text(-0.4, min_cut_x+0.4, "{:.2f}".format(min_cut_x))
    axs[0].set_xlabel('Metrics')
    
    axs[0].plot( [1-0.4, 1+0.2], [min_cut_y, min_cut_y], color='black');
    axs[0].text(1-0.4, min_cut_y+0.6, "{:.2f}".format(min_cut_y))
    axs[0].set_xlabel('Metrics')
    
    for ix, item in data[data.point_diff_cut].iterrows():
        axs[0].plot(
            [0, 1], 
            [item[ref_x], item[ref_y]],
            color='black', alpha=0.2
        )
    
    # Distribution
    sns.kdeplot(
        data = data[(data[ref_x] >= min_cut_x) & data.point_diff_cut], 
        x = OS,
        ax = axs[1],
        label='Target Population'.format(percentil=int(100*percentil), t1=ref_x, t2=ref_y), 
        fill=True,
        color='black',
        alpha=0.05
    );

    sns.kdeplot(
        data = data[(data[ref_x] >= min_cut_x) & (data.point_diff_cut == False)], 
        x = OS,
        ax = axs[1],
        label = 'Hi Population'.format(percentil=int(100*percentil), t1=ref_x),
        fill=True, 
        color='#5151bd',
        alpha=0.05
    )
    
    axs[1].legend();

def variable_change_percentil_summary(data, ref_x, ref_y, event='event', OS='OS', axs=[], ref_x_lab='TG'):
    
    percentiles = [i/100 for i in range(10, 91, 5)]
    perf = []
    
    for percentile in percentiles:
    
        min_cut_x = np.quantile(np.array(data[ref_x]), percentile)
        min_cut_y = np.quantile(np.array(data[ref_y]), percentile)


        target_group = (data[ref_x] >= min_cut_x) & (data[ref_y] <= min_cut_y)


        # Survival plot DeltaP vs < upper quartile
        ref_y_lab = 'High'
        
        svr = PlotSurvival(data, time=OS, censor=event)
        
        ix = target_group
        noix =   (data[ref_x] >= min_cut_x) & ~target_group
        
        svr.add(ix, '{}'.format(ref_x_lab))
        svr.add(noix, '{}'.format(ref_y_lab))

        ref = '{}'.format(ref_y_lab)
        tar = '{}'.format(ref_x_lab)
        svr.plot(axs, ref = ref, targets=[ref, tar], colors=['black', '#5151bd'], line_styles=['-', '-'], table=False, plot=False, legend=False)
        
        perf.append({
            "Experiment": ref_x_lab, 
            "Cutoff": percentile, 
            "Condition": '>',
            "Reference": ref, 
            'Target': tar, 
            'HR': svr.kmfs[tar][3][0], 
            'Pvalue': svr.kmfs[tar][3][1], 
            'CI_Low': svr.kmfs[tar][3][2], 
            'CI_High': svr.kmfs[tar][3][3], 
            'Reference_population': int(svr.counts[0.0][ref]),
            'Target_population': int(svr.counts[0.0][tar]),
        })
        
    
    perf = pd.DataFrame(perf)    
    forestplot(perf, axs=axs, orderby='Cutoff')
    
    return perf

def variable_survival_change(data, var, ref, treat, arm, event='event', OS='OS', axs=[], run_name='R', low_p=25, high_p=91):
    
    percentiles = [i/100 for i in range(low_p, high_p, 5)]
    perf = []
    
    for percentile in percentiles:
        svr = PlotSurvival(data, time=OS, censor=event)
        for treatment in [ref, treat]:
            min_cut_tx = np.quantile(np.array(data[data[arm] == treatment][var]), percentile)    
        
            ix = (data[var] >= min_cut_tx) & (data[arm] == treatment)
            noix =   (data[var] < min_cut_tx) & (data[arm] == treatment)
        
            svr.add(ix, '{} (H)'.format(treatment))
            svr.add(noix, '{} (L)'.format(treatment))

        
        ref_ = '{} (H)'.format(ref)
        tar  = '{} (H)'.format(treat)
        svr.plot(axs, ref = ref_, targets=[ref_, tar], colors=['black', '#5151bd'], line_styles=['-', '-'], table=False, plot=False, legend=False)
        
        perf.append({
            "Experiment": run_name, 
            "Cutoff": percentile, 
            "Condition": '>',
            "Reference": ref_, 
            'Target': tar, 
            'HR': svr.kmfs[tar][3][0], 
            'Pvalue': svr.kmfs[tar][3][1], 
            'CI_Low': svr.kmfs[tar][3][2], 
            'CI_High': svr.kmfs[tar][3][3], 
            'Reference_population': int(svr.counts[0.0][ref_]),
            'Target_population': int(svr.counts[0.0][tar]),
        })
        
    
    perf = pd.DataFrame(perf)    
    forestplot(perf, axs=axs, orderby='Cutoff')
    
    return perf

def variable_survival_change_2_metrics(data, var_x, var_y, treatment, arm, event='event', OS='OS', axs=[], run_name='R', low_p=25, high_p=91):
    
    percentiles = [i/100 for i in range(low_p, high_p, 5)]
    perf = []
    
    for percentile in percentiles:
        svr = PlotSurvival(data, time=OS, censor=event)
        
        min_cut_tx = np.quantile(np.array(data[data[arm] == treatment][var_x]), percentile)   
        min_cut_ty = np.quantile(np.array(data[data[arm] == treatment][var_y]), percentile)    
    
        ix = (data[var_x] >= min_cut_tx) & (data[arm] == treatment)
        noix =   (data[var_y] >= min_cut_ty) & (data[arm] == treatment)
    
        svr.add(ix, '{} (H)'.format(var_x))
        svr.add(noix, '{} (H)'.format(var_y))

        
        ref_ = '{} (H)'.format(var_x)
        tar  = '{} (H)'.format(var_y)
        svr.plot(axs, ref = ref_, targets=[ref_, tar], colors=['black', '#5151bd'], line_styles=['-', '-'], table=False, plot=False, legend=False)
        
        perf.append({
            "Experiment": run_name, 
            "Cutoff": percentile, 
            "Condition": '>',
            "Reference": ref_, 
            'Target': tar, 
            'HR': svr.kmfs[tar][3][0], 
            'Pvalue': svr.kmfs[tar][3][1], 
            'CI_Low': svr.kmfs[tar][3][2], 
            'CI_High': svr.kmfs[tar][3][3], 
            'Reference_population': int(svr.counts[0.0][ref_]),
            'Target_population': int(svr.counts[0.0][tar]),
        })
        
    
    perf = pd.DataFrame(perf)    
    forestplot(perf, axs=axs, orderby='Cutoff')
    
    return perf

def regplot_stats(mutation_counts, x=[], y=[], ax=[], filter=[], label_reg=False, color='black'):
    slope, intercept, r_value, p_value, std_err = stats.linregress(mutation_counts.fillna(0)[x], mutation_counts.fillna(0)[y])
    
    r_value = np.corrcoef(mutation_counts.fillna(0)[x], mutation_counts.fillna(0)[y])[1, 0]
    r_value, p_value = stats.spearmanr(mutation_counts.fillna(0)[x], mutation_counts.fillna(0)[y])
    
    label = "{} = {:.2f}*{} + {:.2f}\nSpearman = {:.2f}\np = {:.2e}".format(y, slope, x, intercept, r_value, p_value) if label_reg == True else "Spearman = {:.2f}\np = {:.2e}".format(r_value, p_value)

    sns.regplot(
        y=y, 
        x=x, 
        data = mutation_counts.fillna(0),
        ax=ax,
        label = label,
        color=color
    );
    
def mutation_plot(data, classifier, axs=[]):
    data['mut'] = data.Reference_Allele + '>' + data.Tumor_Allele_1

    sns.barplot(
        data = data.groupby(['mut', classifier], as_index=False).count()[['mut', classifier, 'counter']] ,
        x='mut',
        y='counter',
        hue= classifier,   
        order=['C>A', 'C>G', 'C>T',  'T>A', 'T>C', 'T>G'],
        ax = axs
    );

def variable_change_interval(data, axs=[], x_lo=5, x_hi=10, var_x='', var_y='', population=0.25, OS='OS', event='event'):
    
    ''' From two surrogate metrics x and y, take an interval on x [x_lo, x_hi]. 
    Then, sort the difference per patient between the two distributions and '''
    
    data.loc[:, 'point_diff'] = data[var_x] - data[var_y]
    
    data['point_diff_cut'] = (data[var_x] >= x_lo) & (data[var_x] <= x_hi)
    
    subset = data[data.point_diff_cut].reset_index(drop=True)
    subset.sort_values('point_diff', ascending=False, inplace=True)
    
    subset['groups'] = [True if i <= subset.shape[0]*population else False for i in range(subset.shape[0])]
    subset['negative_groups'] = [True if i >= subset.shape[0]*(1-population) else False for i in range(subset.shape[0])]
    
    sns.stripplot(data=data[[
            var_x, var_y
        ]], linewidth=1, ax=axs[0], orient='v', size=6);
    
    sns.violinplot(data=data[[
            var_x, var_y
        ]], linewidth=1, ax=axs[0], orient='v', size=6);
    
    axs[0].plot( [-0.4, 0.2], [x_lo, x_lo], color='black');
    axs[0].text(-0.4, x_lo+1.6, "{:.2f}".format(x_lo))
    axs[0].set_xlabel('Metrics')
    
    axs[0].plot( [-0.4, +0.2], [x_hi, x_hi], color='black');
    axs[0].text(-0.4, x_hi+1.6, "{:.2f}".format(x_hi))
    axs[0].set_xlabel('Metrics')
    
    for ix, item in subset.iterrows():
        if not item.groups: continue
            
        axs[0].plot(
            [0, 1], 
            [item[var_x], item[var_y]],
            color='black', alpha=0.2
        )
    
    # Survival
    svr = PlotSurvival(subset, time=OS, censor=event)
    
    ix = np.array(subset.groups)
    noix = np.array(subset.negative_groups)

    svr.add(ix, '{}'.format('Flip'))
    svr.add(noix, '{}'.format('No Flip'))

    ref = '{}'.format('Flip')
    tar = '{}'.format('No Flip')
    svr.plot(axs[1], ref = ref, targets=[ref, tar], colors=['black', '#5151bd'], line_styles=['-', '-'], table=True, plot=True, legend=True)
    
    return subset 