import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from collections import Counter
from sklearn.metrics import confusion_matrix

from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines import CoxPHFitter
import six 
import seaborn as sns
from lifelines.utils import median_survival_times
import itertools

import logging
import sys

logging.basicConfig(format='%(levelname)s\t%(asctime)s\t%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

from lifelines import KaplanMeierFitter
import seaborn as sns
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from ..plot.pyplot import subplots

from matplotlib import pyplot as plt
# plt.rcParams['font.family'] = 'monospace'
# plt.rcParams['font.family'] = 'sans-serif'

def fix_string(s, v, m):
    s = s.split('\t')
    sx = ''
    for ix, i in enumerate(s):
        sx += i + ' '*int(1.*(m[ix] - v[ix])) + ' '
    
    return sx

def set_template(ax, **kwargs):

    template_color = kwargs.get('template_color', 'black')
    xy_font_size = kwargs.get('xy_font_size', 10)
    title_size = kwargs.get('title_size', 12)
    title_weight = kwargs.get('title_weight', 120)


    ax.set_ylabel(kwargs.get('ylabel', ''), weight='bold', fontsize=xy_font_size, color=template_color)
    ax.set_xlabel(kwargs.get('xlabel', ''), weight='bold', fontsize=xy_font_size, color=template_color)
    ax.tick_params(axis='x', colors=template_color)
    ax.tick_params(axis='y', colors=template_color)
    
    ax.xaxis.set_tick_params(labelsize=xy_font_size-1)
    ax.yaxis.set_tick_params(labelsize=xy_font_size-1)
    
    ax.set_title(kwargs.get('title', ''), fontsize=title_size, color=template_color, weight=title_weight)
    
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(1.0)
        ax.spines[axis].set_color(template_color)

    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0.0)

def compute_char_positions_ascii(font_size=12, xycoords=True):
    fig, ax = plt.subplots()
    ax.axis('off')

    # Define the characters for which we want to compute positions
    # This example uses lowercase letters; expand the set as needed
    chars = """ abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    
    positions = {}

    for char in chars:
        # Add a text annotation for the character
        text_obj = ax.text(0, 0, char, transform=ax.transAxes, fontsize=font_size)
        
        # Draw the figure to make the renderer available
        fig.canvas.draw()
        
        # Get bounding box of the text in axes coordinates
        if xycoords:
            bbox = text_obj.get_window_extent().transformed(ax.transAxes.inverted())
        else:
            bbox = text_obj.get_window_extent()

        # Store start and end x-coordinates
        start_x = bbox.x0
        end_x = bbox.x1
        
        positions[char] = end_x
        
        # Remove the text object to plot the next character
        text_obj.remove()

    plt.close(fig)  # Close the figure
    return positions

def get_end_position(label, positions):
    return sum([positions.get(i, 0) for i in label])

class KMPlot():
    def __init__(self, data, time, event, label, **kwargs):
        '''
        Example: 

        axs = subplots(cols=1, rows=1, w=6, h=4)
        KMPlot(data, time=time, event=event, label=['bin_risk', 'Treatment']).plot(
            labels = ['GP_{}'.format('IO'), 'GN_{}'.format('IO'), 'GP_{}'.format('Chemo'), 'GN_{}'.format('Chemo')],
            ax=axs[1],
            comparisons=[
                ['GP_{}'.format('IO'), 'GP_{}'.format('Chemo'), 'GP(IO vs Chemo): '], 
                ['GN_{}'.format('IO'), 'GN_{}'.format('Chemo'), 'GN(IO vs Chemo): '], 
                ['GP_{}'.format('IO'), 'GN_{}'.format('IO'), 'IO(GP vs GN): '], 
                ['GP_{}'.format('Chemo'), 'GN_{}'.format('Chemo'), 'Chemo(GP vs GN): ']
            ],
            title='PFS - IO vs Chemo',
        );

        '''
        
        self.colors = {}
        self.label__ = label
        self.plot_type = kwargs.get('plot_type', 'survival')
        
        self.fit(data, time, event, label, **kwargs)
    
    def extract_hr(self, to_compare, **kwargs):

        xcompare = [[('__label__', '__label__'), "\tHR\t(95% CI)\tP value", '']]
        xinfo = [[len(i) for i in "\tHR\t(95% CI)\tP value".split('\t')]]
        xinfo_space = []

        if to_compare == []: return xcompare, xinfo, xinfo_space
        self.comparisons_table = []
        for cx, item in enumerate(to_compare):
            
            if len(item) == 3:
                [tar, ref, hr_label] = item
            else: 
                [tar, ref] = item
                hr_label = '{} - {}: '.format(tar, ref)

            x = self.data[self.data.__label__.isin([tar, ref])][[self.time, self.event, '__label__']].copy().reset_index(drop=True)
            labs__ = {ref: 0, tar: 1}
            numerical_label = [labs__[i] for i in x.__label__]
            x['__label__'] = numerical_label

            cph = CoxPHFitter().fit(x, duration_col = self.time, event_col = self.event) 
            cph = cph.summary[['exp(coef)', 'p', 'exp(coef) lower 95%', 'exp(coef) upper 95%']].reset_index().to_dict()
            cph = {
                "HR": cph.get('exp(coef)').get(0),
                "HR_lo": cph.get('exp(coef) lower 95%').get(0),
                "HR_hi": cph.get('exp(coef) upper 95%').get(0),
                "P": cph.get('p').get(0),
            }
            
            # color for HR 
            hr_color = kwargs.get('hr_color', self.colors.get(tar, 'black'))
            
            # xinfo_ = '{}\tHR={:.2f}\t(CI 95%: {:.2f} - {:.2f})\tP-value={:.2e}'.format(hr_label, cph.get('HR'), cph.get('HR_lo'), cph.get('HR_hi'), cph.get('P'))
            xinfo_list = {"Treatment": tar, "Control": ref,  "HR": cph.get('HR'), "HR_lo": cph.get('HR_lo'), "HR_hi": cph.get('HR_hi'), "pval": cph.get('P')} #'{}\t{:.2f}\t({:.2f}-{:.2f})\t{:.2e}'.format(hr_label, cph.get('HR'), cph.get('HR_lo'), cph.get('HR_hi'), cph.get('P'))
            xinfo_string = '{}\t{:.2f}\t({:.2f}-{:.2f})\t{:.2e}'.format(hr_label, cph.get('HR'), cph.get('HR_lo'), cph.get('HR_hi'), cph.get('P'))

            xinfo.append([len(i) for i in xinfo_string.split('\t')])
            xinfo_space.append(
                [get_end_position("_"+ii+"_._", kwargs.get('char_positions', {}) ) for ii in xinfo_string.split('\t')]
            )

            xcompare.append([
                (tar, ref), xinfo_string, xinfo_list
            ])

            self.comparisons_table.append(xinfo_list)

        self.hr_table = pd.DataFrame(self.comparisons_table)

        try:
            kpi = self.medians_table.pivot_table(index=self.label__[0], columns=self.label__[1], values='counts')

            counts = []
            for ix, i in self.hr_table.iterrows():
                t, b = i.Treatment.split('_')
                k1 = kpi.loc[t, b]
                
                t, b = i.Control.split('_')
                k2 = kpi.loc[t, b]
                
                counts.append([k1, k2])
                
            self.hr_table[['Treatment_N', 'Control_N']] = counts
        except: 
            pass
            
        return xcompare, xinfo, xinfo_space
    
    def plot(self, labels=None, **kwargs):
        '''
        Plot the Kaplan-Meier survival curves.

        Parameters:
            - labels (list, optional): Labels to plot. Default is None.
            - display_labels (list, optional): Labels to display. Default is None.
            - ax (matplotlib.axes.Axes, optional): Matplotlib axes object. Default is False.
            - colors (list or str, optional): Colors for each label. Default is determined by sns.color_palette.
            - linestyle (list, optional): Line styles for each label. Default is ['-']*len(plot_labels).
            - xy_font_size (int, optional): Font size for x and y axis labels. Default is 12.
            - label_height_adj (float, optional): Adjustment for space between labels on the y-axis. Default is 0.05.
            - template_color (str, optional): Color for template. Default is 'black'.
            - comparisons (list, optional): List of comparisons between two curves. Default is [].
            - title_weight (str, optional): Font weight for title. Default is 'normal'.
            - title_size (int, optional): Font size for title. Default is 14.
            - hr_color (str, optional): Color for hazard ratio layer. Default is determined by self.colors.
            - x_legend (float, optional): X-coordinate for legend. Default is 0.15.
            - y_legend (float, optional): Y-coordinate for legend. Default is 0.95.
            - legend_font_size (int, optional): Font size for legend. Default is 10.
            - show_censor (bool, optional): Whether to show censor. Default is True.
            - ylabel (str, optional): Label for y-axis. Default is 'Survival Probability'.
            - xlabel (str, optional): Label for x-axis. Default is 'Timeline'.
            - ci_show (bool, optional): Whether to show confidence interval. Default is False.
            - label_weight (str, optional): Font weight for labels. Default is 'bold'.
            - x_hr_legend (float, optional): X-coordinate for hazard ratio legend. Default is 0.0.
            - y_hr_legend (float, optional): Y-coordinate for hazard ratio legend. Default is -0.3.
            - hr_font_size (int, optional): Font size for hazard ratio legend. Default is 10.
            - saturation (float, optional): Saturation level for colors. Default is 1.0.
            - linewidth (float, optional): Line width for the curves. Default is 1.5.
            - palette (str, optional): Palette for color selection. Default is 'Paired'.
            - show_cesor (bool, optional): Whether to show censor. Default is True.

        Returns:
            - ax: Matplotlib axes object.
        
        '''

        
        label = labels

        if label == None:
            plot_labels = self.labels
        elif type(label) == list:
            plot_labels = label
        else:
            plot_labels = [label]
        
        display_labels = kwargs.get('display_labels', None)
        ax = kwargs.get('ax', False)
        if ax == False:
            ax = subplots(cols=1, rows=1, w=6, h=4)[0]
        
        colors = kwargs.get('colors', sns.color_palette(kwargs.get('palette', 'Paired'), 100, desat=kwargs.get('saturation', 1)))
        linestyle = kwargs.get('linestyle', ['-']*len(plot_labels))
        xy_font_size = kwargs.get('xy_font_size', 12)
        label_height_adj = kwargs.get('label_height_adj', 0.05)
        hr_height_adj = kwargs.get('hr_height_adj', 0.05)
        template_color = kwargs.get('template_color', 'black')
        to_compare = kwargs.get('comparisons', [])
        title_weight = kwargs.get('title_weight', 'normal')
        title_size = kwargs.get('title_size', 14)
        
        if type(colors) == str:
            colors = [colors]

        label_max_l = [self.label_names_size['__label__']]
        for lx, label_ in enumerate(plot_labels):
            label_max_l.append(self.label_names_size[label_])
            self.colors[label_] = colors[lx]

            if self.plot_type == 'survival':
                self.kmfs[label_].plot(
                    ci_show=kwargs.get('ci_show', False), 
                    show_censors=kwargs.get("show_censor", True),
                    color = colors[lx],
                    linestyle = linestyle[lx],
                    linewidth = kwargs.get('linewidth', 1.5),
                    ax=ax
                )
            if self.plot_type == 'cumulative_density':
                self.kmfs[label_].plot_cumulative_density(
                    ci_show=kwargs.get('ci_show', False), 
                    show_censors=kwargs.get("show_censor", True),
                    color = colors[lx],
                    linestyle = linestyle[lx],
                    linewidth = kwargs.get('linewidth', 1.5),
                    ax=ax
                )
            
            # median survival time
            ax.axvline(self.kmfs[label_].median_survival_time_, 0, 0.5, ls = '--', color = self.colors[label_], lw = 1)
            ax.plot((0, self.kmfs[label_].median_survival_time_), (0.5, 0.5),  ls = '--', color = '#a19595', lw = 1)
            sns.scatterplot(x=[self.kmfs[label_].median_survival_time_], y=[0.5], ax=ax, s=50, color='white', edgecolor=self.colors[label_], alpha=1.0)
            
        
        self.colors['__label__'] = 'black'

        # plt.rcParams['font.family'] = kwargs.get('font_family_labels', 'Roboto Mono for Powerline')
        x_legend=kwargs.get('x_legend', 0.15)
        y_legend=kwargs.get('y_legend', 0.95)
        legend_font_size=kwargs.get('legend_font_size', 10)

        x_hr_legend=kwargs.get('x_hr_legend', 0)
        y_hr_legend=kwargs.get('y_hr_legend', -0.3)
        hr_font_size=kwargs.get('hr_font_size', 10)

        char_positions = compute_char_positions_ascii(font_size=max([legend_font_size, hr_font_size]))

        pos = []
        for label in [i.split('\t') for i in self.label_names_list.values()]:
            posi = []
            for i in label:
                posi.append(get_end_position("_._" + i + '__', char_positions))
            pos.append(posi)

        max_pos = np.max(np.array(pos, dtype=object), axis=0)
        max_pos = np.cumsum(max_pos)
        max_pos = [0] + list(max_pos)

        if kwargs.get("legend", True):
            label_max_l = np.array(label_max_l).max(axis=0)
            for lx, label_ in enumerate(['__label__'] + plot_labels):                
                # vrx = fix_string(self.label_names_list[label_], self.label_names_size[label_], label_max_l)

                for lix, vx in enumerate(self.label_names_list[label_].split('\t')):
                    ax.annotate(
                        vx, 
                        xy=(max_pos[lix] + x_legend, y_legend -lx*label_height_adj), 
                        xycoords='axes fraction', 
                        xytext=(max_pos[lix] + x_legend, y_legend -lx*label_height_adj),
                        weight=kwargs.get("label_weight", 'bold'),
                        size=legend_font_size, 
                        color=self.colors[label_],
                        bbox=dict(fc='white', lw=0, alpha=0.9, boxstyle='round,pad=0.1')
                    )

        # print(max_pos)
        # rect=mpatches.Rectangle((0, 0), 1,1, 
        #                 fill = True,
        #                 color = "purple",
        #                 linewidth = 2, zorder=0)

        # ax.add_patch(rect)

        # Cox PH Fitters for HR estimation

        xcompare, xinfo, xinfo_space = self.extract_hr(to_compare, char_positions=char_positions, **kwargs)

        if xinfo_space != []:
            max_pos_hr = np.max(np.array(xinfo_space, dtype=object), axis=0)
            max_pos_hr = np.cumsum(max_pos_hr)
            max_pos_hr = [0] + list(max_pos_hr)

        

        max_values = np.array(xinfo)
        
        for ix, [k, v, _] in enumerate(xcompare):
            
            tar, ref = k 
            if len(xcompare) == 1: continue
            hr_color = kwargs.get('hr_color', self.colors[tar])
            
            # vx = fix_string(v, max_values[ix], max_values.max(axis=0))
            for lix, vx in enumerate(v.split('\t')):
                ax.annotate(
                    vx, 
                    xy=(x_hr_legend + max_pos_hr[lix], y_hr_legend - ix*hr_height_adj), 
                    xycoords='axes fraction', 
                    xytext=(x_hr_legend + max_pos_hr[lix], y_hr_legend - ix*hr_height_adj),
                    weight='bold', 
                    size=hr_font_size, 
                    color=hr_color,
                    bbox=dict(fc='white', lw=0, alpha=0.9, boxstyle='round,pad=0.1')
                )

        # plt.rcParams['font.family'] = kwargs.get('font_family', '')   
        
        
        ax.set_ylim([-0.03, 1])
        ax.set_ylabel(kwargs.get('ylabel', 'Survival Probability'), weight='bold', fontsize=xy_font_size, color=template_color)
        ax.set_xlabel(kwargs.get('xlabel', 'Timeline'), weight='bold', fontsize=xy_font_size, color=template_color)
        ax.tick_params(axis='x', colors=template_color)
        ax.tick_params(axis='y', colors=template_color)
        
        ax.xaxis.set_tick_params(labelsize=xy_font_size-1)
        ax.yaxis.set_tick_params(labelsize=xy_font_size-1)
        
        ax.set_title(kwargs.get('title', ''), fontsize=title_size, color=template_color, weight=title_weight)

        ax.set_yticks(ax.get_yticks()[-6:])
        
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(1.0)
            ax.spines[axis].set_color(template_color)

        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0.0)
            
        ax.get_legend().remove()
        
    def fit(self, data, time, event, label, **kwargs):
        
        self.time = time
        self.event = event

        label_column = label

        data = data.copy()
        if type(label) == str:
            label = [label]
            data['__label__'] = data[label].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        else:
            data['__label__'] = data[label].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

        kmfs = {}
        
        self.labels = sorted(list(set(data.__label__)))
        self.counts = Counter(data.__label__)
        self.label_names = {}
        self.label_names_list = {}
        self.label_names_size = {}
        self.survival_medians = []
        for label in self.labels:
            kmf = KaplanMeierFitter()
            ix = data.__label__ == label
            kmfs[label] = kmf.fit(data[ix][time], data[ix][event], label='{}'.format( label ))
            
            # lo = list((kmfs[label].confidence_interval_ -  0.5).abs().sort_values('{}_lower_0.95'.format(label)).index)[0]
            # hi = list((kmfs[label].confidence_interval_ -  0.5).abs().sort_values('{}_upper_0.95'.format(label)).index)[0]

            cis = median_survival_times(kmfs[label].confidence_interval_)
            lo, hi = np.array(cis)[0]
            
            # self.label_names[label] = '{}: N={}; Q2={:.1f}'.format(label, self.counts[label], kmfs[label].median_survival_time_)
            # self.label_names[label] = '{}: N={}; Q2={:.2f} (CI 95% {:.2f} - {:.2f})'.format(label, self.counts[label], kmfs[label].median_survival_time_, lo, hi)
            # self.label_names_list[label] = '{}\tN={}\tQ2={:.2f} (CI 95% {:.2f} - {:.2f})'.format(label, self.counts[label], kmfs[label].median_survival_time_, lo, hi)

            self.label_names[label] = '{}: {} {:.1f} ({:.1f} - {:.1f})'.format(label, self.counts[label], kmfs[label].median_survival_time_, lo, hi)
            self.label_names_list[label] = '{}:\t{}\t{:.1f}\t({:.1f} - {:.1f})'.format(label, self.counts[label], kmfs[label].median_survival_time_, lo, hi)
            self.label_names_size[label] = [len(k) for k in self.label_names_list[label].split('\t')]

            
            label_column_values = label.split('_')
            med_ = {label_column[ix]: label_column_values[ix] for ix in range(len(label_column_values))}
            med_.update(dict(counts=self.counts[label], median_time=kmfs[label].median_survival_time_, lo_median_time=lo, hi_median_time=hi))
            

            self.survival_medians.append(
                med_
            )
        
        self.label_names['__label__'] = ['N Median (95%CI)']
        self.label_names_list['__label__'] = ' \tN\tMedian\t(95% CI)'
        self.label_names_size['__label__'] = [len(k) for k in [' ', 'N', 'Median','(95%CI)']]

        self.data = data[[time, event, '__label__']]
        self.kmfs = kmfs

        self.medians_table = pd.DataFrame(self.survival_medians)

        

class PlotSurvival:

    '''
    -----------
    Input: 
        data: 
        time: 
        censor:
    -----------
    Methods:

    -----------
    Output: 
     
    '''

    def __init__(self, data, time, censor):
        self.OS_AVAL = time
        self.OS_CNSR = censor
        self.data_o = data
        self.kmfs = {}
        
    def hazard(self):
        hrs = []
        for k in self.kmfs:
            self.kmfs[k][-1] = ['', '']
        
        for tx, target_name in enumerate(self.targets):
            dfs = []
            kmf, ix, color, hr = self.kmfs[target_name]
            
            if target_name != self.ref:
                logger.debug((self.ref, target_name))
                noix = self.kmfs[self.ref][1]

                dfs.append(pd.DataFrame({'E': self.data_o[ix][self.OS_CNSR], 'T': self.data_o[ix][self.OS_AVAL], '{} vs {}'.format(target_name, self.ref): 1}))
                dfs.append(pd.DataFrame({'E': self.data_o[noix][self.OS_CNSR], 'T': self.data_o[noix][self.OS_AVAL], '{} vs {}'.format(target_name, self.ref): 0}))
                    
                df = pd.concat(dfs)
                logger.debug(df.shape)

                cph = CoxPHFitter().fit(df, 'T', 'E')
                
                self.kmfs[target_name][-1]= np.array(cph.summary[['exp(coef)', 'p', 'exp(coef) lower 95%', 'exp(coef) upper 95%']])[0]

                self.cph = cph

                logger.debug(self.kmfs[target_name])
    
    def add(self, ix, name=''):
        kmf = KaplanMeierFitter();
        self.kmfs[name] =  [kmf.fit(self.data_o[ix][self.OS_AVAL], self.data_o[ix][self.OS_CNSR], label='{}'.format( name )), ix, '', ['', '']]
    
    def table(self, axs):
        survs = [ pd.DataFrame([{"Time": i} for i in axs.get_xticks()]) ]
        
        for f in self.plots:
            counts = []
            for tick in axs.get_xticks():
                event_table_slice = (
                                f[1].event_table.assign(at_risk=lambda x: x.at_risk - x.removed)
                                .loc[:tick, ["at_risk",]]
                                .agg({"at_risk": "min",})
                            )
                if np.isnan(event_table_slice.at_risk): 
                    continue
        
                counts.extend([{"{}".format(f[0]): int(c)} for c in event_table_slice.loc[["at_risk", ]] ])
            survs.append(pd.DataFrame(counts))

        survs = pd.concat(survs, axis=1)
        survs.Time += np.abs(np.min(survs.Time))
        survs = survs[np.sum(survs.iloc[:, 1:], axis=1) > 0]

        self.counts = np.transpose(survs)
        self.counts.columns = self.counts.iloc[0, :]
        self.counts = self.counts.iloc[1:, :]
        
    def plot(self, ax, ref='', targets=[], colors=[], line_styles='-', table=True, plot=True, legend=True, linewidth=2, legend_font_size=4, legend_weight='bold', labelspacing=0.1, table_font_size=10, label_font_size=10, bbox=[-0.0,-0.38, 1, 0.2], lbox=[0.05, 0.01]):
        
        remove_plot = False
        if not ax:
            f, ax = plt.subplots(1, 1, figsize=(4,4))
            remove_plot = True
        
        self.ref = ref
        self.targets = targets
        if not targets: self.targets = list(self.kmfs.keys())
        if not ref: self.ref = targets[0]
        
        self.hazard()
        
        linest = line_styles

        kmfs = []
        for ixx,key in enumerate(self.targets):
            kmf, ix, col, hr = self.kmfs[key]
            if colors:
                col = colors[ixx]

            if line_styles != '-':
                linest = line_styles[ixx]

            if hr[0]:
                if plot:
                    label='{} ({})'.format(key, Counter(ix)[1])
                    label2='HR: {:.2f} [{:.2f} - {:.2f}] P: {:.2e}'.format(hr[0], hr[2], hr[3], hr[1])
                    t = ax.text(0.0,(ixx*lbox[0]) + lbox[1], label2, color=col, weight=legend_weight, fontsize=legend_font_size, transform=ax.transAxes)
                    # t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))
                    
                    kmf.plot(ax=ax, ci_show=False, show_censors=True, color=col, linestyle = linest, linewidth=linewidth, label=label);
                kmfs.append([key, kmf, ix, col, hr])
            else:
                if plot:
                    label='{} ({})'.format(key, Counter(ix)[1])
                    kmf.plot(ax=ax, ci_show=False, show_censors=True, color=col, linestyle = linest, label=label, linewidth=linewidth);
                kmfs.append([key, kmf, ix, col, hr])
            
            if plot:
                ax.set_ylim([0, 1])
                ax.set_ylabel('Survival Probability', weight='bold', fontsize=label_font_size)
                ax.set_xlabel('Timeline', weight='bold', fontsize=label_font_size)
                
                for axis in ['bottom','left']:
                    ax.spines[axis].set_linewidth(0.5)
                    
                for axis in ['top','right']:
                    ax.spines[axis].set_linewidth(0.0)
            
            if legend: 
                ax.legend(loc='upper right')
                ax.legend(labelcolor='linecolor', frameon=False, labelspacing=labelspacing, prop={'size': legend_font_size, 'weight': legend_weight, 'stretch': 1000})
                
            else:
                try:
                    ax.get_legend().remove()
                except:
                    pass
        
        self.plots = kmfs
        self.table(ax)
        
        

        if table: 
            bbox=bbox
            # pd.plotting.table(ax, self.counts.iloc[:, ], loc='bottom', colLoc='right', rowLoc='right', edges='open', bbox=bbox);
            
            mpl_table = ax.table(cellText=self.counts.values.astype(int), bbox=bbox, colLabels=self.counts.columns, rowLabels=self.counts.index, edges='open')
            
            mpl_table.auto_set_font_size(False)
            mpl_table.set_fontsize(table_font_size)
            
            row_colors = ['white'] + colors
            header_columns=0
            header_color='w'
            edge_color='w'
            
            ax.set_xlim([0, np.max(list(self.counts)) + 5])
            
            for k, cell in six.iteritems(mpl_table._cells):
                
                cell.set_edgecolor(edge_color)
                if k[0] == 0 or k[1] < header_columns:
                    cell.set_text_props(weight='bold', color=row_colors[k[0]])
                else:
                    cell.set_text_props(color=row_colors[k[0]])   
        
        if remove_plot:
            plt.close()

class PrettyPlotSurvival:

    '''
    -----------
    Input: 
        data: 
        time: 
        censor:
    -----------
    Methods:

    -----------
    Output: 
     
    '''

    def __init__(self, data, time, censor):
        self.OS_AVAL = time
        self.OS_CNSR = censor
        self.data_o = data
        self.kmfs = {}
        
    def hazard(self):
        hrs = []
        for k in self.kmfs:
            self.kmfs[k][-1] = ['', '']
        
        for tx, target_name in enumerate(self.targets):
            dfs = []
            kmf, ix, color, hr = self.kmfs[target_name]
            
            if target_name != self.ref:
                logger.debug((self.ref, target_name))
                noix = self.kmfs[self.ref][1]

                dfs.append(pd.DataFrame({'E': self.data_o[ix][self.OS_CNSR], 'T': self.data_o[ix][self.OS_AVAL], '{} vs {}'.format(target_name, self.ref): 1}))
                dfs.append(pd.DataFrame({'E': self.data_o[noix][self.OS_CNSR], 'T': self.data_o[noix][self.OS_AVAL], '{} vs {}'.format(target_name, self.ref): 0}))
                    
                df = pd.concat(dfs)
                logger.debug(df.shape)

                cph = CoxPHFitter().fit(df, 'T', 'E')
                
                self.kmfs[target_name][-1]= np.array(cph.summary[['exp(coef)', 'p', 'exp(coef) lower 95%', 'exp(coef) upper 95%']])[0]

                self.cph = cph

                logger.debug(self.kmfs[target_name])
    
    def add(self, ix, name=''):
        kmf = KaplanMeierFitter();
        self.kmfs[name] =  [kmf.fit(self.data_o[ix][self.OS_AVAL], self.data_o[ix][self.OS_CNSR], label='{}'.format( name )), ix, '', ['', '']]
    
    def table(self, axs):
        survs = [ pd.DataFrame([{"Time": i} for i in axs.get_xticks()]) ]
        
        for f in self.plots:
            counts = []
            for tick in axs.get_xticks():
                event_table_slice = (
                                f[1].event_table.assign(at_risk=lambda x: x.at_risk - x.removed)
                                .loc[:tick, ["at_risk",]]
                                .agg({"at_risk": "min",})
                            )
                if np.isnan(event_table_slice.at_risk): 
                    continue
        
                counts.extend([{"{}".format(f[0]): int(c)} for c in event_table_slice.loc[["at_risk", ]] ])
            survs.append(pd.DataFrame(counts))

        survs = pd.concat(survs, axis=1)
        survs.Time += np.abs(np.min(survs.Time))
        survs = survs[np.sum(survs.iloc[:, 1:], axis=1) > 0]

        self.counts = np.transpose(survs)
        self.counts.columns = self.counts.iloc[0, :]
        self.counts = self.counts.iloc[1:, :]
        
    def plot(self, ax, ref='', targets=[], colors=[], line_styles='-', table=False, plot=True, legend=False, linewidth=2, legend_font_size=12, legend_weight='bold', labelspacing=0.1, table_font_size=10, label_font_size=10, bbox=[-0.0,-0.38, 1, 0.2], medians_offset=[0, 0], labels_offset=[0, 0, 0, 0], fill_color='white', fill_alpha=0.1):
        

        remove_plot = False
        if not ax:
            f, ax = plt.subplots(1, 1, figsize=(4,4))
            remove_plot = True
        
        self.ref = ref
        self.targets = targets
        if not targets: self.targets = list(self.kmfs.keys())
        if not ref: self.ref = targets[0]
        
        self.hazard()
        
        linest = line_styles

        kmfs = []
        for ixx,key in enumerate(self.targets):
            kmf, ix, col, hr = self.kmfs[key]
            if colors:
                col = colors[ixx]

            if line_styles != '-':
                linest = line_styles[ixx]

            if hr[0]:
                if plot:
                    label='{} (N={})'.format(key, Counter(ix)[1])
                    label2='HR={:.2f} (95% CI: {:.2f}, {:.2f}); P={:.2e}'.format(hr[0], hr[2], hr[3], hr[1])
                    # ax.text(0,(ixx*lbox[0]) + lbox[1], label2, color=col, weight=legend_weight, fontsize=legend_font_size, transform=ax.transAxes)
                    ax.text(0.0, -0.02, label2, color=col, weight=legend_weight, fontsize=legend_font_size, transform=ax.transAxes)
                    # t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))
                    
                    kmf.plot(
                        ax=ax, ci_show=False, show_censors=True, 
                        color=col, linestyle = linest, linewidth=linewidth, 
                        label=label
                    )
                    
                    # median survival time 
                    sns.scatterplot(x=[kmf.median_survival_time_], y=[0.5], ax=ax, s=100, color=col, alpha=1.0)
                    ax.text(kmf.median_survival_time_ + kmf.median_survival_time_*medians_offset[0], 0.53, '{:.1f}'.format(kmf.median_survival_time_), size=20, weight='bold', color=col)
                    ax.axvline(kmf.median_survival_time_, 0, 0.5, ls = '--', color = '#a19595', lw = 1)
                    ax.plot((0, kmf.median_survival_time_), (0.5, 0.5),  ls = '--', color = '#a19595', lw = 1)
                    
                    x_, y_ = list(kmf.survival_function_[key].items())[-1]
                    ax.text(labels_offset[0], labels_offset[1], label, weight='bold', color=col)

                kmfs.append([key, kmf, ix, col, hr])
            else:
                if plot:
                    label='{} (N={})'.format(key, Counter(ix)[1])
                    
                    kmf.plot(
                        ax=ax, ci_show=False, show_censors=True, 
                        color=col, linestyle = linest, label=label, 
                        linewidth=linewidth
                    )
                    
                    # median survival time
                    sns.scatterplot(x=[kmf.median_survival_time_], y=[0.5], ax=ax, s=100, color=col, alpha=1.0)
                    ax.text(kmf.median_survival_time_ - kmf.median_survival_time_*medians_offset[1], 0.4, '{:.1f}'.format(kmf.median_survival_time_), size=20, weight='bold', color=col)
                    ax.axvline(kmf.median_survival_time_, 0, 0.5, ls = '--', color = '#a19595', lw = 1)
                    ax.plot((0, kmf.median_survival_time_), (0.5, 0.5),  ls = '--', color = '#a19595', lw = 1)
                    
                    x_, y_ = list(kmf.survival_function_[key].items())[-1]
                    ax.text(labels_offset[2], labels_offset[3], label, weight='bold', color=col)
                                        
                kmfs.append([key, kmf, ix, col, hr])
            
            if plot:
                ax.set_ylim([0, 1])
                ax.set_ylabel('Survival Probability', weight='bold', fontsize=label_font_size, color='#7a7974')
                ax.set_xlabel('Timeline', weight='bold', fontsize=label_font_size, color='#7a7974')
                ax.tick_params(axis='x', colors='#7a7974')
                ax.tick_params(axis='y', colors='#7a7974')
                
                for axis in ['bottom','left']:
                    ax.spines[axis].set_linewidth(1.0)
                    ax.spines[axis].set_color('#7a7974')
                    
                for axis in ['top','right']:
                    ax.spines[axis].set_linewidth(0.0)
            
            if legend: 
                ax.legend(loc='upper right')
                ax.legend(labelcolor='linecolor', frameon=False, labelspacing=labelspacing, prop={'size': legend_font_size, 'weight': legend_weight, 'stretch': 1000})
                
            else:
                try:
                    ax.get_legend().remove()
                except:
                    pass
        
        kmf0 = kmfs[0]
        kmf1 = kmfs[1]
        dfs = pd.concat([kmf0[1].survival_function_, kmf1[1].survival_function_], axis = 1 )
        dfs.fillna(method='ffill', inplace=True)
        
        ax.fill_between(dfs.index, dfs[kmf0[0]], dfs[kmf1[0]], alpha = fill_alpha, color = fill_color, edgecolor='white')

        ax.set_ylim([-0.1, 1.05])

        self.plots = kmfs
        self.table(ax)
        
        

        if table: 
            bbox=bbox
            # pd.plotting.table(ax, self.counts.iloc[:, ], loc='bottom', colLoc='right', rowLoc='right', edges='open', bbox=bbox);
            
            mpl_table = ax.table(cellText=self.counts.values.astype(int), bbox=bbox, colLabels=self.counts.columns, rowLabels=self.counts.index, edges='open')
            
            mpl_table.auto_set_font_size(False)
            mpl_table.set_fontsize(table_font_size)
            
            row_colors = ['white'] + colors
            header_columns=0
            header_color='w'
            edge_color='w'
            
            ax.set_xlim([0, np.max(list(self.counts)) + 5])
            
            for k, cell in six.iteritems(mpl_table._cells):
                
                cell.set_edgecolor(edge_color)
                if k[0] == 0 or k[1] < header_columns:
                    cell.set_text_props(weight='bold', color=row_colors[k[0]])
                else:
                    cell.set_text_props(color=row_colors[k[0]])   
        
        if remove_plot:
            plt.close()

def andop(x):
    if sum(x) == len(x):
        return True
    else: 
        return False

# def generate_combined_list(*lists):
#     combined_list = list(itertools.product(*lists))
#     return combined_list

def generate_combined_list(lists):
    combined_list = [list(item) for item in itertools.product(*lists)]
    return combined_list

def forestplot(data, **kwargs):
    '''
    Generates a custom forest plot from the provided data.

    Parameters:
    -----------
    data : pandas.DataFrame The data to be plotted.
    
    marker : str,  Marker style for the scatter plot points. Default is 'o'.
    marker_s : int,  Size of the markers. Default is 20.
    color_column : str,  Column name in the data used to color the markers. Default is None.
    linewidth : float,  Line width for the horizontal lines. Default is 1.
    hr : str,  Column name for the hazard ratio. Default is 'HR'.
    hr_hi : str,  Column name for the upper confidence interval of the hazard ratio. Default is 'HR_hi'.
    hr_lo : str,  Column name for the lower confidence interval of the hazard ratio. Default is 'HR_lo'.

    groupby : str,  Column name to group the data by. Default is None.
    groupby_order : list,  Order of the groups for plotting. Default is None.
    group_xlabel_offset : float,  Offset for the x-axis label position. Default is the lower xlim value.
    group_ylabel_offset : float,  Offset for the y-axis label position. Default is 0.
    
    table : list,  List of labels for the plot. Default is an empty list.
    xtable_pos : list, relative position for the labels. Default is None.
    xtable_offset: float, , position of the table in the x axis.
    ytable_offset : float,  Offset for the group positions. Default is 1.
    table_fontsize : int,  Font size for the table. Default is 10.
    
    index : List of columns to use as index for the plot. Default is None.
    index_order : Order of the index for plotting. Default is None.

    xlabel : str,  Label for the x-axis. Default is 'Hazard Ratio'.
    xlabel_fontsize : int,  Font size for the labels. Default is 10.
    xlim : list,  Limits for the x-axis. Default is [0.25, 1.5].
    xticks_labelsize : int,  Font size for the x-axis tick labels. Default is 10.

    ax : matplotlib.axes.Axes,  The axes on which to plot. Default is an empty string.
    
    Returns:
    --------
    None
        The function modifies the provided matplotlib axes to create a forest plot.

    Example:
    --------
    f, axs = subplots(cols=1, rows=1, w=5, h=1.6, return_f=True)
    forestplot_simple(
        data = study_level.query('split == "train"'),
        groupby = 'tool',
        groupby_order = ['SIDES', 'VT', 'PBMF'],
        index = ['tool', 'biomarker'],
        index_order = [['PBMF', 'VT', 'SIDES'], ['B+', 'B-']],
        table=['biomarker', 'HR_label'],
        xtable_pos = [0, 0.1, 0.17, 0.9],
        xlim=[-0.2, 2.1],
        ax = axs[0],
        ytable_offset=-2,
        marker_s=40,
        color_column = 'color',
        table_fontsize = 9,
    )
    '''
    
    data = data.copy()
    
    marker = kwargs.get('marker', 'o')
    s = kwargs.get('marker_s', 20)
    marker_edgecolor = kwargs.get('marker_edgecolor', '#000000')
    color_column = kwargs.get('color_column', None)

    try:
        data[color_column]
        data['marker_edgecolor'] = data[color_column]
    except:
        data['marker_edgecolor'] = marker_edgecolor
    
    marker_c = kwargs.get('marker_c', '#000000')
    label_fontsize = kwargs.get('xlabel_fontsize', 10)
    table_fontsize = kwargs.get('table_fontsize', 10)
    xlabel = kwargs.get('xlabel', 'Hazard Ratio')
    xlim=kwargs.get('xlim', [0.25, 1.5])

    xticks=kwargs.get('xticks', None)
    xticks_labelsize = kwargs.get('xticks_labelsize', 10)
    
    group = kwargs.get('groupby', None)
    groupby_order = kwargs.get('groupby_order', None)
    if type(groupby_order) == list:
        groups = np.array(groupby_order)
    else:
        groups = data[group].drop_duplicates().values
    
    index = kwargs.get('index', None)
    index_order = kwargs.get('index_order', None)
    ordered_index = generate_combined_list(index_order)

    label=kwargs.get('table', [])
    manual_table_position = kwargs.get('xtable_pos', None)
    manual_table_position = {k: v for k, v in zip(label, manual_table_position)}
    
    hr = kwargs.get('hr', 'HR')
    hi = kwargs.get('hr_hi', 'HR_hi')
    lo = kwargs.get('hr_lo', 'HR_lo')
    ax = kwargs.get('ax', '')

    if not xticks:
        xticks = [ (i/100) for i in range(0, int(100*np.max(data[hi])), 50)]

    ix = 0
    offset = kwargs.get('ytable_offset', 1)
    for gix, gi in enumerate(groups):
        
        gix = gix*len(ordered_index) + offset * gix
        gv_ = data[data[group] == gi].reset_index(drop=True)[index].values
        
        vix = 0
        for ixx, idx in enumerate(ordered_index):

                i = (data[index + [group]] == list(idx) +[gi])
                idxi = i.sum(axis=1) == len(index) + 1
                i = data[idxi]

                if i.shape[0] == 0: continue
                icolor = i['marker_edgecolor'].values[0]
                
                ax.hlines(vix + gix, i[lo], i[hi], color=icolor, linewidth=kwargs.get('linewidth', 1))
                ax.scatter(i[hr], vix + gix, c=icolor, marker=marker, s=s, zorder=100, edgecolors='white')

                ax.set_xlabel(xlabel, fontweight='bold', fontsize=label_fontsize)

                ttx = kwargs.get('xtable_offset', xlim[0])
                for ilab in label:
                    ax.text(
                        ttx+manual_table_position[ilab], 
                        vix+gix-0.25, 
                        i[ilab].values[0], 
                        fontsize=kwargs.get('table_fontsize', 10) - 2, zorder=100,
                        color=icolor
                    )
                
                if vix % 4 == 0:
                    # ax.hlines(vix+gix, xlim[0], xlim[1], color='black', linewidth=1, zorder=-1000)
                    # ax.hlines(vix+gix+2, xlim[0], xlim[1], color='black', linewidth=1, zorder=-1000)
                    ax.fill_between(xlim, vix+gix-0.5, vix+gix+2-0.5, color='gray', alpha=0.05, edgecolor='white')
    
                vix+=1
                
                
        ax.text(
            kwargs.get('group_xlabel_offset', xlim[0]), 
            vix + gix + kwargs.get('group_ylabel_offset', 0.),
            gi, 
            fontsize=kwargs.get('table_fontsize', 10)+1,
            horizontalalignment=kwargs.get('ylabel_align', 'left'),
            color='black'
        )            

    ax.axvline(1, color=kwargs.get('one_line_color', 'gray'), zorder=0, linestyle='--')
    ax.set_xticks(xticks)
    ax.tick_params(axis='x', labelsize=xticks_labelsize)
    ax.set_xlim(xlim);

    # ylim=[-0.25, gix*(vix-1) + 0.5]
    # ax.set_ylim(ylim)

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis = "y", which = "both", left = False, right = False)
    ax.set_yticklabels([])


#### Plot survival for multiple iterations of the data. Training / Testing splits. 

def simple_survival_plot(data, time, event, label, score, **kwargs):
    
    kmfs = {}

    for label in label:
        kmf = KaplanMeierFitter()
        kmfs[label] = kmf.fit(data[ix][self.OS_AVAL], self.data_o[ix][self.OS_CNSR], label='{}'.format( name )), ix, '', ['', '']


def cox_functions(data, predictor='label', control_arm_label=None, iteration_column=None, time='time', event='event', **kwargs):
    """
    Performs survival analysis using Cox Proportional Hazards model across different subgroups or iterations within the data.

    This function is designed to handle survival data, allowing the comparison of a treatment effect across various populations defined by the `predictor`. It computes the hazard ratio for each subgroup within the specified `iteration_column` and provides statistics for each group in comparison to a control arm.

    Parameters:
    - data (pd.DataFrame): The dataset containing survival data, predictors, and possibly multiple iterations or subgroups.
    - predictor (str or list, optional): The column name(s) in `data` used to define the treatment groups. Defaults to 'label'. If a string is provided, it's converted to a list.
    - control_arm_label (str, optional): The label of the control arm group within the predictor column(s). Rows with this label are treated as the baseline in hazard ratio calculations.
    - iteration_column (str, optional): The column name in `data` used to separate the data into different subgroups or iterations for analysis. If specified, the function performs separate analyses for each unique value in this column.
    - time (str, optional): The column name in `data` that specifies the survival time. Defaults to 'time'.
    - event (str, optional): The column name in `data` that indicates the event occurrence (e.g., death, failure). Defaults to 'event'.
    - **kwargs: Additional keyword arguments for customization. Supports 'prefix' (str) to add a prefix to group labels in the output DataFrame.

    Returns:
    - pd.DataFrame: A DataFrame containing the hazard ratio (`hr`), its 95% confidence intervals (`hr_lo`, `hr_hi`), and the p-value (`pval`) for each population defined by `predictor` and `iteration_column`. Also includes counts (`N_<label>`) for each label in the population.

    Raises:
    - Logs an error for any iteration where the Cox Proportional Hazards model fitting fails, providing the fold number and error details.

    Note:
    - The function expects that `data` is preprocessed appropriately, with no missing values in `time` and `event` columns for accurate Cox model fitting.
    - It manipulates `data` to create a temporary `__label__` column for internal computations but does not modify the input DataFrame outside its scope.

    Example:
    ```
    data = pd.DataFrame({...})
    results = cox_functions(data, predictor='treatment_group', control_arm_label='placebo', iteration_column='study_id', time='follow_up_time', event='occurrence')
    print(results)
    ```
    """
    stats = []

    if type(predictor) == str:
        predictor = [predictor]

    data['__label__'] = data[predictor].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    labels = set(data['__label__'])
    folds = set(data[iteration_column])
    prefix = kwargs.get('prefix', 'C')
    
    for fold in folds:
        try:
            data_ = data[(data[iteration_column] == fold)].reset_index(drop=True).copy()
            
            # Set the control as 0 and the target arm as 1
            data_['{}'.format(fold)] = (data_['{}'.format('__label__')] != control_arm_label).astype(np.int)  

            cph = CoxPHFitter().fit(data_[[time, event, '{}'.format(fold)]], time, event)
            sm = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].reset_index(drop=False)
            sm.columns = ['Population', 'hr', 'hr_lo', 'hr_hi', 'pval']
            
            # for label in labels:
                # sm['prev_{}'.format(label)] = data['prev_{}'.format(label)].mean()
            Ns = Counter(data_['__label__'])
            for label in labels:
                sm['N_{}'.format(label)] = Ns[label]
                
            stats.append(sm)
        except Exception as inst:
            logger.error('Fold {} has an error'.format(fold))
            logger.error(inst)
            pass
    
    return pd.concat(stats).reset_index(drop=True)

def kmf_survival_functions(data, predictor='label', iteration_column=None, time='time', event='event'):
    '''Retrieve all KM survival functions using input data'''

    if type(predictor) == str:
        predictor = [predictor]

    data['__label__'] = data[predictor].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    labels = set(data['__label__'])
    folds = set(data[iteration_column])
    survival_functions = {}
    for label in labels:
        survival_functions_fold = []
        for fold in folds:
            try:
                data_ = data[(data[iteration_column] == fold) & (data['__label__'] == label)].reset_index(drop=True)
                kmf = KaplanMeierFitter( label=fold ).fit(data_[time], data_[event])
                survival_functions_fold.append(kmf.survival_function_)
            except:
                pass
        kmfi = pd.concat(survival_functions_fold, axis=1).fillna(method='ffill', inplace=False)
        kmfi_ci = pd.concat([kmfi.median(axis = 1), kmfi.mean(axis=1), kmfi.quantile([0.025, 0.975], axis = 1).transpose()], axis = 1)
        kmfi_ci.columns = ['median', 'mean', 'CI95LO','CI95HI']
        
        survival_functions[label] = kmfi_ci
    
    
    
    
    return survival_functions

def median_plot_survival(kmfs, cox=[], label='', agg='mean', color='#573411', ax=[], linewidth=2, plot_medians=True, linecolor=None, alpha=0.25, label_font_size=12,hr_font_size=10, hr_label_offset=-0.03, median_time_offset=[0, 0], name_offset=[0, 0], median_time_fontsize=20):
    '''
    Plot survival for multiple iterations of the data. Training / Testing splits. 
    '''
    if linecolor == None:
        linecolor = color
    
    ax.plot(kmfs[label].index, kmfs[label][agg], color = linecolor, linewidth=linewidth)
    ax.fill_between(kmfs[label].index, kmfs[label]['CI95HI'], kmfs[label]['CI95LO'], alpha = alpha, color = color)
    
    
    # labels
    if cox.shape[0] > 1: 
        cox = cox.mean()
        hr_label = 'HR={:.2f} (95% CI: {:.2f}, {:.2f}); P={:.2e}'.format(cox.hr, cox.hr_lo, cox.hr_hi, cox.pval)
        ax.text(0.0, hr_label_offset, hr_label, color='black', weight='bold', fontsize=hr_font_size, transform=ax.transAxes)

     
    # medians
    if plot_medians:
        if len(list(kmfs[label][kmfs[label][agg] == 0.5].index)) == 0:
            try:
                median_time = np.mean([kmfs[label][kmfs[label][agg] > 0.5].iloc[-1].name, kmfs[label][kmfs[label][agg] < 0.5].iloc[0].name])
            except:
                median_time = None
        else:
            median_time = list(kmfs[label][kmfs[label][agg] == 0.5].index)[0]
        
        # median label
        if not median_time == None:
            ax.text(median_time + median_time_offset[0], 0.53 + median_time_offset[1], '{:.1f}'.format(median_time), size=median_time_fontsize, weight='bold', color=color)
            ax.axvline(median_time, 0, 0.5, ls = '--', color = '#a19595', lw = 1)
            sns.scatterplot(x=[median_time], y=[0.5], ax=ax, s=100, color=color, alpha=0.8)
            ax.plot((-0, median_time), (0.5, 0.5),  ls = '--', color = '#a19595', lw = 1)
    
    # name 
    x_, y_ = list(kmfs[label][agg].items())[-1]
    if cox.shape[0] > 1: 
        ax.text(name_offset[0], name_offset[1], "{} (N={:.0f})".format(label, cox['N_{}'.format(label)]), weight='bold', color=color)
    else:
        ax.text(name_offset[0], name_offset[1], "{}".format(label), weight='bold', color=color)
    
    # general aspect
    ax.set_ylim([-0.12, 1])
    ax.set_ylabel('Survival Probability', weight='bold', fontsize=label_font_size, color='#7a7974')
    ax.set_xlabel('Timeline', weight='bold', fontsize=label_font_size, color='#7a7974')
    ax.tick_params(axis='x', colors='#7a7974')
    ax.tick_params(axis='y', colors='#7a7974')

    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(1.0)
        ax.spines[axis].set_color('#7a7974')

    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0.0)
    