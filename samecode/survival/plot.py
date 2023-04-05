import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.metrics import confusion_matrix

from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines import CoxPHFitter
import six 
import seaborn as sns
from lifelines.utils import median_survival_times

import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

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

        Optional: 
        
        saturation=1.0,
        linewidth=1.5,
        palette='Paired',
        template_color = 'black', xy_font_size = 12,
        hr_color = 'black',
        x_legend = 0.15, y_legend = 0.95, legend_font_size=10,
        label_height_adj=0.055,
        x_hr_legend = 0.0, y_hr_legend = -0.3, hr_font_size=10,
        

        Contact: gaarangoa
        '''
        
        self.colors = {}
        
        self.fit(data, time, event, label, **kwargs)
    
    def compare(self, ):
        pass
    
    def plot(self, labels=None, **kwargs):
        '''
        label[optional]: Label(s) to plot
        linestyle[optional]:  list same dim as labels
        color[optional]:  list same dim as labels
        linewidth[optional]: list same dim as labels
        legend_font_size[optional]: font size for legend
        legend_labelspacing[optional]: 
        saturation[optional]
        label_height_adj: adjust space between labels (y axis)
        xy_font_size: font size of x and y labels
        comparisons: make comparisons between two curves [[tar, ref], [io, soc], [D, D+T]]
        palette: "Paired"
        template_color: '#7a7974'
        adj_label_loc: 0.1
        hr_color: 'black' # Color for hr layer
        display_labels = [comp1, comp2]

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

        Optional: 
        
        saturation=1.0,
        linewidth=1.5,
        palette='Paired',
        template_color = 'black', xy_font_size = 12,
        hr_color = 'black',
        x_legend = 0.15, y_legend = 0.95, legend_font_size=10,
        label_height_adj=0.055,
        x_hr_legend = 0.0, y_hr_legend = -0.3, hr_font_size=10,


        '''

        plt.rcParams['font.family'] = kwargs.get('font_family', '')
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
        template_color = kwargs.get('template_color', 'black')
        to_compare = kwargs.get('comparisons', [])
        
        if type(colors) == str:
            colors = [colors]

        label_max_l = [self.label_names_size['__label__']]
        for lx, label_ in enumerate(plot_labels):
            label_max_l.append(self.label_names_size[label_])
            self.colors[label_] = colors[lx]
            self.kmfs[label_].plot(
                ci_show=kwargs.get('ci_show', False), 
                show_censors=True,
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

        label_max_l = np.array(label_max_l).max(axis=0)
        for lx, label_ in enumerate(['__label__'] + plot_labels):                
            vx = fix_string(self.label_names_list[label_], self.label_names_size[label_], label_max_l)

            ax.annotate(
                vx, 
                xy=(x_legend, y_legend -lx*label_height_adj), 
                xycoords='axes fraction', 
                xytext=(x_legend, y_legend -lx*label_height_adj),
                weight='bold', 
                size=legend_font_size, 
                color=self.colors[label_],
                # bbox=dict(fc='white', lw=0, alpha=0.3)
            )

        # ax.annotate(
        #     '', 
        #     xy=(-0.01, y_legend -(lx)*label_height_adj), 
        #     xycoords='axes fraction', 
        #     xytext=(1, y_legend -(lx)*label_height_adj), 
        #     arrowprops=dict(arrowstyle="-", color='k'),
        # )

        # Cox PH Fitters for HR estimation
        xcompare = [[('__label__', '__label__'), "\tHR\t(95% CI)\tP value"]]
        xinfo = [[len(i) for i in "\tHR\t(95% CI)\tP value".split('\t')]]
        for cx, item in enumerate(to_compare):
            
            if len(item) == 3:
                [tar, ref, hr_label] = item
            else: 
                [tar, ref] = item
                hr_label = '{} - {}: '.format(tar, ref)

            x = self.data[self.data.__label__.isin([tar, ref])][[self.time, self.event, '__label__']].copy().reset_index(drop=True)
            x.__label__.replace(ref, 0, inplace=True)
            x.__label__.replace(tar, 1, inplace=True)
            x.__label__ = x.__label__.astype(float)

            cph = CoxPHFitter().fit(x, duration_col = self.time, event_col = self.event) 
            cph = cph.summary[['exp(coef)', 'p', 'exp(coef) lower 95%', 'exp(coef) upper 95%']].reset_index().to_dict()
            cph = {
                "HR": cph.get('exp(coef)').get(0),
                "HR_lo": cph.get('exp(coef) lower 95%').get(0),
                "HR_hi": cph.get('exp(coef) upper 95%').get(0),
                "P": cph.get('p').get(0),
            }
            
            # color for HR 
            hr_color = kwargs.get('hr_color', self.colors[tar])
            
            # xinfo_ = '{}\tHR={:.2f}\t(CI 95%: {:.2f} - {:.2f})\tP-value={:.2e}'.format(hr_label, cph.get('HR'), cph.get('HR_lo'), cph.get('HR_hi'), cph.get('P'))
            xinfo_ = '{}\t{:.2f}\t({:.2f}-{:.2f})\t{:.2e}'.format(hr_label, cph.get('HR'), cph.get('HR_lo'), cph.get('HR_hi'), cph.get('P'))
            xinfo.append([len(i) for i in xinfo_.split('\t')])

            xcompare.append([
                (tar, ref), xinfo_
            ])

        

        x_hr_legend=kwargs.get('x_hr_legend', 0)
        y_hr_legend=kwargs.get('y_hr_legend', -0.3)
        hr_font_size=kwargs.get('hr_font_size', 10)

        max_values = np.array(xinfo)
        for ix, [k, v] in enumerate(xcompare):
            
            tar, ref = k 
            if len(xcompare) == 1: continue
            hr_color = kwargs.get('hr_color', self.colors[tar])
            
            vx = fix_string(v, max_values[ix], max_values.max(axis=0))

            ax.annotate(
                vx, 
                xy=(x_hr_legend, y_hr_legend - ix*label_height_adj), 
                xycoords='axes fraction', 
                xytext=(x_hr_legend, y_hr_legend - ix*label_height_adj),
                weight='bold', 
                size=hr_font_size, 
                color=hr_color,
                # bbox=dict(fc='white', lw=0, alpha=0.5)
            )

        # plt.rcParams['font.family'] = kwargs.get('font_family', '')   
        
        
        ax.set_ylim([-0.03, 1])
        ax.set_ylabel(kwargs.get('ylab', 'Survival Probability'), weight='bold', fontsize=xy_font_size, color=template_color)
        ax.set_xlabel(kwargs.get('xlab', 'Timeline'), weight='bold', fontsize=xy_font_size, color=template_color)
        ax.tick_params(axis='x', colors=template_color)
        ax.tick_params(axis='y', colors=template_color)
        
        ax.xaxis.set_tick_params(labelsize=xy_font_size-1)
        ax.yaxis.set_tick_params(labelsize=xy_font_size-1)
        
        ax.set_title(kwargs.get('title', ''), fontsize=xy_font_size, color=template_color, weight='bold')

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

            self.label_names[label] = '{}: {} {:.2f} ({:.2f} - {:.2f})'.format(label, self.counts[label], kmfs[label].median_survival_time_, lo, hi)
            self.label_names_list[label] = '{}:\t{}\t{:.2f}\t({:.2f} - {:.2f})'.format(label, self.counts[label], kmfs[label].median_survival_time_, lo, hi)
            self.label_names_size[label] = [len(k) for k in self.label_names_list[label].split('\t')]
        
        self.label_names['__label__'] = ['N Median (95%CI)']
        self.label_names_list['__label__'] = ' \tN\tMedian\t(95% CI)'
        self.label_names_size['__label__'] = [len(k) for k in [' ', 'N', 'Median','(95%CI)']]

        self.data = data[[time, event, '__label__']]
        self.kmfs = kmfs

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

def forestplot(perf, figsize=(8, 3), ax=[], hr='hr', hi='', lo='', name='', condition='', cutoff='', target='', reference='', sort_by='hr', xticks=[0.25, 1, 1.5], xlim=[0.25, 1.5], median_time_fontsize=20, **kwargs):
    # plt.style.use('default')

    marker = kwargs.get('marker', 'D')
    s = kwargs.get('marker_s', 80)
    marker_edgecolor = kwargs.get('marker_edgecolor', '#004489')
    marker_c = kwargs.get('marker_c', 'white')

    for ix,i in perf.sort_values(sort_by).reset_index(drop=True).iterrows():
    #     plt.plot([i.CI_Low, i.CI_High], ["{} {}".format(i.Experiment, i.Cutoff), "{} {}".format(i.Experiment, i.Cutoff)], c='black')

        ax.hlines("{} {} {}".format(i[name], i[condition], i[cutoff]), i[lo], i[hi], color='black' if i[hr] < 1 else "#D48139")

        ax.scatter(i[lo], "{} {} {}".format(i[name], i[condition], i[cutoff]), c='#004489' if i[hr] < 1 else "#D48139", marker='|')
        ax.scatter(i[hi], "{} {} {}".format(i[name], i[condition], i[cutoff]), c='#004489' if i[hr] < 1 else "#D48139", marker='|')
        ax.scatter(i[hr], "{} {} {}".format(i[name], i[condition], i[cutoff]), c=marker_c, marker=marker, s=s, zorder=100, edgecolors=marker_edgecolor if i[hr] < 1 else "#D48139")

        ax.axvline(1, color='black', zorder=0)
        ax.set_xlabel('Hazard Ratio', fontweight='normal', fontsize=12)

        try:
            ax.text(
                xlim[1], "{} {} {}".format(i[name], i[condition], i[cutoff]), 
                "HR: {:.2f} CI: [{:.2f} - {:.2f}] ({}) N:({:.0f}, {:.0f})".format(i[hr], i[lo], i[hi], i[name], i['nN'], i['nW']), 
                fontsize=8, color='#004489' if i[hr] < 1 else "#D48139"
            )
        except:
             ax.text(
                xlim[1], "{} {} {}".format(i[name], i[condition], i[cutoff]), 
                "HR: {:.2f} CI: [{:.2f} - {:.2f}] ({})".format(i[hr], i[lo], i[hi], i[name]), 
                fontsize=8, color='#004489' if i[hr] < 1 else "#D48139"
            )

    ax.set_xticks(xticks)
    ax.set_xlim(xlim);

    ylim=[-0.5, ix+0.5]
    ax.set_ylim(ylim)

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

def cox_functions(data, predictor='label', control_arm_label=None, iteration_column=None, time='time', event='event' ):
    stats = []
    labels = set(data[predictor])
    folds = set(data[iteration_column])

    for fold in folds:
        # try:
            data_ = data[(data[iteration_column] == fold)].reset_index(drop=True).copy()
            
            # Set the control as 0 and the target arm as 1
            data_['pred__{}'.format(fold)] = (data_['{}'.format(predictor)] != control_arm_label).astype(np.int)  

            cph = CoxPHFitter().fit(data_[[time, event, 'pred__{}'.format(fold)]], time, event)
            sm = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].reset_index(drop=False)
            sm.columns = ['cluster', 'hr', 'hr_lo', 'hr_hi', 'pval']
            
            # for label in labels:
                # sm['prev_{}'.format(label)] = data['prev_{}'.format(label)].mean()
            Ns = Counter(data_[predictor])
            for label in labels:
                sm['N_{}'.format(label)] = Ns[label]
                
            stats.append(sm)
        # except Exception as inst:
        #     print(inst)
        #     pass
    
    return pd.concat(stats).reset_index(drop=True)

def kmf_survival_functions(data, predictor='label', iteration_column=None, time='time', event='event'):
    '''Retrieve all KM survival functions using input data'''
    
    labels = set(data[predictor])
    folds = set(data[iteration_column])
    survival_functions = {}
    for label in labels:
        survival_functions_fold = []
        for fold in folds:
            try:
                data_ = data[(data[iteration_column] == fold) & (data[predictor] == label)].reset_index(drop=True)
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
    