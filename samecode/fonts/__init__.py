import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt

def set_font(GlobalFont = 'arial', **kwargs):
    font_path = '{}/{}.ttf'.format(__file__.replace('__init__.py', ''), GlobalFont)

    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = prop.get_name()
    plt.rcParams['pdf.fonttype'] = 42
    font = {'family':'Arial', 'weight':'normal', 'size':12}
    matplotlib.rc('font', **font)
