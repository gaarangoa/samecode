import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import os

def set_font(global_font='poppins', font_file_path=None, **kwargs):
    """
    Sets the global font for matplotlib. The font can be from system fonts or a custom font file.
    
    Parameters:
        global_font (str): Name of the font (for system fonts) or the font name to be applied to all plots.
        font_file_path (str): Path to a custom .ttf font file (if any). If None, it will use the system font.
    """
    
    font_file_path_ = '{}/{}.ttf'.format(__file__.replace('__init__.py', ''), global_font)
    if os.path.exists(font_file_path_):
        font_file_path = font_file_path_

    
    try:
        if font_file_path and os.path.exists(font_file_path):
            # Load the custom font from the given file path
            font_manager.fontManager.addfont(font_file_path)
            prop = font_manager.FontProperties(fname=font_file_path)
            font_name = prop.get_name()  # Extract the font name from the file
        else:
            # Use the system font
            font_name = global_font

        # Set the font family and apply it to all future plots
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = font_name
        #plt.rcParams['pdf.fonttype'] = 42  # To embed fonts correctly in PDF files
        plt.rcParams['svg.fonttype'] = 'path'  # Better font handling in SVG
        font = {'family': font_name, 'weight': 'normal', 'size': 12}
        
        # Apply the font to matplotlib's global font settings
        matplotlib.rc('font', **font)
        
        # print(f"Font '{font_name}' has been successfully set for all plots.")
    except Exception as e:
        print(f"Failed to set font. Error: {e}")
    
