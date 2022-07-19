"""
    hplot.py : module contains fancy plots.
"""


import matplotlib.pyplot as plt
import numpy as np

def plot(x, y, label=['x','f(x)'],
         label_fontsize =20,  		 
         style = '-',
         xlim = [0,0],
         ylim = [0,0],
         hide_tick=True,
         hide_tick_label=True,
         save_fig=' ',
         dpi=150, 
         plot_xaxis = False,
         plot_yaxis = False,
         remove_axis = False,
         legend_label=' ',
         legend_loc=-1,
         tight_layout = True,		 
         y2=' ', 
         style2 ='r-.',
         legend_label2=' ',	
         y3=' ',
         style3 ='g--',
         legend_label3=' ',
         texts = [[0,0,' ']]
         ):
    #NAME:
    #    plot
    r"""plot given data x,y up to 3 graphs
    
    parameters
    ----------
    x : array_like 
        data x-axis 
    y : array_like
        data y-axis
    label : list, optional 
        label of axes, e.g. ['x',f(x)'].
    style  : str, optional
        style of plot, similar to matplotlib.pyplot.plot style. 
    xlim   : list, optional
        limit of x-axis, list of lower and upper limits	
    ylim   : list
        limit of y axis, list of lower and upper limits
    hide_tick : bool, optional
        do you want to hide tick?
    hide_tick_label : bool, optional
        do you want to hide tick label   
    save_fig  : str, optional
        filename for saving figure 
    dpi  : int, optional
        number of dots per inch 
    plot_xaxis : bool, optional
        do you want to plot x-axis (y=0)?		
    plot_yaxis : bool, optional
        plot y-axis (x=0)?
    remove_axis: bool, optional
        remove_axis?
    legend_label: str, optional
        label for legend   
    legend_loc: int, optional
        location of legend   
    tight_layout: bool, optional
        do you want to tight layout			
    y2 : array_like, optional
        second data y        
    style2 : str, optional
        style of 2nd data y
    legend_label2 : str, optional
        label for legend 2nd function.
    y3 : array_like, optional
        third data y.
    style3 : str, optional
        style of 3rd data y.          
    legend_label3 : str, optional
        label for legend 3rd data y.
    texts : list, optional
        list of [x pos text,y pos text,text]. 
    
    returns
    -------	
    figure : matplotlib_object
        show figure and save it if save_fig given.	
	

    |
    """	
    #HISTORY:
    #    15-12-2020   start writing  Hasanuddin
       
    #Location String  ->	Location Code
    #'best'	               0
    #'upper right'	       1
    #'upper left'	       2
    #'lower left'	       3
    #'lower right'	       4
    #'right'	           5
    #'center left'	       6
    #'center right'	       7
    #'lower center'	       8
    #'upper center'	       9
    #'center'	          10
    
    #plot
    plt.plot(x,y,style,label=legend_label)
	#axis
    if plot_xaxis:
        plt.hlines(0,xlim[0],xlim[1],color='k')
    if plot_yaxis:
        plt.vlines(0,ylim[0],ylim[1],color='k')	
    # plot another function
    if y2 !=' ':
        plt.plot(x,y2,style2,label=legend_label2)
    if y3 !=' ':
        plt.plot(x,y3,style3,label=legend_label3)
    #axes label
    plt.xlabel(label[0],fontsize=label_fontsize)
    plt.ylabel(label[1],fontsize=label_fontsize)
    # remove axis?
    if remove_axis:
        plt.axis('off')
	#axes limit
    if xlim != [0,0]:
        plt.xlim(xlim)
    if ylim != [0,0]:
        plt.ylim(ylim)
    #adding text
    for text in texts:
        plt.text(text[0],text[1],text[2],fontsize=label_fontsize)
    #add legend
    if legend_loc!=-1:
        plt.legend(loc=legend_loc)
    #get current axis
    frame = plt.gca()
	#hide tick label or not
    if hide_tick & hide_tick_label:
        frame.axes.get_xaxis().set_ticks([])
        frame.axes.get_yaxis().set_ticks([])
    #tight_layout
    if tight_layout:
        plt.tight_layout()
    #saving_figure
    if save_fig != ' ':
        plt.savefig(save_fig,dpi=dpi,bbox_inches='tight')
    plt.show()
		 

###############################################################################
def plotf(func, 
         label=['x','f(x)'],
         label_fontsize =20,  		 
         style = '-',
         xlim = [0,0],
         ylim = [0,0],
         points = 100,
         hide_tick=True,
         hide_tick_label=True,
         save_fig=' ',
         dpi=150, 
         plot_xaxis = False,
         plot_yaxis = False,
         remove_axis = False,
         legend_label=' ',
         legend_loc=-1,
         tight_layout = True,		 
         func2=' ',
         style2 ='r-.',
         legend_label2=' ',	
         func3=' ',
         style3 ='g--',
         legend_label3=' ',
         texts = [[0,0,' ']]
         ):
    #Name : plotf
    r""" plot given function
    
    parameters
    ----------
    func : function 
        function 
    label : list, optional 
        label of axes, e.g. ['x',f(x)'].
    style  : str, optional
        style of plot, similar to matplotlib.pyplot.plot style. 
    xlim   : list, optional
        limit of x-axis, list of lower and upper limits	
    ylim   : list
        limit of y axis, list of lower and upper limits
    points : int
        number of datum.
    hide_tick : bool, optional
        do you want to hide tick?
    hide_tick_label : bool, optional
        do you want to hide tick label   
    save_fig  : str, optional
        filename for saving figure 
    dpi  : int, optional
        number of dots per inch 
    plot_xaxis : bool, optional
        do you want to plot x-axis (y=0)?		
    plot_yaxis : bool, optional
        plot y-axis (x=0)?
    remove_axis: bool, optional
        remove_axis?
    legend_label: str, optional
        label for legend   
    legend_loc: int, optional
        location of legend   
    tight_layout: bool, optional
        do you want to tight layout			
    func2 : function, optional
        second function         
    style2 : str, optional
        style of 2nd function
    legend_label2 : str, optional
        label for legend 2nd function.
    func3 : function, optional
        third function y.
    style3 : str, optional
        style of 3rd function.          
    legend_label3 : str, optional
        label for legend 3rd function.
    texts : list, optional
        list of [x pos text,y pos text,text]. 
    
    returns
    -------	
    figure : matplotlib_object
        show figure and save it if save_fig given.	
	

    |
    """	
        
	    	
    #HISTORY:
    #    15-12-2020   start writing  Hasanuddin
       
    #    #Location String  ->	Location Code
    #    #'upper right'	       1
    #    #'upper left'	       2
    #    #'lower left'	       3
    #    #'lower right'	       4
    #    #'right'	           5
    #    #'center left'	       6
    #    #'center right'	   7
    #    #'lower center'	   8
    #    #'upper center'	   9
    #    #'center'	           10

    
    #generate data#
    x = np.linspace(xlim[0],xlim[1],points)
    y = func(x)
    y2=' '
    if func2 !=' ':
        y2 = func2(x)
    y3=' '
    if func3 !=' ':
        y3 = func3(x)
    #plot
    plot(x, y, label=label,
         label_fontsize =label_fontsize,  		 
         style = style,
         xlim = xlim,
         ylim = ylim,
         hide_tick=hide_tick,
         hide_tick_label=hide_tick_label,
         save_fig=save_fig,
         dpi=dpi, 
         plot_xaxis = plot_xaxis,
         plot_yaxis = plot_yaxis,
         remove_axis = remove_axis,
         legend_label=legend_label,
         legend_loc=legend_loc,
         tight_layout = tight_layout,		 
         y2=y2,
         style2 =style2,
         legend_label2=legend_label2,	
         y3=y3,
         style3 =style3,
         legend_label3=legend_label3,
         texts = texts)
    	
###############################################################################	
