'''
    hplot.py
	fancy plots
'''
import matplotlib.pyplot as plt
import numpy as np

def plot(func, 
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
    '''
        func   : function in numpy
        label  : label
        style  : style of plot
        xlim   : limit of x axis
        ylim   : limit of y axis
        points : number of datum
        #Location String	Location Code
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
    ''' 
    #generate data#
    x = np.linspace(xlim[0],xlim[1],points)
    y = func(x)
    #plot
    plt.plot(x,y,style,label=legend_label)
	#axis
    if plot_xaxis:
        plt.hlines(0,xlim[0],xlim[1],color='k')
    if plot_yaxis:
        plt.vlines(0,ylim[0],ylim[1],color='k')	
    # plot another function
    if func2 !=' ':
        plt.plot(x,func2(x),style2,label=legend_label2)
    if func3 !=' ':
        plt.plot(x,func3(x), style3,label=legend_label3)
    #axes label
    plt.xlabel(label[0],fontsize=label_fontsize)
    plt.ylabel(label[1],fontsize=label_fontsize)
    # remove axis?
    if remove_axis:
        plt.axis('off')
	#axes limit
    plt.xlim(xlim)
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
    	
	
    	
    