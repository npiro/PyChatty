<<<<<<< Updated upstream
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 19:53:48 2016

@author: Client
"""

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

class MplFigure(object):
    def __init__(self, parent):
        self.figure = plt.figure(facecolor='white')
        self.canvas = FigureCanvas(self.figure)
=======
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 19:53:48 2016

@author: Client
"""

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

class MplFigure(object):
    def __init__(self, parent):
        self.figure = plt.figure(facecolor='white')
        self.canvas = FigureCanvas(self.figure)
>>>>>>> Stashed changes
        self.toolbar = NavigationToolbar(self.canvas, parent)