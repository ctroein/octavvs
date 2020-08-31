#from PyQt5 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from PyQt5.QtCore import pyqtSignal, pyqtSlot
import numpy as np



class MplWidget(Canvas):
    def __init__(self, parent=None):
        self.fig = Figure(tight_layout=dict(pad=0.4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = self
        Canvas.__init__(self, self.fig)

    def Invert(self):
        # print('called')
        self.ax.invert_xaxis()


class MplWidgetC(Canvas):
    def __init__(self, parent=None):
        self.fig = Figure(tight_layout=dict(pad=0.4))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off() 
        self.ax.set_position([0,0,1,1])
        self.canvas = self
        Canvas.__init__(self, self.fig)
 
   


class Mpl_Proj(MplWidget):
    def __init__(self, parent=None):
        MplWidget.__init__(self, parent)
        self.image = None
        self.artists = []
        self.points = None
        self.ax.set_axis_off()
        self.ax.set_position([0,0,1,1])
        self.image = None

    @pyqtSlot(np.ndarray, str)
    def setImage(self, img, cmap):
#        self.ax.clear()
#        self.artists = []
#        self.fig.tight_layout()
#        self.draw_idle()
        if self.image is None:
            self.image = self.ax.imshow(img, cmap)
        else:
            self.image.set_data(img)
            self.image.set_extent((0, img.shape[1], img.shape[0], 0))
            self.ax.set_xlim(0, img.shape[1])
            self.ax.set_ylim(img.shape[0], 0)
            self.setCmap(cmap)
        self.image.autoscale()

    @pyqtSlot(str)
    def setCmap(self, cmap):
        self.image.set_cmap(cmap)
        self.draw_idle()

    @pyqtSlot()
    def clearMarkings(self):
        for a in self.artists:
            a.remove()
        self.artists = []
        if self.points is not None:
            self.points.remove()
            self.points = None

    @pyqtSlot(list)
    def setRoi(self, pts):
        self.clearMarkings()
        if len(pts) > 1:
            npts = np.array(pts + [pts[0]]).T
            self.artists = self.ax.plot(npts[0]+.5, npts[1]+.5, 'r')
        self.draw_idle()

    def addPoints(self, pts):
        if self.points is not None:
            self.points.remove()
        self.points = self.ax.scatter(pts.T[0,:], pts.T[1,:], marker='p', color='black')
        self.draw_idle()


class MPL_ROI(Canvas):

    captured = pyqtSignal(list)

    def __init__(self, parent=None):
        self.fig = Figure(tight_layout=False)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.ax.set_position([0,0,1,1])
        Canvas.__init__(self, self.fig)
        self.mpl_connect('button_press_event', self.onclick)

        self.points = []
        self.artists = []
        self.wh = (1,1)
        self.mainimg = self.ax.imshow([[0]], cmap='gray', zorder=0, aspect='equal')

    def setImage(self, img, cmap=None):
        self.mainimg.set_data(img)
        self.mainimg.set_cmap(cmap)
        self.mainimg.autoscale()
        self.mainimg.set_extent((0, self.wh[1], self.wh[0], 0))
        self.ax.autoscale(False)
        self.ax.set_position([0,0,1,1])
        self.ax.set_xlim(0, self.wh[1])
        self.ax.set_ylim(self.wh[0], 0)
        self.draw_idle()

    def setCmap(self, cmap):
        self.mainimg.set_cmap(cmap)
        self.draw_idle()

    def setSize(self, wh):
        if wh != self.wh:
            self.resetAll()
        self.wh = wh


    def onclick(self, event):
        x, y = event.xdata, event.ydata
        if (x is None) or (y is None):
            return

        x = int(x)
        y = int(y)
        if not (0 <= x < self.wh[1] and 0 <= y < self.wh[0]):
            print('click outside',x,y,self.wh)
            return
        if not len(self.points):
            self.artists.append(self.ax.plot(x+.5, y+.5, 'y*'))
        else:
            self.artists.append(self.ax.plot((self.points[-1][0]+.5, x+.5), (self.points[-1][1]+.5, y+.5), 'r-'))

        self.points.append((x, y))
        self.draw_idle()
        self.captured.emit(self.points)

    @pyqtSlot()
    def resetAll(self):
        if len(self.points):
            self.points = []
            for la in self.artists:
                for a in la:
                    a.remove()
            self.artists = []
            self.draw_idle()
        self.captured.emit(self.points)

    @pyqtSlot()
    def removeOne(self):
        if len(self.points):
            self.points.pop()
            for a in self.artists.pop():
                a.remove()
            self.draw_idle()

        self.captured.emit(self.points)



