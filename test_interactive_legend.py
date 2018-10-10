import matplotlib.pyplot as plt
import numpy as np

def main():
    fig, ax = plt.subplots()
    x = np.arange(10)
    y = np.zeros((5,len(x)))
    lines1=[]
    lines2=[]
    for idx in range(y.shape[0]):
        y[idx] = x + idx
        line1,=ax.plot(x, y[idx], 'o', label=str(idx))
        line2,=ax.plot(x, y[idx],'--k')
        lines1.append(line1)
        lines2.append(line2)
    ax.legend(title='click to remove')
    leg = interactive_legend(ax,lines1=lines1,lines2=lines2,data=y)
    plt.show()

def interactive_legend(ax=None,lines1=None,lines2=None,data=None):
    if ax is None:
        ax = plt.gca()
    if ax.legend_ is None:
        ax.legend()

    return InteractiveLegend(ax.legend_,lines1,lines2,data)#,legend_title,ax

class InteractiveLegend(object):
    def __init__(self,legend,lines1,lines2,data):#,legend_title,ax
        self.legend = legend
        #self.legend_title = legend_title
        self.fig = legend.axes.figure
        self.ax = plt.gca()
        self.lines1 = lines1
        self.lines2 = lines2
        self.data = data

        self.lookup_artist, self.lookup_handle = self._build_lookups(legend)
        self._setup_connections()

        self.update_legend()

    def _setup_connections(self):
        for artist in self.legend.texts + self.legend.legendHandles:
            artist.set_picker(10) # 10 points tolerance

        self.fig.canvas.mpl_connect('pick_event', self.on_pick)

    def _build_lookups(self, legend):
        labels = [t.get_text() for t in legend.texts]
        handles = legend.legendHandles
        label2handle = dict(zip(labels, handles))
        handle2text = dict(zip(handles, legend.texts))

        lookup_artist = {}
        lookup_handle = {}
        for artist in legend.axes.get_children():
            if artist.get_label() in labels:
                handle = label2handle[artist.get_label()]
                lookup_handle[artist] = handle
                lookup_artist[handle] = artist
                lookup_artist[handle2text[handle]] = artist

        lookup_handle.update(zip(handles, handles))
        lookup_handle.update(zip(legend.texts, handles))

        return lookup_artist, lookup_handle

    def on_pick(self, event):
        handle = event.artist
        if handle in self.lookup_artist:
            artist = self.lookup_artist[handle]
            if not self.lines1==None:
                line_idx = self.lines1.index(artist)
                self.data = np.delete(self.data,line_idx,axis=0) # remove data from array
                self.lines1[line_idx].remove()
            if not self.lines2==None: # remove companion line
                self.lines2[line_idx].remove()
            self.update_legend() # remove line from legend

    def update_legend(self):
        import matplotlib as mpl

        l = self.legend

        defaults = dict(
            loc = l._loc,
            numpoints = l.numpoints,
            markerscale = l.markerscale,
            scatterpoints = l.scatterpoints,
            scatteryoffsets = l._scatteryoffsets,
            prop = l.prop,
            # fontsize = None,
            borderpad = l.borderpad,
            labelspacing = l.labelspacing,
            handlelength = l.handlelength,
            handleheight = l.handleheight,
            handletextpad = l.handletextpad,
            borderaxespad = l.borderaxespad,
            columnspacing = l.columnspacing,
            ncol = l._ncol,
            mode = l._mode,
            fancybox = type(l.legendPatch.get_boxstyle())==mpl.patches.BoxStyle.Round,
            shadow = l.shadow,
            title = l.get_title().get_text() if l._legend_title_box.get_visible() else None,
            framealpha = l.get_frame().get_alpha(),
            bbox_to_anchor = l.get_bbox_to_anchor()._bbox,
            bbox_transform = l.get_bbox_to_anchor()._transform,
            frameon = l._drawFrame,
            handler_map = l._custom_handler_map,
        )
        mpl.pyplot.legend(**defaults)
        self.legend = self.ax.legend_
        self.lookup_artist, self.lookup_handle = self._build_lookups(self.legend)
        self._setup_connections()
        self.fig.canvas.draw()

    def show(self):
        plt.show()

if __name__ == '__main__':
    main()