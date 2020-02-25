import numpy as np
import plotly.graph_objs as go

def layout(dark_mode=False,axis_range=None):
    
    scene = dict(xaxis = dict(nticks=10, range = (0,192), showticklabels=True, title='x'),
                 yaxis = dict(nticks=10, range = (0,192), showticklabels=True, title='y'),
                 zaxis = dict(nticks=10, range = (0,192), showticklabels=True, title='z'),
                 aspectmode='cube')
    if axis_range is not None:
        if len(np.shape(axis_range)) == 1:
            scene['xaxis']['range'] = axis_range
            scene['yaxis']['range'] = axis_range
            scene['zaxis']['range'] = axis_range
        elif len(np.shape(axis_range)) == 2:
            axis_names=['xaxis','yaxis','zaxis']
            for idx, values in enumerate(axis_range):
                scene[axis_names[idx]]['range'] = values
        else:
            sys.stdout.write('Error: axis_range %s is not supported shape (2) or (N,2) with N<4\n' % axis_range)
            raise ValueError
    
    layout = go.Layout(
    showlegend=True,
    legend=dict(x=1.01,y=0.95),
    width=1024,
    height=768,
    hovermode='closest',
    margin=dict(l=0,r=0,b=0,t=0),                                                                                                                                  
    uirevision = 'same',
    scene = scene,
    )
    if dark_mode:
        layout.template='plotly_dark'
    return layout
