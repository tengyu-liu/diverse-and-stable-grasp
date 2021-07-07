import numpy as np
import plotly.graph_objects as go

def plot_obj(mesh):
    return go.Mesh3d(
        x=mesh.vertices[:,0], 
        y=mesh.vertices[:,1], 
        z=mesh.vertices[:,2], 
        i=mesh.faces[:,0], 
        j=mesh.faces[:,1], 
        k=mesh.faces[:,2], 
        color='lightblue')

def plot_hand(verts, faces):
    return go.Mesh3d(
        x=verts[:,0], 
        y=verts[:,1], 
        z=verts[:,2], 
        i=faces[:,0], 
        j=faces[:,1], 
        k=faces[:,2], 
        color='lightpink')

def plot_contact_points(pts, grad):
    pts = pts.detach().cpu().numpy()
    grad = grad.detach().cpu().numpy()
    grad = grad / np.linalg.norm(grad, axis=-1, keepdims=True)
    return go.Cone(x=pts[:,0], y=pts[:,1], z=pts[:,2], u=-grad[:,0], v=-grad[:,1], w=-grad[:,2], anchor='tip',
                            colorscale=[(0,'lightpink'), (1,'lightpink')], sizemode='absolute', sizeref=0.2, opacity=0.5)