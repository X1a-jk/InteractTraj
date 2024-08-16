import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({'figure.max_open_warning': 0})
from scipy.ndimage.filters import gaussian_filter
import matplotlib.colors as mcolors
import matplotlib.cm as cm


def get_heatmap(x, y, prob, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=prob, density=True)
    heatmap = gaussian_filter(heatmap, sigma=s)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent


def draw_heatmap(vector, vector_prob, gt_idx):
    _, ax = plt.subplots(figsize=(10, 10))
    vector_prob = vector_prob.cpu().numpy()

    for j in range(vector.shape[0]):
        if j in gt_idx:
            color = (0, 0, 1)
        else:
            grey_scale = max(0, 0.9 - vector_prob[j])
            color = (0.9, grey_scale, grey_scale)

        x0, y0, x1, y1, = vector[j, :4]
        ax.plot((x0, x1), (y0, y1), color=color, linewidth=2)

    return plt

def draw_seq_map(center,  other=None, heat_map=False, save_np=False, save=False, edge=None, path='../vis'):
    plt.switch_backend('agg')    
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.axis('equal')


    lane_color = 'green'
    bound_color = 'red'
    rest_color = 'blue'
    alpha = 0.12
    linewidth = 3
    ax.axis('off')

    for j in range(center.shape[0]):
        traf_state = center[j, -1]
        x0, y0, x1, y1, = center[j, :4]
        if x0 == 0: break
        ax.plot((x0, x1), (y0, y1), '--', color=lane_color, linewidth=1, alpha=0.2)

        if traf_state == 1:
            color = 'red'
            ax.plot((x0, x1), (y0, y1), color=color, alpha=alpha, linewidth=linewidth, zorder=5000)
        elif traf_state == 2:
            color = 'yellow'
            ax.plot((x0, x1), (y0, y1), color=color, alpha=alpha, linewidth=linewidth, zorder=5000)
        elif traf_state == 3:
            color = 'green'
            ax.plot((x0, x1), (y0, y1), color=color, alpha=alpha, linewidth=linewidth, zorder=5000)

    
    if edge is not None:
        colors = ['red', 'blue', 'orange', 'green', 'purple', 'brown']
        for j in range(len(edge)):
            cr = colors[j % len(colors)]
            # if lane[j, k, -1] == 0: continue
            x0, y0, x1, y1, = edge[j, :4]
            if x0 == 0: break
            ax.plot((x0, x1), (y0, y1), cr, linewidth=1.5)
            

    if other is not None:
        for j in range(len(other)):
            x0, y0, x1, y1, = other[j, :4]
            if x0 == 0: break
            ax.plot((x0, x1), (y0, y1), rest_color, linewidth=0.7, alpha=0.9)


    plt.autoscale()
    if save:
        fig.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0)
    elif save_np:
        fig.tight_layout()
        fig.canvas.draw()
        return np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return plt

def draw_seq(t, center, agents, traj=None, other=None, heat_map=False, save_np=False, save=False, edge=None, path='../vis'):
    plt.switch_backend('agg')
    
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.axis('equal')

    shapes = []
    collide = []
    poly = agents[0].get_polygon()[0]
    shapes.append(poly)
    for i in range(1, len(agents)):
        intersect = False
        poly = agents[i].get_polygon()[0]
        for shape in shapes:
            if poly.intersects(shape):
                intersect = True
                collide.append(i)
                break
        if not intersect:
            shapes.append(poly)

    colors = ['tab:red',
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan']

    lane_color = 'black'
    alpha = 0.12
    linewidth = 3

    if heat_map:
        lane_color = 'white'
        alpha = 0.2
        linewidth = 6
    ax.axis('off')
    
    for j in range(center.shape[0]):
        traf_state = center[j, -1]

        x0, y0, x1, y1, = center[j, :4]

        if x0 == 0: break
        ax.plot((x0, x1), (y0, y1), '--', color=lane_color, linewidth=1, alpha=0.2)
        if traf_state == 1:
            color = 'red'
            ax.plot((x0, x1), (y0, y1), color=color, alpha=alpha, linewidth=linewidth, zorder=5000)
        elif traf_state == 2:
            color = 'yellow'
            ax.plot((x0, x1), (y0, y1), color=color, alpha=alpha, linewidth=linewidth, zorder=5000)
        elif traf_state == 3:
            color = 'green'
            ax.plot((x0, x1), (y0, y1), color=color, alpha=alpha, linewidth=linewidth, zorder=5000)

    if edge is not None:
        for j in range(len(edge)):
            x0, y0, x1, y1, = edge[j, :4]
            if x0 == 0: break
            ax.plot((x0, x1), (y0, y1), lane_color, linewidth=1.5)
            
    if other is not None:
        for j in range(len(other)):
            x0, y0, x1, y1, = other[j, :4]
            if x0 == 0: break
            ax.plot((x0, x1), (y0, y1), lane_color, linewidth=0.7, alpha=0.9)
    
    for i in range(len(agents)):

        if i == 0:
            col = colors[0]
        else:
            ind = (i-1) % 9 + 1
            col = colors[ind]
        if traj is not None:
            traj_i = traj[:, i]
            len_t = traj_i.shape[0] - 1
            for j in range(len_t):
                x0, y0 = traj_i[j]
                x1, y1 = traj_i[j + 1]
                
                # only partial
                if j > t:
                    continue
                
                if abs(x0) < 100 and abs(y0) < 100 and abs(x1) < 100 and abs(y1) < 100:
                    ax.plot((x0, x1), (y0, y1), '-', color=col, linewidth=4.0, marker='.', markersize=10)
                

        agent = agents[i]
        agent_type = int(agent.type[0][0])
        colors_type = ["black", "red", "yellow"]
        rect = agent.get_rect()[0]

        rect = plt.Polygon(rect, edgecolor=colors_type[agent_type-1],
                           facecolor=col, linewidth=0.5, zorder=10000)
        ax.add_patch(rect)
    
    plt.autoscale()
    plt.xlim([-100, 100])
    plt.ylim([-100, 100])
    
    if save:
        fig.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0)
    elif save_np:
        fig.tight_layout()
        fig.canvas.draw()
        return np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return plt

def draw_traj(traj, save_np=False, save=False, edge=None, path='../vis'):
    plt.switch_backend('agg')    
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.axis('equal')
    collide = []
    colors = list(mcolors.TABLEAU_COLORS)
    plt.xlim([-100, 100])
    plt.ylim([-100, 100])
    ax.axis('on')

    for i in range(traj.shape[1]):
        if i in collide: continue
        ind = i % 10
        col = colors[ind]
        traj_i = traj[:, i]
        len_t = traj_i.shape[0] - 1
        for j in range(len_t):
            x0, y0 = traj_i[j]
            x1, y1 = traj_i[j + 1]
            if abs(x0) < 80 and abs(y0) < 80 and abs(x1) < 80 and abs(y1) < 80:
                ax.plot((x0, x1), (y0, y1), '-', color=col, linewidth=2.0, marker='.', markersize=3)

    if save:
        fig.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0)
    elif save_np:
        fig.canvas.draw()
        return np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return plt


def draw(center, agents, other, heat_map=None, save=False, edge=None, path='../vis', vis_range=60, save_np=False, showup=True, figsize=(10, 10), draw_traf_state=True):
    if not showup:
        plt.switch_backend('agg')

    fig, ax = plt.subplots(figsize=figsize)
    plt.axis('equal')

    colors = list(mcolors.TABLEAU_COLORS)
    lane_color = 'black'
    alpha = 0.12
    linewidth = 8
    if heat_map:
        lane_color = 'white'
        ax.imshow(heat_map[0], extent=heat_map[1], alpha=1, origin='lower', cmap=cm.jet)
        alpha = 0.5
        linewidth = linewidth
        plt.xlim(heat_map[1][:2])
        plt.ylim(heat_map[1][2:])
    ax.axis('on')

    for j in range(center.shape[0]):
        traf_state = center[j, -1]

        x0, y0, x1, y1, = center[j, :4]

        if x0 == 0: break
        ax.plot((x0, x1), (y0, y1), '--', color=lane_color, linewidth=1, alpha=0.2)

        if draw_traf_state:
            if traf_state == 1:
                color = 'red'
                ax.plot((x0, x1), (y0, y1), color=color, alpha=alpha, linewidth=linewidth, zorder=5000)
            elif traf_state == 2:
                color = 'yellow'
                ax.plot((x0, x1), (y0, y1), color=color, alpha=alpha, linewidth=linewidth, zorder=5000)
            elif traf_state == 3:
                color = 'green'
                ax.plot((x0, x1), (y0, y1), color=color, alpha=alpha, linewidth=linewidth, zorder=5000)

    if edge is not None:
        for j in range(len(edge)):
            x0, y0, x1, y1, = edge[j, :4]
            if x0 == 0: break
            ax.plot((x0, x1), (y0, y1), lane_color, linewidth=1)

    if other is not None:
        for j in range(len(other)):
            x0, y0, x1, y1, = other[j, :4]
            if x0 == 0: break
            ax.plot((x0, x1), (y0, y1), lane_color, linewidth=0.7)

    for i in range(len(agents)):
        ind = i % 10
        col = colors[ind]
        agent = agents[i]
        center = agent.position[0]
        if abs(center[0]) > (vis_range - 7) or abs(center[1]) > (vis_range - 7): continue
        vel = agent.velocity[0]
        rect = agent.get_rect()[0]
        rect = plt.Polygon(rect, edgecolor=lane_color,
                           facecolor=col, linewidth=0.5, zorder=10000)
        if abs(vel[0] + center[0]) < (vis_range - 2) and abs(vel[1] + center[1]) < (vis_range - 2):
            ax.plot([center[0], vel[0] + center[0]], [center[1], vel[1] + center[1]], '.-', color='lime', linewidth=1.2,
                    markersize=2, zorder=10000)
        ax.add_patch(rect)

    plt.autoscale()
    if save:
        fig.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0)
    elif save_np:
        fig.tight_layout()
        fig.canvas.draw()
        return np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return plt

