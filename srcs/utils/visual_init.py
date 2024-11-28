import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({'figure.max_open_warning': 0})
from scipy.ndimage.filters import gaussian_filter
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as patches

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
    
    fig, ax = plt.subplots(figsize=(9, 9))
    plt.axis('equal')
    ax.axis('off') 
    
    background_rect = patches.Rectangle(
        (0, 0), 1, 1, transform=ax.transAxes, color='white', zorder=0
    )
    ax.add_patch(background_rect)

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

    # Generate 32 distinct dark colors using Dark2 colormap
    cmap = plt.get_cmap('Set1')  # Use 'Set1' colormap for bright and high-contrast colors
    colors = [cmap(i % 9) for i in range(32)]  # Generate 32 colors using 'Set1'
    poly_color = '#D3D3D3' 
    lane_color = 'black'
    alpha = 0.12
    linewidth = 3

    if heat_map:
        lane_color = 'black'
        alpha = 0.2
        linewidth = 6
    ax.axis('off')

    
    for j in range(center.shape[0]):
    

        traf_state = center[j, -1]

        x0, y0, x1, y1 = center[j, :4]

        if x0 == 0:
            break

        vec_x = x1 - x0
        vec_y = y1 - y0
    
        norm = np.sqrt(vec_x**2 + vec_y**2)
        if norm == 0:
            continue

        perp_x = -vec_y / norm
        perp_y = vec_x / norm
    

        radius = 1.9
        half_width = radius
    

        p1 = (x0 + perp_x * half_width, y0 + perp_y * half_width)
        p2 = (x0 - perp_x * half_width, y0 - perp_y * half_width)
        p3 = (x1 - perp_x * half_width, y1 - perp_y * half_width)
        p4 = (x1 + perp_x * half_width, y1 + perp_y * half_width)
    

        connector_polygon = patches.Polygon([p1, p2, p3, p4], closed=True, edgecolor=None,
                               facecolor=poly_color, alpha=1.0, zorder=2000)
        ax.add_patch(connector_polygon)

        circle1 = patches.Circle((x0, y0), radius=half_width, edgecolor=None,
                             facecolor=poly_color, alpha=1.0, zorder=2000)
        ax.add_patch(circle1)

        circle2 = patches.Circle((x1, y1), radius=half_width, edgecolor=None,
                             facecolor=poly_color, alpha=1.0, zorder=2000)
        ax.add_patch(circle2)

        ax.plot((x0, x1), (y0, y1), '--', color=lane_color, linewidth=3, alpha=0.2,zorder=5000)

        
       
    '''
    if traf_state == 1:
        color = 'red'
        ax.plot((x0, x1), (y0, y1), color=color, alpha=0.7, linewidth=lane_width, zorder=5000)
    elif traf_state == 2:
        color = 'yellow'
        ax.plot((x0, x1), (y0, y1), color=color, alpha=0.7, linewidth=lane_width, zorder=5000)
    elif traf_state == 3:
        color = 'green'
        ax.plot((x0, x1), (y0, y1), color=color, alpha=0.7, linewidth=lane_width, zorder=5000)
    '''
    if edge is not None:
        for j in range(len(edge)):
            x0, y0, x1, y1 = edge[j, :4]

            if x0 == 0:
                break
        
            arrow = patches.FancyArrowPatch((x0, y0), (x1, y1),
                                        arrowstyle='->',
                                        color=lane_color,
                                        mutation_scale=10, 
                                        linewidth=1.5,
                                        zorder=5000) 
            ax.add_patch(arrow)
            
    if other is not None:
        for j in range(len(other)):
            x0, y0, x1, y1, = other[j, :4]
            if x0 == 0: break
            ax.plot((x0, x1), (y0, y1), lane_color, linewidth=0.7, alpha=0.9,zorder=5000)
    
    for i in range(len(agents)):
        agent = agents[i]
        agent_type = int(agent.type[0][0])
        col = colors[i % 32]  # Use one of the dark colors from Dark2 colormap
        if traj is not None:
            
            traj_i = traj[:, i]
            # print(i,traj_i,"visual_init")
            len_t = traj_i.shape[0] - 1
            for j in range(len_t):
                x0, y0 = traj_i[j]
                x1, y1 = traj_i[j + 1]
                
                # 设置 alpha 随时间步变浅
                alpha = 1 - 0.8 * (j / len_t)  # 这里 alpha 随 j 递减，初始值为1，结束时接近0

                # 获取颜色并设置透明度
                col_with_alpha = list(col)  # 复制颜色
                col_with_alpha[-1] = alpha  # 修改 alpha 通道

                if abs(x0) < 1000 and abs(y0) < 1000 and abs(x1) < 1000 and abs(y1) < 1000:
                    ax.plot((x0, x1), (y0, y1), '-', color=col_with_alpha, linewidth=10.0,zorder=7000)# marker='.', markersize=10)
                

        colors_type = ["black", "red", "yellow"]
        # 获取多边形的顶点坐标
        rect_coords = agent.get_rect()[0]  # 确保这里获取的是顶点的坐标数组

        # 创建多边形
        polygon = plt.Polygon(rect_coords, edgecolor="black",
                          facecolor=col, linewidth=1.5, zorder=10000)
        ax.add_patch(polygon)

        # 计算多边形的质心
        centroid_x = np.mean(rect_coords[:, 0])
        centroid_y = np.mean(rect_coords[:, 1])
        

        # 在多边形中心添加数字 i 并设置为粗体
        ax.text(centroid_x, centroid_y, str(i), fontsize=12, fontweight='bold', 
                color='black', ha='center', va='center', zorder=10001)

    if save_np:
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img_array

    if save:
        fig.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    return plt
def draw_traj(traj, save_np=False, save=False, edge=None, path='../vis', abn_idx=None):
    plt.switch_backend('agg')
    
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.axis('equal')

    shapes = []
    collide = []
    colors = list(mcolors.TABLEAU_COLORS)
    lane_color = 'black'
    alpha = 0.12
    linewidth = 2.0

    plt.xlim([-100, 100])
    plt.ylim([-100, 100])

    ax.axis('on')

    for i in range(traj.shape[1]):
        if i in collide: continue
        #     face_color = col
        ind = i % 10
        col = colors[ind]

        # color='red'
        # face_color = 'black' #col

        traj_i = traj[:, i]
        len_t = traj_i.shape[0] - 1
        for j in range(len_t):
            # if j<3:
            #     color='red'
            #     #face_color = 'black' #col
            # else:
            #     #break
            #     color = 'red'
            x0, y0 = traj_i[j]
            x1, y1 = traj_i[j + 1]

            if abs(x0) < 80 and abs(y0) < 80 and abs(x1) < 80 and abs(y1) < 80:
                ax.plot((x0, x1), (y0, y1), '-', color=col, linewidth=2.0, marker='.', markersize=3)
    # ax.set_facecolor('black')
    # plt.autoscale()
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

