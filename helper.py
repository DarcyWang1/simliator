import numpy as np
import matplotlib.pyplot as plt

def draw(data):
    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a colormap where True values are colored and False values are not
    colors = np.empty(data.shape, dtype=object)
    colors[data] = 'red'

    # Create the voxel plot
    ax.voxels(data, facecolors=colors, edgecolor='k')
    print(3)
    plt.show()

def draw_edge(indexes: np.ndarray, start: np.ndarray, end: np.ndarray):
    v = end - start  # 3
    indexes=np.moveaxis(indexes,0,-1)
    w = indexes - start  # m*n*z*3
    l = np.sum(np.abs(w)>np.abs(v),axis=-1)==0
    s = np.sum(np.abs(indexes-end)>np.abs(v),axis=-1)==0

    # Calculate the distance d from P to the closest point on the line
    d = (w @ v) / np.dot(v, v)  # m*n*z

    # Calculate the closest point Y on the line to X
    Y = start + v * np.stack([d] * 3, axis=-1)  # m*n*z*3
    #print('i',indexes[0,-1,0])
    #print('Y',Y[0,-1,0])
    #print('s',start,'e',end)
    msk = np.sum((Y - indexes) ** 2, axis=-1) <= 1  # m*n*z
    #print(numpy.sum((Y - indexes) ** 2, axis=-1)[0,19,0])
    #print(msk[0,19,0])
    return msk*l*s

if __name__ == '__main__':
    # Generate a random 3D boolean array
    #data = np.random.choice([0, 1], size=(500,500,500))
    z = np.indices((100,100,100))
    print(1)
    start = np.asarray([10,20,50])
    end = np.asarray([20,30,30])
    draw(draw_edge(z,start,end))
