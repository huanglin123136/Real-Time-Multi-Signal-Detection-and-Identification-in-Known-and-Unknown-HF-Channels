import matplotlib.pyplot as plt
import numpy as np

def draw_plot(seqs, coords):
    seqs = np.array(seqs)
    coords = np.array(coords)
    seqs = seqs[0]
    scale = 1e5
    gap = scale / len(seqs)
    plt.figure()
    coords = coords.reshape(-1, 3)
    #    x = np.linspace(scale * idx + start, scale * (idx + 1)+ start, len(seqs + 1))
    x = np.linspace(0, 1e+05, seqs.shape[0])
    plt.plot(x, seqs, c='blue', linewidth=0.5)
    for i in range(len(coords)):
        pt0 = int(coords[i][0] * scale / gap)
        pt1 = int(coords[i][1] * scale / gap)
        #        if pt1 - pt0 > 1200 or pt1 - pt0 < 100:
        #            continue
        value_true = np.mean(seqs[pt0: pt1]) * np.ones(pt1 - pt0 + 1)
        x2 = np.array([i * gap for i in range(pt0, pt1 + 1)])
        #        x2 = np.array([i * gap + scale * idx for i in range(pt0, pt1 + 1)])
        plt.plot(x2, value_true, color='black', linewidth=3)
    plt.legend(loc=3, ncol=1)
    plt.xlabel('Hz')
    plt.ylabel('Spec')
    # plt.title('prediction for seq {}'.format(idx))
    # plt.xlim(scale * idx + start, scale * (idx + 1) + start)
    #    plt.xlim(scale * idx, scale * (idx + 1))
    # plt.xlim(0, 1)
    plt.ylim(-1, 1)
    # plt.savefig('sample.png', format='png', transparent=True, dpi=300, pad_inches=0)
    plt.show()



def draw_plotv2(seqs, coords, index):
    seqs = np.array(seqs)
    coords = np.array(coords)
    seqs = seqs[0]
    scale = 1e5
    gap = scale / len(seqs)
    plt.figure()
    coords = coords.reshape(-1, 3)
    #    x = np.linspace(scale * idx + start, scale * (idx + 1)+ start, len(seqs + 1))
    x = np.linspace(0, 1e+05, seqs.shape[0])
    plt.plot(x, seqs, c='blue', linewidth=0.5)
    for i in range(len(coords)):
        pt0 = int(coords[i][0] * scale / gap)
        pt1 = int(coords[i][1] * scale / gap)
        #        if pt1 - pt0 > 1200 or pt1 - pt0 < 100:
        #            continue
        value_true = np.mean(seqs[pt0: pt1]) * np.ones(pt1 - pt0 + 1)
        x2 = np.array([i * gap for i in range(pt0, pt1 + 1)])
        #        x2 = np.array([i * gap + scale * idx for i in range(pt0, pt1 + 1)])
        if coords[i][2] == 0:
            plt.plot(x2, value_true, color='red', linewidth=3)
        elif coords[i][2] == 1:
            plt.plot(x2, value_true, color='gray', linewidth=3)
        else:
            plt.plot(x2, value_true, color='black', linewidth=3)


    # plt.legend(loc=3, ncol=1)
    plt.xlabel('Hz')
    plt.ylabel('Spec')
    # plt.title('prediction for seq {}'.format(idx))
    # plt.xlim(scale * idx + start, scale * (idx + 1) + start)
    #    plt.xlim(scale * idx, scale * (idx + 1))
    # plt.xlim(0, 1)
    plt.ylim(-1, 1)
    plt.savefig('./pics/{}.png'.format(index), format='png', transparent=True, dpi=300, pad_inches=0)
    plt.show()