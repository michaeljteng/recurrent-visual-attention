import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

from utils import denormalize, bounding_box

#  if args.viz:
#      makedirs('png')
#      import matplotlib.pyplot as plt
#      fig = plt.figure(figsize=(12, 4), facecolor='white')
#      ax_traj = fig.add_subplot(131, frameon=False)
#      ax_phase = fig.add_subplot(132, frameon=False)
#      ax_vecfield = fig.add_subplot(133, frameon=False)
#      plt.show(block=False)
#
#
#  def visualize(true_y, pred_y, odefunc, itr):
#
#      if args.viz:
#
#          ax_traj.cla()
#          ax_traj.set_title('Trajectories')
#          ax_traj.set_xlabel('t')
#          ax_traj.set_ylabel('x,y')
#          ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 0], t.numpy(), true_y.numpy()[:, 0, 1], 'g-')
#          ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 0], '--', t.numpy(), pred_y.numpy()[:, 0, 1], 'b--')
#          ax_traj.set_xlim(t.min(), t.max())
#          ax_traj.set_ylim(-2, 2)
#          ax_traj.legend()
#
#          ax_phase.cla()
#          ax_phase.set_title('Phase Portrait')
#          ax_phase.set_xlabel('x')
#          ax_phase.set_ylabel('y')
#          ax_phase.plot(true_y.numpy()[:, 0, 0], true_y.numpy()[:, 0, 1], 'g-')
#          ax_phase.plot(pred_y.numpy()[:, 0, 0], pred_y.numpy()[:, 0, 1], 'b--')
#          ax_phase.set_xlim(-2, 2)
#          ax_phase.set_ylim(-2, 2)
#
#          ax_vecfield.cla()
#          ax_vecfield.set_title('Learned Vector Field')
#          ax_vecfield.set_xlabel('x')
#          ax_vecfield.set_ylabel('y')
#
#          y, x = np.mgrid[-2:2:21j, -2:2:21j]
#          dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2))).cpu().detach().numpy()
#          mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
#          dydt = (dydt / mag)
#          dydt = dydt.reshape(21, 21, 2)
#
#          ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
#          ax_vecfield.set_xlim(-2, 2)
#          ax_vecfield.set_ylim(-2, 2)
#
#          fig.tight_layout()
#          plt.savefig('png/{:03d}'.format(itr))
#          plt.draw()
#          plt.pause(0.001)

def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument("--plot_dir", type=str, required=True,
                     help="path to directory containing pickle dumps")
    arg.add_argument("--epoch", type=int, required=True,
                     help="epoch of desired plot")
    args = vars(arg.parse_args())
    return args['plot_dir'], args['epoch']


def main(plot_dir, epoch):

    # read in pickle files
    glimpses = pickle.load(
        open(plot_dir + "g_{}.p".format(epoch), "rb")
    )
    locations = pickle.load(
        open(plot_dir + "l_{}.p".format(epoch), "rb")
    )

    glimpses = np.concatenate(glimpses)

    # grab useful params
    size = int(plot_dir.split('gsize:')[1].split('x')[0])
    #  size = 128
    num_anims = len(locations)
    num_cols = glimpses.shape[0]
    img_shape = glimpses.shape[1]

    # denormalize coordinates
    coords = [denormalize(img_shape, l) for l in locations]

    fig, axs = plt.subplots(nrows=1, ncols=num_cols)
    # fig.set_dpi(100)

    # plot base image
    if len(glimpses[0].shape) == 2:
        grayscale = True
    else:
        grayscale = False
    for j, ax in enumerate(axs.flat):
        #  import pdb; pdb.set_trace()
        if grayscale:
            ax.imshow(glimpses[j], cmap="Greys_r")
        else:
            ax.imshow(np.transpose(glimpses[j]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    #
    #      color = 'r'
    #      # coords indexed by num_glimpses
    #      image_coords = coords[0]
    #
    #      # get the glimpse for the example image
    #      single_glimpse_loc = image_coords[j]
    #      for p in ax.patches:
    #          p.remove()
    #      rect = bounding_box(single_glimpse_loc[0], single_glimpse_loc[1], 32, color)
    #      ax.add_patch(rect)
    #  plt.show()


    def updateData(i):
        color = 'r'
        co = coords[i]
        for j, ax in enumerate(axs.flat):
            for p in ax.patches:
                p.remove()
            c = co[j]
            rect = bounding_box(
                c[0], c[1], size, color
            )
            ax.add_patch(rect)

    # animate
    anim = animation.FuncAnimation(
        fig, updateData, frames=num_anims, interval=1000, repeat=True
    )

    # save as mp4
    plt.show()
    #  name = plot_dir + 'epoch_{}.mp4'.format(epoch)
    #  anim.save(name, extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])


if __name__ == "__main__":
    args = parse_arguments()
    main(*args)
