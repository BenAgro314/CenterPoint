# argument: file path like 
# work_dirs/waymo_centerpoint_voxelnet_3epoch/*.log.json

import sys
import os
import matplotlib.pyplot as plt
import json


def main():
    # read in file path as first arg
    file_path = os.path.abspath(sys.argv[1])

    dir_path = os.path.split(file_path)
    f = open(file_path, "r")
    iters = []
    loss = []
    hm_loss = []
    loc_loss = []
    x_loc_loss = []
    y_loc_loss = []
    z_loc_loss = []
    l_loc_loss = []
    w_loc_loss = []
    h_loc_los = []
    sin_loc_loss = []
    cos_loc_loss = []

    lr = []

    iters_per_epoch = 0
    for line in f:
        data = json.loads(line)
        epoch = data["epoch"]
        if epoch == 1:
            iters.append(data["iter"])
            iters_per_epoch = data["iter"]
        else:
            iters.append(data["iter"] + iters_per_epoch * epoch)
        loss.append(data["loss"])
        hm_loss.append(data["hm_loss"])
        loc_loss.append(data["loc_loss"])
        x_loc_loss.append(data["loc_loss_elem"][0][0])
        y_loc_loss.append(data["loc_loss_elem"][0][1])
        z_loc_loss.append(data["loc_loss_elem"][0][2])
        l_loc_loss.append(data["loc_loss_elem"][0][3])
        w_loc_loss.append(data["loc_loss_elem"][0][4])
        h_loc_los.append(data["loc_loss_elem"][0][5])
        sin_loc_loss.append(data["loc_loss_elem"][0][6])
        cos_loc_loss.append(data["loc_loss_elem"][0][7])
        
        lr.append(data["lr"])

    plt.plot(iters, loss, label="Total Loss")
    plt.plot(iters, hm_loss, label="Heatmap Loss")
    plt.plot(iters, loc_loss, label="Localization Loss")
    plt.plot(iters, x_loc_loss, label="X Localization Loss")
    plt.plot(iters, y_loc_loss, label="Y Localization Loss")
    plt.plot(iters, z_loc_loss, label="Z Localization Loss")
    plt.plot(iters, l_loc_loss, label="L Localization Loss")
    plt.plot(iters, w_loc_loss, label="W Localization Loss")
    plt.plot(iters, h_loc_los, label="H Localization Loss")
    plt.plot(iters, sin_loc_loss, label="Sin Localization Loss")
    plt.plot(iters, cos_loc_loss, label="Cos Localization Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    loss_image = os.path.join(dir_path[0], "loss.png")
    plt.savefig(loss_image)
    plt.close("all")
    plt.plot(iters, lr)
    plt.savefig(os.path.join(dir_path[0], "lr.png"))
    plt.close("all")


if __name__ == "__main__":
    main()