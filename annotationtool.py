import numpy as np
import numpy.random.common
import numpy.random.bounded_integers
import numpy.random.entropy
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.backend_bases import NavigationToolbar2
import os

GOLBALEPREFIX_EXTENSTION = "Distance_"
GOLBALEPREFIX_FORCE = "Tension_"


def read_folder(path, endswith=".itx", type="."):
    return [os.path.join(root, name)
            for root, dirs, files in os.walk(path)
            for name in files
            if (name.endswith(endswith) and name.find(type) >= 0)]


def read_itx(filename):
    force = []
    with open(filename) as file1:
        lines = file1.readlines()
        lines = lines[3:-3]
        for each in lines:
            force.append(float(each.strip()))
    force = np.array(force)
    return force


def generate_displayset():
    raw_data = read_folder("data", type=GOLBALEPREFIX_EXTENSTION)
    generated_dataset_atlas = []
    for each in raw_data:
        generated_dataset_atlas.append(
            each.replace("/", ";").replace("\\", ";")+".npy")
    dataset_data = read_folder("output", ".npy")
    for each in dataset_data:
        each = each.replace("output", "").replace("\\", "").replace("/", "")
        if not each in generated_dataset_atlas:
            print("[INFO] adding extra dataset {} into display set. ".format(each))
            generated_dataset_atlas.append(each)
    return generated_dataset_atlas


def convert_smfs(name_of_generated_dataset_atlas, marked_index_this=set([]), distance_prefix="Distance_", force_prefix="Tension_"):
    print("[INFO] dataset file not found, building it from atlas.",
          name_of_generated_dataset_atlas)
    extension_this = read_itx(
        name_of_generated_dataset_atlas.replace(";", "/").replace(".npy", ""))
    force_this = read_itx(name_of_generated_dataset_atlas.replace(
        ";", "/").replace(distance_prefix, force_prefix).replace(".npy", ""))
    mask_this = np.zeros((extension_this.shape[0]))
    for each in marked_index_this:
        mask_this[each] = 1
    np.save("output/"+name_of_generated_dataset_atlas,
            np.vstack((extension_this, force_this, mask_this)))
    return extension_this, force_this, mask_this


def generate_csv():
    dataset_data = read_folder("output", ".npy")
    with open("output.csv", "w") as file1:
        file1.write("data,force,ext,orig\n")
        for each in dataset_data:
            loaded = np.load(each)
            extension_this = loaded[0]
            force_this = loaded[1]
            marked_index_this = []
            for i in range(0, len(loaded[2])):
                if loaded[2][i] > 0:
                    marked_index_this.append(i)
            extension_this = extension_this[marked_index_this]
            force_this = force_this[marked_index_this]
            file1.write("{},{},{},{}\n".format(each.replace("output", "").replace("\\", "").replace("/", ""),
                                               str(extension_this).replace("\n ", ""), str(force_this).replace("\n ", ""), each.replace("output", "").replace("\\", "").replace("/", "").replace(".npy", "").replace(";", "/")))


if __name__ == "__main__":

    generated_dataset_atlas = generate_displayset()

    def read_smfs(graphid):
        global extension_this
        global force_this
        global marked_index_this

        try:
            loaded = np.load("./output/"+generated_dataset_atlas[graphid])
            extension_this = loaded[0]
            force_this = loaded[1]
            mask_this = loaded[2]
        except Exception as ex:
            print(ex)
            extension_this, force_this, mask_this = convert_smfs(
                generated_dataset_atlas[graphid], distance_prefix=GOLBALEPREFIX_EXTENSTION, force_prefix=GOLBALEPREFIX_FORCE)

        marked_index_this = []
        for i in range(0, len(mask_this)):
            if mask_this[i] > 0:
                marked_index_this.append(i)
        marked_index_this = set(marked_index_this)
        print("=================================")
        print("makred index", marked_index_this)
        print("sample points for this graph is ", len(extension_this))
        print("max distance", np.max(extension_this))
        print("min distance", np.min(extension_this))
        print("stanard speed (nm/Hz) at Constant Speed mode",
              (np.max(extension_this)-np.min(extension_this))/len(extension_this))

    def reload(new_graphid):
        global ax
        global plt
        global fig
        global graphid
        global markers_obj_this
        global marked_index_this
        global generated_dataset_atlas
        global extension_this
        global force_this
        global extension_display
        global force_display

        if new_graphid >= len(generated_dataset_atlas):
            return
        if ('extension_this' in globals() and 'marked_index_this' in globals()):
            mask_this = np.zeros((extension_this.shape[0]))
            for each in marked_index_this:
                mask_this[each] = 1
            np.save("output/"+generated_dataset_atlas[graphid],
                    np.vstack((extension_this, force_this, mask_this)))

            print(extension_this[list(marked_index_this)])
            print(force_this[list(marked_index_this)])

        graphid = new_graphid
        plt.cla()
        ax.set_title('SMFS (o:zoom, p:pan, r:reset, right click to set point)')
        fig.canvas.set_window_title(
            generated_dataset_atlas[graphid]+" - SMFS event marker (Zuzeng Lin, ver 20200329)")

        read_smfs(graphid)

        if os.path.exists("smoothing.txt"):
            extension_display = moving_average(extension_this)
            force_display = moving_average(force_this)
        else:
            extension_display = extension_this
            force_display = force_this
        ax.plot(extension_display, force_display,
                picker=1)

        markers_obj_this = ax.plot(extension_display[list(
            marked_index_this)], force_display[list(marked_index_this)], 'r+')[0]
        fig.canvas.draw()

    def moving_average(x, w=5):
        ret = np.convolve(x, np.ones(w), 'same') / w
        ret[0:w//2] = ret[w]
        ret[-w//2:] = ret[-w]

        return ret

    def onpick(event):
        global markers_obj_this
        global marked_index_this
        global extension_display
        global force_display
        if event.mouseevent.button == 1:
            for each in event.ind:
                marked_index_this.add(each)
        else:
            for each in event.ind:
                marked_index_this.discard(each)

        if markers_obj_this is not None:
            markers_obj_this.remove()
        markers_obj_this = ax.plot(extension_display[list(
            marked_index_this)], force_display[list(marked_index_this)], 'r+')[0]
        fig.canvas.draw()
        return True

    def new_forward(self, *args, **kwargs):
        global graphid
        reload(graphid+1)

    def new_back(self, *args, **kwargs):
        global graphid
        if graphid > 0:
            reload(graphid-1)

    NavigationToolbar2.back = new_back
    NavigationToolbar2.forward = new_forward
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('pick_event', onpick)
    graphid = 0
    reload(graphid)
    plt.show()
    generate_csv()
