import os
import pickle
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

pickle_dir = '/home/bagro/saved_outputs/'

# list all files in directory
files = os.listdir(pickle_dir)

objects = metrics_pb2.Objects()

files = sorted(files, key=lambda x: (int(x.split("_")[0]), int(x.split('_')[1].split(".")[0])))

for file in files:
    assert file.endswith(".pkl")
    seq_id = file.split('_')[0]
    frame_id = file.split('_')[1].split(".")[0]
    # if frame_id not in ["183", "184", "185"]:
    #     continue
    # if int(frame_id) > 7:
    #     break
    print(f'Parsing {seq_id}, {frame_id}')
    with open(os.path.join(pickle_dir, file), 'rb') as f:
        data = pickle.load(f)
        for ob in data:
            o = metrics_pb2.Object()
            o.context_name = ob["context_name"]
            o.frame_timestamp_micros = ob["frame_timestamp_micros"]
            box = label_pb2.Label.Box()
            box.center_x = ob["box"]["x"]
            box.center_y = ob["box"]["y"]
            box.center_z = ob["box"]["z"]
            box.length = ob["box"]["l"]
            box.width = ob["box"]["w"]
            box.height = ob["box"]["h"]
            box.heading = ob["box"]["theta"]
            o.object.box.CopyFrom(box)
            o.score = ob["score"]
            o.object.type = ob["object_type"]

            objects.objects.append(o)

path = 'waabi_pred.bin'
print(f"results saved to {path}")
f = open(path, 'wb')
f.write(objects.SerializeToString())
f.close()