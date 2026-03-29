import numpy as np
import os

# Data directory: set DATA_DIR environment variable or defaults to ./data
data_path = os.environ.get('DATA_DIR', './data')

def data_prep(data):
    import numpy as np
    if data == "MNIST":
        X = np.load(data_path + '/mnist_images.npy', allow_pickle=True).reshape(70000, 28 * 28)
        labels = np.load(data_path + '/mnist_labels.npy', allow_pickle=True)
    elif data == "FMNIST":
        X = np.load(data_path + '/fmnist_images.npy', allow_pickle=True).reshape(70000, 28 * 28)
        labels = np.load(data_path + '/fmnist_labels.npy', allow_pickle=True)
    elif data == "USPS":
        X = np.load(data_path + '/USPS.npy', allow_pickle=True)
        labels = np.load(data_path + '/USPS_labels.npy', allow_pickle=True)
    elif data == "20NG":
        X = np.load(data_path + '/20NG.npy', allow_pickle=True)
        labels = np.load(data_path + '/20NG_labels.npy', allow_pickle=True)
    elif data == "COIL20":
        X = np.load(data_path + '/coil_20.npy', allow_pickle=True).reshape(1440, 128 * 128)
        labels = np.load(data_path + '/coil_20_labels.npy', allow_pickle=True)
        labels = labels-1
    elif data == "AirPlane":
        X = np.load(data_path + '/airplane.npy', allow_pickle=True)
        labels = np.load(data_path + '/airplane_label.npy', allow_pickle=True)
    elif data == "Mammoth":
        import json
        with open(data_path + "/mammoth_umap.json", 'r') as load_f:
            data_json = json.loads(load_f.read())
        labels = np.array(data_json["labels"])
        X = np.array(data_json["3d"])
    elif data == "kang":
        X = np.load(data_path + "/kang_log_pca.npy", allow_pickle=True)
        labels = np.load(data_path + "/kang_labels.npy", allow_pickle=True)
    elif data == 'seurat':
        X = np.load(data_path + "/seurat_bmnc_rna_70.npy", allow_pickle=True)
        labels_ = np.load(data_path + "/seurat_bmnc_label.npy", allow_pickle=True)
        label_dic = []
        labels = np.zeros(labels_.shape, dtype=np.int32)
        num = 0
        for i in range(len(labels_)):
            if labels_[i] not in label_dic:
                label_dic.append(labels_[i])
                num += 1
            labels[i] = label_dic.index(labels_[i])
    elif data =="neurips2021_total":
        X = np.load(data_path + "/neurips2021_pca.npy", allow_pickle=True)
        labels = np.load(data_path + "/neurips2021_new_labels.npy", allow_pickle=True)
    elif data == "human_cortex_version3":
        X = np.load(data_path + "/human_cortex_pca_version3.npy", allow_pickle=True)
        labels = np.load(data_path + "/human_cortex_new_labels_version3.npy", allow_pickle=True)
    elif data == "seurat_new":
        X = np.load(data_path + "/seurat_data.npy", allow_pickle=True)
        labels = np.load(data_path + "/seurat_label.npy", allow_pickle=True)
    elif data == "fico":
        X_path = os.path.join(data_path, "FICO_data.npy")
        label_path = os.path.join(data_path, "FICO_label.npy")
        X = np.load(X_path).astype(np.float32)
        labels = np.load(label_path)
        X = X[labels > 0]
        labels = labels[labels > 0]
    return X, labels