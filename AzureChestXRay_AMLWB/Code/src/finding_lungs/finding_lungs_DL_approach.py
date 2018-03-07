import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
from keras.models import load_model
from skimage import exposure, img_as_float
from skimage import transform
import numpy as np

from skimage import measure
import lungs_finder as lf
import cv2

# for lung detection
left_edge = 0
right_edge = 256
top_edge = 0
bottom_edge = 256
margin = 12

row_size = 256
col_size = 256

# Path to csv-file. File should contain X-ray filenames as first column,
# mask filenames as second column.

out_folder_matched_img = os.path.join("/mnt", "MyAzureFileShare", "Data", "ChestXRay", "images_centered")
out_folder_mismatched_image = os.path.join("/mnt", "MyAzureFileShare", "Data", "ChestXRay",
                                           "images_centered_mismatched_by_both")
csv_path = os.path.join("/mnt", "MyAzureFileShare", "Data", "ChestXRay", "Data_Entry_2017.csv")
# Path to the folder with images. Images will be read from path + path_from_csv
img_path = os.path.join("/mnt", "MyAzureFileShare", "Data", "ChestXRay", "images")
mis_detected_csv_path = os.path.join("/mnt", "MyAzureFileShare", "Data", "ChestXRay", "mis_detected.csv")
df = pd.read_csv(csv_path)

# Load test data
im_shape = (256, 256)

# Load model
# plt.figure(figsize=(10, 10))
model_name = './trained_model.hdf5.bak'
UNet = load_model(model_name)

threshold = 0.85
# list to save the mis detected images
image_misdetect_list = []


def finding_lungs_non_DL_approach_and_save(image, file_name):
    # print(row.columns.values)
    # file_name = row[0]
    # print("line is", file_name, image.shape)
    # when reading from txt there is something in the end so we need to eliminate that
    # image = cv2.imread(os.path.join("Z:\\", "Data", "ChestXRay", "images", file_name), 0)

    img_height = image.shape[0]
    img_width = image.shape[1]
    # Get both lungs image. It uses HOG as main method,
    # but if HOG found nothing it uses HAAR or LBP.
    found_lungs = lf.get_lungs(image)

    # this can be written in a more concise way but we just keep it a bit redundant for easy reading
    if found_lungs is not None and found_lungs.shape[0] > img_height / 2 and found_lungs.shape[1] > img_width / 2:
        # print(found_lungs.shape)
        found_lungs_resized = cv2.resize(found_lungs, im_shape)
        # cv2.imshow(file_name, found_lungs)
        # code = cv2.waitKey(0)
        cv2.imwrite(os.path.join(out_folder_matched_img, file_name), found_lungs_resized)
        return True
    else:
        cv2.imwrite(os.path.join(out_folder_mismatched_image, file_name), cv2.resize(image, im_shape))
        return False


for index, item in df.iterrows():
    # X, y = loadDataGeneral(current_df, path, im_shape)
    raw_img = cv2.imread(os.path.join(img_path, item['Image Index']))

    img = img_as_float(raw_img)[:, :, 0]
    img = transform.resize(img, im_shape)
    img = exposure.equalize_hist(img)
    # img = np.expand_dims(img, -1)
    img -= img.mean()
    img /= img.std()

    file_name = item['Image Index']
    X = np.expand_dims(img, axis=0)
    X = np.expand_dims(X, axis=-1)
    n_test = X.shape[0]
    inp_shape = X[0].shape

    # img = exposure.rescale_intensity(np.squeeze(X), out_range=(0, 1))

    # print("size of img is", img.shape)
    prediction = UNet.predict(X)[..., 0].reshape(inp_shape[:2])

    thresh_img = np.where(prediction > threshold, 1.0, 0.0)  # threshold the image

    labels = measure.label(thresh_img)  # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    # print(label_vals)
    regions = measure.regionprops(labels)
    good_labels = []
    global_B_box = []
    for prop in regions:
        B = prop.bbox
        if B[2] - B[0] > row_size / 4 and B[3] - B[1] > col_size / 6:  # make sure size of lung to avoid small areas
            good_labels.append(prop.label)
            global_B_box.append(B)

    # print(len(good_labels))

    DL_failed_detect_flag = False
    if len(good_labels) == 2:

        left_edge = np.clip(min(global_B_box[0][1] - margin, global_B_box[1][1] - margin), a_min=0, a_max=256)
        right_edge = np.clip(max(global_B_box[0][3] + margin, global_B_box[1][3] + margin), a_min=0, a_max=256)
        top_edge = np.clip(min(global_B_box[0][0] - margin, global_B_box[1][0] - margin), a_min=0, a_max=256)
        bottom_edge = np.clip(max(global_B_box[0][2] + margin * 3, global_B_box[1][2] + margin * 4), a_min=0,
                              a_max=256)  # leave more margins at the bottom
    else:
        # print(file_name)

        DL_failed_detect_flag = True

    if DL_failed_detect_flag:
        img_name = os.path.join(out_folder_mismatched_image, file_name)
        if not finding_lungs_non_DL_approach_and_save(raw_img, file_name):
            # save file name only if both methods are not detected
            image_misdetect_list.append(file_name)
            print(file_name)
    else:
        img_name = os.path.join(out_folder_matched_img, file_name)
        cropped = cv2.resize(raw_img, im_shape)[top_edge:bottom_edge, left_edge:right_edge]
        # print(cropped)
        resized_cropped = cv2.resize(cropped, im_shape)
        cv2.imwrite(img_name, resized_cropped)

    # if mis_detected_flag:
    #     mis_detected_flag = False
    #     fig, ax = plt.subplots(2, 2, figsize=[12, 12])
    #     ax[0, 0].set_title("Original " + file_name)
    #     ax[0, 0].imshow(raw_img, cmap='gray')
    #     ax[0, 0].axis('off')
    #     ax[0, 1].set_title("Threshold " + file_name)
    #     ax[0, 1].imshow(thresh_img, cmap='gray')
    #     #         ax[0, 1].imshow(prediction, cmap='gray')
    #     ax[0, 1].axis('off')
    #     ax[1, 0].set_title("Color Labels " + file_name)
    #     ax[1, 0].imshow(labels)
    #     ax[1, 0].axis('off')
    #     ax[1, 1].set_title("Apply Mask on Original " + file_name)
    #
    #     ax[1, 1].imshow(resized_cropped, cmap='gray')
    #     ax[1, 1].axis('off')

    if index > 112120:  # for debug purpose
        break

    if index % 100 == 0:
        df = pd.DataFrame({'col': image_misdetect_list})
        df.to_csv(mis_detected_csv_path, header=False, index=False)

df = pd.DataFrame({'col': image_misdetect_list})
df.to_csv(mis_detected_csv_path, header=False, index=False)
