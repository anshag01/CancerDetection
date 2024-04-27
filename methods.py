import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from PIL import Image
from scipy.stats import kurtosis, skew
from skimage import color, feature
from skimage.draw import disk
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor_kernel
from sklearn.decomposition import PCA
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from tqdm import tqdm

import methods

GABOR_FREQUENCIES = [0.05, 0.15, 0.25]
GABOR_THETAS = [0, np.pi / 4, np.pi / 2]
GABOR_SIGMAS = [1, 3]

GLCM_DISTANCES = [1, 3, 5, 10, 50, 100]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def get_labels(repo_dir):
    label = pd.read_csv(
        os.path.join(repo_dir, "../dataverse_files/", "HAM10000_metadata.csv")
    )

    # label = label.set_index('image_id')
    cancerous = ["akiec", "bcc", "mel"]
    non_cancerous = ["bkl", "df", "nv", "vasc"]
    label["cancer"] = False
    label.loc[label["dx"].isin(cancerous), "cancer"] = True
    label.loc[label["dx"].isin(non_cancerous), "cancer"] = False

    return label


def z_normalization(image: np.array, normalize=True) -> np.array:
    if normalize:
        mean = np.mean(image, axis=(0, 1))
        std_dev = np.std(image, axis=(0, 1))
        if np.any(std_dev > 0):
            normalized_image = (image - mean) / std_dev
        else:
            normalized_image = image - mean
        return normalized_image


def convert_grayscale(image_path):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resized_img = cv2.resize(gray_img, (600, 450))
    return gray_img


def convert_rgb(image_path):
    """
    Converts the image to normalised RGB format
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def color_distribution(img):
    """
    Returns the color distribution of the image
    """
    df = pd.DataFrame(img.reshape(-1, 3), columns=["R", "G", "B"])
    color_distribution = df.describe()
    return color_distribution


def create_rgb_histogram(img):
    """
    Create one RGB histogram based on the distribution of the image
    """
    img = cv2.imread(img)
    color = ("b", "g", "r")
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()
    return histr


def create_histogram(image, color_space="RGB"):
    """
    :param image: cv2 RGB input image
    :return: Create one RGB histogram based on the distribution of the image.
    """

    if color_space == "RGB":
        pass
    elif color_space == "HSV":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    else:
        assert False

    # Split the image into its respective channels
    channels = cv2.split(image)

    # Calculate histograms
    histograms = []
    for channel in channels:
        hist, bins = np.histogram(channel, bins=256, range=[0, 256])
        histograms.append(hist)

    return histograms


def get_contrast(img):
    """
    Returns the contrast of the image
    """
    img = convert_grayscale(img)
    std_dev = np.std(img)  # Contrast
    return std_dev


def load_preprocess_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def extract_contours(thresh):
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    return contours


def classify_shapes(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * (area / (perimeter**2))
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

    # print("Circularity:", circularity, "Vertices:", len(approx))
    if len(approx) > 4 and circularity > 0.6:
        return "circular"
    elif len(approx) < 5 and circularity < 0.7:
        return "pointed"
    else:
        return "irregular"


def lbp_features(image_path, radius=1, n_points=8, method="uniform"):
    image = cv2.imread(image_path)
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(image, n_points, radius, method)

    hist, _ = np.histogram(
        lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2)
    )
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-7

    return hist


def calculate_glcm_features_for_blob(gray_image, blob):
    """
    Function to calculate GLCM features for a specific region (blob) in the image

    :param gray_image:
    :param blob:
    :return: feature vector
    """
    y, x, r = blob

    # generate a mask for the blob area
    rr, cc = disk((y, x), r, shape=gray_image.shape)
    mask = np.zeros_like(gray_image, dtype=bool)
    mask[rr, cc] = True

    # use the mask to select the region of interest
    blob_region = gray_image[mask]

    # compute the GLCM for the selected region
    roi_image = np.zeros_like(gray_image)
    roi_image[rr, cc] = blob_region

    # compute the GLCM
    glcm = graycomatrix(
        roi_image,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        symmetric=True,
        normed=True,
    )

    # feature extraction
    properties = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    ]

    feature_vector = []
    for prop in properties:
        # Flatten to convert from 2D to 1D
        temp = graycoprops(glcm, prop).flatten()

        # Taking mean across different angles
        feature_vector.append(np.mean(temp))

    return feature_vector


def calculate_glcm_features(image: np.ndarray) -> list:
    """
    Compute the Gray-Level Co-Occurrence Matrix (GLCM)

    :param image:
    :return: feature vector
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Normalize pixel values to 0 - 255
    gray_image = np.uint8(
        (gray_image - gray_image.min()) / (gray_image.max() - gray_image.min()) * 255
    )

    glcm = graycomatrix(
        gray_image,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        symmetric=True,
        normed=True,
    )

    # Feature extraction
    properties = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    ]
    feature_vector = []

    for prop in properties:
        temp = graycoprops(glcm, prop).flatten()  # Flatten to convert from 2D to 1D
        feature_vector.append(np.mean(temp))  # Taking mean across different angles

    return feature_vector


def detect_significant_blob(image, plot_image=False, plot_chosen_transformation=False):
    """
    Loosely based on https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html
    Goal: detect a significant circular blob inside an image

    :param image: cv2 in RGB (!) format
    :param plot_image: produce plot of detected blob overlayed on input image
    :param plot_chosen_transformation: if chosen the transformed image is plotted (i.e., saturation channel)
    :return: (y, x, r) of detected blob
    """

    # transformed_image = color.rgb2gray(image)
    transformed_image = color.rgb2hsv(image)[
        :, :, 1
    ]  # extract the hue channel or the channel of interest
    transformed_image = (transformed_image - np.min(transformed_image)) / (
        np.max(transformed_image) - np.min(transformed_image)
    )

    # detect blobs using transformed image
    blobs = []
    i = 0
    threshold = 0.02
    decrement = 0.95
    max_sigma = 500  # tweak these values a bit
    height, width = transformed_image.shape[:2]

    while len(blobs) <= 3:
        blobs = feature.blob_doh(
            transformed_image, max_sigma=max_sigma, threshold=threshold
        )
        i += 1
        max_sigma *= 0.97
        threshold *= decrement

        # select only the blobs that are fully inside the image
        blobs = np.array(
            [
                (y, x, r)
                for (y, x, r) in blobs
                if (r > 40 and x > 0 and y > 0 and r < max_sigma)
                or (
                    y - r >= 0
                    and x - r >= 0
                    and r >= 10
                    and y + r <= height
                    and x + r <= width
                )
            ]
        )
        if i == 50:
            break

    if blobs.size == 0:
        print("No blobs detected.")
        blobs = [(height / 2, width / 2, 150)]

    calculate_stds_within_blob = []
    cost_fct = []
    plot_blobs = []

    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), 1 * r, color="w", linestyle=":", linewidth=2, fill=False)

        # if multiple patches should be plotted
        plot_blobs.append(c)

        std_in_blob = calculate_std_within_blob(transformed_image, blob)
        cost_fct.append(std_in_blob * r)
        calculate_stds_within_blob.append(std_in_blob)

    # calculate the cost function of the image inside the blob -> we want to
    # select the largest one
    significant_blob = blobs[np.argmax(cost_fct)]

    y, x, r = significant_blob
    c = plt.Circle((x, y), 1 * r, color="r", linewidth=3, fill=False)
    plot_blobs.append(c)
    if plot_image:
        fig, ax = plt.subplots(1, 1)

        for c in plot_blobs:
            ax.add_patch(c)

        ax.imshow(image)
        if plot_chosen_transformation:
            ax.imshow(transformed_image, cmap=plt.cm.gray)
        plt.show()

    return significant_blob


def calculate_std_within_blob(image_channel, blob):
    """
    Calculate the standard deviation of pixel values inside the detected blob.

    :param image_channel: the image channel (2D array) on which to perform the calculation
    :param blob: a tuple (row, column, radius) defining the blob location and size.
    :return: standard deviation of the pixel values inside the blob.
    """
    # extract the center and radius of the blob
    y, x, r = blob

    # get the indices for the pixels inside the blob
    rr, cc = disk((y, x), r, shape=image_channel.shape)

    # extract pixel values inside the blob
    pixel_values_inside_blob = image_channel[rr, cc]

    # calculate the std
    std_dev = np.std(pixel_values_inside_blob)

    return std_dev


def apply_gabor_filters_and_extract_features(image, frequencies, thetas, sigmas):
    """
    Apply Gabor filters using skimage's gabor_kernel and extract features.
    """
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        image = color.rgb2gray(image)

    # Prepare filter bank kernels
    kernels = []
    for theta in thetas:
        for sigma in sigmas:
            for frequency in frequencies:
                # Use skimage's gabor_kernel function
                kernel = gabor_kernel(
                    frequency, theta=theta, sigma_x=sigma, sigma_y=sigma
                )
                kernels.append(kernel)

    # Apply Gabor filters at each combination of frequency and theta
    feature_vector = []
    for kernel in kernels:
        # Filter the image using the real part of the kernel
        real_kernel = np.real(kernel)
        filtered_image = ndi.convolve(image, real_kernel, mode="wrap")

        # Calculate statistical features from the filtered image
        mean_val = np.mean(filtered_image)
        std_val = np.std(filtered_image)
        skew_val = skew(filtered_image.ravel())
        kurt_val = kurtosis(filtered_image.ravel())

        # Append the features to the feature vector
        feature_vector.extend([mean_val, std_val, skew_val, kurt_val])

    return feature_vector


class ImageHeuristicFeatureExtractor:
    def __init__(self, data_folder_path: str, label: pd.DataFrame):
        self.data_folder_path = data_folder_path
        self.label = label
        self.histograms_rgb = []
        self.histograms_hsv = []
        self.graycomatrix_features = []
        self.gabor_features = []
        self.list_images = []
        self.list_filenames = []
        self.df = pd.DataFrame  # sample dataframe that contains file names

    def extract_features(self):
        for image_name in tqdm(os.listdir(self.data_folder_path)):
            if ".DS_Store" in image_name:
                continue

            image_path = os.path.join(self.data_folder_path, image_name)
            if os.path.exists(image_path):
                filename = image_name.split(".")[0]
                self.list_filenames.append(filename)

                if "augmented" in filename:
                    clean_image_name = "_".join(filename.split("_")[-2:])
                else:
                    clean_image_name = filename

                self.list_images.append(clean_image_name)

                image_rgb = self.load_image(image_path)
                self.process_image(image_rgb)

        return self.merge_features()

    def load_image(self, image_path: str):
        return load_image(image_path, BGR2RGB=True)

    def process_image(self, image_rgb):
        self.histograms_rgb.append(create_histogram(image_rgb, "RGB"))
        self.histograms_hsv.append(create_histogram(image_rgb, "HSV"))
        self.graycomatrix_features.append(calculate_glcm_features(image_rgb))
        # self.gabor_features.append(apply_gabor_filters_and_extract_features(image_rgb, GABOR_FREQUENCIES, GABOR_THETAS, GABOR_SIGMAS))

    def merge_features(self):

        tmp_rgb = pd.DataFrame(
            pd.DataFrame(
                np.array(self.histograms_rgb).reshape(len(self.histograms_rgb), -1),
                index=self.list_images,
            )
        )
        tmp_rgb["filename"] = self.list_filenames
        tmp_rgb["image_id"] = self.list_images

        tmp_hsv = pd.DataFrame(
            pd.DataFrame(
                np.array(self.histograms_hsv).reshape(len(self.histograms_hsv), -1),
                index=self.list_images,
            )
        )
        tmp_hsv["filename"] = self.list_filenames
        tmp_hsv["image_id"] = self.list_images

        tmp_glcm = pd.DataFrame(
            np.array(self.graycomatrix_features), index=self.list_images
        )
        tmp_glcm["filename"] = self.list_filenames
        tmp_glcm["image_id"] = self.list_images

        df_rgb = pd.merge(tmp_rgb, self.label, left_index=True, right_index=True)
        df_hsv = pd.merge(tmp_hsv, self.label, left_index=True, right_index=True)
        df_glcm = pd.merge(tmp_glcm, self.label, left_index=True, right_index=True)

        try:
            tmp_gabor = pd.DataFrame(
                np.array(self.gabor_features), index=self.list_images
            )
            tmp_gabor["filename"] = self.list_filenames
            tmp_gabor["image_id"] = self.list_images

            df_gabor = pd.merge(
                tmp_gabor, self.label, left_index=True, right_index=True
            )
        except ValueError:
            df_gabor = pd.DataFrame()

        self.df = df_glcm
        return df_rgb, df_hsv, df_glcm, df_gabor

    def get_feature_and_label_arrays(self):
        dfs = self.extract_features()
        feature_label_pairs = {}
        for i, df in enumerate(dfs):
            if len(df) == 0:
                continue

            feature_type = ["rgb", "hsv", "glcm", "gabor"][i]
            num_features = [3 * 256, 3 * 256, 6, 72][i]
            x = df.iloc[:, :num_features].to_numpy()
            y = df["cancer"].to_numpy()
            feature_label_pairs[feature_type] = (x, y)
        return feature_label_pairs

    def return_one_df(self):
        return self.df

    def return_list_filenames(self):
        return self.list_filenames


def standardize_features(
    features: np.ndarray, use_pca: bool = False, n_components: int = None
):
    """
    Standardizes the features and optionally applies PCA for dimensionality reduction.

    :param features: The feature matrix to process.
    :param use_pca: Whether to apply PCA. Default is False.
    :param n_components: The number of principal components to keep if PCA is applied. If None and use_pca is True, all components are kept.
    :return: The processed feature matrix.
    """

    # Initialize the StandardScaler
    scaler = StandardScaler()
    # Standardize the features
    standardized_features = scaler.fit_transform(features)

    # Check if PCA should be applied
    if use_pca:
        # Initialize PCA with the specified number of components
        pca = PCA(n_components=n_components)
        # Apply PCA
        processed_features = pca.fit_transform(standardized_features)
        return processed_features
    else:
        return standardized_features


def load_image(image_path: str, BGR2RGB=True) -> np.ndarray:
    img = cv2.imread(image_path)
    if BGR2RGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def extract_individual_features(df, original_folder_path):
    histograms_rgb, histograms_hsv, graycomatrix_features, Y = [], [], [], []

    for image_name in tqdm(os.listdir(original_folder_path)):

        image_path = os.path.join(original_folder_path, image_name)

        if os.path.exists(image_path):
            # Read the image using OpenCV
            image_name = image_name.split(".")[0]
            if "augmented" in image_name:
                image_name = image_name.split("_")[-2] + "_" + image_name.split("_")[-1]

            image_rgb = load_image(image_path, BGR2RGB=True)

            # Feature Extraction
            # Histograms
            hist_rgb = create_histogram(image_rgb, color_space="RGB")
            hist_hsv = create_histogram(image_rgb, color_space="HSV")
            histograms_rgb.append(hist_rgb)
            histograms_hsv.append(hist_hsv)

            # Structure: GLCM Matrix
            graycomatrix_features.append(calculate_glcm_features(image_rgb))

            # Append the label
            cancer = df.loc[df["image_id"] == image_name, "cancer"].values[0]
            Y.append(cancer)

    return (
        np.array(histograms_rgb),
        np.array(histograms_hsv),
        np.array(graycomatrix_features),
        np.array(Y),
    )


def generate_feature_vector(train_vectors: list, test_vectors: list):
    def process_vectors(vectors):
        x_combined = np.array([])

        for vec in vectors:
            if vec.ndim == 3:
                vec = vec.reshape(vec.shape[0], -1)

            if len(x_combined) == 0:
                x_combined = vec
            else:
                x_combined = np.concatenate((x_combined, vec), axis=1)

        return x_combined

    x_train = process_vectors(train_vectors)
    x_test = process_vectors(test_vectors)

    return x_train, x_test


class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith((".jpg", ".png"))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Do preprocessing here
        image_path = self.images[idx]
        rgb_image_arr = methods.convert_rgb(image_path)
        normalised_img = methods.z_normalization(rgb_image_arr)
        image = Image.fromarray(normalised_img.astype("uint8"), "RGB")
        image_tensor = self.transform(image) if self.transform else image
        key = os.path.basename(image_path).removesuffix(".jpg").removesuffix(".png")
        return key, image_tensor


def extract_image_ids(filenames: list) -> pd.DataFrame:
    """Extract image IDs from filenames assuming format includes ID as the last two underscore-separated parts."""
    image_ids = ["_".join(name.split("_")[-2:]) for name in filenames]
    return pd.DataFrame({"filename": filenames, "image_id": image_ids})


def load_features(features_path: str) -> pd.DataFrame:
    """Load features from a JSON file and transpose the dataframe."""
    if not features_path.endswith(".json"):
        raise ValueError("Invalid file type: JSON expected")
    features = pd.read_json(features_path).T
    return features


def merge_features_with_labels(
    features_path: str,
    labels_df: pd.DataFrame,
    export: bool = False,
) -> pd.DataFrame:
    """Merge image features with labels into a single DataFrame."""
    features = load_features(features_path)

    filenames = features.T.columns.to_numpy()
    temp_files = extract_image_ids(filenames)

    label_data = temp_files.merge(labels_df, on="image_id", how="left")

    merged_data = features.merge(label_data, left_index=True, right_on="image_id")

    if export:
        export_path = os.path.join(
            os.path.dirname(features_path),
            features_path.split(".")[-2].split("/")[-1] + "_pandas.csv",
        )
        merged_data.to_csv(export_path)

    return merged_data


def not_oversampled_images(features_dataframe: pd.DataFrame) -> list[bool]:
    """
    Generate a list indicating whether each image in the dataframe should be included
    in testing based on whether it has been augmented.

    :param features_dataframe: A DataFrame containing image metadata with columns 'filename' and 'image_id'.
    :return: A list of booleans where True indicates the image has not been augmented,
                and False indicates it has.
    """

    # Extract augmented image IDs
    augmented_ids = set(
        "_".join(
            filename.split("_")[-2:]
        )  # Assuming the ID is in the last two parts of the filename
        for filename in features_dataframe.filename
        if "augmented" in filename
    )

    # Determine inclusion in testing for each image
    include_in_testing = [
        image_id not in augmented_ids for image_id in features_dataframe.image_id
    ]

    return include_in_testing


def calculate_test_size(dataframe, test_size, include_in_testing):
    """
    Calculate the adjusted test size for splitting the dataset, excluding oversampled entries.

    :param dataframe: The complete dataframe.
    :param test_size: The desired proportion of the test set size relative to the unique images.
    :param include_in_testing: Boolean array indicating which images are not oversampled.
    :return: Adjusted test size.
    """
    unique_image_count = len(np.unique(dataframe.image_id))
    valid_image_count = np.sum(include_in_testing)
    return test_size * unique_image_count / valid_image_count


def calculate_metrics(y_test, y_pred):
    """
    Calculates and prints the accuracy, precision, recall, and F1 score for the given test labels and predictions.

    :param y_test: True labels for the test set.
    :param y_pred: Predicted labels for the test set.
    :return:
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")


def plot_confusion_matrix(y_test, y_pred, print_metrics=True):
    """
    Plots the confusion matrix for the given test labels and predictions.

    :param y_test: True labels for the test set.
    :param y_pred: Predicted labels for the test set.
    :return:
    """
    if print_metrics:
        calculate_metrics(y_test, y_pred)

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Initialize the ConfusionMatrixDisplay object with the confusion matrix
    cmd = ConfusionMatrixDisplay(conf_matrix)

    # Plot the confusion matrix
    cmd.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


def plot_low_dim_components(
    x_train_low_dim, y_train, label="PCA", component_1=0, component_2=1
):
    # Scatter plot of the first two PCA components
    # Here, X_pca[:, 0] is the first component, X_pca[:, 1] is the second component
    plt.figure(figsize=(10, 7))
    plt.scatter(
        x_train_low_dim[y_train == 0, component_1],
        x_train_low_dim[y_train == 0, component_2],
        c="blue",
        label="Non-Cancerous",
        alpha=0.5,
    )

    # Non-cancerous in blue
    plt.scatter(
        x_train_low_dim[y_train == 1, component_1],
        x_train_low_dim[y_train == 1, component_2],
        c="red",
        label="Cancerous",
        alpha=0.5,
    )  # Cancerous labeled in red

    # Adding labels and title
    plt.xlabel(f"{label} Component {component_1}")
    plt.ylabel(f"{label} Component {component_2}")
    plt.title(f"{label} of Image Data")
    plt.legend()
