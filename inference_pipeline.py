# Load classifier and vocab
clf = ...     # trained classifier from above
vocab = np.load('bovw_vocab.npy')

# Feature extraction functions OMITTED for brevity; use your earlier code

image = cv2.imread('image.jpg')
windows = []
window_size = (64, 128)  # Change as suited
step_size = 32

for (x, y, window) in sliding_window(image, step_size, window_size):
    if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
        continue
    gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
    sift_vec = extract_sift_bovw(gray)     # From your earlier function
    gabor_vec = extract_gabor_features(gray)
    lbp_vec = extract_lbp_features(gray)
    hsv_vec = extract_hsv_histogram(window)
    features = np.hstack([sift_vec, gabor_vec, lbp_vec, hsv_vec])
    pred = clf.predict([features])[0]
    if pred == 1:   # If 'Bottle'
        windows.append((x, y, x + window_size[0], y + window_size[1]))
