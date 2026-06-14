
    delta_sq = (hash1 - hash2) ** 2
    if len(distinctiveness_weights) != len(hash1):
        distinctiveness_weights = cv2.resize(distinctiveness_weights.reshape(64,64), (8, 8)).flatten()
    return np.sqrt(np.sum(distinctiveness_weights * delta_sq))

scaled_threshold = LSH_HASH_SIZE * 0.15