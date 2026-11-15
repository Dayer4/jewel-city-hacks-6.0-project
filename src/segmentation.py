
import cv2
import numpy as np


def segment_image(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert dark handwriting into white foreground
    _, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find connected components
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    components = []

    for i in range(1, count):

        x, y, w, h, area = stats[i]

        # Ignore tiny noise
        if area < 5:
            continue

        components.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": int(area)
        })

    if not components:
        raise ValueError("No handwriting detected")

    median_h = np.median([c["h"] for c in components])

    # Group components into lines
    lines = []

    for component in sorted(components, key=lambda c: c["y"] + c["h"] / 2):

        center_y = component["y"] + component["h"] / 2

        matching_line = None

        for line in lines:

            line_center = np.mean([
                c["y"] + c["h"] / 2 for c in line
            ])

            if abs(center_y - line_center) < median_h * 0.8:
                matching_line = line
                break

        if matching_line is None:
            lines.append([component])
        else:
            matching_line.append(component)

    results = []

    for line in lines:

        line.sort(key=lambda c: c["x"])

        widths = [c["w"] for c in line]
        median_w = np.median(widths)

        space_threshold = max(median_h * 0.6, median_w * 0.8)

        characters = []

        previous_right = None

        for component in line:

            x = component["x"]
            y = component["y"]
            w = component["w"]
            h = component["h"]

            # Crop original grayscale image
            crop = image[y:y+h, x:x+w]

            # Detect possible word spaces
            space_before = (
                previous_right is not None
                and x - previous_right > space_threshold
            )

            characters.append({
                "image": crop,
                "bbox": (x, y, w, h),
                "space_before": space_before
            })

            previous_right = x + w

        results.append(characters)

    return results
