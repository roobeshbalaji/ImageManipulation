"""
Non-Semantic Local Manipulation (NSLM) patch generator.

Reads upright face images, finds a face box (landmarks → Haar → centered proxy),
then places two small non-overlapping patches inside the box, rotates them by 180°,
and lightly blurs the patch borders to blend.
"""

import os
import random
import math
import cv2

from facial_landmark_detection import get_image_facial_landmarks
from main import get_bounding_rectangle

# === CONFIG ===
INPUT_DIR   = "../archive/img_align_celeba/upright_normal"
OUTPUT_DIR  = "../archive/img_align_celeba/output_non_semantic_blended"
H_SIZE      = (20, 40)   # (patch_height, patch_width)
V_SIZE      = (40, 20)
PATCH_COUNT = 2          # always two patches per face
BORDER_SIZE = 3          # width of blending border, in pixels
RNG_SEED    = 1337       # set for reproducibility
# ==============

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(RNG_SEED)


def clamp_face_box(face_box, H, W, pad=0):
    """Clamp a (y1,x1)-(y2,x2) face box to image bounds with optional padding.

    Args:
        face_box: [[y1, x1], [y2, x2]] integer coords.
        H, W: image height/width.
        pad: expand box by this many pixels before clamping.

    Returns:
        A valid [[y1, x1], [y2, x2]] with y1<=y2 and x1<=x2; never degenerate.
    """
    (y1, x1), (y2, x2) = face_box
    y1 = max(0, min(H - 1, y1 - pad))
    x1 = max(0, min(W - 1, x1 - pad))
    y2 = max(0, min(H - 1, y2 + pad))
    x2 = max(0, min(W - 1, x2 + pad))
    if y2 < y1: y1, y2 = y2, y1
    if x2 < x1: x1, x2 = x2, x1
    if (y2 - y1) < 4 or (x2 - x1) < 4:
        cy, cx = H // 2, W // 2
        hh, ww = max(8, H // 3), max(8, W // 3)
        y1, y2 = max(0, cy - hh // 2), min(H - 1, cy + hh // 2)
        x1, x2 = max(0, cx - ww // 2), min(W - 1, cx + ww // 2)
    return [[y1, x1], [y2, x2]]


def haar_face_box(img):
    """Detect the largest frontal face with OpenCV Haar cascade.

    Args:
        img: BGR uint8 image.

    Returns:
        [[y1, x1], [y2, x2]] if found, else None.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    )
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60, 60)
    )
    if faces is None or len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return [[y, x], [y + h, x + w]]


def central_face_proxy_box(H, W):
    """Fallback face region centered in the image (works for CelebA because the images are mostly center aligned). Might have to be changed if the dataset changes.

    Args:
        H, W: image height/width.

    Returns:
        [[y1, x1], [y2, x2]] conservative centered box.
    """
    box_h = max(40, int(0.60 * H))
    box_w = max(40, int(0.50 * W))
    y1 = (H - box_h) // 2
    x1 = (W - box_w) // 2
    return [[y1, x1], [y1 + box_h, x1 + box_w]]


def rotate_180_in_place(img, rect):
    """Rotate a rectangular region by 180° in-place.

    Args:
        img: BGR image.
        rect: (y1, x1, y2, x2) integer coords.
    """
    y1, x1, y2, x2 = rect
    patch = img[y1:y2, x1:x2].copy()
    rotated = cv2.rotate(patch, cv2.ROTATE_180)
    img[y1:y2, x1:x2] = rotated


def blur_orthogonal_border(image, blurred_image, x1, y1, x2, y2, border_size):
    """Copy a thin orthogonal strip (row/col) from a blurred version to soften edges."""
    if x1 == x2:
        for x in range(max(0, x1 - border_size), min(image.shape[0] - 1, x1 + border_size) + 1):
            for y in range(y1, y2 + 1):
                image[x][y] = blurred_image[x][y]
    if y1 == y2:
        for y in range(max(0, y1 - border_size), min(image.shape[1] - 1, y1 + border_size) + 1):
            for x in range(x1, x2 + 1):
                image[x][y] = blurred_image[x][y]


def blur_rectangle_border(image, y1, x1, y2, x2, border_size=2):
    """Feather the border of a rectangle by blending with a blurred copy.

    Args:
        image: BGR image.
        y1,x1,y2,x2: rectangle coords.
        border_size: blend width in pixels.
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    blur_orthogonal_border(image, blurred, y1, x1, y1, x2, border_size)  # top
    blur_orthogonal_border(image, blurred, y2, x1, y2, x2, border_size)  # bottom
    blur_orthogonal_border(image, blurred, y1, x1, y2, x1, border_size)  # left
    blur_orthogonal_border(image, blurred, y1, x2, y2, x2, border_size)  # right


def can_fit(sizes, box_h, box_w):
    """Check that each (h,w) in sizes individually fits inside box_h×box_w."""
    for ph, pw in sizes:
        if ph > box_h or pw > box_w:
            return False
    return True


def try_sample_two_nonoverlapping(box, sizes, max_tries=1000):
    """Sample two non-overlapping rectangles of given sizes inside a box.

    Args:
        box: [[y1,x1],[y2,x2]] container.
        sizes: list of (h,w) sizes to place in order.
        max_tries: attempts per rectangle.

    Returns:
        ((y1,x1,y2,x2), (y1,x1,y2,x2)) on success, else None.
    """
    (fy1, fx1), (fy2, fx2) = box
    box_h, box_w = fy2 - fy1, fx2 - fx1
    rects = []
    for ph, pw in sizes:
        placed = False
        for _ in range(max_tries):
            if box_h - ph <= 0 or box_w - pw <= 0:
                break
            y1 = random.randint(fy1, fy2 - ph)
            x1 = random.randint(fx1, fx2 - pw)
            y2, x2 = y1 + ph, x1 + pw
            if all(y2 <= oy1 or oy2 <= y1 or x2 <= ox1 or ox2 <= x1
                   for (oy1, ox1, oy2, ox2) in rects):
                rects.append((y1, x1, y2, x2))
                placed = True
                break
        if not placed:
            return None
    return rects[0], rects[1]


def sample_two_patches_robust(face_box, sizes, img_shape):
    """Place two non-overlapping patches inside face_box.

    Args:
        face_box: [[y1,x1],[y2,x2]] initial face region.
        sizes: list of (h,w) desired sizes; orientation is preserved.
        img_shape: image shape for clamping.

    Returns:
        Two rectangles ((y1,x1,y2,x2), (y1,x1,y2,x2)) or None if placement fails.
    """
    H, W = img_shape[:2]
    face_box = clamp_face_box(face_box, H, W, pad=2)
    (fy1, fx1), (fy2, fx2) = face_box
    box_h, box_w = fy2 - fy1, fx2 - fx1

    scales = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3]
    MIN_HW = 6

    for s in scales:
        scaled = []
        for ph, pw in sizes:
            ph_s = max(MIN_HW, int(round(ph * s)))
            pw_s = max(MIN_HW, int(round(pw * s)))
            scaled.append((ph_s, pw_s))
        if not can_fit(scaled, box_h, box_w):
            continue
        rects = try_sample_two_nonoverlapping(face_box, scaled)
        if rects is not None:
            return rects

    pad_y = max(2, int(0.05 * H))
    pad_x = max(2, int(0.05 * W))
    expanded = clamp_face_box([[fy1 - pad_y, fx1 - pad_x], [fy2 + pad_y, fx2 + pad_x]], H, W, pad=0)
    return try_sample_two_nonoverlapping(expanded, scaled)


def get_face_box_robust(img, img_path):
    """Get a reliable face box: landmarks → Haar → centered proxy.

    Args:
        img: BGR image.
        img_path: path to the same image (for landmark code).

    Returns:
        [[y1,x1],[y2,x2]] clamped and padded.
    """
    H, W = img.shape[:2]

    lm = get_image_facial_landmarks(img_path)
    if lm and len(lm) == 68:
        box = get_bounding_rectangle(lm)
        return clamp_face_box(box, H, W, pad=2)

    box = haar_face_box(img)
    if box is not None:
        return clamp_face_box(box, H, W, pad=2)

    return clamp_face_box(central_face_proxy_box(H, W), H, W, pad=0)


def process_one(img_path, out_path):
    """Apply the NSLM manipulation to a single image and write it out.

    Args:
        img_path: input image path.
        out_path: output image path.

    Returns:
        True if written successfully, else False.
    """
    img = cv2.imread(img_path)
    if img is None:
        return False

    H, W = img.shape[:2]
    face_box = get_face_box_robust(img, img_path)

    sizes = random.sample([H_SIZE, V_SIZE] * PATCH_COUNT, k=PATCH_COUNT)
    rects = sample_two_patches_robust(face_box, sizes, img.shape)
    if rects is None:
        proxy_box = central_face_proxy_box(H, W)
        rects = sample_two_patches_robust(proxy_box, sizes, img.shape)

    for (y1, x1, y2, x2) in rects:
        rotate_180_in_place(img, (y1, x1, y2, x2))
        blur_rectangle_border(img, y1, x1, y2, x2, border_size=BORDER_SIZE)

    ok = cv2.imwrite(out_path, img)
    return bool(ok)


def is_image_file(fname):
    """Return True if fname has a common image extension."""
    ext = os.path.splitext(fname)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    """Process all images in INPUT_DIR and write outputs to OUTPUT_DIR."""
    names = [f for f in os.listdir(INPUT_DIR) if is_image_file(f)]
    names.sort()

    total = len(names)
    written = 0
    for i, fname in enumerate(names, 1):
        in_path  = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)
        success  = process_one(in_path, out_path)
        if not success:
            base, ext = os.path.splitext(out_path)
            alt = f"{base}__ns_{i}{ext}"
            success = process_one(in_path, alt)
        written += int(success)

    print(f"Requested outputs: {total}, Written: {written}")
    if written != total:
        print("WARNING: Some files could not be processed (likely unreadable/corrupt).")


if __name__ == "__main__":
    main()
