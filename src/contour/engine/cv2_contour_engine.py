"""CV2 implementation of contour extraction engine."""

import cv2
import numpy as np

from .contour_engine import ContourExtractionEngine


class CV2ContourEngine(ContourExtractionEngine):
    """Extracts contours using OpenCV.

    Attributes:
        min_length: Minimum length of a contour to be kept.
        threshold_value: Threshold value for binarization.
        threshold_max_value: Max value for binarization.
        morph_kernel_size: Kernel size for morphological operations.
        morph_iterations: Number of iterations for morphological operations.
        epsilon_factor: Factor for approximation accuracy.
    """

    def __init__(
        self,
        min_length: float = 50.0,
        threshold_value: int = 127,
        threshold_max_value: int = 255,
        morph_kernel_size: tuple[int, int] = (3, 3),
        morph_iterations: int = 2,
        epsilon_factor: float = 0.005,
    ):
        """Initializes the CV2 contour engine.

        Args:
            min_length: Minimum length of a contour to be kept.
            threshold_value: Threshold value for binarization.
            threshold_max_value: Max value for binarization.
            morph_kernel_size: Kernel size for morphological operations.
            morph_iterations: Number of iterations for morphological operations.
            epsilon_factor: Factor for approximation accuracy.
        """
        self.min_length = min_length
        self.threshold_value = threshold_value
        self.threshold_max_value = threshold_max_value
        self.morph_kernel_size = morph_kernel_size
        self.morph_iterations = morph_iterations
        self.epsilon_factor = epsilon_factor

    def extract_contours(self, mask_path: str) -> list[np.ndarray]:
        """Extracts contours from a binary mask file.

        Args:
            mask_path: Path to the binary mask image.

        Returns:
            List of contours, where each contour is a numpy array of shape (N, 1, 2).

        Raises:
            FileNotFoundError: If the mask file cannot be read.
        """
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask at {mask_path}")

        _, binary = cv2.threshold(
            mask, self.threshold_value, self.threshold_max_value, cv2.THRESH_BINARY
        )

        # Morphological closing to merge close lines
        kernel = np.ones(self.morph_kernel_size, np.uint8)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations
        )

        # Skeletonize to get single-pixel width lines
        binary = self.skeletonize(binary)

        # RETR_LIST: Retrieve all contours without establishing hierarchy
        # CHAIN_APPROX_SIMPLE: Compress horizontal, vertical, and diagonal segments
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = []
        for cnt in contours:
            length = cv2.arcLength(cnt, closed=False)

            if length >= self.min_length:
                epsilon = self.epsilon_factor * length
                approx = cv2.approxPolyDP(cnt, epsilon, closed=False)
                filtered_contours.append(approx)

        return filtered_contours

    def skeletonize(self, img: np.ndarray) -> np.ndarray:
        """Reduces binary image lines to single pixel width using Zhang-Suen thinning.

        Args:
            img: Binary image.

        Returns:
            Skeletonized binary image.
        """
        # Ensure image is binary (0 or 1)
        _, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        img = img.astype(np.uint8)

        # Precompute LUTs for the two steps of Zhang-Suen
        lut1 = np.zeros(256, dtype=np.uint8)
        lut2 = np.zeros(256, dtype=np.uint8)

        # Neighbors indices:
        # P9 P2 P3
        # P8 P1 P4
        # P7 P6 P5
        # We map them to bits:
        # P2: bit 0, P3: bit 1, P4: bit 2, P5: bit 3
        # P6: bit 4, P7: bit 5, P8: bit 6, P9: bit 7

        for i in range(256):
            # Extract neighbors
            neighbors = [(i >> j) & 1 for j in range(8)]
            # P2, P3, P4, P5, P6, P7, P8, P9 = neighbors

            # B(P1): Number of non-zero neighbors
            b = sum(neighbors)

            # A(P1): Number of 0->1 transitions in P2, P3, ..., P9, P2
            # Sequence: P2, P3, P4, P5, P6, P7, P8, P9, P2
            transitions = 0
            for j in range(8):
                current = neighbors[j]
                next_val = neighbors[(j + 1) % 8]
                if current == 0 and next_val == 1:
                    transitions += 1
            a = transitions

            # Conditions
            # a) 2 <= B <= 6
            # b) A == 1
            cond_a_b = (2 <= b <= 6) and (a == 1)

            if cond_a_b:
                p2, _p3, p4, _p5, p6, _p7, p8, _p9 = neighbors

                # Step 1 conditions:
                # c) P2 * P4 * P6 == 0
                # d) P4 * P6 * P8 == 0
                cond_c = (p2 * p4 * p6) == 0
                cond_d = (p4 * p6 * p8) == 0
                if cond_c and cond_d:
                    lut1[i] = 1  # Mark for deletion

                # Step 2 conditions:
                # c') P2 * P4 * P8 == 0
                # d') P2 * P6 * P8 == 0
                cond_c_prime = (p2 * p4 * p8) == 0
                cond_d_prime = (p2 * p6 * p8) == 0
                if cond_c_prime and cond_d_prime:
                    lut2[i] = 1  # Mark for deletion

        # Kernel to compute neighborhood index
        # Weights corresponding to bits 0..7 for P2..P9
        # P9(128) P2(1)   P3(2)
        # P8(64)  P1      P4(4)
        # P7(32)  P6(16)  P5(8)
        kernel = np.array(
            [[128, 1, 2], [64, 0, 4], [32, 16, 8]], dtype=float
        )  # float for filter2D

        skel = img.copy()

        while True:
            changed = False

            # Step 1
            # Compute neighborhood indices
            # filter2D computes sum(kernel * image).
            # We need to ensure border handling is correct (constant 0)
            neighborhoods = cv2.filter2D(
                skel, -1, kernel, borderType=cv2.BORDER_CONSTANT
            )

            # Apply LUT to find pixels to remove
            # lut1[neighborhoods] gives 1 if should remove, 0 otherwise
            # But we only remove if the pixel itself is 1
            to_remove = np.take(lut1, neighborhoods.astype(int))
            to_remove = to_remove & skel

            if cv2.countNonZero(to_remove) > 0:
                skel = cv2.subtract(skel, to_remove)
                changed = True

            # Step 2
            neighborhoods = cv2.filter2D(
                skel, -1, kernel, borderType=cv2.BORDER_CONSTANT
            )
            to_remove = np.take(lut2, neighborhoods.astype(int))
            to_remove = to_remove & skel

            if cv2.countNonZero(to_remove) > 0:
                skel = cv2.subtract(skel, to_remove)
                changed = True

            if not changed:
                break

        return skel * 255
