"""U-Net based contour extraction engine."""

from pathlib import Path

import cv2
import numpy as np
import torch

from .contour_engine import ContourExtractionEngine


class UNetContourEngine(ContourExtractionEngine):
    """Extracts contours using a trained U-Net segmentation model.

    This engine uses a U-Net to predict a binary segmentation mask from
    an input image (which may contain text, terrain, etc.), then extracts
    individual contours from the predicted mask.

    Attributes:
        model_path: Path to the trained model checkpoint.
        device: Torch device for inference.
        threshold: Probability threshold for binary prediction.
        min_length: Minimum contour length to keep.
        epsilon_factor: Factor for contour approximation.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        threshold: float = 0.5,
        min_length: float = 50.0,
        epsilon_factor: float = 0.005,
    ):
        """Initialize the U-Net contour engine.

        Args:
            model_path: Path to the trained model checkpoint (.pt file).
            device: Device for inference ("cuda", "mps", or "cpu").
            threshold: Probability threshold for binarizing predictions.
            min_length: Minimum contour length to keep.
            epsilon_factor: Factor for approximation accuracy.
        """
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.min_length = min_length
        self.epsilon_factor = epsilon_factor

        # Setup device
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Load model
        self.model = self._load_model()
        self.model.eval()

        # ImageNet normalization stats
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _load_model(self) -> torch.nn.Module:
        """Load the trained U-Net model from checkpoint.

        Returns:
            Loaded model ready for inference.
        """
        try:
            import segmentation_models_pytorch as smp
        except ImportError as e:
            msg = (
                "segmentation-models-pytorch not installed. "
                "Run: pip install segmentation-models-pytorch"
            )
            raise ImportError(msg) from e

        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Get model config from checkpoint
        config = checkpoint.get("config", {})
        encoder_name = config.get("encoder_name", "resnet34")
        # Don't load pretrained weights for inference
        encoder_weights = config.get("encoder_weights", None)

        # Create model
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)

        return model

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model inference.

        Args:
            image: Input image as numpy array (H, W, 3) in RGB format.

        Returns:
            Preprocessed tensor ready for the model.
        """
        # Resize to model input size if needed
        # The model should handle various sizes due to fully convolutional architecture
        # but we'll resize to 512x512 for consistency
        original_size = image.shape[:2]
        image = cv2.resize(image, (512, 512))

        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std

        # Convert to tensor [1, 3, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        image = image.to(self.device)

        return image, original_size

    def _postprocess_mask(
        self, logits: torch.Tensor, original_size: tuple[int, int]
    ) -> np.ndarray:
        """Postprocess model output to binary mask.

        Args:
            logits: Model output logits [1, 1, H, W].
            original_size: Original image size (H, W) to resize back to.

        Returns:
            Binary mask as numpy array (H, W).
        """
        # Apply sigmoid and threshold
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        mask = (probs > self.threshold).astype(np.uint8) * 255

        # Resize back to original size
        mask = cv2.resize(mask, (original_size[1], original_size[0]))

        return mask

    def extract_contours(self, image_path: str) -> list[np.ndarray]:
        """Extract contours from an image using U-Net prediction.

        Note: Despite the parameter name 'image_path' (for interface compatibility),
        this method expects an INPUT IMAGE (with terrain, text, etc.), not a mask.
        The U-Net predicts the mask internally.

        Args:
            image_path: Path to the input image (not a pre-existing mask).

        Returns:
            List of contours, where each contour is a numpy array of shape (N, 1, 2).

        Raises:
            FileNotFoundError: If the image file cannot be read.
        """
        mask = self.predict_mask(image_path)

        # Extract contours from predicted mask
        return self._extract_contours_from_mask(mask)

    def _extract_contours_from_mask(self, mask: np.ndarray) -> list[np.ndarray]:
        """Extract individual contours from a binary mask.

        Uses skeletonization and cv2.findContours, similar to CV2ContourEngine.

        Args:
            mask: Binary mask (H, W) with values 0 or 255.

        Returns:
            List of contours.
        """
        # Skeletonize to get single-pixel width lines
        skeleton = self._skeletonize(mask)

        # Find contours
        contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and approximate
        filtered_contours = []
        for cnt in contours:
            length = cv2.arcLength(cnt, closed=False)

            if length >= self.min_length:
                epsilon = self.epsilon_factor * length
                approx = cv2.approxPolyDP(cnt, epsilon, closed=False)
                filtered_contours.append(approx)

        return filtered_contours

    def _skeletonize(self, img: np.ndarray) -> np.ndarray:
        """Reduce binary image to single pixel width skeleton.

        Uses morphological thinning.

        Args:
            img: Binary image (0 or 255).

        Returns:
            Skeletonized binary image.
        """
        # Ensure binary
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Use OpenCV's thinning (Zhang-Suen algorithm)
        # cv2.ximgproc.thinning requires opencv-contrib-python
        try:
            skeleton = cv2.ximgproc.thinning(binary)
        except AttributeError:
            # Fallback: simple morphological skeleton
            skeleton = self._morphological_skeleton(binary)

        return skeleton

    def _morphological_skeleton(self, img: np.ndarray) -> np.ndarray:
        """Fallback morphological skeletonization.

        Args:
            img: Binary image.

        Returns:
            Skeletonized image.
        """
        skeleton = np.zeros_like(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            img = eroded.copy()

            if cv2.countNonZero(img) == 0:
                break

        return skeleton

    def predict_mask(self, image_path: str) -> np.ndarray:
        """Predict segmentation mask for an image.

        Useful for visualization and debugging.

        Args:
            image_path: Path to input image.

        Returns:
            Predicted binary mask as numpy array.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor, original_size = self._preprocess_image(image)

        with torch.no_grad():
            logits = self.model(input_tensor)

        return self._postprocess_mask(logits, original_size)
