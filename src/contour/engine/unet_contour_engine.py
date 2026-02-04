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
        model_path: Path to the trained model checkpoint (local or HF Hub).
        device: Torch device for inference.
        threshold: Probability threshold for binary prediction.
        min_length: Minimum contour length to keep.
        epsilon_factor: Factor for contour approximation.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cuda",
        threshold: float = 0.5,
        min_length: float = 50.0,
        epsilon_factor: float = 0.005,
        hf_repo_id: str | None = None,
        hf_filename: str = "best_model.pt",
    ):
        """Initialize the U-Net contour engine.

        Args:
            model_path: Path to a local model checkpoint (.pt file).
                If None, will download from Hugging Face Hub.
            device: Device for inference ("cuda", "mps", or "cpu").
            threshold: Probability threshold for binarizing predictions.
            min_length: Minimum contour length to keep.
            epsilon_factor: Factor for approximation accuracy.
            hf_repo_id: Hugging Face Hub repository ID.
                Used if model_path is None.
            hf_filename: Filename of the model in the HF repo.
        """
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

        # Resolve model path
        if model_path is not None:
            self.model_path = Path(model_path)
        else:
            # Download from Hugging Face Hub
            self.model_path = self._download_from_hub(
                hf_repo_id or "mattiaskvist/topovision-unet",
                hf_filename,
            )

        # Load model
        self.model = self._load_model()
        self.model.eval()

        # ImageNet normalization stats
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _download_from_hub(self, repo_id: str, filename: str) -> Path:
        """Download model weights from Hugging Face Hub.

        Args:
            repo_id: Repository ID on Hugging Face Hub.
            filename: Name of the model file to download.

        Returns:
            Path to the downloaded model file.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            msg = "huggingface_hub not installed. Run: pip install huggingface-hub"
            raise ImportError(msg) from e

        print(f"Downloading model from {repo_id}/{filename}...")
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )
        print(f"Model downloaded to: {model_path}")
        return Path(model_path)

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

        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=True
        )

        # Get model config from checkpoint
        config = checkpoint.get("config", {})
        encoder_name = config.get("encoder_name", "resnet34")

        # Create model without pretrained weights - we'll load our trained
        # weights from the checkpoint, which would overwrite them anyway
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )

        # Load trained weights from checkpoint
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
