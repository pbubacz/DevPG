import base64
import logging
import mimetypes
import os
import re
from mimetypes import guess_type
from typing import Any, Iterator, List, Optional

import fitz  # PyMuPDF
from PIL import Image
from openai import AzureOpenAI

from langchain_community.document_loaders.base import BaseBlobParser, BaseLoader
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

aoai_api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
aoai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "data/cropped")


aoai_api_version = os.getenv(
    "AZURE_OPENAI_API_VERSION"
)  # this might change in the future
MAX_TOKENS = 2000


def find_figure_indices(text):
    """
    Find the indices of figures in the Markdown text.

    Args:
        text (str): The Markdown text.

    Returns:
        List[int]: A list of indices of figures found in the text.
    """
    pattern = r"!\[\]\(figures/(\d+)\)"
    matches = re.findall(pattern, text)
    indices = [int(match) for match in matches]
    return indices


# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    """
    Encode a local image file into a data URL.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The data URL of the image.
    """
    # Guess the MIME type of the image based on the file extension
    try:
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = "application/octet-stream"  # Default MIME type if none is found

        # Read and encode the image file
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

        # Construct the data URL
        return f"data:{mime_type};base64,{base64_encoded_data}"
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        raise
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        raise


def crop_image_from_image(image_path, page_number, bounding_box):
    """
    Crop a region from an image and return it as a PIL Image.

    Args:
        image_path (str): The path to the image file.
        page_number (int): The page number (for PDF and TIFF files, 0-indexed).
        bounding_box (tuple): The bounding box coordinates in the format (x0, y0, x1, y1).

    Returns:
        Image: A PIL Image of the cropped area.
    """
    with Image.open(image_path) as img:
        if img.format == "TIFF":
            # Open the TIFF image
            img.seek(page_number)
            img = img.copy()

        # The bounding box is expected to be in the format (left, upper, right, lower).
        cropped_image = img.crop(bounding_box)
        return cropped_image


def crop_image_from_pdf_page(pdf_path, page_number, bounding_box):
    """
    Crop a region from a PDF page and return it as a PIL Image.

    Args:
        pdf_path (str): The path to the PDF file.
        page_number (int): The page number (0-indexed).
        bounding_box (tuple): The bounding box coordinates in the format (x0, y0, x1, y1).

    Returns:
        Image: A PIL Image of the cropped area.
    """
    print(f"Crop image from PDF page - PDF path: {pdf_path}")
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)

    # Cropping the page. The rect requires the coordinates in the format (x0, y0, x1, y1).
    bbx = [x * 72 for x in bounding_box]
    rect = fitz.Rect(bbx)
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), clip=rect)

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    doc.close()

    return img


def crop_image_from_file(
    file_path: str, page_number: int, bounding_box: tuple
) -> Image.Image:
    """
    Crop an image from a file.

    Args:
        file_path (str): The path to the file.
        page_number (int): The page number (for PDF and TIFF files, 0-indexed).
        bounding_box (tuple): The bounding box coordinates in the format (x0, y0, x1, y1).

    Returns:
        A PIL Image of the cropped area.
    """
    print(f"Crop image from file - File path: {file_path}")

    mime_type = mimetypes.guess_type(file_path)[0]

    if mime_type == "application/pdf":
        return crop_image_from_pdf_page(file_path, page_number, bounding_box)
    else:
        return crop_image_from_image(file_path, page_number, bounding_box)


def understand_image_with_gptv(image_path, caption):
    """
    Generates a description for an image using the GPT-4V model.

    Parameters:
    - api_base (str): The base URL of the API.
    - api_key (str): The API key for authentication.
    - deployment_name (str): The name of the deployment.
    - api_version (str): The version of the API.
    - image_path (str): The path to the image file.
    - caption (str): The caption for the image.

    Returns:
    - img_description (str): The generated description for the image.
    """
    aoai_deployment_name = os.getenv("AZURE_OPENAI_MODEL_GPT4o", "gpt-4o")

    client = AzureOpenAI(
        api_key=aoai_api_key,
        api_version=aoai_api_version,
        base_url=f"{aoai_api_base}openai/deployments/{aoai_deployment_name}",
    )

    data_url = local_image_to_data_url(image_path)
    print("Getting image description for image caption:", caption)

    response = client.chat.completions.create(
        model=aoai_deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Describe this image (note: it has image caption: {caption}):"
                            if caption
                            else "Describe this image:"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        max_tokens=MAX_TOKENS,
    )
    img_description = response.choices[0].message.content
    return img_description, data_url


def update_figure_description(md_content, img_description, idx):
    """
    Updates the figure description in the Markdown content.

    Args:
        md_content (str): The original Markdown content.
        img_description (str): The new description for the image.
        idx (int): The index of the figure.

    Returns:
        str: The updated Markdown content with the new figure description.
    """

    # The substring you're looking for
    start_substring = f"![](figures/{idx})"
    end_substring = "</figure>"
    new_string = f'<!-- FigureContent="{img_description}" -->'

    new_md_content = md_content
    # Find the start and end indices of the part to replace
    start_index = md_content.find(start_substring)
    if start_index != -1:  # if start_substring is found
        start_index += len(
            start_substring
        )  # move the index to the end of start_substring
        end_index = md_content.find(end_substring, start_index)
        if end_index != -1:  # if end_substring is found
            # Replace the old string with the new string
            new_md_content = (
                md_content[:start_index] + new_string + md_content[end_index:]
            )

    return new_md_content


def include_figure_in_md(input_file_path, result, output_folder=OUTPUT_FOLDER):
    """
    Include figure descriptions in the Markdown content.

    Args:
        input_file_path (str): The path to the input file.
        result (Any): The result object containing figures and content.
        output_folder (str): The folder where cropped images will be saved.

    Returns:
        str: The updated Markdown content with figure descriptions.
        dict: A dictionary mapping figure indices to their image URLs.
    """

    md_content = result.content
    fig_metadata = {}
    if result.figures:
        print("Figures:")
        for idx, figure in enumerate(result.figures):
            figure_content = ""
            img_description = ""
            print(f"Figure #{idx} has the following spans: {figure.spans}")
            for i, span in enumerate(figure.spans):
                print(f"Span #{i}: {span}")
                figure_content += md_content[span.offset : span.offset + span.length]
            print(f"Original figure content in markdown: {figure_content}")

            # Note: figure bounding regions currently contain both the bounding region of figure caption and figure body
            if figure.caption:
                caption_region = figure.caption.bounding_regions
                print(f"\tCaption: {figure.caption.content}")
                print(f"\tCaption bounding region: {caption_region}")
                for region in figure.bounding_regions:
                    if region not in caption_region:
                        print(f"\tFigure body bounding regions: {region}")
                        # To learn more about bounding regions, see https://aka.ms/bounding-region
                        boundingbox = (
                            region.polygon[0],  # x0 (left)
                            region.polygon[1],  # y0 (top)
                            region.polygon[4],  # x1 (right)
                            region.polygon[5],  # y1 (bottom)
                        )
                        print(
                            f"\tFigure body bounding box in (x0, y0, x1, y1): {boundingbox}"
                        )

                        cropped_image = crop_image_from_file(
                            input_file_path, region.page_number - 1, boundingbox
                        )  # page_number is 1-indexed

                        # Get the base name of the file
                        base_name = os.path.basename(input_file_path)
                        # Remove the file extension
                        file_name_without_extension = os.path.splitext(base_name)[0]

                        output_file = (
                            f"{file_name_without_extension}_cropped_image_{idx}.png"
                        )
                        cropped_image_filename = os.path.join(
                            output_folder, output_file
                        ).replace("\\", "/")

                        print(
                            f"\tFigure {idx} cropped and is to be saved as {cropped_image_filename}"
                        )

                        cropped_image.save(cropped_image_filename)
                        print(
                            f"\tFigure {idx} cropped and saved as {cropped_image_filename}"
                        )
                        img_description, image_url = understand_image_with_gptv(
                            cropped_image_filename, figure.caption.content
                        )

                        print(f"\tDescription of figure {idx}: {img_description}")
            else:
                print("\tNo caption found for this figure.")
                for region in figure.bounding_regions:
                    print(f"\tFigure body bounding regions: {region}")
                    # To learn more about bounding regions, see https://aka.ms/bounding-region
                    boundingbox = (
                        region.polygon[0],  # x0 (left)
                        region.polygon[1],  # y0 (top
                        region.polygon[4],  # x1 (right)
                        region.polygon[5],  # y1 (bottom)
                    )
                    print(
                        f"\tFigure body bounding box in (x0, y0, x1, y1): {boundingbox}"
                    )

                    cropped_image = crop_image_from_file(
                        input_file_path, region.page_number - 1, boundingbox
                    )  # page_number is 1-indexed

                    # Get the base name of the file
                    base_name = os.path.basename(input_file_path)
                    # Remove the file extension
                    file_name_without_extension = os.path.splitext(base_name)[0]

                    output_file = (
                        f"{file_name_without_extension}_cropped_image_{idx}.png"
                    )
                    cropped_image_filename = os.path.join(
                        output_folder, output_file
                    ).replace("\\", "/")

                    print(
                        f"\tFigure {idx} cropped and saved as {cropped_image_filename}"
                    )
                    # cropped_image_filename = f"data/cropped/image_{idx}.png"
                    cropped_image.save(cropped_image_filename)
                    print(
                        f"\tFigure {idx} cropped and saved as {cropped_image_filename}"
                    )
                    img_description, image_url = understand_image_with_gptv(
                        cropped_image_filename, ""
                    )
                    print(f"\tDescription of figure {idx}: {img_description}")

            md_content = replace_figure_description(
                md_content, figure_content, img_description
            )
            fig_metadata[idx] = image_url
            # md_content = update_figure_description(md_content, img_description, idx)
    return md_content, fig_metadata


def replace_figure_description(md_content, figure_content, img_description):
    """
    Replace the figure description in the Markdown content.

    Args:
        md_content (str): The original Markdown content.
        figure_content (str): The content of the figure.
        img_description (str): The description of the image.

    Returns:
        str: The updated Markdown content with the new figure description.
    """
    start = md_content.find(figure_content)
    end = start + len(figure_content)
    new_content = (
        md_content[:start]
        + f'<!-- FigureContent="{img_description}" -->'
        + md_content[end:]
    )
    return new_content


class AzureAIDocumentIntelligenceParser(BaseBlobParser):
    """Parses a PDF with Azure Document Intelligence"""

    def __init__(
        self,
        api_endpoint: str,
        api_key: str,
        api_version: Optional[str] = None,
        api_model: str = "prebuilt-layout",
        mode: str = "markdown",
        analysis_features: Optional[List[str]] = None,
    ):
        """
        Initialize the object for file processing with Azure Document Intelligence

        This constructor initializes a AzureAIDocumentIntelligenceParser object to be
        used for parsing files using the Azure Document Intelligence API. The lazy_parse
        method allows for processing files in a memory-efficient manner.

        Args:
            api_endpoint: The API endpoint to use for DocumentIntelligenceClient construction.
            api_key: str
                The API key to use for DocumentIntelligenceClient construction.
            api_version: Optional[str]
                The API version for DocumentIntelligenceClient. Setting None to use
                the default value from `azure-ai-documentintelligence` package.
            api_model: str
                The model to use for document analysis. Default is "prebuilt-layout".
            mode: str
                The mode of operation, can be "single", "page", or "markdown". Default is "markdown".
            analysis_features: Optional[List[str]]
                List of optional analysis features to enable, as a str that conforms to the enum
                in the `azure-ai-documentintelligence` package. Default value is None.

        Examples:
        ---------
        >>> obj = AzureAIDocumentIntelligenceParser(api_endpoint="...", api_key="...")
        """

        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.ai.documentintelligence.models import DocumentAnalysisFeature
        from azure.core.credentials import AzureKeyCredential

        kwargs = {}
        if api_version is not None:
            kwargs["api_version"] = api_version

        if analysis_features is not None:
            _SUPPORTED_FEATURES = [
                DocumentAnalysisFeature.OCR_HIGH_RESOLUTION,
            ]

            analysis_features = [
                DocumentAnalysisFeature(feature) for feature in analysis_features
            ]
            if any(
                [feature not in _SUPPORTED_FEATURES for feature in analysis_features]
            ):
                logger.warning(
                    "The current supported features are: %s. Using other features may result in unexpected behavior.",
                    [f.value for f in _SUPPORTED_FEATURES],
                )

        self.client = DocumentIntelligenceClient(
            endpoint=api_endpoint,
            credential=AzureKeyCredential(api_key),
            # headers={"x-ms-useragent": "langchain-parser/1.0.0"},
            features=analysis_features,
            **kwargs,
        )
        self.api_model = api_model
        self.mode = mode
        assert self.mode in ["single", "page", "markdown"]

    def _generate_docs_page(self, result: Any) -> Iterator[Document]:
        """
        Generate Documents for each page in the result.

        Args:
            result (Any): The result object from the Azure Document Intelligence API.

        Yields:
            Document: A Document object representing a page in the result.
        """
        for p in result.pages:
            content = " ".join([line.content for line in p.lines])

            d = Document(
                page_content=content,
                metadata={
                    "page": p.page_number,
                },
            )
            yield d

    def _generate_docs_single(self, file_path: str, result: Any) -> Iterator[Document]:
        """
        Generate a single Document for the entire result.

        Args:
            file_path (str): The path to the input file.
            result (Any): The result object from the Azure Document Intelligence API.

        Yields:
            Document: A Document object representing the entire result.
        """
        md_content, fig_metadata = include_figure_in_md(file_path, result)
        yield Document(page_content=md_content, metadata={"images": fig_metadata})

    def lazy_parse(self, file_path: str) -> Iterator[Document]:
        """
        Parse a file using Azure Document Intelligence in a memory-efficient manner.

        Args:
            file_path (str): The path to the file to parse.

        Yields:
            Document: A Document object representing the parsed content.
        """

        blob = Blob.from_path(file_path)
        with blob.as_bytes_io() as file_obj:
            poller = self.client.begin_analyze_document(
                self.api_model,
                file_obj,
                content_type="application/octet-stream",
                output_content_format="markdown" if self.mode == "markdown" else "text",
            )
            result = poller.result()

            if self.mode in ["single", "markdown"]:
                yield from self._generate_docs_single(file_path, result)
            elif self.mode in ["page"]:
                yield from self._generate_docs_page(result)
            else:
                raise ValueError(f"Invalid mode: {self.mode}")

    def parse_url(self, url: str) -> Iterator[Document]:
        """
        Parse a file from a URL using Azure Document Intelligence.

        Args:
            url (str): The URL to the file to parse.

        Yields:
            Document: A Document object representing the parsed content.
        """
        from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

        poller = self.client.begin_analyze_document(
            self.api_model,
            AnalyzeDocumentRequest(url_source=url),
            # content_type="application/octet-stream",
            output_content_format="markdown" if self.mode == "markdown" else "text",
        )
        result = poller.result()

        if self.mode in ["single", "markdown"]:
            yield from self._generate_docs_single(url, result)
        elif self.mode in ["page"]:
            yield from self._generate_docs_page(result)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")


class AzureAIDocumentIntelligenceLoader(BaseLoader):
    """
    A loader for parsing files using Azure Document Intelligence.
    """

    def __init__(
        self,
        api_endpoint: str,
        api_key: str,
        file_path: Optional[str] = None,
        url_path: Optional[str] = None,
        api_version: Optional[str] = None,
        api_model: str = "prebuilt-layout",
        mode: str = "markdown",
        *,
        analysis_features: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the object for file processing with Azure Document Intelligence
        (formerly Form Recognizer).

        This constructor initializes a AzureAIDocumentIntelligenceParser object to be
        used for parsing files using the Azure Document Intelligence API. The load
        method generates Documents whose content representations are determined by the
        mode parameter.

        Parameters:
        -----------
        api_endpoint: str
            The API endpoint to use for DocumentIntelligenceClient construction.
        api_key: str
            The API key to use for DocumentIntelligenceClient construction.
        file_path : Optional[str]
            The path to the file that needs to be loaded.
            Either file_path or url_path must be specified.
        url_path : Optional[str]
            The URL to the file that needs to be loaded.
            Either file_path or url_path must be specified.
        api_version: Optional[str]
            The API version for DocumentIntelligenceClient. Setting None to use
            the default value from `azure-ai-documentintelligence` package.
        api_model: str
            Unique document model name. Default value is "prebuilt-layout".
            Note that overriding this default value may result in unsupported
            behavior.
        mode: Optional[str]
            The type of content representation of the generated Documents.
            Use either "single", "page", or "markdown". Default value is "markdown".
        analysis_features: Optional[List[str]]
            List of optional analysis features, each feature should be passed
            as a str that conforms to the enum `DocumentAnalysisFeature` in
            `azure-ai-documentintelligence` package. Default value is None.

        Examples:
        ---------
        >>> obj = AzureAIDocumentIntelligenceLoader(
        ...     file_path="path/to/file",
        ...     api_endpoint="https://endpoint.azure.com",
        ...     api_key="APIKEY",
        ...     api_version="2023-10-31-preview",
        ...     api_model="prebuilt-layout",
        ...     mode="markdown"
        ... )
        """

        assert (
            file_path is not None or url_path is not None
        ), "file_path or url_path must be provided"
        self.file_path = file_path
        self.url_path = url_path

        self.parser = AzureAIDocumentIntelligenceParser(
            api_endpoint=api_endpoint,
            api_key=api_key,
            api_version=api_version,
            api_model=api_model,
            mode=mode,
            analysis_features=analysis_features,
        )

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """
        Load the file using Azure Document Intelligence in a memory-efficient manner.

        Yields:
            Document: A Document object representing the loaded content.
        """
        if self.file_path is not None:
            yield from self.parser.parse(self.file_path)
        else:
            yield from self.parser.parse_url(self.url_path)  # type: ignore[arg-type]
