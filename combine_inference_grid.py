#!/usr/bin/env python3
"""Combine two sets of inference images into a labeled comparison grid.

The script expects two folders that each contain an equal number of images.
The images are arranged in two rows: the top row shows the images from the
first directory (e.g. 8-step inference) and the bottom row shows the images
from the second directory (e.g. 50-step inference). Prompt captions are shown
above each column, and row descriptions are rendered on the left.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

from PIL import Image, ImageDraw, ImageFont

DEFAULT_PROMPTS: Sequence[str] = (
    "((masterpiece,best quality)) , ((1girl)), ((school uniform)),brown blazer, black skirt, necktie,red plaid skirt,looking at viewer, masterpiece, best quality, ultra-detailed, 8k resolution, high dynamic range, absurdres, stunningly beautiful, intricate details, sharp focus, detailed eyes, cinematic color grading, high-resolution texture,photorealistic portrait, nails",
    "a painting of a virus monster playing guitar",
    "an image of an animal half mouse half octopus",
    "a painting of a squirrel eating a burger",
    "a painting of a chair that looks like an octopus",
    "a man wear a shirt with the inscription that write I love generative models!",
    "a street sign that reads latent diffusion",
)

ROW_DESCRIPTIONS: Sequence[str] = (
    "Generated with 8-step inference (ours)",
    "Generated with 50-step inference",
)

PADDING = 40
COLUMN_SPACING = 20
ROW_SPACING = 20
PROMPT_FONT_SIZE = 60
ROW_LABEL_FONT_SIZE = 70
PROMPT_LINE_SPACING = 6
ROW_LINE_SPACING = 4
PROMPT_AREA_PADDING_Y = 10
ROW_LABEL_PADDING_X = 20
ROW_LABEL_PADDING_Y = 10
ROW_LABEL_MAX_WIDTH = 420
TEXT_COLOR = (30, 30, 30)
BACKGROUND_COLOR = (245, 245, 245)
TEXT_STROKE_WIDTH = 2
MAX_OUTPUT_LENGTH = 3072

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

BOLD_FONT_CANDIDATES = (
    "arialbd.ttf",
    "Arial Bold.ttf",
    "Arial-Bold.ttf",
    "Arial-BoldMT.ttf",
    "DejaVuSans-Bold.ttf",
    "LiberationSans-Bold.ttf",
)

REGULAR_FONT_CANDIDATES = (
    "arial.ttf",
    "Arial.ttf",
    "DejaVuSans.ttf",
    "LiberationSans-Regular.ttf",
)


def load_images(directory: Path) -> List[Image.Image]:
    image_paths = sorted(
        path for path in directory.iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No images with supported extensions found in {directory}.")

    images: List[Image.Image] = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            images.append(img.convert("RGB"))
    return images


def get_font(size: int, fallback: str | None = None, *, bold: bool = False) -> ImageFont.ImageFont:
    candidate_paths: List[str] = []
    if fallback:
        candidate_paths.append(str(fallback))
    if bold:
        candidate_paths.extend(BOLD_FONT_CANDIDATES)
    candidate_paths.extend(REGULAR_FONT_CANDIDATES)

    for candidate in candidate_paths:
        try:
            return ImageFont.truetype(candidate, size=size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def measure_text_width(font: ImageFont.ImageFont, text: str) -> float:
    if hasattr(font, "getlength"):
        return font.getlength(text)  # type: ignore[attr-defined]
    return font.getsize(text)[0]


def line_height(font: ImageFont.ImageFont) -> int:
    if hasattr(font, "getbbox"):
        bbox = font.getbbox("Ay")  # type: ignore[attr-defined]
        return bbox[3] - bbox[1]
    return font.getsize("Ay")[1]


def wrap_text_to_width(text: str, font: ImageFont.ImageFont, max_width: float) -> List[str]:
    paragraphs = [segment.strip() for segment in text.split("\n") if segment.strip()]
    if not paragraphs:
        return []

    wrapped_lines: List[str] = []
    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            wrapped_lines.append("")
            continue
        current_line = words[0]
        for word in words[1:]:
            trial_line = f"{current_line} {word}"
            if measure_text_width(font, trial_line) <= max_width:
                current_line = trial_line
            else:
                wrapped_lines.append(current_line)
                current_line = word
        wrapped_lines.append(current_line)
    return wrapped_lines


def text_block_height(num_lines: int, font: ImageFont.ImageFont, spacing: int) -> int:
    if num_lines <= 0:
        return 0
    return num_lines * line_height(font) + (num_lines - 1) * spacing


def ensure_equal_image_sizes(rows: Sequence[Sequence[Image.Image]]) -> None:
    first_width, first_height = rows[0][0].size
    for row in rows:
        for image in row:
            if image.size != (first_width, first_height):
                raise ValueError("All source images must share the same dimensions.")


def build_canvas(
    rows: Sequence[Sequence[Image.Image]],
    prompts: Sequence[Sequence[str]],
    row_labels: Sequence[Sequence[str]],
    prompt_font: ImageFont.ImageFont,
    row_font: ImageFont.ImageFont,
) -> Image.Image:
    cols = max(len(row) for row in rows)
    img_width, img_height = rows[0][0].size

    prompt_block_heights = [text_block_height(len(lines), prompt_font, PROMPT_LINE_SPACING) for lines in prompts]
    max_prompt_height = max(prompt_block_heights) if prompt_block_heights else 0
    prompt_area_height = max_prompt_height + 2 * PROMPT_AREA_PADDING_Y if max_prompt_height else 0

    row_block_heights = [text_block_height(len(lines), row_font, ROW_LINE_SPACING) for lines in row_labels]
    max_row_block_height = max(row_block_heights) if row_block_heights else 0
    row_label_width = max(
        (max(measure_text_width(row_font, line) for line in lines) if lines else 0)
        for lines in row_labels
    )
    row_label_width = int(row_label_width) + 2 * ROW_LABEL_PADDING_X

    grid_width = cols * img_width + (cols - 1) * COLUMN_SPACING
    grid_height = len(rows) * img_height + (len(rows) - 1) * ROW_SPACING

    canvas_width = PADDING + row_label_width + grid_width + PADDING
    canvas_height = PADDING + prompt_area_height + grid_height + PADDING
    if max_row_block_height > img_height:
        # Allow extra vertical space if the row label is taller than the image height.
        additional_row_height = max_row_block_height - img_height
        canvas_height += additional_row_height

    canvas = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)

    grid_origin_x = PADDING + row_label_width
    grid_origin_y = PADDING + prompt_area_height

    # Draw prompts above columns.
    if prompts:
        for col_index, lines in enumerate(prompts):
            if not lines:
                continue
            text_height = text_block_height(len(lines), prompt_font, PROMPT_LINE_SPACING)
            column_center_x = grid_origin_x + col_index * (img_width + COLUMN_SPACING) + img_width / 2
            text_y = PADDING + PROMPT_AREA_PADDING_Y + (max_prompt_height - text_height) / 2
            for line in lines:
                text_width = measure_text_width(prompt_font, line)
                text_x = column_center_x - text_width / 2
                draw.text(
                    (text_x, text_y),
                    line,
                    fill=TEXT_COLOR,
                    font=prompt_font,
                    stroke_width=TEXT_STROKE_WIDTH,
                    stroke_fill=TEXT_COLOR,
                )
                text_y += line_height(prompt_font) + PROMPT_LINE_SPACING

    # Draw row labels.
    for row_index, lines in enumerate(row_labels):
        if not lines:
            continue
        text_height = text_block_height(len(lines), row_font, ROW_LINE_SPACING)
        row_top = grid_origin_y + row_index * (img_height + ROW_SPACING)
        row_center_y = row_top + img_height / 2
        text_y = row_center_y - text_height / 2
        for line in lines:
            text_width = measure_text_width(row_font, line)
            text_x = PADDING + (row_label_width - text_width) / 2
            draw.text(
                (text_x, text_y),
                line,
                fill=TEXT_COLOR,
                font=row_font,
                stroke_width=TEXT_STROKE_WIDTH,
                stroke_fill=TEXT_COLOR,
            )
            text_y += line_height(row_font) + ROW_LINE_SPACING

    # Paste images.
    for row_index, row in enumerate(rows):
        for col_index, image in enumerate(row):
            x = grid_origin_x + col_index * (img_width + COLUMN_SPACING)
            y = grid_origin_y + row_index * (img_height + ROW_SPACING)
            canvas.paste(image, (int(x), int(y)))

    return canvas


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine inference results into a labeled grid.")
    parser.add_argument(
        "--eight-step-dir",
        type=Path,
        default=Path("gen_img_val_xl/samples-customedXL-8-retrain-free-full-trick-1-7.5"),
        help="Directory containing 8-step inference images.",
    )
    parser.add_argument(
        "--fifty-step-dir",
        type=Path,
        default=Path("gen_img_val_xl/samples-org-50-notNPNet"),
        help="Directory containing 50-step inference images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gen_img_val_xl/comparison_grid.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=None,
        help="Optional TTF font file for rendering text.",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Optional text file with one prompt per line to override the defaults.",
    )
    parser.add_argument(
        "--prompt-font-size",
        type=int,
        default=PROMPT_FONT_SIZE,
        help="Font size for the prompt captions.",
    )
    parser.add_argument(
        "--row-font-size",
        type=int,
        default=ROW_LABEL_FONT_SIZE,
        help="Font size for the row descriptions.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_OUTPUT_LENGTH,
        help=(
            "Strict upper bound (in pixels) for the longest side of the output image. "
            "The generated image will have both width and height less than this value."
        ),
    )
    return parser.parse_args(argv)


def read_prompts(args: argparse.Namespace) -> Sequence[str]:
    if args.prompts_file is None:
        return DEFAULT_PROMPTS
    with args.prompts_file.open("r", encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip()]
    if not prompts:
        raise ValueError("Prompts file is empty.")
    return prompts


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)

    prompts = read_prompts(args)

    eight_step_dir = args.eight_step_dir.resolve()
    fifty_step_dir = args.fifty_step_dir.resolve()

    if not eight_step_dir.is_dir():
        raise FileNotFoundError(f"8-step directory not found: {eight_step_dir}")
    if not fifty_step_dir.is_dir():
        raise FileNotFoundError(f"50-step directory not found: {fifty_step_dir}")

    eight_step_images = load_images(eight_step_dir)
    fifty_step_images = load_images(fifty_step_dir)

    if len(eight_step_images) != len(fifty_step_images):
        raise ValueError("Both directories must contain the same number of images.")
    if len(prompts) != len(eight_step_images):
        raise ValueError(
            "The number of prompts must match the number of images per row. "
            f"Got {len(prompts)} prompts but {len(eight_step_images)} images."
        )

    rows: Sequence[Sequence[Image.Image]] = (eight_step_images, fifty_step_images)
    ensure_equal_image_sizes(rows)

    prompt_font = get_font(
        args.prompt_font_size,
        fallback=str(args.font_path) if args.font_path else None,
        bold=True,
    )
    row_font = get_font(
        args.row_font_size,
        fallback=str(args.font_path) if args.font_path else None,
        bold=True,
    )

    prompt_lines = [wrap_text_to_width(prompt, prompt_font, rows[0][0].width) for prompt in prompts]
    row_label_lines = [
        wrap_text_to_width(label, row_font, ROW_LABEL_MAX_WIDTH)
        for label in ROW_DESCRIPTIONS
    ]

    canvas = build_canvas(rows, prompt_lines, row_label_lines, prompt_font, row_font)

    if args.max_length <= 1:
        raise ValueError("--max-length must be greater than 1.")

    max_canvas_dimension = max(canvas.size)
    if max_canvas_dimension >= args.max_length:
        target_length = args.max_length - 1
        scale = target_length / max_canvas_dimension
        new_size = (
            max(1, int(round(canvas.width * scale))),
            max(1, int(round(canvas.height * scale))),
        )
        canvas = canvas.resize(new_size, resample=Image.LANCZOS)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Saved comparison grid to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
