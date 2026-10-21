"""Draw the notebook's training examples and shared-input comparison."""

from PIL import Image, ImageDraw

from src.helper.shared_inputs import prepare_image


def training_examples_figure(inventory_df, example_ids):
    """Show the chosen readable training images with their labels and IDs."""
    training_by_id = inventory_df.loc[
        inventory_df["train"] & inventory_df["is_readable"]
    ].set_index("mapillary_image_id")
    figure = Image.new("RGB", (900, 1400), "white")
    labels = ImageDraw.Draw(figure)
    for row, (surface_type, ids) in enumerate(example_ids.items()):
        for column, image_id in enumerate(ids):
            example = training_by_id.loc[image_id]
            assert example["surface_type"] == surface_type
            with Image.open(example["file_path"]) as image:
                image.thumbnail((430, 245))
                figure.paste(image, (column * 450 + 10, row * 280 + 5))
            labels.text(
                (column * 450 + 10, row * 280 + 252),
                f"{surface_type} | {image_id}",
                fill="black",
            )
    return figure


def crop_comparison_figure(split_df, sizes):
    """Show the first included training image of each size beside its prepared pixels."""
    training_examples = split_df.loc[split_df["split"] == "train"]
    comparison = Image.new("RGB", (570, 480), "white")
    caption = ImageDraw.Draw(comparison)
    for row, size in enumerate(sizes):
        for record in training_examples.itertuples():
            with Image.open(record.file_path) as source:
                if source.size != size:
                    continue
                original = source.convert("RGB")
                prepared = prepare_image(record.file_path)
            comparison.paste(original, (10, row * 240 + 25))
            comparison.paste(
                Image.fromarray((prepared * 255).astype("uint8")),
                (330, row * 240 + 25),
            )
            caption.text(
                (10, row * 240 + 5),
                f"Original {size} | {record.mapillary_image_id}",
                fill="black",
            )
            caption.text(
                (330, row * 240 + 5), "Prepared (128, 96)", fill="black"
            )
            break
        else:
            raise ValueError(f"No included training example with size {size}")
    return comparison
