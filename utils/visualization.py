from PIL import Image, ImageDraw, ImageOps

from .image_decode import decode_image


def save_collage(query_bytes, results, output, cell_width=240, cell_height=240):
    width, height = int(cell_width), int(cell_height)
    if width < 32 or height < 32:
        raise ValueError("visualization cells must be at least 32 pixels")

    columns = min(4, max(1, len(results)))
    rows = max(1, (len(results) + columns - 1) // columns)
    canvas = Image.new("RGB", (width * (columns + 1), height * rows), "white")
    draw = ImageDraw.Draw(canvas)

    def put(data, x, y, label):
        image = ImageOps.contain(decode_image(data), (width - 16, height - 34))
        canvas.paste(image, (x + (width - image.width) // 2,
                             y + 8 + (height - 34 - image.height) // 2))
        draw.text((x + 8, y + height - 23), label[:34], fill="black")

    put(query_bytes, 0, 0, "Query")
    for pos, (name, data) in enumerate(results):
        row, col = divmod(pos, columns)
        put(data, (col + 1) * width, row * height, f"{pos + 1}. {name}")
    canvas.save(output)
