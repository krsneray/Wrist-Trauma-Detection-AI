def flip_bounding_box_horizontal(x_center, image_width):
    if x_center < 0 or x_center > image_width:
        raise ValueError("Koordinat görüntü sınırları dışında olamaz!")
    return image_width - x_center