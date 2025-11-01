def draw_bounding_boxes(image, boxes, color=(0,255,0), thickness=2):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

image = cv2.imread('image.jpg')
image_with_boxes = draw_bounding_boxes(image, windows)
cv2.imwrite('output_with_boxes.jpg', image_with_boxes)
cv2.imshow('Detection', image_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()
