import cv2


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Mouse click at coordinates : ', x, ',', y)
        param.append((x, y))

def get_click_coordinates(image):
    # create an empty list to store the coordinates
    coordinates = []

    # display the image in a window
    cv2.imshow('image', image)

    # set the callback function for mouse events
    cv2.setMouseCallback('image', click_event, param=coordinates)

    # wait until any key is pressed
    cv2.waitKey(0)

    # return the coordinates
    return coordinates