import math
import os
import pathlib

import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils

BLANK_WIDTH = 600
BLANK_HEIGHT = 830

RESULT_X_START = BLANK_WIDTH // 4
RESULT_X_END = int(BLANK_WIDTH * 0.9)
RESULT_WIDTH = RESULT_X_END - RESULT_X_START

RESULT_Y_START = BLANK_HEIGHT // 4
RESULT_Y_END = int(BLANK_HEIGHT * 0.7)
RESULT_HEIGHT = RESULT_Y_END - RESULT_Y_START

DRAW = False
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
LAYOUT_IMAGE_SIZE = 300


# DRAW = True


def do_magic(chat_id, bot):
    directory = f"bot/{chat_id}/"
    skipped = 0
    files = os.listdir(directory)
    print(files)
    files_count = len(files)
    if files_count <= 0:
        if chat_id:
            bot.send_message(chat_id, f"I didn't have any images.. fuck off.")
            return

    is_video = files_count == 1
    print(is_video)

    # load the input image, resize it, and convert it to grayscale
    images = list()
    pathlib.Path(f"temp/{chat_id}/").mkdir(parents=True, exist_ok=True)
    os.system(f"rm -f temp/{chat_id}/*")

    out = cv2.VideoWriter(
        "outpy.mp4",
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        20,
        (RESULT_WIDTH, RESULT_HEIGHT),
    )

    frames = get_frames(directory, files, is_video)

    image_count = len(frames)
    print("Total images: ", image_count)
    message = None
    if chat_id:
        bot.send_message(
            chat_id, f"Ok i got {image_count} images to go thought stick with me!"
        )
        message = bot.send_message(chat_id, f"Let me think")

    i = 0
    base_scale = 0
    for image in frames:
        scale = 1
        if message:
            bot.edit_message_text(
                message_id=message.message_id,
                chat_id=chat_id,
                text=f"{i}th image is done. {image_count - i} remaining. skipped {skipped}",
            )
        i += 1
        if i % 10 == 0:
            print(i, end=", ")
        if i % 100 == 0:
            print("")
        image = imutils.resize(image, width=BLANK_WIDTH)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale image
        rects = detector(gray, 1)
        # loop over the face detections
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        # cv2.imshow('img', image)
        # cv2.waitKey(0)
        if len(rects) <= 0:
            if not is_video:
                skipped += 1
                print("skipped no face", end=". ")
                continue
        else:
            rect = rects[len(rects) // 2]
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            left_eye = shape[36:41]
            right_eye = shape[42:48]
            right_eye_x = right_eye[0][0]
            right_eye_y = right_eye[0][1]
            left_eye_x = left_eye[-1][0]
            left_eye_y = left_eye[-1][1]

            if base_scale == 0:
                base_scale = abs(right_eye_x - left_eye_x) // 1.1

            scale = base_scale / (right_eye_x - left_eye_x)
            # if scale > 2.5:
            # 	print("skipped because of scale")
            # 	continue
            # scale += i * scale_up
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (rect_x, rect_y, rect_w, rect_h) = face_utils.rect_to_bb(rect)
            if DRAW:
                cv2.rectangle(
                    image,
                    (rect_x, rect_y),
                    (rect_x + rect_w, rect_y + rect_h),
                    (0, 255, 0),
                    2,
                )
                # show the face number
                cv2.putText(
                    image,
                    "Face #1",
                    (rect_x - 10, rect_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw them on the image
                for (x, y) in shape:
                    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

            adj = math.sqrt(
                (right_eye_x - left_eye_x) ** 2 + (right_eye_y - left_eye_y) ** 2
            )
            opp = right_eye_y - left_eye_y
            rotation_angle = math.degrees(math.tan(opp / adj))
            center = ((right_eye_x + left_eye_x) // 2, (right_eye_y + left_eye_y) // 2)

            temp = imutils.rotate(image, rotation_angle, scale=scale)
            # cv2.imshow('img', temp)
            # cv2.waitKey(0)
            # brightness_rating = sum(
            # 	cv2.cvtColor(
            # 		temp[max(center[1] - 40, 0): center[1] + 40, max(center[0] - 100, 0): center[0] + 100],
            # 		cv2.COLOR_BGR2HSV
            # 	)[:, :, 0][2]
            # )

            if DRAW:
                cv2.circle(image, center, 1, (0, 255, 0), -1)
        if len(rects) <= 0 and is_video:
            # images.append((imutils.resize(image, width=RESULT_WIDTH), 0))
            center = (image.shape[1] // 2, image.shape[0] // 2)
            rotation_angle = 0

        blank_image = np.zeros((BLANK_WIDTH, BLANK_HEIGHT, 3), np.uint8)
        x_start_pos = blank_image.shape[0] // 2 - center[1]
        y_start_pos = blank_image.shape[1] // 2 - center[0]
        x_end_pos = min(x_start_pos + image.shape[0], blank_image.shape[0])
        y_end_pos = min(y_start_pos + image.shape[1], blank_image.shape[1])
        blank_image[
            max(0, x_start_pos) : x_end_pos,
            max(0, y_start_pos) : y_end_pos,
        ] = image[
            abs(min(0, x_start_pos)) : x_end_pos
            - max(x_start_pos, 0)
            + abs(min(0, x_start_pos)),
            abs(min(0, y_start_pos)) : y_end_pos
            - max(y_start_pos, 0)
            + abs(min(0, y_start_pos)),
        ]

        if DRAW:
            cv2.circle(
                blank_image,
                (blank_image.shape[1] // 2, blank_image.shape[0] // 2),
                5,
                (255, 0, 255),
                -1,
            )

        blank_image = imutils.rotate(blank_image, rotation_angle, scale=scale)
        if DRAW:
            cv2.imshow("img", blank_image)
            cv2.waitKey(0)

        images.append(
            (
                blank_image[RESULT_X_START:RESULT_X_END, RESULT_Y_START:RESULT_Y_END],
                # brightness_rating
            )
        )
        out.write(image)

    if message:
        bot.edit_message_text(
            message_id=message.message_id,
            chat_id=chat_id,
            text=f"Now i know what to do give me some tim.. Again",
        )
    out.release()
    # images = sorted(images, key=lambda tup: tup[1])

    print("saving files")

    if bot:
        bot.edit_message_text(
            message_id=message.message_id,
            chat_id=chat_id,
            text=f"saving files :) hold up!",
        )

    files = list()
    for image in range(0, len(images)):
        file_name = f"temp/{chat_id}/{'{:04d}'.format(image)}.jpg"
        cv2.imwrite(file_name, images[image][0])
        files.append(file_name)

    print("generating gif")
    if bot:
        bot.edit_message_text(
            message_id=message.message_id, chat_id=chat_id, text=f"Generating Gif"
        )

    delay = 20
    if is_video:
        delay = 4

    if is_video:
        os.system(f"rm gifsbot/{chat_id}.mp4")
        os.system(
            f"ffmpeg -framerate 25 -i temp/{chat_id}/%04d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p gifsbot/{chat_id}.mp4"
        )
    else:
        os.system(
            f"convert -delay {delay} -resize 95% -loop 0 temp/{chat_id}/*.jpg gifsbot/{chat_id}.gif"
        )

    if bot:
        bot.edit_message_text(
            message_id=message.message_id,
            chat_id=chat_id,
            text=f"Uploading....",
        )
        output_format = "gif"
        if is_video:
            output_format = "mp4"
        bot.send_animation(
            chat_id, animation=open(f"gifsbot/{chat_id}.{output_format}", "rb")
        )
    print("done :)")


def do_me_out(chat_id):
    directory = f"bot/{chat_id}/"
    pathlib.Path(f"gifsbot").mkdir(parents=True, exist_ok=True)

    images_temp = get_frames(directory, os.listdir(directory), False)
    images = list()
    print(directory, os.listdir(directory))
    for image in images_temp:
        if len(image) > len(image[0]):
            images.append(imutils.resize(image, width=LAYOUT_IMAGE_SIZE))
        else:
            images.append(imutils.resize(image, height=LAYOUT_IMAGE_SIZE))

    # widths, heights = zip(*(i.size for i in images))
    image_count = len(images)
    count_ = int(math.sqrt(image_count) + 0.5)
    max_size = min(image_count * LAYOUT_IMAGE_SIZE, LAYOUT_IMAGE_SIZE * count_)
    row = 0
    column = 0
    # new_im = Image.new("RGB", (max_size, max_size), color="#FFF")
    blank_image = np.ones((max_size, max_size, 3), np.uint8)

    for i in range(image_count):
        image = images[i]
        x_size = column * LAYOUT_IMAGE_SIZE
        y_size = row * LAYOUT_IMAGE_SIZE

        # print(
        #     i,
        #     "row:",
        #     row,
        #     "column:",
        #     column,
        #     (LAYOUT_IMAGE_SIZE * column, LAYOUT_IMAGE_SIZE * row),
        #     (x_size, x_size + len(image)),
        #     len(image),
        #     (y_size, y_size + len(image[0])),
        #     len(image[0]),
        # )

        blank_image[
            x_size : x_size + LAYOUT_IMAGE_SIZE,
            y_size : y_size + LAYOUT_IMAGE_SIZE,
        ] = image[0:LAYOUT_IMAGE_SIZE, 0:LAYOUT_IMAGE_SIZE]
        row += 1
        if row == count_:
            row = 0
            column += 1
    cv2.imwrite(f"gifsbot/{chat_id}.jpg", blank_image)


def get_frames(directory, files, is_video):
    if is_video:
        frames = list()
        cap = cv2.VideoCapture(f"{directory}/{files[0]}")
        if not cap.isOpened():
            print("Error opening video stream or file")

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
        cap.release()
        return frames
    return [cv2.imread(f"{directory}/{files[i]}") for i in range(0, len(files))]
