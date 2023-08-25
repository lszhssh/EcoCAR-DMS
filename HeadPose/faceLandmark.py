import cv2
import dlib
import numpy as np

cap = cv2.VideoCapture(0)

hog_face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("/Users/sizheli/GitHub/EcoCAR-DMS/HeadPose/shape_predictor_68_face_landmarks.dat")

# 3-D model coords from a generic head model
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = frame.shape
    facial_coords = np.empty([6, 2], dtype="double")

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)

        for n in range(0, 68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            if n == 33:          # nose tip
                facial_coords[0] = [x, y]
                cv2.circle(frame, (x, y), 1, (255, 0, 0), 1)
            elif n == 8:         # chin
                facial_coords[1] = [x, y]
                cv2.circle(frame, (x, y), 1, (255, 0, 0), 1)
            elif n == 36:        # left eye left corner
                facial_coords[2] = [x, y]
                cv2.circle(frame, (x, y), 1, (255, 0, 0), 1)
            elif n == 45:        # right eye right corner
                facial_coords[3] = [x, y]
                cv2.circle(frame, (x, y), 1, (255, 0, 0), 1)
            elif n == 48:        # left mouth corner
                facial_coords[4] = [x, y]
                cv2.circle(frame, (x, y), 1, (255, 0, 0), 1)
            elif n == 54:        # right mouth corner
                facial_coords[5] = [x, y]
                cv2.circle(frame, (x, y), 1, (255, 0, 0), 1)
            else:
                cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )

    print("Camera Matrix :\n {0}".format(camera_matrix));

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, facial_coords, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    print("Rotation Vector:\n {0}".format(rotation_vector));
    print("Translation Vector:\n {0}".format(translation_vector));


    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    for p in facial_coords:
        cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)


    p1 = ( int(facial_coords[0][0]), int(facial_coords[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    cv2.line(frame, p1, p2, (255,0,0), 2)

    cv2.imshow("Face Landmarks", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()