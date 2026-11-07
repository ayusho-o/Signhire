"""
find_camera.py
--------------
Scans camera indices 0-5 and opens a preview window for each one found.
Press Q to close each window and move to the next camera.
Run this to identify which index corresponds to your HP built-in camera.

    python find_camera.py
"""
import cv2

for index in range(6):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"  Index {index}: not found")
        cap.release()
        continue

    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"  Index {index}: opened but couldn't read a frame")
        cap.release()
        continue

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Index {index}: FOUND  ({w}x{h}) — showing preview, press Q to close")

    window = f"Camera index {index}  ({w}x{h}) — press Q to close"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 640, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(window, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

print("\nDone. Use the index that showed your HP camera with --camera <index>")
