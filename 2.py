import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 尝试常见的内角点尺寸范围
    found = None
    for w in range(4, 10):
        for h in range(4, 10):
            ret_cb, corners = cv2.findChessboardCorners(gray, (w, h), None)
            if ret_cb:
                found = (w, h)
                cv2.drawChessboardCorners(frame, (w, h), corners, ret_cb)
                break
        if found:
            break
    cv2.putText(frame, f"Detected: {found}" if found else "Not detected", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Check', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()