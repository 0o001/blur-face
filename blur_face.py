import cv2

def blur_img(img, factor = 1):
        kW = int(img.shape[1] / factor)
        kH = int(img.shape[0] / factor)
    
        if kW % 2 == 0:
                kW = kW - 1
        if kH % 2 == 0:
                kH = kH - 1
    
        blurred_img = cv2.GaussianBlur(img, (kW , kH), 0)
        return blurred_img

def main():
        cap = cv2.VideoCapture(0)

        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        while(True):
                ret, frame = cap.read()

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = faceCascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30)
                )

                for x, y, w, h in faces:
                        detected_face = frame[int(y): int(y + h), int(x): int(x + w)]
                        detected_face_blurred = blur_img(detected_face, factor = 3)
                        frame[y: y + h, x: x + w] = detected_face_blurred
                        
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        cap.release()
        cv2.destroyAllWindows()

main()