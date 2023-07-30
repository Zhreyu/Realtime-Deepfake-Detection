import cv2
from face_detection import detect_bounding_box
from deepfake_detection import predict

def main():
    video_capture = cv2.VideoCapture(0)
    while True:
        result, video_frame = video_capture.read()
        if result is False:
            break
        
        faces = detect_bounding_box(video_frame)
        video_frame = predict(video_frame)

        cv2.imshow("My Face Detection Project", video_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
