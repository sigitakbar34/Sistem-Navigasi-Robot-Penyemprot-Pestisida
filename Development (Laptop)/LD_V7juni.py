import cv2
import numpy as np
import utils_V7juni
from ultralytics import YOLO
import time
import csv
import config
last_mask_time = time.time()

start_time = None

# untuk menyimpan hasil deteksi ke dalam file CSV
def save_to_csv(class_name, confidence, inference_time, fps):
    # global start_time
    fields = ['Detik', 'Jalur', 'Ujung Jalur', 'Confidence Jalur', 'Confidence Ujung Jalur', 'Inference Time', 'FPS', 'errorPos']
    data_row = ['', '', '', '', '', '', '', '']  # Inisialisasi baris data dengan nilai kosong
    errorPos = config.errorPos
    # Hitung waktu detik sejak proses dimulai
    # current_time = time.time()
    # detik = round((current_time - start_time),2)
    detik = utils_V7juni.get_elapsed_time()
    # detik = elapsed_time_global
    print(f"Elapsed Time (detik): {detik}") 

    if class_name == 'jalur':
        data_row[1] = class_name
        data_row[3] = f"{confidence:.2f}"
    elif class_name == 'ujung-jalur':
        data_row[2] = class_name
        data_row[4] = f"{confidence:.2f}"

    data_row[0] = detik  # Simpan waktu detik
    data_row[5] = f"{inference_time:.1f}ms"
    data_row[6] = f"{fps:.2f}"
    data_row[7] = f"{errorPos:.2f}"

    with open('detected_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        # Tulis header jika file masih kosong
        if file.tell() == 0:
            writer.writerow(fields)
        # Tulis baris data
        writer.writerow(data_row)


def getLane(results, display=2):
    global last_mask_time
    frameCopy = frame.copy()

    mask_np = utils_V7juni.masking(results, frame)
    if mask_np is not None:
        last_mask_time = time.time()  
        points = utils_V7juni.valTrackbars()
        hT, wT, c = frame.shape
        
        frameWarp = utils_V7juni.warping(mask_np, points, wT, hT)
        if frameWarp is None or frameWarp.size == 0:
            print("Error: frameWarp is invalid or has zero size.")
            return
        else: 
            print("ada nilai framewarp")

        frameWarpPoints = utils_V7juni.drawPoints(frameCopy, points)
        processedImage = utils_V7juni.isEndOfLane(results, model, frameWarp, display=True)
        if processedImage is None or processedImage.size == 0:
            print("processedImage tidak ada nilainya ")
            return
        else: 
            print("ada nilai processed image")
            

        if display == 1:
            stack_img = utils_V7juni.stackImages(0.7, ([tampilResults, frameWarpPoints], [mask_np, frameWarp]))
            cv2.imshow('Stacked Image', stack_img)
            cv2.imshow('Processed Image', processedImage)


        elif display == 2:
            stack_img = utils_V7juni.stackImages(0.7, ([frameWarpPoints, mask_np], [frameWarp, processedImage]))
            cv2.imshow('Stacked Image', stack_img)

    else:
        # Tutup window jika tidak ada nilai baru dalam 10 detik
        current_time = time.time()
        if current_time - last_mask_time > 15:
            if cv2.getWindowProperty("Masking", cv2.WND_PROP_VISIBLE) > 0:
                cv2.destroyWindow("Masking")
            if cv2.getWindowProperty("Warp", cv2.WND_PROP_VISIBLE) > 0:
                cv2.destroyWindow("Warp")
            if cv2.getWindowProperty("Warp Points", cv2.WND_PROP_VISIBLE) > 0:
                cv2.destroyWindow("Warp Points")
            if cv2.getWindowProperty("Processed Image", cv2.WND_PROP_VISIBLE) > 0:
                cv2.destroyWindow("Processed Image")
            if cv2.getWindowProperty("Stacked Image", cv2.WND_PROP_VISIBLE) > 0:
                cv2.destroyWindow("Stacked Image")
            if cv2.getWindowProperty("Result Instance Segmentation", cv2.WND_PROP_VISIBLE) > 0:
                cv2.destroyWindow("Result Instance Segmentation")


if __name__ == '__main__':
    model = YOLO("/home/sgt/TA/Coding_Almost/PID/416ao.pt")
    video_path = ("/home/sgt/TA/Coding_Almost/PID/WIN_20240530_16_26_19_Pro.mp4")

    # cap = cv2.VideoCapture(video_path) 
        
    cap = cv2.VideoCapture(2)    
    if not cap.isOpened():
        print("Gagal membuka video.")
        exit()
    initializeTrackBarVals = [8, 128, 0, 179]
    utils_V7juni.initializeTrackbars(initializeTrackBarVals)
    frameCounter = 0
    start_time = time.time()
    
    prev_time = 0
    new_time = 0

    while True:
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0

        ret, video = cap.read()
        if not ret:
            print("Gagal membaca frame dari video.")
            break
        
        frame = utils_V7juni.tambahkan_bingkai(video)
        frame = cv2.resize(frame, (512, 256))
        results = model.predict(frame, imgsz=416, conf=0.5)

        new_time = time.time()
        fps = round((1/(new_time-prev_time)),2)
        config.fps = fps
        prev_time = new_time
        print("FPS main: "+str(fps))
        # print("fps utils: ", utils_V7juni.fps_value)

        for result in results: 
            for box in result.boxes:
                class_name = result.names[int(box.cls)]
                confidence = float(box.conf) 
                # fps = utils_V7juni.fps_value
                inference_time = result.speed['inference']
                print(f"Detected: {class_name}, confidence: {confidence:.2f}, inference time (ms): {inference_time:.1f}, fps: {fps:.2f},")
                save_to_csv(class_name, confidence, inference_time, fps)
        tampilResults = results[0].plot()
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(tampilResults, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        getLane(results, display=2)
        cv2.imshow("Result Instance Segmentation", tampilResults)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
