# import cv2
# import numpy as np
# import torch
# import time 
# import serial 
# import csv

# esp_ser = serial.Serial(
#     port = '/dev/ttyUSB0',
#     baudrate = 115200,
#     bytesize=serial.EIGHTBITS,
#     parity=serial.PARITY_NONE,
#     stopbits=serial.STOPBITS_ONE,
#     timeout=0.1,
#     # timeout=1,
# )

# start_detection_time = None
# ujung_jalur_detected = False

# turn_movement = None  
# straight_movement = None

# last_detected_movement = None

# detected_jalur = False
# detected_ujung_jalur = False

# waktupid = None
# fps_value = 0.0

# def tambahkan_bingkai(video, lebar_bingkai_vertikal=128, tinggi_bingkai_horizontal=64):
#     h, w, _ = video.shape

#     # Mengatur ulang ukuran video menjadi lebih kecil
#     new_h = h - 2 * tinggi_bingkai_horizontal
#     new_w = w - 2 * lebar_bingkai_vertikal
#     video_kecil = cv2.resize(video, (new_w, new_h))

#     # Membuat bingkai vertikal (sisi kiri dan kanan)
#     bingkai_vertikal = np.zeros((new_h, lebar_bingkai_vertikal, 3), dtype=np.uint8)
#     bingkai_vertikal[:] = (0, 0, 0)  # Set warna bingkai ke hitam
    
#     # Membuat bingkai horizontal (sisi atas dan bawah)
#     bingkai_horizontal = np.zeros((tinggi_bingkai_horizontal, new_w + 2 * lebar_bingkai_vertikal, 3), dtype=np.uint8)
#     bingkai_horizontal[:] = (0, 0, 0)  # Set warna bingkai ke hitam

#     # Menambahkan bingkai di sisi kiri dan kanan
#     video_dengan_bingkai_vertikal = np.concatenate((bingkai_vertikal, video_kecil, bingkai_vertikal), axis=1)

#     # Menambahkan bingkai di sisi atas dan bawah
#     frame_dengan_bingkai = np.concatenate((bingkai_horizontal, video_dengan_bingkai_vertikal, bingkai_horizontal), axis=0)
    
#     return frame_dengan_bingkai


# # def masking(results):
# #     all_masks = None

# #     for result in results:
# #         if len(result) < 1:
# #             continue

# #         masks = result.masks.data
# #         boxes = result.boxes.data
# #         clss = boxes[:, 5]

# #         for class_id in range(int(torch.max(clss).item()) + 1):
# #             class_indices = torch.where(clss == class_id)
# #             class_masks = masks[class_indices]
# #             class_mask = torch.any(class_masks, dim=0).int()

# #             if all_masks is None:
# #                 all_masks = torch.zeros_like(class_mask)

# #             all_masks += class_mask


# #     if all_masks is None:
# #         return None

# #     all_masks *= 255
# #     all_masks = all_masks.to('cpu')  # Memindahkan tensor ke CPU
# #     mask_np = all_masks.numpy().astype('uint8')
# #     mask_np = cv2.resize(mask_np, (512, 256))
# #     # mask_np = cv2.resize(mask_np, (480, 240))

# #     return mask_np

# def masking(results, frame):
#     all_masks = None

#     for result in results:
#         if len(result) < 1:
#             continue
        
#         for result in results:
#             if len(result) < 1:
#                 continue

#             for ci, c in enumerate(result):
#                 label = c.names[c.boxes.cls.tolist().pop()]

#                 mask_np = np.zeros(frame.shape[:2], np.uint8)

#                 # Create contour mask 
#                 contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
#                 _ = cv2.drawContours(mask_np, [contour], -1, (255, 255, 255), cv2.FILLED)

#                 # cv2.imshow("Masking", mask_np)
#                 return mask_np


# def warping(frame, points, w, h, inv=False):
#     points1 = np.float32(points)
#     points2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
#     if inv:
#         matrix = cv2.getPerspectiveTransform(points2, points1)
#     else:
#         matrix = cv2.getPerspectiveTransform(points1, points2)

#     frameWarp = cv2.warpPerspective(frame, matrix, (w, h))
#     return frameWarp


# def nothing(a):
#     pass


# def initializeTrackbars(intialTracbarVals, wT=512, hT=256):
# # def initializeTrackbars(intialTracbarVals, wT=480, hT=240):
#     cv2.namedWindow("Trackbars")
#     cv2.resizeWindow("Trackbars", 512, 256)
#     # cv2.resizeWindow("Trackbars", 480, 240)
#     cv2.createTrackbar("Width Top", "Trackbars",
#                        intialTracbarVals[0], wT//2, nothing)
#     cv2.createTrackbar("Height Top", "Trackbars",
#                        intialTracbarVals[1], hT, nothing)
#     cv2.createTrackbar("Width Bottom", "Trackbars",
#                        intialTracbarVals[2], wT//2, nothing)
#     cv2.createTrackbar("Height Bottom", "Trackbars",
#                        intialTracbarVals[3], hT, nothing)


# def valTrackbars(wT=512, hT=256):
# # def valTrackbars(wT=480, hT=240):
#     widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
#     heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
#     widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
#     heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
#     points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop),
#                          (widthBottom, heightBottom), (wT-widthBottom, heightBottom)])
#     return points


# def drawPoints(frame, points):
#     for x in range(4):
#         cv2.circle(frame, (int(points[x][0]), int(
#             points[x][1])), 8, (0, 0, 255), cv2.FILLED)
#     return frame
    

# def stackImages(scale, imgArray):
#     rows = len(imgArray)
#     cols = len(imgArray[0])
#     rowsAvailable = isinstance(imgArray[0], list)
#     width = imgArray[0][0].shape[1]
#     height = imgArray[0][0].shape[0]
#     if rowsAvailable:
#         for x in range(0, rows):
#             for y in range(0, cols):
#                 if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
#                     imgArray[x][y] = cv2.resize(
#                         imgArray[x][y], (0, 0), None, scale, scale)
#                 else:
#                     imgArray[x][y] = cv2.resize(
#                         imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
#                 if len(imgArray[x][y].shape) == 2:
#                     imgArray[x][y] = cv2.cvtColor(
#                         imgArray[x][y], cv2.COLOR_GRAY2BGR)
#         imageBlank = np.zeros((height, width, 3), np.uint8)
#         hor = [imageBlank]*rows
#         hor_con = [imageBlank]*rows
#         for x in range(0, rows):
#             hor[x] = np.hstack(imgArray[x])
#         ver = np.vstack(hor)
#     else:
#         for x in range(0, rows):
#             if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
#                 imgArray[x] = cv2.resize(
#                     imgArray[x], (0, 0), None, scale, scale)
#             else:
#                 imgArray[x] = cv2.resize(
#                     imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
#             if len(imgArray[x].shape) == 2:
#                 imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
#         hor = np.hstack(imgArray)
#         ver = hor
#     return ver
 

# def isEndOfLane(results, model, frameWarp, display=True):
#     global waktupid, start_detection_time, ujung_jalur_detected, turn_movement, straight_movement, last_detected_movement, detected_jalur, detected_ujung_jalur, last_detection_time, fps_value
#     names = model.names
#     detected_jalur = False
#     detected_ujung_jalur = False

#     for r in results:
#         for classes in r.boxes.cls:
#             clss = names[int(classes)]
#             print("class yg dideteksi: ", clss)

#             if clss == 'jalur':
#                 detected_jalur = True
#             elif clss == 'ujung-jalur':
#                 detected_ujung_jalur = True

#     current_time = time.time()

#     if waktupid is None:
#         waktupid = current_time

#     elapsed_time = round(current_time - waktupid, 2)


#     if detected_jalur and not detected_ujung_jalur:
#         print('hanya jalur')
#         frameWarpCopy = frameWarp.copy()
        
#         _, thresholded = cv2.threshold(frameWarpCopy, 200, 255, cv2.THRESH_BINARY)
        
#         bottom_area_start = 3 * frameWarpCopy.shape[0] // 4
#         bottom_area_end = frameWarpCopy.shape[0]

#         white_columns = np.where(thresholded[bottom_area_start:bottom_area_end, :] == 255)[1]

#         if len(white_columns) > 0:
#             centerline_x = int(np.mean(white_columns))
#             left_sum = np.sum(thresholded[:, :centerline_x], dtype=np.int64)
#             right_sum = np.sum(thresholded[:, centerline_x:], dtype=np.int64)

#             errorJalur = round((right_sum - left_sum) / (right_sum + left_sum) * 100, 2)
#             print("errorJalur: ", errorJalur)

#             # Hitung kesalahan posisi
#             height, width = frameWarpCopy.shape[:2]
#             x_pot = int(width / 2)
#             y_pot = int(height*0.875)

#             # errorFrame = centerline_x - x_pot
#             errorFrame = round((centerline_x - x_pot)*0.4, 2)
#             print("errorFrame: ", errorFrame)

#             # Gabungkan kesalahan posisi dengan kesalahan perbandingan jumlah piksel
#             errorPos = round(errorJalur + errorFrame, 2)
#             print("errorPos: ", errorPos)
            
#             kirimData(esp_ser, errorPos, None, None)

#             # Save error to CSV
#             with open('errors.csv', mode='a', newline='') as file:
#                 writer = csv.writer(file)
#                 writer.writerow([elapsed_time, errorPos])

#             # Simpan histori pergerakan terakhir
#             if turn_movement or straight_movement:
#                 last_detected_movement = (turn_movement, straight_movement)

#             if display:
#                 frameWarpCopy_bgr = cv2.cvtColor(frameWarpCopy, cv2.COLOR_GRAY2BGR)
#                 frameWarpCopy_bgr[thresholded == 255] = [255, 255, 0]
                

#                 cv2.circle(frameWarpCopy_bgr, (x_pot, y_pot), 10, (0, 0, 255), -1)
#                 # cv2.line(frameWarpCopy_bgr, (x_pot, y_pot), (x_pot, 0), (0, 0, 255), 2)

#                 area_height = frameWarpCopy.shape[0] // 4

#                 # # Area 1 (atas)
#                 # cv2.rectangle(frameWarpCopy_bgr, (0, 0), (frameWarpCopy.shape[1], area_height), (255, 0, 0), 2)

#                 # # Area 2 (kedua dari atas)
#                 # cv2.rectangle(frameWarpCopy_bgr, (0, area_height), (frameWarpCopy.shape[1], 2 * area_height), (0, 255, 0), 2)

#                 # # Area 3 (kedua dari bawah)
#                 # cv2.rectangle(frameWarpCopy_bgr, (0, 2 * area_height), (frameWarpCopy.shape[1], 3 * area_height), (255, 0, 255), 2)

#                 # Area 4 (bawah)
#                 cv2.rectangle(frameWarpCopy_bgr, (0, 3 * area_height), (frameWarpCopy.shape[1], frameWarpCopy.shape[0]), (0, 0, 255), 2)

#                 # Tambahkan titik hijau yang menandai titik tengah pada area yang dihitung (paling bawah)
#                 center_x = centerline_x
#                 center_y = (bottom_area_start + bottom_area_end) // 2
#                 cv2.circle(frameWarpCopy_bgr, (center_x, center_y), 10, (0, 255, 255), -1)

#                 cv2.line(frameWarpCopy_bgr, (x_pot, y_pot), (center_x, center_y), (0, 0, 255), 2)

#                 # Tarik garis dari titik hijau ke sisi atas gambar (panduan saja)
#                 cv2.line(frameWarpCopy_bgr, (center_x, center_y), (center_x, 0), (255, 0, 0), 2)

#                 if -10 <= errorPos <= 10:
#                     text = "Maju Lurus"
#                     straight_movement = "Lurus"
#                 elif -25 <= errorPos < -10:
#                     text = "Belok Kiri"
#                 elif 10 < errorPos <= 25:
#                     text = "Belok Kanan"
#                 elif errorPos < -25:
#                     turn_movement = "KiriTajam"
#                     text = "Kiri Tajam"
#                 elif errorPos > 25:
#                     text = "Kanan Tajam"
#                     turn_movement = "KananTajam"

#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 org = (10, 30)
#                 font_scale = 1
#                 font_color = (255, 0, 255)
#                 font_thickness = 2

#                 org_error = (10, 60)
#                 text_error = f"Error: {errorPos}"

#                 current_time = time.time()
#                 fps = 1 / (current_time - isEndOfLane.last_time) if hasattr(isEndOfLane, 'last_time') else 0
#                 isEndOfLane.last_time = current_time

#                 fps_value = fps

#                 org_fps = (300, 30)
#                 text_fps = f"FPS: {fps:.2f}"
                
#                 cv2.putText(frameWarpCopy_bgr, text, org, font, font_scale, font_color, font_thickness)
#                 cv2.putText(frameWarpCopy_bgr, text_error, org_error, font, font_scale, font_color, font_thickness)
#                 cv2.putText(frameWarpCopy_bgr, text_fps, org_fps, font, font_scale, font_color, font_thickness)

#                 if turn_movement and straight_movement:
#                     print("INPO: ['{}', '{}']".format(turn_movement, straight_movement))

#             return frameWarpCopy_bgr

#     elif detected_jalur and detected_ujung_jalur:
#         print('jalur dan uj')
#         UJ = "MD\n"
#         kirimData(esp_ser, None, None, UJ)

#         start_detection_time = None
#         ujung_jalur_detected = False
#         last_detection_time = None

#         return

#     elif detected_ujung_jalur and not detected_jalur:
#         if not ujung_jalur_detected:
#             start_detection_time = current_time
#             ujung_jalur_detected = True
#             last_detection_time = current_time
#             print("Deteksi awal ujung jalur, start_detection_time diatur ke:", start_detection_time)
#         else:
#             # Jika deteksi ujung jalur telah hilang sebelumnya, reset waktunya
#             if last_detection_time and (current_time - last_detection_time > 0.5):  # Anggap 0.5 detik sebagai durasi untuk reset deteksi
#                 start_detection_time = current_time
#                 print("Deteksi ujung jalur di-reset, start_detection_time diatur ke:", start_detection_time)

#             last_detection_time = current_time

#             UJ = "MD\n"
#             kirimData(esp_ser, None, None, UJ)
#             selisih_waktu = current_time - start_detection_time
#             print("Selisih waktu: ", selisih_waktu)
#             if selisih_waktu > 1.5:
#                 if last_detected_movement:
#                     print("Info 2 last movement: ['{}', '{}']".format(last_detected_movement[0], last_detected_movement[1]))
#                     kirimData(esp_ser, None, last_detected_movement, None)

#                     turn_movement = None
#                     straight_movement = None
#                     last_detected_movement = None
#                     ujung_jalur_detected = False
#                     start_detection_time = None
#                     last_detection_time = None
#             else:
#                 print("Selisih waktu belum mencapai 2,2 detik. Tidak mengirim data last detected movement.")


# def kirimData(esp_ser, errorPos, last_detected_movement, UJ):
#     global detected_jalur, detected_ujung_jalur
    
#     # Prioritas pertama: kirim UJ jika tersedia
#     if UJ:
#         data_to_send = UJ
#         print("Mengirim data ke ESP32: {}".format(data_to_send))
#         if esp_ser.is_open:
#             esp_ser.write(data_to_send.encode())
#             print("Data berhasil dikirim ke ESP32.")
#         else:
#             print("Gagal mengirim data. Port serial tidak terbuka.")
#         return

#     # Prioritas kedua: kirim last_detected_movement jika tersedia
#     if last_detected_movement:
#         if last_detected_movement[0] == "KananTajam" and last_detected_movement[1] == "Lurus":
#             data_to_send = "KI\n"
#         elif last_detected_movement[0] == "KiriTajam" and last_detected_movement[1] == "Lurus":
#             data_to_send = "KA\n"
#         else:
#             data_to_send = None

#         if data_to_send:
#             print("Mengirim data ke ESP32: {}".format(data_to_send))
#             if esp_ser.is_open:
#                 esp_ser.write(data_to_send.encode())
#                 print("Data berhasil dikirim ke ESP32.")
#             else:
#                 print("Gagal mengirim data. Port serial tidak terbuka.")
#             return

#     # Prioritas terakhir: kirim error jika tidak ada UJ dan last_detected_movement yang valid
#     if not UJ and not last_detected_movement and detected_jalur and not detected_ujung_jalur:
#         data_to_send = "error_{}\n".format(errorPos)
#         print("Mengirim data ke ESP32: {}".format(data_to_send))
#         if esp_ser.is_open:
#             esp_ser.write(data_to_send.encode())
#             print("Data berhasil dikirim ke ESP32.")
#         else:
#             print("Gagal mengirim data. Port serial tidak terbuka.")

import cv2
import numpy as np
import torch
import time 
import serial 
import csv
import config

# esp_ser = serial.Serial(
#     port = '/dev/ttyUSB0',
#     baudrate = 115200,
#     bytesize=serial.EIGHTBITS,
#     parity=serial.PARITY_NONE,
#     stopbits=serial.STOPBITS_ONE,
#     timeout=0.1,
#     # timeout=1,
# )

start_detection_time = None
ujung_jalur_detected = False

turn_movement = None  
straight_movement = None

last_detected_movement = None

detected_jalur = False
detected_ujung_jalur = False

elapsed_time_global = 0

waktupid = None
fps_value = 0.0

def tambahkan_bingkai(video, lebar_bingkai_vertikal=128, tinggi_bingkai_horizontal=64):
    h, w, _ = video.shape

    # Mengatur ulang ukuran video menjadi lebih kecil
    new_h = h - 2 * tinggi_bingkai_horizontal
    new_w = w - 2 * lebar_bingkai_vertikal
    video_kecil = cv2.resize(video, (new_w, new_h))

    # Membuat bingkai vertikal (sisi kiri dan kanan)
    bingkai_vertikal = np.zeros((new_h, lebar_bingkai_vertikal, 3), dtype=np.uint8)
    bingkai_vertikal[:] = (0, 0, 0)  # Set warna bingkai ke hitam
    
    # Membuat bingkai horizontal (sisi atas dan bawah)
    bingkai_horizontal = np.zeros((tinggi_bingkai_horizontal, new_w + 2 * lebar_bingkai_vertikal, 3), dtype=np.uint8)
    bingkai_horizontal[:] = (0, 0, 0)  # Set warna bingkai ke hitam

    # Menambahkan bingkai di sisi kiri dan kanan
    video_dengan_bingkai_vertikal = np.concatenate((bingkai_vertikal, video_kecil, bingkai_vertikal), axis=1)

    # Menambahkan bingkai di sisi atas dan bawah
    frame_dengan_bingkai = np.concatenate((bingkai_horizontal, video_dengan_bingkai_vertikal, bingkai_horizontal), axis=0)
    
    return frame_dengan_bingkai


# def masking(results):
#     all_masks = None

#     for result in results:
#         if len(result) < 1:
#             continue

#         masks = result.masks.data
#         boxes = result.boxes.data
#         clss = boxes[:, 5]

#         for class_id in range(int(torch.max(clss).item()) + 1):
#             class_indices = torch.where(clss == class_id)
#             class_masks = masks[class_indices]
#             class_mask = torch.any(class_masks, dim=0).int()

#             if all_masks is None:
#                 all_masks = torch.zeros_like(class_mask)

#             all_masks += class_mask


#     if all_masks is None:
#         return None

#     all_masks *= 255
#     all_masks = all_masks.to('cpu')  # Memindahkan tensor ke CPU
#     mask_np = all_masks.numpy().astype('uint8')
#     mask_np = cv2.resize(mask_np, (512, 256))
#     # mask_np = cv2.resize(mask_np, (480, 240))

#     return mask_np

def masking(results, frame):
    all_masks = None

    for result in results:
        if len(result) < 1:
            continue
        
        for result in results:
            if len(result) < 1:
                continue

            for ci, c in enumerate(result):
                label = c.names[c.boxes.cls.tolist().pop()]

                mask_np = np.zeros(frame.shape[:2], np.uint8)

                # Create contour mask 
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                _ = cv2.drawContours(mask_np, [contour], -1, (255, 255, 255), cv2.FILLED)

                # cv2.imshow("Masking", mask_np)
                return mask_np


def warping(frame, points, w, h, inv=False):
    points1 = np.float32(points)
    points2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(points2, points1)
    else:
        matrix = cv2.getPerspectiveTransform(points1, points2)

    frameWarp = cv2.warpPerspective(frame, matrix, (w, h))
    return frameWarp


def nothing(a):
    pass


def initializeTrackbars(intialTracbarVals, wT=512, hT=256):
# def initializeTrackbars(intialTracbarVals, wT=480, hT=240):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 512, 256)
    # cv2.resizeWindow("Trackbars", 480, 240)
    cv2.createTrackbar("Width Top", "Trackbars",
                       intialTracbarVals[0], wT//2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars",
                       intialTracbarVals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars",
                       intialTracbarVals[2], wT//2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars",
                       intialTracbarVals[3], hT, nothing)


def valTrackbars(wT=512, hT=256):
# def valTrackbars(wT=480, hT=240):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop),
                         (widthBottom, heightBottom), (wT-widthBottom, heightBottom)])
    return points


def drawPoints(frame, points):
    for x in range(4):
        cv2.circle(frame, (int(points[x][0]), int(
            points[x][1])), 8, (0, 0, 255), cv2.FILLED)
    return frame
    

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(
                        imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(
                    imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(
                    imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver
 

def isEndOfLane(results, model, frameWarp, display=True):
    global waktupid, start_detection_time, ujung_jalur_detected, turn_movement, straight_movement, last_detected_movement, detected_jalur, detected_ujung_jalur, last_detection_time, fps_value, elapsed_time_global
    names = model.names
    detected_jalur = False
    detected_ujung_jalur = False

    for r in results:
        for classes in r.boxes.cls:
            clss = names[int(classes)]
            print("class yg dideteksi: ", clss)

            if clss == 'jalur':
                detected_jalur = True
            elif clss == 'ujung-jalur':
                detected_ujung_jalur = True

    current_time = time.time()

    if waktupid is None:
        waktupid = current_time

    elapsed_time = round(current_time - waktupid, 2)
    elapsed_time_global = elapsed_time

    if detected_jalur and not detected_ujung_jalur:
        print('hanya jalur')
        frameWarpCopy = frameWarp.copy()
        
        _, thresholded = cv2.threshold(frameWarpCopy, 200, 255, cv2.THRESH_BINARY)
        
        bottom_area_start = 3 * frameWarpCopy.shape[0] // 4
        bottom_area_end = frameWarpCopy.shape[0]

        white_columns = np.where(thresholded[bottom_area_start:bottom_area_end, :] == 255)[1]

        if len(white_columns) > 0:
            centerline_x = int(np.mean(white_columns))
            left_sum = np.sum(thresholded[:, :centerline_x], dtype=np.int64)
            right_sum = np.sum(thresholded[:, centerline_x:], dtype=np.int64)

            errorJalur = round((right_sum - left_sum) / (right_sum + left_sum) * 100, 2)
            print("errorJalur: ", errorJalur)

            # Hitung kesalahan posisi
            height, width = frameWarpCopy.shape[:2]
            x_pot = int(width / 2)
            y_pot = int(height*0.875)

            # errorFrame = centerline_x - x_pot
            errorFrame = round((centerline_x - x_pot)*0.4, 2)
            print("errorFrame: ", errorFrame)

            # Gabungkan kesalahan posisi dengan kesalahan perbandingan jumlah piksel
            errorPos = round(errorJalur + errorFrame, 2)
            print("errorPos: ", errorPos)

            config.errorPos = errorPos
            
            # kirimData(esp_ser, errorPos, None, None)

            # Save error to CSV
            fields = ['waktu', 'error']
            data_row = ['', '']
            data_row[0] = elapsed_time  # Simpan waktu detik
            data_row[1] = errorPos

            # with open('errors.csv', mode='a', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow([elapsed_time, errorPos])

            with open('errors.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                # Tulis header jika file masih kosong
                if file.tell() == 0:
                    writer.writerow(fields)
                # Tulis baris data
                writer.writerow(data_row)

            # Simpan histori pergerakan terakhir
            if turn_movement or straight_movement:
                last_detected_movement = (turn_movement, straight_movement)

            if display:
                frameWarpCopy_bgr = cv2.cvtColor(frameWarpCopy, cv2.COLOR_GRAY2BGR)
                frameWarpCopy_bgr[thresholded == 255] = [255, 255, 0]
                

                cv2.circle(frameWarpCopy_bgr, (x_pot, y_pot), 10, (0, 0, 255), -1)
                # cv2.line(frameWarpCopy_bgr, (x_pot, y_pot), (x_pot, 0), (0, 0, 255), 2)

                area_height = frameWarpCopy.shape[0] // 4

                # # Area 1 (atas)
                # cv2.rectangle(frameWarpCopy_bgr, (0, 0), (frameWarpCopy.shape[1], area_height), (255, 0, 0), 2)

                # # Area 2 (kedua dari atas)
                # cv2.rectangle(frameWarpCopy_bgr, (0, area_height), (frameWarpCopy.shape[1], 2 * area_height), (0, 255, 0), 2)

                # # Area 3 (kedua dari bawah)
                # cv2.rectangle(frameWarpCopy_bgr, (0, 2 * area_height), (frameWarpCopy.shape[1], 3 * area_height), (255, 0, 255), 2)

                # Area 4 (bawah)
                cv2.rectangle(frameWarpCopy_bgr, (0, 3 * area_height), (frameWarpCopy.shape[1], frameWarpCopy.shape[0]), (0, 0, 255), 2)

                # Tambahkan titik hijau yang menandai titik tengah pada area yang dihitung (paling bawah)
                center_x = centerline_x
                center_y = (bottom_area_start + bottom_area_end) // 2
                cv2.circle(frameWarpCopy_bgr, (center_x, center_y), 10, (0, 255, 255), -1)

                cv2.line(frameWarpCopy_bgr, (x_pot, y_pot), (center_x, center_y), (0, 0, 255), 2)

                # Tarik garis dari titik hijau ke sisi atas gambar (panduan saja)
                cv2.line(frameWarpCopy_bgr, (center_x, center_y), (center_x, 0), (255, 0, 0), 2)

                if -10 <= errorPos <= 10:
                    text = "Maju Lurus"
                    straight_movement = "Lurus"
                elif -25 <= errorPos < -10:
                    text = "Belok Kiri"
                elif 10 < errorPos <= 25:
                    text = "Belok Kanan"
                elif errorPos < -25:
                    turn_movement = "KiriTajam"
                    text = "Kiri Tajam"
                elif errorPos > 25:
                    text = "Kanan Tajam"
                    turn_movement = "KananTajam"

                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (10, 30)
                font_scale = 1
                font_color = (255, 0, 255)
                font_thickness = 2

                org_error = (10, 60)
                text_error = f"Error: {errorPos}"

                # current_time = time.time()
                # fps = 1 / (current_time - isEndOfLane.last_time) if hasattr(isEndOfLane, 'last_time') else 0
                # isEndOfLane.last_time = current_time

                # fps_value = fps

                fps = f"FPS: {config.fps}"

                org_fps = (300, 30)
                # text_fps = f"FPS: {fps:.2f}"
                text_fps = f"{fps}"
                print("fps utils: ", fps)
                cv2.putText(frameWarpCopy_bgr, text, org, font, font_scale, font_color, font_thickness)
                cv2.putText(frameWarpCopy_bgr, text_error, org_error, font, font_scale, font_color, font_thickness)
                cv2.putText(frameWarpCopy_bgr, text_fps, org_fps, font, font_scale, font_color, font_thickness)

                if turn_movement and straight_movement:
                    print("INPO: ['{}', '{}']".format(turn_movement, straight_movement))

            return frameWarpCopy_bgr

    elif detected_jalur and detected_ujung_jalur:
        print('jalur dan uj')
        UJ = "MD\n"
        # kirimData(esp_ser, None, None, UJ)

        start_detection_time = None
        ujung_jalur_detected = False
        last_detection_time = None

        return

    elif detected_ujung_jalur and not detected_jalur:
        if not ujung_jalur_detected:
            start_detection_time = current_time
            ujung_jalur_detected = True
            last_detection_time = current_time
            print("Deteksi awal ujung jalur, start_detection_time diatur ke:", start_detection_time)
        else:
            # Jika deteksi ujung jalur telah hilang sebelumnya, reset waktunya
            if last_detection_time and (current_time - last_detection_time > 0.5):  # Anggap 0.5 detik sebagai durasi untuk reset deteksi
                start_detection_time = current_time
                print("Deteksi ujung jalur di-reset, start_detection_time diatur ke:", start_detection_time)

            last_detection_time = current_time

            UJ = "MD\n"
            # kirimData(esp_ser, None, None, UJ)
            selisih_waktu = current_time - start_detection_time
            print("Selisih waktu: ", selisih_waktu)
            if selisih_waktu > 1.5:
                if last_detected_movement:
                    print("Info 2 last movement: ['{}', '{}']".format(last_detected_movement[0], last_detected_movement[1]))
                    # kirimData(esp_ser, None, last_detected_movement, None)

                    turn_movement = None
                    straight_movement = None
                    last_detected_movement = None
                    ujung_jalur_detected = False
                    start_detection_time = None
                    last_detection_time = None
            else:
                print("Selisih waktu belum mencapai 2,2 detik. Tidak mengirim data last detected movement.")

def get_elapsed_time():
    return elapsed_time_global

def kirimData(esp_ser, errorPos, last_detected_movement, UJ):
    global detected_jalur, detected_ujung_jalur
    
    # Prioritas pertama: kirim UJ jika tersedia
    if UJ:
        data_to_send = UJ
        print("Mengirim data ke ESP32: {}".format(data_to_send))
        if esp_ser.is_open:
            esp_ser.write(data_to_send.encode())
            print("Data berhasil dikirim ke ESP32.")
        else:
            print("Gagal mengirim data. Port serial tidak terbuka.")
        return

    # Prioritas kedua: kirim last_detected_movement jika tersedia
    if last_detected_movement:
        if last_detected_movement[0] == "KananTajam" and last_detected_movement[1] == "Lurus":
            data_to_send = "KI\n"
        elif last_detected_movement[0] == "KiriTajam" and last_detected_movement[1] == "Lurus":
            data_to_send = "KA\n"
        else:
            data_to_send = None

        if data_to_send:
            print("Mengirim data ke ESP32: {}".format(data_to_send))
            if esp_ser.is_open:
                esp_ser.write(data_to_send.encode())
                print("Data berhasil dikirim ke ESP32.")
            else:
                print("Gagal mengirim data. Port serial tidak terbuka.")
            return

    # Prioritas terakhir: kirim error jika tidak ada UJ dan last_detected_movement yang valid
    if not UJ and not last_detected_movement and detected_jalur and not detected_ujung_jalur:
        data_to_send = "error_{}\n".format(errorPos)
        print("Mengirim data ke ESP32: {}".format(data_to_send))
        if esp_ser.is_open:
            esp_ser.write(data_to_send.encode())
            print("Data berhasil dikirim ke ESP32.")
        else:
            print("Gagal mengirim data. Port serial tidak terbuka.")

