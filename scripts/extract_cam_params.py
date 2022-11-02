import os
import logging
import subprocess
import sqlite3
import numpy as np
import cv2
import shutil
from numpy.lib import recfunctions as rf
from tqdm import tqdm
from cv2 import aruco
from glob import glob
from argparse import ArgumentParser

logging.getLogger().setLevel(logging.INFO)
    
MARKER_AREA_THRESHOLD = 1000 # Parameter
MAX_IMAGE_ID = 2**31 - 1

debug = True

def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2

def array_to_blob(array):
    return array.tobytes()

def blob_to_array(blob, dtype, shape=(-1,)):
    return np.frombuffer(blob, dtype=dtype).reshape(*shape)

def add_descriptors(self, image_id, descriptors):
    descriptors = np.ascontiguousarray(descriptors, np.uint8)
    self.execute(
        "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
        (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

def get_images(args, db_path):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("SELECT image_id,name FROM images;")
    image_ids, image_paths = zip(*cursor.fetchall())
    image_ids = list(image_ids)
    image_paths = [os.path.join(args.root_dir, "calib", image_path) for image_path in image_paths]
    cursor.close()
    connection.close()
    return image_ids, image_paths

def add_features(image_ids, image_paths, db_path):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)  # Using few bits per marker for better detection
    parameters = aruco.DetectorParameters_create()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001) # Sub-pixel detection

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("DELETE FROM keypoints;")
    cursor.execute("DELETE FROM descriptors;")
    cursor.execute("DELETE FROM matches;")
    connection.commit()

    logging.info("Extracting features from ChArUco")
    for image_id, image_path in tqdm(zip(image_ids, image_paths)):
        frame = cv2.imread(image_path)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        orig_corners, orig_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if len(orig_corners) <= 0:
            logging.warning("No markers found in {image_path}")
            continue
        else:
            corners = []
            ids = []
            for (corner, id) in zip(orig_corners, orig_ids):
                area = cv2.contourArea(corner)
                if area >= MARKER_AREA_THRESHOLD:  # PARAM
                    ids.append(id)
                    corners.append(corner)

            if len(orig_corners) - len(corners) > 0:
                logging.warning(f'Ignoring {len(orig_corners) - len(corners)} sliver markers.')

            ids = np.asarray(ids).flatten()

            for i in range(len(corners)):
                if np.all(corners[i] >= 0):
                    cv2.cornerSubPix(gray, corners[i], winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria)
                    corners[i] = corners[i].squeeze()
                else:
                    raise NotImplementedError

            # Insert keypoints
            keypoints = np.concatenate(corners)
            cursor.execute(
                "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
                (image_id,) + keypoints.shape + (array_to_blob(keypoints.astype(np.float32)),)
            )

            ids = np.repeat(ids, 4)
            for i in range(4):
                ids[i::4] += (i*12)

            ids = np.ascontiguousarray(np.tile(ids, (128, 1)).T, np.uint8)
            cursor.execute(
                "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
                (image_id,) + ids.shape + (array_to_blob(ids),)
            )

        connection.commit()

    cursor.close()
    connection.close()

def exhaustive_match(image_ids, image_paths, db_path, args):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("DELETE FROM matches;")
    connection.commit()

    logging.info("Matching features")
    image_pairs = []
    for i in tqdm(range(len(image_ids))):
        for j in range(len(image_ids)):
            if image_ids[i] >= image_ids[j]:
                continue
            
            cursor.execute(f"SELECT data FROM descriptors WHERE image_id == {image_ids[i]};")
            desc1 = blob_to_array(cursor.fetchall()[0][0], np.uint8, (-1, 128))[:,0]
            cursor.execute(f"SELECT data FROM descriptors WHERE image_id == {image_ids[j]};")
            desc2 = blob_to_array(cursor.fetchall()[0][0], np.uint8, (-1, 128))[:,0]

            cursor.execute(f"SELECT data FROM keypoints WHERE image_id == {image_ids[i]};")
            cursor.execute(f"SELECT data FROM keypoints WHERE image_id == {image_ids[j]};")

            # Find matches
            matches = []
            for k in range(desc1.shape[0]):
                for l in range(desc2.shape[0]):
                    if desc1[k] == desc2[l]:
                        matches.append([k, l])

            # Insert into database
            pair_id = image_ids_to_pair_id(image_ids[i], image_ids[j])
            if not matches:
                continue

            image_pairs.append([image_paths[i], image_paths[j]])
            matches = np.asarray(matches, np.uint32)
            
            cursor.execute("INSERT INTO matches VALUES (?, ?, ?, ?)",
                           (pair_id,) + matches.shape + (array_to_blob(matches),))

            connection.commit()

    cursor.close()
    connection.close()

    logging.info(f"Writing image pairs at {os.path.join(args.root_dir, 'match_list.txt')}")
    with open(os.path.join(args.root_dir, "match_list.txt"), "w") as f:
        for pair in image_pairs:
            f.write((" ".join(pair)).replace(os.path.join(args.root_dir, "calib") + "/", "") + "\n")

def main(args):
    db_path = os.path.join(args.root_dir, "db.db")
    image_path = os.path.join(args.root_dir, "calib")

    # Read intinsics
    with open(os.path.join(args.root_dir, "intrinsics.txt")) as f:
        intr = np.eye(3)
        for i in range(3):
            intr[i] = [float(k) for k in f.readline().split(',')]

    # Read distortion coefficients
    with open(os.path.join(args.root_dir, "distortion.txt")) as f:
        dist = np.asarray([float(k) for k in f.readlines()])

    if os.path.exists(db_path):
        logging.warning("Previous database found, deleting.")
        os.remove(db_path)
        try:
            os.remove(os.path.join(args.root_dir, "db.db-wal"))
            os.remove(os.path.join(args.root_dir, "db.db-shm"))
        except FileNotFoundError:
            pass

    cam_params = [intr[0,0], intr[1,1], intr[0,2], intr[1,2], *dist[:4]]
    cam_params = [str(param) for param in cam_params]
    logging.info("Importing images in database")
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", db_path,
        "--image_path", image_path,
        "--ImageReader.single_camera_per_folder", "1",
        "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.camera_params", ",".join(cam_params)]
    )

    image_ids, image_paths = get_images(args, db_path)
    add_features(image_ids, image_paths, db_path)
    exhaustive_match(image_ids, image_paths, db_path, args)

    logging.info("Performing geometric verification")
    subprocess.run([
        "colmap", "matches_importer",
        "--database_path", db_path,
        "--match_list_path", f"{os.path.join(args.root_dir, 'match_list.txt')}",
        "--match_type", "pairs",
        "--SiftMatching.min_num_inliers", "1"
    ])

    logging.info("Reconstructing")
    # TODO: Make this work cross platform
    tmp_path = "/tmp/colmap_reconstruction"
    shutil.rmtree(tmp_path, ignore_errors=True)
    os.makedirs(tmp_path)
    subprocess.run([
        "colmap", "mapper",
        "--database_path", db_path,
        "--image_path", image_path,
        "--output_path", tmp_path,
    ])

    logging.warning(f"Deleting previous reconstruction found at {args.out_dir}")
    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir)
    subprocess.run([
        "colmap", "model_converter",
        "--input_path", os.path.join(tmp_path, "0"),
        "--output_path", args.out_dir,
        "--output_type", "TXT"
    ])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-r", "--root_dir", required=True, help="Base directory")
    parser.add_argument("-o", "--out_dir", default="output", help="Output directory")

    args = parser.parse_args()

    # main(args)

    # Read images
    image_params = []
    with open(os.path.join(args.out_dir, "images.txt")) as f:
        skip_next = False
        for line in f.readlines():
            if skip_next:
                skip_next = False
                continue
            if line.startswith("#"):
                continue
            data = line.split()
            param = []
            param.append(int(data[8]))
            param.append(data[9].split("/")[0])
            param += [float(datum) for datum in data[1:8]]
            image_params.append(tuple(param))
            skip_next = True

    images = np.array(image_params, dtype=[
        ('cam_id', int), ('cam_name', '<U10'), 
        ('qvecw', float), ('qvecx', float), ('qvecy', float), ('qvecz', float),
        ('tvecx', float), ('tvecy', float), ('tvecz', float)
    ])

    # Read cameras
    cam_params = []
    with open(os.path.join(args.out_dir, "cameras.txt")) as f:
        for line in f.readlines():
            if line.startswith("#"):
                continue
            data = line.split()
            param = []
            param.append(int(data[0]))
            param.append(int(data[2]))
            param.append(int(data[3]))
            param += [float(datum) for datum in data[4:]]
            cam_params.append(tuple(param))
    cameras = np.array(cam_params, dtype=[
        ('cam_id', int),
        ('width', int), ('height', int),
        ('fx', float), ('fy', float),
        ('cx', float), ('cy', float),
        ('k1', float), ('k2', float),
        ('p1', float), ('p2', float),
    ])

    img_cams = rf.join_by('cam_id', cameras, images)
    np.savetxt(os.path.join(args.out_dir, "params.txt"), img_cams, fmt="%s", header=" ".join(img_cams.dtype.fields))

