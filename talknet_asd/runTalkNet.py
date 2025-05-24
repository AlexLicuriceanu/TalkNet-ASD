import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, pdb, math, python_speech_features
from pathlib import Path
from typing import Optional, Dict, Any

from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
from scenedetect import open_video
from scenedetect.scene_manager import SceneManager
from scenedetect.detectors import ContentDetector
import os, pickle

from .model.faceDetector.s3fd import S3FD
from .talkNet import talkNet
from .utils import *
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# Get the directory where this file is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)
PATH_PRETRAINED = os.path.join(MODEL_DIR, 'pretrain_TalkSet.model')

s = talkNet()
if os.path.isfile(PATH_PRETRAINED) == False:
    Link = "1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt"
    cmd = "gdown --id %s -O %s"%(Link, PATH_PRETRAINED)
    subprocess.call(cmd, shell=True, stdout=None)

s.loadParameters(PATH_PRETRAINED)
sys.stderr.write("Loaded pretrained TalkNet model\n")

def run_talknet(
    video_path: str,
    output_dir: Optional[str] = None,
    n_data_loader_thread: int = 10,
    facedet_scale: float = 0.25,
    min_track: int = 10,
    num_failed_det: int = 10,
    min_face_size: int = 1,
    crop_scale: float = 0.40,
    start_time: int = 0,
    duration: int = 0,
    debug: bool = False,
    batch_size: int = 16,
    visualize: bool = False,
):
    
    # Create a class to hold arguments
    class Args:
        def __init__(self):
            self.videoPath = video_path
            self.nDataLoaderThread = n_data_loader_thread
            self.facedetScale = facedet_scale
            self.minTrack = min_track
            self.numFailedDet = num_failed_det
            self.minFaceSize = min_face_size
            self.cropScale = crop_scale
            self.start = start_time
            self.duration = duration
            self.debug = debug
            self.batchSize = batch_size
            self.visualize = visualize
            
            # Set up paths
            if output_dir is None:
                self.savePath = str(Path(video_path).parent / Path(video_path).stem)
            else:
                self.savePath = output_dir
                
            self.pyaviPath = os.path.join(self.savePath, 'pyavi')
            self.pyframesPath = os.path.join(self.savePath, 'pyframes')
            self.pyworkPath = os.path.join(self.savePath, 'pywork')
            self.pycropPath = os.path.join(self.savePath, 'pycrop')
            
            # Video and audio paths
            self.videoFilePath = os.path.join(self.pyaviPath, 'video.avi')
            self.audioFilePath = os.path.join(self.pyaviPath, 'audio.wav')

    args = Args()

    # Create output directories
    if os.path.exists(args.savePath):
        rmtree(args.savePath)
    os.makedirs(args.pyaviPath, exist_ok=True)
    os.makedirs(args.pyframesPath, exist_ok=True)
    os.makedirs(args.pyworkPath, exist_ok=True)
    os.makedirs(args.pycropPath, exist_ok=True)

    # Extract video
    if args.duration == 0:
        command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" % 
                  (args.videoPath, args.nDataLoaderThread, args.videoFilePath))
    else:
        command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic" % 
                  (args.videoPath, args.nDataLoaderThread, args.start, args.start + args.duration, args.videoFilePath))
    subprocess.call(command, shell=True, stdout=None)
    
    if args.debug:
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the video and save in %s \r\n" %(args.videoFilePath))

    # Extract audio
    command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % 
              (args.videoFilePath, args.nDataLoaderThread, args.audioFilePath))
    subprocess.call(command, shell=True, stdout=None)
    
    if args.debug:
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" %(args.audioFilePath))

    # Extract video frames
    command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" % 
              (args.videoFilePath, args.nDataLoaderThread, os.path.join(args.pyframesPath, '%06d.jpg')))
    subprocess.call(command, shell=True, stdout=None)
    if args.debug:
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the frames and save in %s \r\n" %(args.pyframesPath))

    # Scene detection
    scene = scene_detect(args)
    if args.debug:
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scene detection and save in %s \r\n" %(args.pyworkPath))

    # Face detection
    faces = inference_video_yolo(args)
    if args.debug:
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face detection and save in %s \r\n" %(args.pyworkPath))

    # Face tracking
    allTracks, vidTracks = [], []
    for shot in scene:
        if shot[1] - shot[0] >= args.minTrack:
            allTracks.extend(track_shot(args, faces[shot[0]:shot[1]]))
            
    if args.debug:
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face track and detected %d tracks \r\n" %len(allTracks))

    # Face clips cropping
    # for ii, track in tqdm.tqdm(enumerate(allTracks), total=len(allTracks)):
    #     vidTracks.append(crop_video(args, track, os.path.join(args.pycropPath, '%05d'%ii)))

    # Preload all frames to memory
    flist = sorted(glob.glob(os.path.join(args.pyframesPath, '*.jpg')))
    frame_cache = {i: cv2.imread(f) for i, f in enumerate(flist)}

    # Load audio once
    sr, full_audio = wavfile.read(args.audioFilePath)

    # Parallel crop using ThreadPool
    from multiprocessing.pool import ThreadPool
    from functools import partial

    crop_fn = partial(crop_video, args=args, frame_cache=frame_cache, full_audio=full_audio, sr=sr)

    args_list = [(track, os.path.join(args.pycropPath, f"{ii:05d}")) for ii, track in enumerate(allTracks)]

    with ThreadPool(processes=os.cpu_count()) as pool:
        vidTracks = list(tqdm.tqdm(pool.starmap(crop_fn, args_list), total=len(args_list)))



    savePath = os.path.join(args.pyworkPath, 'tracks.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(vidTracks, fil)
        
    if args.debug:
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face Crop and saved in %s tracks \r\n" %args.pycropPath)

    # Active Speaker Detection
    files = glob.glob("%s/*.avi"%args.pycropPath)
    files.sort()
    scores = evaluate_network(files, args)
    savePath = os.path.join(args.pyworkPath, 'scores.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(scores, fil)
        
    if args.debug:
        sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scores extracted and saved in %s \r\n" %args.pyworkPath)

    if args.visualize:
        visualization(vidTracks, scores, args)

def scene_detect(args):
    # Open video
    video = open_video(args.videoFilePath)

    # Set up scene detection
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video)

    # Convert scenes to list of (start_frame, end_frame)
    scene_list_raw = scene_manager.get_scene_list()
    scene_list = [(s[0].get_frames(), s[1].get_frames()) for s in scene_list_raw]

    # Fallback: if no scenes detected, default to whole video range
    if not scene_list:
        total_frames = int(video.frame_rate * video.duration.get_seconds())
        scene_list = [(0, total_frames)]

    # Save to pickle
    save_path = os.path.join(args.pyworkPath, 'scene.pckl')
    with open(save_path, 'wb') as fil:
        pickle.dump(scene_list, fil)

    if args.debug:
        sys.stderr.write(f"{args.videoFilePath} - scenes detected: {len(scene_list)}\n")

    return scene_list

def inference_video(args):
    # GPU: Face detection, output is the list contains the face location and score in this frame
    DET = S3FD(device='cuda')
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    dets = []
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale])
        dets.append([])
        for bbox in bboxes:
            dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info
        
        if args.debug:
            sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))
    savePath = os.path.join(args.pyworkPath,'faces.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(dets, fil)
    return dets



def inference_video_yolo(args):
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'yolov11n-face.pt')
    model = YOLO(model_path).to('cuda')

    flist = sorted(glob.glob(os.path.join(args.pyframesPath, '*.jpg')))
    dets = []

    batch_size = args.batchSize

    for i in range(0, len(flist), batch_size):
        batch_files = flist[i:i + batch_size]
        batch_images = [cv2.imread(f) for f in batch_files]

        # Run YOLOv8 on the batch
        results = model.predict(batch_images, conf=0.5, verbose=False)

        for j, result in enumerate(results):
            frame_idx = i + j
            dets.append([])
            boxes = result.boxes.cpu().numpy()

            if boxes and hasattr(boxes, 'xyxy'):
                for bbox, conf in zip(boxes.xyxy, boxes.conf):
                    x1, y1, x2, y2 = bbox.tolist()
                    dets[-1].append({
                        'frame': frame_idx,
                        'bbox': [x1, y1, x2, y2],
                        'conf': float(conf)
                    })

            if args.debug:
                sys.stderr.write(f'{args.videoFilePath}-{frame_idx:05d}; {len(dets[-1])} dets\r')

    # Save detections
    savePath = os.path.join(args.pyworkPath, 'faces.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(dets, fil)

    return dets


def bb_intersection_over_union(boxA, boxB, evalCol = False):
    # CPU: IOU Function to calculate overlap between two image
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if evalCol == True:
        iou = interArea / float(boxAArea)
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def track_shot(args, sceneFaces):
    # CPU: Face tracking
    iouThres  = 0.5     # Minimum IOU between consecutive face detections
    tracks    = []
    while True:
        track     = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > args.minTrack:
            frameNum    = numpy.array([ f['frame'] for f in track ])
            bboxes      = numpy.array([numpy.array(f['bbox']) for f in track])
            frameI      = numpy.arange(frameNum[0],frameNum[-1]+1)
            bboxesI    = []
            for ij in range(0,4):
                interpfn  = interp1d(frameNum, bboxes[:,ij])
                bboxesI.append(interpfn(frameI))
            bboxesI  = numpy.stack(bboxesI, axis=1)
            if max(numpy.mean(bboxesI[:,2]-bboxesI[:,0]), numpy.mean(bboxesI[:,3]-bboxesI[:,1])) > args.minFaceSize:
                tracks.append({'frame':frameI,'bbox':bboxesI})
    return tracks

# def crop_video(args, track, cropFile):
#     # CPU: crop the face clips
#     flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) # Read the frames
#     flist.sort()
#     vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224,224))# Write video
#     dets = {'x':[], 'y':[], 's':[]}
#     for det in track['bbox']: # Read the tracks
#         dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
#         dets['y'].append((det[1]+det[3])/2) # crop center x 
#         dets['x'].append((det[0]+det[2])/2) # crop center y
#     dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
#     dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
#     dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
#     for fidx, frame in enumerate(track['frame']):
#         cs  = args.cropScale
#         bs  = dets['s'][fidx]   # Detection box size
#         bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
#         image = cv2.imread(flist[frame])
#         frame = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
#         my  = dets['y'][fidx] + bsi  # BBox center Y
#         mx  = dets['x'][fidx] + bsi  # BBox center X
#         face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
#         vOut.write(cv2.resize(face, (224, 224)))
#     audioTmp    = cropFile + '.wav'
#     audioStart  = (track['frame'][0]) / 25
#     audioEnd    = (track['frame'][-1]+1) / 25
#     vOut.release()
#     command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
#               (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp)) 
#     output = subprocess.call(command, shell=True, stdout=None) # Crop audio file
#     _, audio = wavfile.read(audioTmp)
#     command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
#               (cropFile, audioTmp, args.nDataLoaderThread, cropFile)) # Combine audio and video file
#     output = subprocess.call(command, shell=True, stdout=None)
#     os.remove(cropFile + 't.avi')
#     return {'track':track, 'proc_track':dets}

def crop_video(track, cropFile, args, frame_cache, full_audio, sr):
    vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224, 224))
    dets = {'x': [], 'y': [], 's': []}

    for det in track['bbox']:
        dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets['y'].append((det[1] + det[3]) / 2)
        dets['x'].append((det[0] + det[2]) / 2)

    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)

    for fidx, frame in enumerate(track['frame']):
        cs = args.cropScale
        bs = dets['s'][fidx]
        bsi = int(bs * (1 + 2 * cs))

        image = frame_cache[frame]
        padded = numpy.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=110)

        my = dets['y'][fidx] + bsi
        mx = dets['x'][fidx] + bsi
        face = padded[int(my - bs):int(my + bs * (1 + 2 * cs)),
                      int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
        vOut.write(cv2.resize(face, (224, 224)))

    vOut.release()

    # Slice audio instead of calling ffmpeg
    audio_start_idx = int((track['frame'][0] / 25) * sr)
    audio_end_idx = int(((track['frame'][-1] + 1) / 25) * sr)
    track_audio = full_audio[audio_start_idx:audio_end_idx]

    audioTmp = cropFile + '.wav'
    wavfile.write(audioTmp, sr, track_audio)

    # Combine video and audio
    command = (
        f"ffmpeg -y -i {cropFile}t.avi -i {audioTmp} -threads {args.nDataLoaderThread} "
        f"-c:v copy -c:a copy {cropFile}.avi -loglevel panic"
    )
    subprocess.call(command, shell=True, stdout=None)
    os.remove(cropFile + 't.avi')

    return {'track': track, 'proc_track': dets}


def extract_MFCC(file, outPath):
    # CPU: extract mfcc
    sr, audio = wavfile.read(file)
    mfcc = python_speech_features.mfcc(audio,sr) # (N_frames, 13)   [1s = 100 frames]
    featuresPath = os.path.join(outPath, file.split('/')[-1].replace('.wav', '.npy'))
    numpy.save(featuresPath, mfcc)

def evaluate_network(files, args):
    # GPU: active speaker detection by pretrained TalkNet
    s.eval()
    allScores = []
    # durationSet = {1,2,4,6} # To make the result more reliable
    # durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
    durationSet = {2, 3, 4}
    for file in tqdm.tqdm(files, total = len(files)):
        fileName = os.path.splitext(file.split('/')[-1])[0] # Load audio and video
        _, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
        audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
        video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + '.avi'))
        videoFeature = []
        while video.isOpened():
            ret, frames = video.read()
            if ret == True:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224,224))
                face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
                videoFeature.append(face)
            else:
                break
        video.release()
        videoFeature = numpy.array(videoFeature)
        length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0] / 25)
        audioFeature = audioFeature[:int(round(length * 100)),:]
        videoFeature = videoFeature[:int(round(length * 25)),:,:]
        allScore = [] # Evaluation use TalkNet
        for duration in durationSet:
            batchSize = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batchSize):
                    inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cuda()
                    inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda()
                    embedA = s.model.forward_audio_frontend(inputA)
                    embedV = s.model.forward_visual_frontend(inputV)	
                    embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                    out = s.model.forward_audio_visual_backend(embedA, embedV)
                    score = s.lossAV.forward(out, labels = None)
                    scores.extend(score)
            allScore.append(scores)
        allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
        allScores.append(allScore)	
    return allScores

def visualization(tracks, scores, args):
    # CPU: visulize the result for video format
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    faces = [[] for i in range(len(flist))]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
            s = numpy.mean(s)
            faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
    firstImage = cv2.imread(flist[0])
    fw = firstImage.shape[1]
    fh = firstImage.shape[0]
    vOut = cv2.VideoWriter(os.path.join(args.pyaviPath, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), 25, (fw,fh))
    colorDict = {0: 0, 1: 255}
    for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
        image = cv2.imread(fname)
        for face in faces[fidx]:
            clr = colorDict[int((face['score'] >= 0))]
            txt = round(face['score'], 1)
            cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(face['x']+face['s']), int(face['y']+face['s'])),(0,clr,255-clr),10)
            cv2.putText(image,'%s'%(txt), (int(face['x']-face['s']), int(face['y']-face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)
        vOut.write(image)
    vOut.release()
    command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
        (os.path.join(args.pyaviPath, 'video_only.avi'), os.path.join(args.pyaviPath, 'audio.wav'), \
        args.nDataLoaderThread, os.path.join(args.pyaviPath,'video_out.avi'))) 
    output = subprocess.call(command, shell=True, stdout=None)

#if __name__ == '__main__':
#    run_talknet(video_path="/home/rhc/demo/001.avi", output_dir="/home/rhc/demo/output")
